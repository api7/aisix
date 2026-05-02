//! kind=bedrock guardrail dispatcher — calls AWS Bedrock's
//! `ApplyGuardrail` API on every chat request and translates the
//! response into a [`GuardrailVerdict`].
//!
//! PRD-09c §6 Phase 2. The cp-api side ships the
//! envelope-encrypted secret; cp-api decrypts at projection time
//! so this module only handles plaintext credentials. We never log
//! the secret.
//!
//! Behavior matrix (failure modes):
//!
//! | Bedrock response                | `fail_open` | Verdict                        |
//! |---------------------------------|-------------|--------------------------------|
//! | `action=NONE`                   | n/a         | Allow                          |
//! | `action=GUARDRAIL_INTERVENED`   | n/a         | Block { reason }               |
//! | 5xx / IO error                  | true        | Bypass { "bedrock_5xx" }       |
//! | 5xx / IO error                  | false       | Block { "bedrock unavailable" } |
//! | timeout (`latency_mode=timed`)  | true        | Bypass { "bedrock_timeout" }   |
//! | timeout (`latency_mode=timed`)  | false       | Block { "bedrock timeout" }    |
//! | throttle (4xx ThrottlingException) | true     | Bypass { "bedrock_throttled" } |
//! | throttle                        | false       | Block { "bedrock throttled" }  |
//!
//! `latency_mode=serial` waits unconditionally — the timeout row
//! never fires.

use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{
    BedrockAWSCredentials, BedrockConfig, BedrockLatencyMode, GuardrailHookPoint,
};
use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_credential_types::Credentials;
use aws_sdk_bedrockruntime::config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::operation::apply_guardrail::ApplyGuardrailError;
use aws_sdk_bedrockruntime::types::{
    GuardrailAction, GuardrailContentBlock, GuardrailContentSource, GuardrailTextBlock,
};
use aws_sdk_bedrockruntime::Client;
use aws_smithy_runtime_api::client::result::SdkError;
use aws_smithy_runtime_api::http::Response;

use crate::{Guardrail, GuardrailVerdict};

/// Deployment-wide override for the Bedrock endpoint URL. Read by
/// `BedrockGuardrail::new` (production code path); unset means the
/// SDK default (real AWS) is used.
///
/// Set this in `e2e/compose.yml` to point the DP at a local
/// fakecloud / LocalStack instance. Operators of self-hosted
/// stacks behind an outbound HTTP proxy can also use this to route
/// through their proxy.
///
/// Scoped to the env var name (not a per-guardrail config field)
/// because endpoint overrides are a deployment concern, not a
/// per-row configuration that a tenant should be able to set.
pub const BEDROCK_ENDPOINT_URL_ENV_VAR: &str = "AISIX_BEDROCK_ENDPOINT_URL";

/// One Bedrock guardrail row, materialised into a request-time
/// dispatcher. Built once per snapshot from
/// [`aisix_core::models::Guardrail`] + plaintext credentials.
pub struct BedrockGuardrail {
    /// Operator-facing row name. Kept for log labels; the trait's
    /// static `name()` returns "bedrock" so the metric cardinality
    /// stays bounded.
    pub row_name: String,
    pub guardrail_id: String,
    pub guardrail_version: String,
    pub hook_point: GuardrailHookPoint,
    pub latency_mode: BedrockLatencyMode,
    pub fail_open: bool,
    /// AWS SDK client, pre-configured with the row's region and
    /// static credentials. Wrapped in `Arc` so swapping snapshots
    /// doesn't drop a client mid-request.
    client: Arc<Client>,
}

impl BedrockGuardrail {
    /// Build the dispatcher from a parsed [`BedrockConfig`]. Caller
    /// owns the row's `name`, `hook_point`, and `fail_open` (they
    /// live on the outer Guardrail struct, not on the kind config).
    ///
    /// Synchronous on purpose: the snapshot rebuild path is sync (a
    /// blocking call from inside the etcd-watch supervisor's
    /// `arc_swap::store`), and `aws_config::defaults().load().await`
    /// is only async because of credential-source discovery (env,
    /// IMDS, file). With static credentials we have nothing to
    /// discover, so we compose `SdkConfig` directly via the builder.
    pub fn new(
        row_name: impl Into<String>,
        cfg: &BedrockConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
    ) -> Self {
        // Honor the deployment-wide endpoint override. This lets the
        // e2e suite point at fakecloud (or any AWS-compatible local
        // mock) without changing the cp-api wire shape, and lets a
        // self-hosted operator route Bedrock through a proxy. Empty
        // string is treated as unset so a `docker run -e
        // AISIX_BEDROCK_ENDPOINT_URL=` doesn't accidentally redirect.
        let endpoint_url = std::env::var(BEDROCK_ENDPOINT_URL_ENV_VAR)
            .ok()
            .filter(|s| !s.is_empty());
        if let Some(url) = endpoint_url.as_ref() {
            // Visible at INFO so an operator inspecting DP logs sees
            // "Bedrock isn't talking to AWS" without having to grep
            // env. Prints once per BedrockGuardrail materialised from
            // the snapshot, which is rare (snapshot rebuild only).
            tracing::info!(
                endpoint = %url,
                guardrail_id = %cfg.guardrail_id,
                "BedrockGuardrail using endpoint URL override (AISIX_BEDROCK_ENDPOINT_URL)",
            );
        }
        Self::with_endpoint(row_name, cfg, hook_point, fail_open, endpoint_url)
    }

    /// Internal constructor that accepts an optional `endpoint_url`
    /// override. Production calls `new()` (None → real AWS Bedrock);
    /// tests pass a wiremock URL to point the SDK at a local
    /// canned-response server. Also used by anyone wiring Bedrock
    /// against a self-hosted gateway / proxy.
    fn with_endpoint(
        row_name: impl Into<String>,
        cfg: &BedrockConfig,
        hook_point: GuardrailHookPoint,
        fail_open: bool,
        endpoint_url: Option<String>,
    ) -> Self {
        let BedrockAWSCredentials::Static {
            access_key_id,
            secret_access_key,
        } = &cfg.aws_credentials;
        // Static credentials provider — no STS, no role assume.
        // Phase 4 will add a kind=role_arn variant.
        let creds = Credentials::new(
            access_key_id.clone(),
            secret_access_key.clone(),
            // No session token (static keys are long-lived).
            None,
            None,
            "aisix-guardrails-bedrock",
        );
        let mut builder = aws_config::SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(cfg.region.clone()))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            // The retry sleep_impl is needed for the SDK's built-in
            // retries; aws-config's default features set this when
            // the rt-tokio feature is on (see workspace Cargo.toml).
            .sleep_impl(aws_smithy_async::rt::sleep::SharedAsyncSleep::new(
                aws_smithy_async::rt::sleep::TokioSleep::new(),
            ));
        if let Some(url) = endpoint_url {
            builder = builder.endpoint_url(url);
        }
        let sdk_cfg = builder.build();
        let client = Client::new(&sdk_cfg);
        Self {
            row_name: row_name.into(),
            guardrail_id: cfg.guardrail_id.clone(),
            guardrail_version: cfg.guardrail_version.clone(),
            hook_point,
            latency_mode: cfg.latency_mode.clone(),
            fail_open,
            client: Arc::new(client),
        }
    }

    /// Run `ApplyGuardrail` against a content block. Wraps the
    /// SDK call with `latency_mode` enforcement and translates the
    /// response/error into a `GuardrailVerdict` per §Behavior matrix.
    async fn apply(&self, source: GuardrailContentSource, text: String) -> GuardrailVerdict {
        let req = self
            .client
            .apply_guardrail()
            .guardrail_identifier(&self.guardrail_id)
            .guardrail_version(&self.guardrail_version)
            .source(source)
            .content(GuardrailContentBlock::Text(
                GuardrailTextBlock::builder()
                    .text(text)
                    .build()
                    .expect("GuardrailTextBlock requires text — set above"),
            ));

        let result = match self.latency_mode {
            BedrockLatencyMode::Serial => req.send().await.map_err(BedrockFailure::from_sdk),
            BedrockLatencyMode::Timed { timeout_ms } => {
                match tokio::time::timeout(
                    Duration::from_millis(timeout_ms as u64),
                    req.send(),
                )
                .await
                {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err(BedrockFailure::from_sdk(e)),
                    Err(_) => Err(BedrockFailure::Timeout),
                }
            }
        };

        match result {
            Ok(resp) => match resp.action() {
                GuardrailAction::GuardrailIntervened => GuardrailVerdict::Block {
                    reason: format!(
                        "bedrock guardrail {} intervened",
                        self.guardrail_id
                    ),
                },
                GuardrailAction::None => GuardrailVerdict::Allow,
                other => {
                    // Forward-compat: an unknown enum variant from a
                    // future SDK upgrade. Treat as no-block (the
                    // safer interpretation since `intervened` is the
                    // active-block signal).
                    tracing::warn!(
                        guardrail_id = %self.guardrail_id,
                        action = ?other,
                        "unknown ApplyGuardrail action; treating as Allow",
                    );
                    GuardrailVerdict::Allow
                }
            },
            Err(failure) => self.handle_failure(failure),
        }
    }

    fn handle_failure(&self, failure: BedrockFailure) -> GuardrailVerdict {
        let reason = failure.bypass_tag();
        tracing::warn!(
            row = %self.row_name,
            guardrail_id = %self.guardrail_id,
            failure = ?failure,
            fail_open = self.fail_open,
            "bedrock ApplyGuardrail call failed",
        );
        if self.fail_open {
            GuardrailVerdict::Bypass {
                reason: reason.into(),
            }
        } else {
            GuardrailVerdict::Block {
                reason: format!("bedrock unavailable ({reason})"),
            }
        }
    }
}

/// Failure cause buckets that map onto `guardrail_bypassed_reason`
/// telemetry tags. `Other` collapses every long-tail SDK error onto
/// `bedrock_5xx` so an unrecognised AWS error doesn't leak its
/// internal shape into our wire schema.
#[derive(Debug)]
enum BedrockFailure {
    Timeout,
    Throttled,
    Other,
}

impl BedrockFailure {
    fn from_sdk(err: SdkError<ApplyGuardrailError, Response>) -> Self {
        // ThrottlingException is the SDK's named throttle variant.
        if let SdkError::ServiceError(svc) = &err {
            if matches!(svc.err(), ApplyGuardrailError::ThrottlingException(_)) {
                return Self::Throttled;
            }
        }
        Self::Other
    }

    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "bedrock_timeout",
            Self::Throttled => "bedrock_throttled",
            Self::Other => "bedrock_5xx",
        }
    }
}

#[async_trait]
impl Guardrail for BedrockGuardrail {
    fn name(&self) -> &'static str {
        // Static name keeps metric cardinality bounded; the row's
        // own name is logged via tracing fields when we hit a
        // failure path.
        "bedrock"
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !matches!(
            self.hook_point,
            GuardrailHookPoint::Input | GuardrailHookPoint::Both
        ) {
            return GuardrailVerdict::Allow;
        }
        let text = collect_input_text(req);
        if text.is_empty() {
            // Empty content is a no-op — Bedrock would 400 on it
            // and we'd needlessly burn a call.
            return GuardrailVerdict::Allow;
        }
        self.apply(GuardrailContentSource::Input, text).await
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        ) {
            return GuardrailVerdict::Allow;
        }
        let text = resp.message.content.clone();
        if text.is_empty() {
            return GuardrailVerdict::Allow;
        }
        self.apply(GuardrailContentSource::Output, text).await
    }
}

/// Concatenate the request's user-visible message contents into one
/// blob. Bedrock's `ApplyGuardrail` takes a single text block per
/// call — combining the messages here avoids paying for one call
/// per turn while keeping the same semantic coverage.
fn collect_input_text(req: &ChatFormat) -> String {
    req.messages
        .iter()
        .map(|m| m.content.as_str())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::models::{BedrockAWSCredentials, BedrockConfig, BedrockLatencyMode};

    fn cfg() -> BedrockConfig {
        BedrockConfig {
            guardrail_id: "abcdefgh1234".into(),
            guardrail_version: "DRAFT".into(),
            region: "us-east-1".into(),
            aws_credentials: BedrockAWSCredentials::Static {
                access_key_id: "AKIAEXAMPLE".into(),
                secret_access_key: "TEST".into(),
            },
            latency_mode: BedrockLatencyMode::Serial,
        }
    }

    /// Pin the failure-tag mapping. Operators see these strings in
    /// `usage_events.guardrail_bypassed_reason`; a regression that
    /// renames `bedrock_5xx` to `bedrock_5xx_error` would
    /// retroactively hide bypass events from the dashboard's filter.
    #[test]
    fn bypass_tags_match_wire_contract() {
        assert_eq!(BedrockFailure::Timeout.bypass_tag(), "bedrock_timeout");
        assert_eq!(BedrockFailure::Throttled.bypass_tag(), "bedrock_throttled");
        assert_eq!(BedrockFailure::Other.bypass_tag(), "bedrock_5xx");
    }

    /// `handle_failure` is the integration point between the SDK
    /// error mapper and the verdict — this test pins both
    /// `fail_open` paths without needing a live Bedrock client.
    /// We construct a BedrockGuardrail with a placeholder client
    /// that we never actually call (.apply() is what would call it).
    #[tokio::test]
    async fn timeout_with_fail_open_true_returns_bypass() {
        let g = build_test(true);
        let v = g.handle_failure(BedrockFailure::Timeout);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_timeout"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_with_fail_open_false_returns_block() {
        let g = build_test(false);
        let v = g.handle_failure(BedrockFailure::Timeout);
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    #[tokio::test]
    async fn throttle_with_fail_open_true_tags_throttled() {
        let g = build_test(true);
        let v = g.handle_failure(BedrockFailure::Throttled);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_throttled"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn other_5xx_with_fail_open_true_tags_5xx() {
        let g = build_test(true);
        let v = g.handle_failure(BedrockFailure::Other);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_5xx"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    /// Hook-point gating: an Output-only row must allow input checks
    /// without ever hitting AWS. We assert via Allow, never reaching
    /// the apply() codepath.
    #[tokio::test]
    async fn output_only_row_skips_input_check() {
        let mut g = build_test(true);
        g.hook_point = GuardrailHookPoint::Output;
        let req = ChatFormat::new("m", vec![aisix_gateway::ChatMessage::user("hello")]);
        // If apply() were reached with bad creds, the SDK would
        // panic at runtime; Allow proves we short-circuited at the
        // hook-point gate.
        let v = g.check_input(&req).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    fn build_test(fail_open: bool) -> BedrockGuardrail {
        BedrockGuardrail::new("test-row", &cfg(), GuardrailHookPoint::Both, fail_open)
    }

    // --- wiremock integration tests --------------------------------
    //
    // The unit tests above exercise the failure-mapping logic
    // (`handle_failure`) directly. These tests stand up a local
    // wiremock server, point the SDK client at it via
    // `endpoint_url`, and exercise `apply()` end-to-end including:
    //
    //   * SDK-level request serialization (ApplyGuardrail HTTP body)
    //   * sigv4 signing + transport
    //   * Response deserialization back into the SDK's typed shape
    //   * Verdict translation (Allow / Block / Bypass)
    //
    // We don't try to assert on the exact HTTP path — different SDK
    // versions encode the URL slightly differently (the v1 shape is
    // POST /guardrail/{id}/version/{ver}/apply); broad path matching
    // keeps the test robust to SDK upgrades.

    use aws_sdk_bedrockruntime::types::GuardrailContentSource;
    use serde_json::json;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn build_with_endpoint(endpoint: String, fail_open: bool) -> BedrockGuardrail {
        BedrockGuardrail::with_endpoint(
            "wiremock-test",
            &cfg(),
            GuardrailHookPoint::Both,
            fail_open,
            Some(endpoint),
        )
    }

    /// Happy-path: Bedrock returns `action=NONE` → Allow. This is the
    /// most common production response (the operator's guardrail
    /// didn't fire). Also pins that the request actually hits
    /// `apply_guardrail` — wiremock's mounted matcher requires
    /// exactly one POST under `/guardrail/.../apply`.
    #[tokio::test]
    async fn apply_returns_allow_on_action_none() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/guardrail/.+/apply$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "action": "NONE",
                "outputs": [],
                "assessments": [],
                "usage": {
                    "topicPolicyUnits": 0,
                    "contentPolicyUnits": 0,
                    "wordPolicyUnits": 0,
                    "sensitiveInformationPolicyUnits": 0,
                    "sensitiveInformationPolicyFreeUnits": 0,
                    "contextualGroundingPolicyUnits": 0
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let g = build_with_endpoint(server.uri(), true);
        let v = g.apply(GuardrailContentSource::Input, "hello".into()).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    /// `action=GUARDRAIL_INTERVENED` → Block. Operator's policy
    /// fired (PII / topic / word filter etc.) and the request
    /// should never reach the upstream LLM.
    #[tokio::test]
    async fn apply_returns_block_on_action_intervened() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/guardrail/.+/apply$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{ "text": "I cannot help with that request." }],
                "assessments": [],
                "usage": {
                    "topicPolicyUnits": 1,
                    "contentPolicyUnits": 0,
                    "wordPolicyUnits": 0,
                    "sensitiveInformationPolicyUnits": 0,
                    "sensitiveInformationPolicyFreeUnits": 0,
                    "contextualGroundingPolicyUnits": 0
                }
            })))
            .mount(&server)
            .await;

        let g = build_with_endpoint(server.uri(), true);
        let v = g.apply(GuardrailContentSource::Input, "leak something".into()).await;
        match v {
            GuardrailVerdict::Block { reason } => {
                assert!(
                    reason.contains("abcdefgh1234"),
                    "block reason should mention guardrail_id, got {reason}",
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    /// HTTP 500 + `fail_open=true` → Bypass tagged `bedrock_5xx`.
    /// The SDK retries 500s a couple of times by default (we set
    /// `sleep_impl` for that). wiremock returns 500 every time, so
    /// the SDK gives up and we map the failure.
    #[tokio::test]
    async fn apply_5xx_with_fail_open_true_returns_bypass() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/guardrail/.+/apply$"))
            .respond_with(ResponseTemplate::new(500).set_body_json(json!({
                "__type": "InternalServerException",
                "message": "Bedrock is having a bad day"
            })))
            .mount(&server)
            .await;

        let g = build_with_endpoint(server.uri(), true);
        let v = g.apply(GuardrailContentSource::Input, "anything".into()).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_5xx"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    /// HTTP 500 + `fail_open=false` → Block. Operator chose to
    /// block on Bedrock outage (correctness over availability).
    #[tokio::test]
    async fn apply_5xx_with_fail_open_false_returns_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/guardrail/.+/apply$"))
            .respond_with(ResponseTemplate::new(500).set_body_json(json!({
                "__type": "InternalServerException",
                "message": "Bedrock is having a bad day"
            })))
            .mount(&server)
            .await;

        let g = build_with_endpoint(server.uri(), false);
        let v = g.apply(GuardrailContentSource::Input, "anything".into()).await;
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    /// 429 ThrottlingException → tagged `bedrock_throttled`. Distinct
    /// from generic 5xx because operators triage AWS-quota issues
    /// differently from Bedrock service outages.
    #[tokio::test]
    async fn apply_429_throttling_with_fail_open_tags_throttled() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/guardrail/.+/apply$"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "__type": "ThrottlingException",
                "message": "Rate exceeded"
            })))
            .mount(&server)
            .await;

        let g = build_with_endpoint(server.uri(), true);
        let v = g.apply(GuardrailContentSource::Input, "anything".into()).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_throttled"),
            other => panic!("expected Bypass(bedrock_throttled), got {other:?}"),
        }
    }

    // The env-var path (`new()` reads `AISIX_BEDROCK_ENDPOINT_URL`)
    // isn't unit-tested here because:
    //   1. `aisix-guardrails` declares `#![forbid(unsafe_code)]` and
    //      `std::env::set_var` is unsafe in modern Rust; the
    //      forbidden lint can't be relaxed for a single test without
    //      moving it to a separate `tests/` integration binary.
    //   2. The override logic is two lines of trivial mapping
    //      (`env::var().ok().filter(non-empty)`) → `with_endpoint`,
    //      which the wiremock tests above already exercise
    //      thoroughly.
    //   3. The deployment-wide override is verified end-to-end by
    //      `dashboard/tests/e2e/bedrock-live.spec.ts` against a real
    //      fakecloud sidecar — see PRD-09c §6.7 + the e2e compose
    //      file. That's the contract test that catches a regression
    //      where the env var stops being read.

    /// `latency_mode=timed` + Bedrock takes longer than the timeout
    /// → tagged `bedrock_timeout`. wiremock's `set_delay` makes the
    /// server hang past our 100ms timeout.
    #[tokio::test]
    async fn apply_timed_mode_aborts_at_timeout_and_tags_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/guardrail/.+/apply$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_millis(800))
                    .set_body_json(json!({
                        "action": "NONE",
                        "outputs": [],
                        "assessments": [],
                        "usage": {
                            "topicPolicyUnits": 0,
                            "contentPolicyUnits": 0,
                            "wordPolicyUnits": 0,
                            "sensitiveInformationPolicyUnits": 0,
                            "sensitiveInformationPolicyFreeUnits": 0,
                            "contextualGroundingPolicyUnits": 0
                        }
                    })),
            )
            .mount(&server)
            .await;

        let mut tight = cfg();
        tight.latency_mode = BedrockLatencyMode::Timed { timeout_ms: 100 };
        let g = BedrockGuardrail::with_endpoint(
            "timed-test",
            &tight,
            GuardrailHookPoint::Both,
            /* fail_open */ true,
            Some(server.uri()),
        );
        let v = g.apply(GuardrailContentSource::Input, "anything".into()).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_timeout"),
            other => panic!("expected Bypass(bedrock_timeout), got {other:?}"),
        }
    }
}
