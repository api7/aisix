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
        let sdk_cfg = aws_config::SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(cfg.region.clone()))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            // The retry sleep_impl is needed for the SDK's built-in
            // retries; aws-config's default features set this when
            // the rt-tokio feature is on (see workspace Cargo.toml).
            .sleep_impl(aws_smithy_async::rt::sleep::SharedAsyncSleep::new(
                aws_smithy_async::rt::sleep::TokioSleep::new(),
            ))
            .build();
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
}
