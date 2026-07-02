//! kind=bedrock guardrail dispatcher — calls AWS Bedrock's
//! `ApplyGuardrail` API on every chat request and translates the
//! response into a [`GuardrailVerdict`].
//!
//! PRD-09c §6 Phase 2. The cp-api side ships the
//! envelope-encrypted secret; cp-api decrypts at projection time
//! so this module only handles plaintext credentials. We never log
//! the secret.
//!
//! Behavior matrix (failure modes). The effective `fail_open` is the
//! outer `Guardrail::fail_open` on the INPUT hook and the independent
//! `BedrockConfig::output_fail_open` (default fail-closed) on the OUTPUT
//! hook, so a Bedrock outage can't release unscanned model output by
//! default:
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
use aws_sdk_bedrockruntime::operation::apply_guardrail::{ApplyGuardrailError, ApplyGuardrailOutput};
use aws_sdk_bedrockruntime::types::{
    GuardrailAction, GuardrailAssessment, GuardrailContentBlock, GuardrailContentSource,
    GuardrailSensitiveInformationPolicyAction, GuardrailTextBlock,
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
    /// Fail-open policy for the INPUT hook (the outer `Guardrail::fail_open`).
    pub fail_open: bool,
    /// Fail-open policy for the OUTPUT hook (`BedrockConfig::output_fail_open`,
    /// default fail-closed). Kept separate so a Bedrock outage can't release
    /// unscanned model output by default.
    pub output_fail_open: bool,
    /// AWS SDK client, pre-configured with the row's region and
    /// static credentials. Wrapped in `Arc` so swapping snapshots
    /// doesn't drop a client mid-request.
    client: Arc<Client>,
}

impl BedrockGuardrail {
    /// Build the dispatcher from a parsed [`BedrockConfig`]. Caller
    /// owns the row's `name`, `hook_point`, and `fail_open` (they
    /// live on the outer Guardrail struct, not on the kind config),
    /// plus the optional deployment-wide `endpoint_url` override
    /// (sourced from `aisix_core::Config::bedrock_endpoint_url` —
    /// `None` means the SDK default, i.e. real AWS Bedrock).
    ///
    /// Empty-string overrides are treated as unset by the caller so
    /// a `docker run -e AISIX_BEDROCK_ENDPOINT_URL=` doesn't
    /// accidentally redirect; this constructor doesn't filter again.
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
        endpoint_url: Option<String>,
    ) -> Self {
        if let Some(url) = endpoint_url.as_ref() {
            // Visible at INFO so an operator inspecting DP logs sees
            // "Bedrock isn't talking to AWS" without having to grep
            // env. Prints once per BedrockGuardrail materialised from
            // the snapshot, which is rare (snapshot rebuild only).
            tracing::info!(
                endpoint = %url,
                guardrail_id = %cfg.guardrail_id,
                "BedrockGuardrail using endpoint URL override (Config.bedrock_endpoint_url)",
            );
        }
        Self::with_endpoint(row_name, cfg, hook_point, fail_open, endpoint_url)
    }

    /// Internal constructor that accepts an optional `endpoint_url`
    /// override. Production calls `new()` with the value forwarded
    /// from `Config::bedrock_endpoint_url`; tests pass a wiremock
    /// URL to point the SDK at a local canned-response server.
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
            output_fail_open: cfg.output_fail_open,
            client: Arc::new(client),
        }
    }

    /// Fail-open policy that governs `source`. The input hook follows the
    /// outer `fail_open`; the output hook follows `output_fail_open` (default
    /// fail-closed) so a Bedrock outage can't release unscanned model output.
    fn fail_open_for(&self, source: &GuardrailContentSource) -> bool {
        match source {
            GuardrailContentSource::Output => self.output_fail_open,
            _ => self.fail_open,
        }
    }

    /// Run `ApplyGuardrail` against a content block. Wraps the
    /// SDK call with `latency_mode` enforcement and translates the
    /// response/error into a `GuardrailVerdict` per §Behavior matrix.
    async fn apply(&self, source: GuardrailContentSource, text: String) -> GuardrailVerdict {
        let fail_open = self.fail_open_for(&source);
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
                match tokio::time::timeout(Duration::from_millis(timeout_ms as u64), req.send())
                    .await
                {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) => Err(BedrockFailure::from_sdk(e)),
                    Err(_) => Err(BedrockFailure::Timeout),
                }
            }
        };

        match result {
            Ok(resp) => match classify_response(&resp, &self.guardrail_id) {
                BedrockOutcome::Allow => GuardrailVerdict::Allow,
                BedrockOutcome::Block => GuardrailVerdict::block(format!(
                    "bedrock guardrail {} intervened",
                    self.guardrail_id
                )),
                // Bedrock ANONYMIZED (masked) the content rather than
                // blocking. The masked replacement text is in the payload.
                // TODO(#932 bedrock follow-up): once the async rewrite
                // channel lands, return a mask verdict and write the masked
                // text back instead of blocking. Until then, block so the
                // un-masked content is never released (fail-safe, and
                // behaviour-preserving vs the pre-classify code).
                BedrockOutcome::Mask(_) => GuardrailVerdict::block(format!(
                    "bedrock guardrail {} anonymized content",
                    self.guardrail_id
                )),
            },
            Err(failure) => self.handle_failure(failure, fail_open),
        }
    }

    fn handle_failure(&self, failure: BedrockFailure, fail_open: bool) -> GuardrailVerdict {
        let (reason, error_detail, error_source) = failure.log_fields();
        tracing::warn!(
            row = %self.row_name,
            guardrail_id = %self.guardrail_id,
            failure_tag = reason,
            error = error_detail,
            source = error_source,
            fail_open = fail_open,
            "bedrock ApplyGuardrail call failed",
        );
        if fail_open {
            GuardrailVerdict::Bypass {
                reason: reason.into(),
            }
        } else {
            GuardrailVerdict::block(format!("bedrock unavailable ({reason})"))
        }
    }
}

/// The masking-aware interpretation of an `ApplyGuardrail` response.
///
/// Bedrock reports `action = GUARDRAIL_INTERVENED` for BOTH a hard block
/// AND a PII anonymization (mask). The two are told apart by the
/// per-policy actions inside `assessments`: a topic/content/word/
/// contextual-grounding policy hit, or a PII/regex entity with
/// `action = BLOCKED`, is a hard block; a PII/regex entity with
/// `action = ANONYMIZED` (and nothing blocking) is a mask, whose
/// replacement text Bedrock returns in `outputs[].text`. Mirrors
/// LiteLLM's `_should_raise_guardrail_blocked_exception` (raise iff any
/// assessment entry is BLOCKED; otherwise apply the masked output).
#[derive(Debug, PartialEq, Eq)]
enum BedrockOutcome {
    /// `action = NONE` — nothing detected.
    Allow,
    /// A hard block: some policy blocked (topic/content/word/contextual)
    /// or a PII/regex entity had `action = BLOCKED`.
    Block,
    /// Only anonymization occurred. Carries the masked replacement text
    /// per `outputs[]` block (real Bedrock returns one).
    Mask(Vec<String>),
}

/// Classify an `ApplyGuardrail` response into allow / block / mask.
/// Secure by default: an intervention that is neither a recognizable
/// block nor accompanied by masked output is treated as a block.
fn classify_response(resp: &ApplyGuardrailOutput, guardrail_id: &str) -> BedrockOutcome {
    match resp.action() {
        GuardrailAction::None => BedrockOutcome::Allow,
        GuardrailAction::GuardrailIntervened => {
            if resp.assessments().iter().any(assessment_has_hard_block) {
                return BedrockOutcome::Block;
            }
            let masked: Vec<String> = resp
                .outputs()
                .iter()
                .filter_map(|o| o.text())
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            if masked.is_empty() {
                // Intervened, no recognizable hard block, no masked
                // output — block rather than risk releasing content whose
                // disposition we can't read.
                BedrockOutcome::Block
            } else {
                BedrockOutcome::Mask(masked)
            }
        }
        other => {
            // Forward-compat: an unknown enum variant from a future SDK
            // upgrade. `intervened` is the active signal, so an unknown
            // action is treated as no-intervention (Allow).
            tracing::warn!(
                guardrail_id = %guardrail_id,
                action = ?other,
                "unknown ApplyGuardrail action; treating as Allow",
            );
            BedrockOutcome::Allow
        }
    }
}

/// True if the assessment carries any BLOCKING disposition — a
/// topic/content/word policy hit (these have no anonymize mode, so their
/// presence is a block), a contextual-grounding filter that BLOCKED, or a
/// PII/regex entity whose action is BLOCKED (as opposed to ANONYMIZED).
fn assessment_has_hard_block(a: &GuardrailAssessment) -> bool {
    let topic_blocked = a.topic_policy().is_some_and(|p| !p.topics().is_empty());
    let content_blocked = a.content_policy().is_some_and(|p| !p.filters().is_empty());
    let word_blocked = a.word_policy().is_some_and(|p| {
        !p.custom_words().is_empty() || !p.managed_word_lists().is_empty()
    });
    let grounding_blocked = a.contextual_grounding_policy().is_some_and(|p| {
        p.filters().iter().any(|f| {
            *f.action()
                == aws_sdk_bedrockruntime::types::GuardrailContextualGroundingPolicyAction::Blocked
        })
    });
    let pii_blocked = a.sensitive_information_policy().is_some_and(|sip| {
        sip.pii_entities()
            .iter()
            .any(|e| *e.action() == GuardrailSensitiveInformationPolicyAction::Blocked)
            || sip
                .regexes()
                .iter()
                .any(|r| *r.action() == GuardrailSensitiveInformationPolicyAction::Blocked)
    });
    topic_blocked || content_blocked || word_blocked || grounding_blocked || pii_blocked
}

/// Failure cause buckets that map onto `guardrail_bypassed_reason`
/// telemetry tags. `Other` collapses every long-tail SDK error onto
/// `bedrock_5xx` so an unrecognised AWS error doesn't leak its
/// internal shape into our wire schema — but it carries the original
/// SDK error detail for the failure log line so operators can tell a
/// network blip from an auth/validation error (#106).
#[derive(Debug)]
enum BedrockFailure {
    Timeout,
    Throttled,
    Other {
        /// Human-readable detail for logs: the modeled service error
        /// (carries the exception code, e.g. AccessDeniedException) or
        /// the dispatch/transport error Display.
        detail: String,
        /// The underlying cause (hyper/rustls/sigv4) when present.
        source: Option<String>,
    },
}

impl BedrockFailure {
    fn from_sdk(err: SdkError<ApplyGuardrailError, Response>) -> Self {
        // ThrottlingException is the SDK's named throttle variant.
        if let SdkError::ServiceError(svc) = &err {
            if matches!(svc.err(), ApplyGuardrailError::ThrottlingException(_)) {
                return Self::Throttled;
            }
            // Surface the modeled service error — its Debug carries the
            // exception code (AccessDeniedException, ValidationException,
            // …) so the failure log is greppable.
            return Self::Other {
                detail: format!("{:?}", svc.err()),
                source: std::error::Error::source(&err).map(|s| s.to_string()),
            };
        }
        // Dispatch / response / construction / timeout: the SdkError
        // Display names the variant; source() reaches the underlying
        // hyper/rustls/sigv4 error.
        Self::Other {
            detail: err.to_string(),
            source: std::error::Error::source(&err).map(|s| s.to_string()),
        }
    }

    fn bypass_tag(&self) -> &'static str {
        match self {
            Self::Timeout => "bedrock_timeout",
            Self::Throttled => "bedrock_throttled",
            Self::Other { .. } => "bedrock_5xx",
        }
    }

    /// Fields for the failure log line: the wire-stable tag plus, for
    /// `Other`, the captured SDK error detail and underlying source.
    fn log_fields(&self) -> (&'static str, Option<&str>, Option<&str>) {
        match self {
            Self::Other { detail, source } => {
                (self.bypass_tag(), Some(detail.as_str()), source.as_deref())
            }
            _ => (self.bypass_tag(), None, None),
        }
    }
}

#[async_trait]
impl Guardrail for BedrockGuardrail {
    /// Its streamed-output hold-back policy applies only when it inspects
    /// output (#466); an input-only attachment must not buffer the response.
    fn runs_on_output(&self) -> bool {
        matches!(
            self.hook_point,
            GuardrailHookPoint::Output | GuardrailHookPoint::Both
        )
    }

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
        let text = resp.guardrail_output_text();
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
        .map(crate::message_scan_text)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::models::{BedrockAWSCredentials, BedrockConfig, BedrockLatencyMode};

    // --- classify_response: block vs mask vs allow (#932 bedrock) ---------
    mod classify {
        use super::super::{assessment_has_hard_block, classify_response, BedrockOutcome};
        use aws_sdk_bedrockruntime::operation::apply_guardrail::ApplyGuardrailOutput;
        use aws_sdk_bedrockruntime::types::{
            GuardrailAction, GuardrailAssessment, GuardrailOutputContent, GuardrailPiiEntityFilter,
            GuardrailPiiEntityType, GuardrailSensitiveInformationPolicyAction as PiiAction,
            GuardrailSensitiveInformationPolicyAssessment, GuardrailTopic, GuardrailTopicPolicyAction,
            GuardrailTopicPolicyAssessment, GuardrailTopicType,
        };

        fn resp(
            action: GuardrailAction,
            outputs: Vec<&str>,
            assessments: Vec<GuardrailAssessment>,
        ) -> ApplyGuardrailOutput {
            ApplyGuardrailOutput::builder()
                .action(action)
                .set_outputs(Some(
                    outputs
                        .into_iter()
                        .map(|t| GuardrailOutputContent::builder().text(t).build())
                        .collect(),
                ))
                .set_assessments(Some(assessments))
                .build()
                .expect("action/outputs/assessments all set")
        }

        fn pii(action: PiiAction) -> GuardrailAssessment {
            let entity = GuardrailPiiEntityFilter::builder()
                .r#match("alice@example.com")
                .r#type(GuardrailPiiEntityType::Email)
                .action(action)
                .build()
                .unwrap();
            let sip = GuardrailSensitiveInformationPolicyAssessment::builder()
                .pii_entities(entity)
                .set_regexes(Some(vec![]))
                .build()
                .unwrap();
            GuardrailAssessment::builder()
                .sensitive_information_policy(sip)
                .build()
        }

        fn topic() -> GuardrailAssessment {
            let t = GuardrailTopic::builder()
                .name("blocked-topic")
                .r#type(GuardrailTopicType::Deny)
                .action(GuardrailTopicPolicyAction::Blocked)
                .build()
                .unwrap();
            let tp = GuardrailTopicPolicyAssessment::builder()
                .topics(t)
                .build()
                .unwrap();
            GuardrailAssessment::builder().topic_policy(tp).build()
        }

        #[test]
        fn action_none_is_allow() {
            let r = resp(GuardrailAction::None, vec![], vec![]);
            assert_eq!(classify_response(&r, "gid"), BedrockOutcome::Allow);
        }

        #[test]
        fn anonymized_pii_with_masked_output_is_mask() {
            let r = resp(
                GuardrailAction::GuardrailIntervened,
                vec!["contact {EMAIL} about the order"],
                vec![pii(PiiAction::Anonymized)],
            );
            assert_eq!(
                classify_response(&r, "gid"),
                BedrockOutcome::Mask(vec!["contact {EMAIL} about the order".to_owned()])
            );
        }

        #[test]
        fn blocked_pii_is_block() {
            let r = resp(
                GuardrailAction::GuardrailIntervened,
                vec!["irrelevant"],
                vec![pii(PiiAction::Blocked)],
            );
            assert_eq!(classify_response(&r, "gid"), BedrockOutcome::Block);
        }

        #[test]
        fn topic_policy_hit_is_block_even_with_masked_output() {
            // A hard block (topic) wins even if masked output is present.
            let r = resp(
                GuardrailAction::GuardrailIntervened,
                vec!["masked text"],
                vec![topic()],
            );
            assert_eq!(classify_response(&r, "gid"), BedrockOutcome::Block);
        }

        #[test]
        fn mixed_anonymized_and_blocked_is_block() {
            let r = resp(
                GuardrailAction::GuardrailIntervened,
                vec!["masked"],
                vec![pii(PiiAction::Anonymized), pii(PiiAction::Blocked)],
            );
            assert_eq!(classify_response(&r, "gid"), BedrockOutcome::Block);
        }

        #[test]
        fn intervened_without_hard_block_or_masked_output_is_block() {
            // Secure default: an intervention we can't read as a mask blocks.
            let r = resp(GuardrailAction::GuardrailIntervened, vec![], vec![]);
            assert_eq!(classify_response(&r, "gid"), BedrockOutcome::Block);
        }

        #[test]
        fn hard_block_helper_only_true_for_blocking_dispositions() {
            assert!(!assessment_has_hard_block(&pii(PiiAction::Anonymized)));
            assert!(assessment_has_hard_block(&pii(PiiAction::Blocked)));
            assert!(assessment_has_hard_block(&topic()));
        }
    }

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
            // Default fail-closed output (cp-api omits the field when unset).
            output_fail_open: false,
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
        assert_eq!(
            BedrockFailure::Other {
                detail: "x".into(),
                source: None,
            }
            .bypass_tag(),
            "bedrock_5xx",
        );
    }

    /// #106: the `Other` bucket keeps the wire-stable `bedrock_5xx`
    /// tag but surfaces the captured SDK error detail + source so the
    /// failure log line is greppable. The non-`Other` buckets carry no
    /// extra detail.
    #[test]
    fn other_failure_log_fields_carry_detail_and_source() {
        let f = BedrockFailure::Other {
            detail: "AccessDeniedException(...)".into(),
            source: Some("dispatch failure: connection refused".into()),
        };
        let (tag, detail, source) = f.log_fields();
        assert_eq!(tag, "bedrock_5xx");
        assert_eq!(detail, Some("AccessDeniedException(...)"));
        assert_eq!(source, Some("dispatch failure: connection refused"));

        let (tag, detail, source) = BedrockFailure::Timeout.log_fields();
        assert_eq!(tag, "bedrock_timeout");
        assert_eq!(detail, None);
        assert_eq!(source, None);
    }

    /// `handle_failure` is the integration point between the SDK
    /// error mapper and the verdict — this test pins both
    /// `fail_open` paths without needing a live Bedrock client.
    /// We construct a BedrockGuardrail with a placeholder client
    /// that we never actually call (.apply() is what would call it).
    #[tokio::test]
    async fn timeout_with_fail_open_true_returns_bypass() {
        let g = build_test(true);
        let v = g.handle_failure(BedrockFailure::Timeout, g.fail_open);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_timeout"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_with_fail_open_false_returns_block() {
        let g = build_test(false);
        let v = g.handle_failure(BedrockFailure::Timeout, g.fail_open);
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    #[tokio::test]
    async fn throttle_with_fail_open_true_tags_throttled() {
        let g = build_test(true);
        let v = g.handle_failure(BedrockFailure::Throttled, g.fail_open);
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_throttled"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn other_5xx_with_fail_open_true_tags_5xx() {
        let g = build_test(true);
        let v = g.handle_failure(
            BedrockFailure::Other {
                detail: "AccessDeniedException(...)".into(),
                source: None,
            },
            g.fail_open,
        );
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_5xx"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    /// The OUTPUT hook follows `output_fail_open`, which defaults to
    /// fail-closed even when the input-side `fail_open` is true. A Bedrock
    /// outage on the output side must therefore Block, not release unscanned
    /// model output. This is the P1-3 fix: the single `fail_open` no longer
    /// governs both hooks.
    #[tokio::test]
    async fn output_hook_defaults_fail_closed_even_when_input_fail_open() {
        // build_test sets input fail_open=true; cfg() leaves output_fail_open
        // at its serde default (false).
        let g = build_test(true);
        assert!(g.fail_open, "input fail_open is true in this fixture");
        assert!(!g.output_fail_open, "output must default fail-closed");
        // Input side bypasses (fail_open=true)...
        assert!(g
            .handle_failure(
                BedrockFailure::Timeout,
                g.fail_open_for(&GuardrailContentSource::Input)
            )
            .is_bypass());
        // ...output side blocks (output_fail_open=false).
        assert!(g
            .handle_failure(
                BedrockFailure::Timeout,
                g.fail_open_for(&GuardrailContentSource::Output),
            )
            .is_block());
    }

    /// Operators can still opt the output hook back into fail-open by setting
    /// `output_fail_open: true` — then an outage bypasses on output too. The
    /// input policy here is the opposite (`fail_open=false`) so the test
    /// proves the output hook uses its OWN policy, not the input one.
    #[tokio::test]
    async fn output_fail_open_true_bypasses_on_output() {
        let mut c = cfg();
        c.output_fail_open = true;
        let g = BedrockGuardrail::new("row", &c, GuardrailHookPoint::Both, false, None);
        // Output opted into fail-open → bypass.
        assert!(g
            .handle_failure(
                BedrockFailure::Timeout,
                g.fail_open_for(&GuardrailContentSource::Output),
            )
            .is_bypass());
        // Input still fails closed → block (proves the two policies are split).
        assert!(g
            .handle_failure(
                BedrockFailure::Timeout,
                g.fail_open_for(&GuardrailContentSource::Input),
            )
            .is_block());
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
        // None → SDK default endpoint (we never call .apply() in
        // these tests, so it doesn't matter that it would point at
        // real AWS).
        BedrockGuardrail::new(
            "test-row",
            &cfg(),
            GuardrailHookPoint::Both,
            fail_open,
            None,
        )
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
        let v = g
            .apply(GuardrailContentSource::Input, "leak something".into())
            .await;
        match v {
            GuardrailVerdict::Block { reason, .. } => {
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
        let v = g
            .apply(GuardrailContentSource::Input, "anything".into())
            .await;
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
        let v = g
            .apply(GuardrailContentSource::Input, "anything".into())
            .await;
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
        let v = g
            .apply(GuardrailContentSource::Input, "anything".into())
            .await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_throttled"),
            other => panic!("expected Bypass(bedrock_throttled), got {other:?}"),
        }
    }

    // The endpoint-override pass-through (Config field →
    // BedrockGuardrail::new → with_endpoint) is exercised by the
    // wiremock tests above, which thread the URL in via the public
    // `new()` constructor. The Config-loading side (env var
    // `AISIX_BEDROCK_ENDPOINT_URL` → `Config::bedrock_endpoint_url`)
    // is covered by `aisix-core`'s config tests. The end-to-end
    // contract — operator sets env var, DP redirects Bedrock calls —
    // is verified by the downstream e2e suite against a real
    // fakecloud sidecar.

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
        let v = g
            .apply(GuardrailContentSource::Input, "anything".into())
            .await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "bedrock_timeout"),
            other => panic!("expected Bypass(bedrock_timeout), got {other:?}"),
        }
    }
}
