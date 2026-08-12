//! Per-ProviderKey telemetry attribution shared by every request handler's
//! usage-event emitter (AISIX-Cloud#867 + non-chat parity follow-up).
//!
//! The five attribution fields — `provider_kind` / `provider_featured` /
//! `branded_provider` / `pk_label` / `byo_label` — are sourced from the
//! resolved ProviderKey's `telemetry_tags` at emit time. Centralising the
//! snapshot lookup AND the wire-field mapping here keeps the handler family
//! (chat / messages / responses / completions / embeddings / rerank / audio /
//! images) from drifting apart again — the exact bug #867 fixed for
//! `/v1/responses` after it had already been fixed for chat + messages.
//!
//! [`ResolvedPk`] is the second half of that anti-drift move (#941): the
//! attempt's ProviderKey row is looked up ONCE per completion and the
//! result is handed to every terminal emitter, so the metric label and
//! the usage-event attribution can neither disagree nor pay for the same
//! `DashMap` read three times.

use std::borrow::Cow;
use std::sync::Arc;

use aisix_core::{AisixSnapshot, ProviderKey, ResourceEntry};
use aisix_obs::UsageEvent;

use crate::chat::sanitize_tag;
use crate::client_ip::ClientContext;
use crate::state::ProxyState;

/// The provider's own response object `id`, read off a JSON response body —
/// OpenAI's `chat.completion.id`, a Responses-API `resp_…`, a legacy
/// completions `cmpl-…`, a Cohere rerank `id`.
///
/// AISIX-Cloud#1289 keeps three ids strictly separate: this one, the
/// gateway's own `request_id`, and the provider's HTTP transport header id.
/// None of them may stand in for another, and an absent id stays absent — a
/// synthesised value would send an operator hunting in the provider's console
/// for a call that never happened there. Empty is a normal outcome:
/// embeddings / audio / images carry no id in their response shape at all, an
/// errored call has no response object, and a cache hit never reached a
/// provider.
///
/// The value is upstream-controlled and reaches both a log line and cp-api's
/// `dpmgr_usage_events`, so it goes through [`sanitize_tag`] (control chars
/// stripped, 256-char cap) — an unescaped newline in an id would otherwise
/// let an upstream forge whole log records.
pub(crate) fn provider_response_id(body: &serde_json::Value) -> String {
    sanitize_provider_response_id(body.get("id").and_then(|v| v.as_str()).unwrap_or_default())
}

/// [`provider_response_id`] for a value already decoded into a typed struct
/// (a bridge `Response.id`, a stream chunk `id`). Every producer must route
/// through one of the two, or the cap and control-char stripping the field's
/// consumers assume are not actually applied to that path.
pub(crate) fn sanitize_provider_response_id(id: &str) -> String {
    if id.is_empty() {
        return String::new();
    }
    sanitize_tag(id.to_string())
}

/// Label value the `provider_key_id` / `provider_key_name` pair falls back
/// to when the request never resolved a ProviderKey. Matches
/// `request_metrics::UNKNOWN` and [`PkLabels::default`].
pub(crate) const UNKNOWN_PK: &str = "unknown";

/// The attempt's ProviderKey, resolved ONCE per completion.
///
/// Three terminal emitters want something off the same row: `record` and
/// `record_usage` want the readable `provider_key_name` label (#890 req-3),
/// and the usage-event emitters want `telemetry_tags` (AISIX-Cloud#867).
/// Each used to look the row up itself — three `DashMap` reads and two
/// `display_name` clones for one request (#941). Resolving here and passing
/// the result down replaces them with one read, and makes the two emits
/// provably agree: they now read the same row observation, not two lookups
/// that a concurrent snapshot swap can separate.
///
/// The borrow is the anti-drift device: [`crate::request_metrics::Upstream`]
/// takes [`PkLabels`], not a bare id, so a new call site cannot reintroduce
/// a per-emit lookup without saying so.
pub(crate) struct ResolvedPk<'a> {
    id: &'a str,
    /// `display_name`, control-char stripped and length-capped via
    /// [`sanitize_tag`]; [`UNKNOWN_PK`] when the id is empty, unresolved
    /// or names a row with a blank display name. Borrowed in the
    /// fallback case so the common pre-dispatch failure allocates nothing.
    name: Cow<'a, str>,
    entry: Option<Arc<ResourceEntry<ProviderKey>>>,
}

impl<'a> ResolvedPk<'a> {
    /// Look the row up once. `id` reaches the metric label verbatim —
    /// including the empty string the pre-dispatch failure paths pass —
    /// so the emitted series is byte-identical to the per-emitter lookups
    /// this replaced.
    pub(crate) fn resolve(snap: &AisixSnapshot, id: &'a str) -> Self {
        let entry = if id.is_empty() {
            None
        } else {
            snap.provider_keys.get_by_id(id)
        };
        let name = match entry.as_ref() {
            Some(e) => {
                let name = sanitize_tag(e.value.display_name.clone());
                if name.is_empty() {
                    Cow::Borrowed(UNKNOWN_PK)
                } else {
                    Cow::Owned(name)
                }
            }
            None => Cow::Borrowed(UNKNOWN_PK),
        };
        Self { id, name, entry }
    }

    /// A completion that never reached a ProviderKey — the pre-dispatch
    /// rejections and the endpoints that have no upstream key at all.
    /// Same labels the per-emitter lookup produced for an empty id, with
    /// no snapshot read.
    pub(crate) fn unresolved() -> ResolvedPk<'static> {
        ResolvedPk {
            id: "",
            name: Cow::Borrowed(UNKNOWN_PK),
            entry: None,
        }
    }

    /// The id + name pair for the metric label set.
    pub(crate) fn labels(&self) -> PkLabels<'_> {
        PkLabels {
            id: self.id,
            name: &self.name,
        }
    }

    /// Attribution tags for the usage event. Cloned on demand: the tag
    /// strings only reach the wire on the paths that emit an event, so a
    /// bare metric emit pays nothing for them. An unresolved key yields
    /// the default (all-empty) tags, which skip-serialize to wire NULL.
    pub(crate) fn telemetry_tags(&self) -> aisix_core::TelemetryTags {
        self.entry
            .as_ref()
            .map(|e| e.value.telemetry_tags.clone())
            .unwrap_or_default()
    }
}

/// The ProviderKey dimensions of a metric label set: the id and the
/// readable name that is 1:1 with it (so the pair adds no series).
///
/// The fields are private on purpose. [`ResolvedPk::labels`] and
/// [`PkLabels::default`] are the only ways to build one, so a name can
/// never be paired with an id it was not read off — which is the whole
/// reason `Upstream` takes this type instead of a bare id.
#[derive(Clone, Copy)]
pub(crate) struct PkLabels<'a> {
    id: &'a str,
    name: &'a str,
}

impl<'a> PkLabels<'a> {
    pub(crate) fn id(self) -> &'a str {
        self.id
    }

    pub(crate) fn name(self) -> &'a str {
        self.name
    }
}

impl Default for PkLabels<'_> {
    fn default() -> Self {
        Self {
            id: UNKNOWN_PK,
            name: UNKNOWN_PK,
        }
    }
}

/// The exporter set a terminal emit fans out to.
///
/// Deliberately NOT read off the request's frozen snapshot (#941 audit
/// M1). Exporter membership is a delivery-authorization decision, not a
/// label: an operator who deletes an exporter — say one configured for
/// full content capture, pointed at the wrong tenant — must stop it
/// receiving events, including captured prompt and response text, from
/// requests that are still in flight. A frozen list would keep feeding it
/// for as long as the longest request runs.
///
/// The zero-config fast path still pays nothing: the emptiness check is
/// one relaxed atomic load on the request's own snapshot
/// (`ResourceTable::is_empty`), so a deployment with no exporters
/// configured never reaches the reload.
pub(crate) fn live_exporters(
    state: &ProxyState,
    snap: &AisixSnapshot,
) -> Vec<Arc<ResourceEntry<aisix_core::ObservabilityExporter>>> {
    if snap.observability_exporters.is_empty() {
        return Vec::new();
    }
    state.snapshot.load().observability_exporters.entries()
}

/// Total token cost of a request as committed against TPM/TPD rate limits
/// (and reported as the prometheus usage total): prompt + completion +
/// Anthropic cache creation/read. Anthropic reports cache tokens as counters
/// SEPARATE from `input_tokens`, so a prompt+completion sum silently
/// undercounts cached traffic — the OpenAI bridge already folds them into
/// `total_tokens` (#679) and the CP display total includes them (#906); this
/// keeps the native `/v1/messages` and `/v1/responses` commits consistent
/// (AISIX-Cloud#995). OpenAI's `cached_tokens` is a subset of
/// `prompt_tokens` and is deliberately NOT an input here.
pub(crate) fn total_tokens_with_cache(
    prompt_tokens: u32,
    completion_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
) -> u64 {
    u64::from(prompt_tokens)
        + u64::from(completion_tokens)
        + u64::from(cache_creation_tokens)
        + u64::from(cache_read_tokens)
}

/// The `model` metric label for a request whose client-supplied `model`
/// field never resolved to a configured model (e.g. model-not-found). See
/// [`metric_model_label`].
pub(crate) const UNRESOLVED_MODEL_LABEL: &str = "unresolved";

/// Bound the `model` metric label to the configured set. A request's `model`
/// field is arbitrary caller-controlled text until it resolves against the
/// snapshot; feeding the raw value into a Prometheus label lets a caller
/// explode metric cardinality. Three outcomes:
/// - an exact configured name (direct or virtual router — both live in
///   `models`) labels as itself;
/// - a name only a WILDCARD row serves labels as that row's `display_name`
///   (`openai/*`) — the caller can mint unlimited concrete suffixes, so the
///   configured row is the only bounded identity, and it keeps success and
///   failure series for one row under one label;
/// - anything else is the fixed [`UNRESOLVED_MODEL_LABEL`] sentinel.
///
/// This is the typed-endpoint analogue of passthrough's
/// `PASSTHROUGH_MODEL_LABEL` guard (#451); `request_metrics::record` /
/// `record_usage` apply it at the emit chokepoint so no handler family
/// member can drift.
pub(crate) fn metric_model_label<'a>(snap: &AisixSnapshot, model_name: &'a str) -> Cow<'a, str> {
    if snap.models.get_by_name(model_name).is_some() {
        return Cow::Borrowed(model_name);
    }
    match crate::model_resolve::wildcard_row_name(snap, model_name) {
        Some(row) => Cow::Owned(row),
        None => Cow::Borrowed(UNRESOLVED_MODEL_LABEL),
    }
}

/// Stamp the five per-PK attribution fields onto an in-progress UsageEvent,
/// sanitising the operator-controlled tag strings (control-char strip + length
/// cap) before they hit the wire. One source of truth for the mapping so the
/// non-chat handlers can't diverge from chat / messages.
pub(crate) fn apply_pk_telemetry(event: &mut UsageEvent, pk: &ResolvedPk<'_>) {
    let tags = pk.telemetry_tags();
    event.provider_kind =
        sanitize_tag(tags.kind.map(|k| k.as_str().to_owned()).unwrap_or_default());
    event.provider_featured = tags.featured;
    event.branded_provider = sanitize_tag(tags.branded_provider.unwrap_or_default());
    event.pk_label = sanitize_tag(tags.pk_label.unwrap_or_default());
    event.byo_label = sanitize_tag(tags.byo_label.unwrap_or_default());
}

/// Stamp the JWT identity attribution fields onto an in-progress
/// UsageEvent (AISIX-Cloud#564). A `None` identity (the API-key path)
/// leaves the fields empty, which skip-serialize to wire NULL. One
/// source of truth for the mapping so the handler family can't drift —
/// same rationale as [`apply_pk_telemetry`]. The values are sanitised
/// like every other externally-influenced tag: the subject is a claim
/// from a verified token, but the identity provider is still not a
/// trusted emitter of control characters or unbounded strings.
pub(crate) fn apply_jwt_identity(
    event: &mut UsageEvent,
    jwt: Option<&std::sync::Arc<crate::auth::JwtIdentity>>,
) {
    let Some(jwt) = jwt else {
        return;
    };
    event.jwt_subject = sanitize_tag(jwt.subject.clone());
    event.jwt_provider = sanitize_tag(jwt.provider.clone());
    event.jwt_claim_mapping = sanitize_tag(jwt.claim_mapping.clone().unwrap_or_default());
}

/// Emit ONE zero-token `UsageEvent` for a FAILED request on a non-chat handler
/// (completions / embeddings / rerank / audio / images / passthrough / jobs /
/// realtime), so the dashboard Logs and budget ledger surface the failure
/// (status and bounded error class) instead of dropping it. Mirrors the #655
/// behavior chat / messages / responses already have: those endpoints emit a
/// zero-token event per failed attempt; the single-attempt non-chat handlers
/// emit one terminal event here.
///
/// `model_id` is intentionally left empty — on the error path the resolved
/// Model id isn't threaded back out of dispatch, but `requested_model`,
/// `api_key_id`, `status_code` and `error_class` are enough for the request to
/// appear in Logs. `label` is the usage_sink bucket (#408).
/// `inbound_protocol` must match what the caller's success path emits (e.g.
/// `"passthrough"` for `/passthrough/...`, `"realtime"` for `/v1/realtime`,
/// `"openai"` for the OpenAI-shaped handlers) so Logs protocol filtering sees
/// failures and successes under the same tag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_error_usage_event(
    state: &ProxyState,
    snap: &AisixSnapshot,
    label: &'static str,
    inbound_protocol: &'static str,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    error_class: &str,
    client: &ClientContext,
) {
    let mut event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        status_code,
        inbound_protocol: inbound_protocol.to_string(),
        error_class: error_class.to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    apply_jwt_identity(&mut event, client.jwt.as_ref());
    state.usage_sink.try_emit(label, event.clone());
    let exporters = live_exporters(state, snap);
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_model_label_three_outcomes() {
        use aisix_core::resource::ResourceEntry;
        use aisix_core::snapshot::ResourceTable;
        let table = ResourceTable::default();
        let wildcard: aisix_core::Model = serde_json::from_value(serde_json::json!({
            "display_name": "openai/*",
            "provider": "openai",
            "model_name": "*",
            "provider_key_id": "pk-1",
        }))
        .unwrap();
        table.insert(ResourceEntry::new("m-star", wildcard, 1));
        let snap = AisixSnapshot {
            models: table,
            ..Default::default()
        };
        // Exact configured name labels as itself (the wildcard row's own
        // name IS an exact name).
        assert_eq!(metric_model_label(&snap, "openai/*"), "openai/*");
        // A caller-minted suffix a wildcard row serves labels as the ROW,
        // not the unbounded concrete string (#451 class on the success
        // path).
        assert_eq!(metric_model_label(&snap, "openai/gpt-4o"), "openai/*");
        assert_eq!(
            metric_model_label(&snap, "openai/anything-else"),
            "openai/*"
        );
        // Unservable text stays the sentinel.
        assert_eq!(
            metric_model_label(&snap, "no-such/model"),
            UNRESOLVED_MODEL_LABEL
        );
    }

    /// AISIX-Cloud#1289: the id is upstream-controlled and reaches a log line
    /// and cp-api's `dpmgr_usage_events`. A newline in it would break the
    /// one-record-per-line shape every log consumer relies on, and an
    /// unbounded id would ride every line for the request. Both entry points
    /// must normalise, because producers use whichever fits their decode:
    /// JSON bodies take the `Value` form, bridge/stream structs the `&str`.
    #[test]
    fn both_entry_points_strip_control_chars_and_cap_length() {
        let injected = "chatcmpl-1\nfake-log-line status=200";
        assert_eq!(
            sanitize_provider_response_id(injected),
            "chatcmpl-1fake-log-line status=200",
        );
        assert_eq!(
            provider_response_id(&serde_json::json!({ "id": injected })),
            "chatcmpl-1fake-log-line status=200",
        );

        let long = "x".repeat(1000);
        assert_eq!(sanitize_provider_response_id(&long).chars().count(), 256);
        assert_eq!(
            provider_response_id(&serde_json::json!({ "id": long }))
                .chars()
                .count(),
            256,
        );
    }

    /// A response with no id — the normal case on several endpoints — must
    /// stay empty rather than becoming a placeholder, and a non-string `id`
    /// must not be coerced into one.
    #[test]
    fn absent_or_non_string_id_stays_empty() {
        assert_eq!(sanitize_provider_response_id(""), "");
        assert_eq!(provider_response_id(&serde_json::json!({})), "");
        assert_eq!(provider_response_id(&serde_json::json!({ "id": 42 })), "");
        assert_eq!(provider_response_id(&serde_json::json!({ "id": null })), "");
    }

    const PK_ID: &str = "22222222-2222-2222-2222-222222222222";

    fn snap_with_pk(display_name: &str, tags: &str) -> AisixSnapshot {
        let json = format!(
            r#"{{"display_name":{},"secret":"sk-up","api_base":"http://up","provider":"openai","adapter":"openai"{}}}"#,
            serde_json::to_string(display_name).unwrap(),
            tags,
        );
        let pk: ProviderKey = serde_json::from_str(&json).unwrap();
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap
    }

    /// The `provider_key_id` / `provider_key_name` pair is a metric label
    /// set — a changed fallback would silently split every series that
    /// carries it. Both halves report `"unknown"` when nothing resolved,
    /// which is what `Upstream::default()` and the pre-dispatch rejection
    /// paths rely on.
    #[test]
    fn unresolved_key_labels_both_halves_unknown() {
        let snap = AisixSnapshot::new();
        for pk in [
            ResolvedPk::unresolved(),
            ResolvedPk::resolve(&snap, ""),
            ResolvedPk::resolve(&snap, UNKNOWN_PK),
            ResolvedPk::resolve(&snap, PK_ID),
        ] {
            assert_eq!(pk.labels().name, "unknown");
            assert_eq!(pk.telemetry_tags(), aisix_core::TelemetryTags::default());
        }
        assert_eq!(PkLabels::default().id, "unknown");
        assert_eq!(PkLabels::default().name, "unknown");
    }

    /// The id reaches the label verbatim even when it resolves to nothing —
    /// an id the operator deleted mid-request still names which key the
    /// request tried to use, and rewriting it to `"unknown"` would merge
    /// those samples with the never-resolved ones.
    #[test]
    fn unresolvable_id_still_reaches_the_label() {
        let snap = AisixSnapshot::new();
        assert_eq!(
            ResolvedPk::resolve(&snap, "pk-deleted").labels().id,
            "pk-deleted"
        );
        assert_eq!(ResolvedPk::resolve(&snap, "").labels().id, "");
    }

    /// One lookup now feeds the metric label AND the wire attribution tags,
    /// so both have to come off the same row.
    #[test]
    fn one_resolve_serves_both_the_label_and_the_tags() {
        let snap = snap_with_pk(
            "prod-openai",
            r#","telemetry_tags":{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod"}"#,
        );
        let pk = ResolvedPk::resolve(&snap, PK_ID);
        assert_eq!(pk.labels().id, PK_ID);
        assert_eq!(pk.labels().name, "prod-openai");
        let tags = pk.telemetry_tags();
        assert!(tags.featured);
        assert_eq!(tags.branded_provider.as_deref(), Some("openai"));
        assert_eq!(tags.pk_label.as_deref(), Some("prod"));
    }

    /// `display_name` is operator-controlled and reaches a Prometheus label,
    /// so it keeps the `sanitize_tag` treatment the per-emit lookup applied:
    /// control characters stripped, 256 chars max. A name that sanitises
    /// away entirely falls back to `"unknown"` rather than emitting a blank
    /// label value.
    #[test]
    fn display_name_is_sanitised_and_blank_falls_back() {
        let snap = snap_with_pk("prod\nopenai", "");
        assert_eq!(
            ResolvedPk::resolve(&snap, PK_ID).labels().name,
            "prodopenai"
        );

        let long = "x".repeat(1000);
        let snap = snap_with_pk(&long, "");
        let pk = ResolvedPk::resolve(&snap, PK_ID);
        assert_eq!(pk.labels().name.chars().count(), 256);

        let snap = snap_with_pk("\u{1}\u{2}", "");
        assert_eq!(ResolvedPk::resolve(&snap, PK_ID).labels().name, "unknown");
    }
}
