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

use aisix_core::AisixSnapshot;
use aisix_obs::UsageEvent;

use crate::chat::sanitize_tag;
use crate::client_ip::ClientContext;
use crate::state::ProxyState;

/// Resolve a ProviderKey's telemetry attribution tags from the live snapshot.
/// An empty `provider_key_id` (pre-dispatch error paths) or an id with no
/// matching row yields the default (all-empty) tags, which serialise to wire
/// NULL — same contract as the chat / messages emitters.
pub(crate) fn provider_telemetry_tags(
    snap: &AisixSnapshot,
    provider_key_id: &str,
) -> aisix_core::TelemetryTags {
    if provider_key_id.is_empty() {
        return Default::default();
    }
    snap.provider_keys
        .get_by_id(provider_key_id)
        .map(|e| e.value.telemetry_tags.clone())
        .unwrap_or_default()
}

/// Resolve the readable provider-key NAME for the #890 req-3 metric label
/// (`provider_key_name`). Returns the ProviderKey's `display_name`
/// (control-char stripped + length-capped via [`sanitize_tag`]) or
/// `"unknown"` when the id is empty / unresolved / blank. 1:1 with the
/// `provider_key_id`, so it adds no metric series. Shared by the chat +
/// messages metric emitters so the value can't drift between handlers.
pub(crate) fn provider_key_metric_name(snap: &AisixSnapshot, provider_key_id: &str) -> String {
    if provider_key_id.is_empty() {
        return "unknown".to_string();
    }
    let name = snap
        .provider_keys
        .get_by_id(provider_key_id)
        .map(|e| sanitize_tag(e.value.display_name.clone()))
        .unwrap_or_default();
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name
    }
}

/// Stamp the five per-PK attribution fields onto an in-progress UsageEvent,
/// sanitising the operator-controlled tag strings (control-char strip + length
/// cap) before they hit the wire. One source of truth for the mapping so the
/// non-chat handlers can't diverge from chat / messages.
pub(crate) fn apply_pk_telemetry(
    event: &mut UsageEvent,
    snap: &AisixSnapshot,
    provider_key_id: &str,
) {
    let tags = provider_telemetry_tags(snap, provider_key_id);
    event.provider_kind =
        sanitize_tag(tags.kind.map(|k| k.as_str().to_owned()).unwrap_or_default());
    event.provider_featured = tags.featured;
    event.branded_provider = sanitize_tag(tags.branded_provider.unwrap_or_default());
    event.pk_label = sanitize_tag(tags.pk_label.unwrap_or_default());
    event.byo_label = sanitize_tag(tags.byo_label.unwrap_or_default());
}

/// Emit ONE zero-token `UsageEvent` for a FAILED request on a non-chat handler
/// (completions / embeddings / rerank / audio / images), so the dashboard Logs
/// and budget ledger surface the failure (status and bounded error class)
/// instead of dropping it. Mirrors the #655 behavior chat / messages /
/// responses already have: those endpoints emit a zero-token event per failed
/// attempt; the single-attempt non-chat handlers emit one terminal event here.
///
/// `model_id` is intentionally left empty — on the error path the resolved
/// Model id isn't threaded back out of dispatch, but `requested_model`,
/// `api_key_id`, `status_code` and `error_class` are enough for the request to
/// appear in Logs. `label` is the usage_sink bucket (#408); all five callers
/// are OpenAI-shaped, so `inbound_protocol` is `"openai"`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_error_usage_event(
    state: &ProxyState,
    label: &'static str,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    error_class: &str,
    client: &ClientContext,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        status_code,
        inbound_protocol: "openai".to_string(),
        error_class: error_class.to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        ..Default::default()
    };
    state.usage_sink.try_emit(label, event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}
