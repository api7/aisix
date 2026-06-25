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
