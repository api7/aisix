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
use aisix_obs::{UsageEvent, UsageEventLabels};

use crate::chat::sanitize_tag;
use crate::client_ip::ClientContext;
use crate::operation::Surface;
use crate::state::ProxyState;

/// The request's ENFORCE-mode guardrail audit handle (AISIX-Cloud#1330),
/// cloned off the resolved chain with
/// [`GuardrailChain::audit_log`](aisix_guardrails::GuardrailChain::audit_log)
/// at the point the chain is resolved.
///
/// Threaded rather than re-derived because the read has to happen at
/// terminal-event time, which is routinely a different frame from the
/// chain's: a streaming emitter runs from a `move` closure after the
/// handler returned, and `chat.rs` erases its chain to `Arc<dyn Guardrail>`
/// before the output hook even runs. `None` — no chain resolved, the
/// dominant guardrail-free deployment — reads identically to an empty
/// snapshot, so no call site needs to branch on it.
pub(crate) type GuardrailAudit = Option<Arc<aisix_guardrails::GuardrailAuditLog>>;

/// Snapshot `audit` into a terminal `UsageEvent`'s
/// `guardrail_enforced_hits`.
///
/// Terminal events only: guardrails run once per request, not once per
/// attempt, so stamping a per-attempt event would report the same hit
/// once per retry.
///
/// Reading does not consume the log, and that is load-bearing rather than
/// incidental: the retrying families call this from emitters that also
/// serve superseded attempts, and a destructive read would leave whichever
/// event happened to go out first holding the only copy.
pub(crate) fn enforced_hits(audit: &GuardrailAudit) -> Vec<aisix_core::GuardrailEnforcedHit> {
    audit.as_ref().map(|a| a.snapshot()).unwrap_or_default()
}

/// Snapshot `audit` into a `UsageEvent`'s `guardrail_bypassed_reason`:
/// the bounded tag of the first guardrail this request failed OPEN on,
/// empty when none did.
///
/// Not terminal-only, unlike [`enforced_hits`]. A bypass says the request
/// went upstream unscreened, which is true of every attempt the request
/// made, and `chat.rs` has stamped it on its per-attempt events since the
/// field existed — a retry that dropped it would read as a screened
/// attempt.
///
/// `chat.rs` keeps threading its own copy instead of calling this: it
/// hands the value to per-attempt and ensemble sub-call emitters at
/// points where the value is deliberately the one captured EARLIER in the
/// request, which a request-scoped snapshot cannot express.
pub(crate) fn bypass_reason(audit: &GuardrailAudit) -> String {
    audit
        .as_ref()
        .and_then(|a| a.bypass_reason())
        .unwrap_or_default()
}

/// Snapshot `audit` into a terminal `UsageEvent`'s `guardrail_scores`
/// (AISIX-Cloud#1467). Same handle, same terminal-only rule and same
/// non-destructive read as [`enforced_hits`] — a scoring guardrail runs
/// once per request, so a per-attempt event would repeat its numbers once
/// per retry and make a single screening look like several.
pub(crate) fn guardrail_scores(audit: &GuardrailAudit) -> Vec<aisix_core::GuardrailScore> {
    audit
        .as_ref()
        .map(|a| a.score_snapshot())
        .unwrap_or_default()
}

/// Whether a request has a guardrail decision worth preserving even when
/// the handler has no token usage to report.
///
/// Merely having an attached guardrail is not enough: unsupported-provider
/// and unparseable-usage paths historically suppress zero-value noise rows.
/// A mask/block, monitor hit, or similarity score is an operator-visible
/// security fact, so those paths must emit a zero-token event instead.
pub(crate) fn has_guardrail_attribution(
    audit: &GuardrailAudit,
    monitor_hits: &[aisix_core::GuardrailMonitorHit],
) -> bool {
    !monitor_hits.is_empty()
        || audit
            .as_ref()
            .is_some_and(|log| !log.snapshot().is_empty() || !log.score_snapshot().is_empty())
}

/// [`guardrail_scores`] for the retrying families — see
/// [`terminal_enforced_hits`].
pub(crate) fn terminal_guardrail_scores(
    terminal: bool,
    audit: &GuardrailAudit,
) -> Vec<aisix_core::GuardrailScore> {
    if terminal {
        guardrail_scores(audit)
    } else {
        Vec::new()
    }
}

/// [`enforced_hits`] for the retrying families (chat / messages /
/// responses), whose emitters serve both the terminal event and the
/// superseded per-attempt ones and are told which they are building.
///
/// Written as one helper rather than an `if` at each emitter so the rule
/// — request-scoped attribution rides the terminal event only — is stated
/// once and cannot be applied inconsistently across the three.
pub(crate) fn terminal_enforced_hits(
    terminal: bool,
    audit: &GuardrailAudit,
) -> Vec<aisix_core::GuardrailEnforcedHit> {
    if terminal {
        enforced_hits(audit)
    } else {
        Vec::new()
    }
}

/// Stamp how the caller established its principal onto a usage event.
///
/// The ordinary credential path leaves the field empty (the shape on the
/// wire for the overwhelming majority of events); a principal inherited
/// from an entry's anonymous configuration is marked, so a key that
/// doubles as an anonymous principal can still be told apart in the
/// usage record. Every handler that can serve an anonymous caller —
/// today `/mcp` and the passthrough routes — calls this on its emitter.
pub(crate) fn apply_auth_type(event: &mut UsageEvent, auth: &crate::auth::AuthenticatedKey) {
    if auth.anonymous {
        event.auth_type = "anonymous".to_string();
    }
}

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
    /// The upstream wire protocol this key dispatches through
    /// (AISIX-Cloud#1403) — [`aisix_gateway::upstream_protocol`] of the
    /// resolved row, [`aisix_gateway::UPSTREAM_PROTOCOL_UNKNOWN`] when
    /// there is no row. Computed HERE rather than at the emit so it
    /// rides along with the id and name the same row produced, on the
    /// one snapshot read this type exists to make.
    protocol: &'static str,
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
        let protocol = entry
            .as_ref()
            .map(|e| aisix_gateway::upstream_protocol(&e.value))
            .unwrap_or(aisix_gateway::UPSTREAM_PROTOCOL_UNKNOWN);
        Self {
            id,
            name,
            protocol,
            entry,
        }
    }

    /// A completion that never reached a ProviderKey — the pre-dispatch
    /// rejections and the endpoints that have no upstream key at all.
    /// Same labels the per-emitter lookup produced for an empty id, with
    /// no snapshot read.
    pub(crate) fn unresolved() -> ResolvedPk<'static> {
        ResolvedPk {
            id: "",
            name: Cow::Borrowed(UNKNOWN_PK),
            protocol: aisix_gateway::UPSTREAM_PROTOCOL_UNKNOWN,
            entry: None,
        }
    }

    /// The ProviderKey dimensions of the metric label set.
    pub(crate) fn labels(&self) -> PkLabels<'_> {
        PkLabels {
            id: self.id,
            name: &self.name,
            protocol: self.protocol,
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

/// The ProviderKey dimensions of a metric label set: the id, the
/// readable name, and the upstream wire protocol — all three functions
/// of one ProviderKey row, so the trio adds no series beyond the id.
///
/// The fields are private on purpose. [`ResolvedPk::labels`] and
/// [`PkLabels::default`] are the only ways to build one, so a name or a
/// protocol can never be paired with an id it was not read off — which
/// is the whole reason `Upstream` takes this type instead of a bare id.
#[derive(Clone, Copy)]
pub(crate) struct PkLabels<'a> {
    id: &'a str,
    name: &'a str,
    protocol: &'static str,
}

impl<'a> PkLabels<'a> {
    pub(crate) fn id(self) -> &'a str {
        self.id
    }

    pub(crate) fn name(self) -> &'a str {
        self.name
    }

    /// The `upstream_protocol` label (AISIX-Cloud#1403).
    pub(crate) fn protocol(self) -> &'static str {
        self.protocol
    }
}

impl Default for PkLabels<'_> {
    fn default() -> Self {
        Self {
            id: UNKNOWN_PK,
            name: UNKNOWN_PK,
            protocol: aisix_gateway::UPSTREAM_PROTOCOL_UNKNOWN,
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
pub const UNRESOLVED_MODEL_LABEL: &str = "unresolved";

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

/// Fixed non-model surface labels the emit chokepoint must pass through
/// byte-identical: rewriting any of these would silently split every
/// series that carries it (the `unknown` note below) and merge distinct
/// surfaces into the `unresolved` bucket. Also an O(1) short-circuit so
/// the tunnel hot paths (passthrough / MCP / A2A / jobs) never pay the
/// wildcard scan.
const FIXED_SURFACE_MODEL_LABELS: &[&str] = &[
    UNRESOLVED_MODEL_LABEL,
    "passthrough",
    "mcp",
    "a2a",
    "unknown",
    "files",
    "batches",
    "fine_tuning",
];

/// The `(model, upstream_model)` label pair for the emit chokepoint —
/// COLLAPSE-ONLY: a name only a wildcard row serves folds to the row's
/// `(display_name, model_name template)` (both halves of a wildcard hit
/// are caller-derived); every other value passes through verbatim.
/// Sentinel fallbacks stay a HANDLER decision (`metric_model_label` on
/// the pre-resolution error paths) — the chokepoint must never rewrite
/// a caller's fixed surface label.
pub(crate) fn metric_model_label_pair<'a>(
    snap: &AisixSnapshot,
    model_name: &'a str,
    upstream_model: &'a str,
) -> (Cow<'a, str>, Cow<'a, str>) {
    if FIXED_SURFACE_MODEL_LABELS.contains(&model_name)
        || snap.models.get_by_name(model_name).is_some()
    {
        return (Cow::Borrowed(model_name), Cow::Borrowed(upstream_model));
    }
    match crate::model_resolve::wildcard_row_identity(snap, model_name) {
        Some((row, template)) => (Cow::Owned(row), Cow::Owned(template)),
        None => (Cow::Borrowed(model_name), Cow::Borrowed(upstream_model)),
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

/// Stamp the caller-identity attribution fields onto an in-progress
/// UsageEvent: the JWT identity (AISIX-Cloud#564) and the org member the
/// authenticating key belongs to (AISIX-Cloud#1389). A `None` in either
/// argument leaves its fields empty, which skip-serialize to wire NULL.
/// One source of truth for the mapping so the handler family can't drift —
/// same rationale as [`apply_pk_telemetry`]. The values are sanitised
/// like every other externally-influenced tag: the subject is a claim
/// from a verified token, but the identity provider is still not a
/// trusted emitter of control characters or unbounded strings.
///
/// `user_id` takes both halves of the auth surface: on the API-key path it
/// is the key's own `user_id`, and on the JWT path it is the `user_id` of
/// the key the token resolved to — a JWT request runs AS a key, so one
/// argument covers both and the member filter sees every credential a
/// member calls through.
///
/// `user_name` is that member's display name for the metric label of the
/// same name (AISIX-Cloud#1455). It is a separate argument rather than a
/// separate call so the compiler makes every emitting path hand over both
/// halves of one ApiKey row: a name sourced anywhere but beside its id is
/// a name that will eventually label the wrong member.
pub(crate) fn apply_caller_identity(
    event: &mut UsageEvent,
    jwt: Option<&std::sync::Arc<crate::auth::JwtIdentity>>,
    user_id: Option<&str>,
    user_name: Option<&str>,
) {
    if let Some(user_id) = user_id {
        event.user_id = sanitize_tag(user_id.to_string());
    }
    if let Some(user_name) = user_name {
        // NOT `sanitize_tag`, unlike the id above. That call exists to keep
        // operator-typed text safe on the cp-api wire and in the log line;
        // `user_name` reaches neither (`#[serde(skip)]`, and
        // `log_provider_call` does not render it). Its only destination is
        // the `user_name` metric label, which the four #890 families
        // already stamp raw off the same ApiKey row — sanitising this one
        // copy would make a long or odd name read differently on
        // `aisix_usage_event*` than on `aisix_llm_*`, and break the
        // cross-family join on the pair for exactly that member. The
        // prometheus exporter escapes exposition characters itself.
        event.user_name = user_name.to_string();
    }
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
///
/// A guardrail refusal reaches this function like any other failure, so
/// `guardrail_blocked` is a REQUIRED argument rather than a default: a
/// handler that forgets it produces a 422 the "Guardrail blocks" view
/// cannot find, which reads to an operator as "the gateway logged no
/// guardrail activity at all" (AISIX-Cloud#1428).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_error_usage_event(
    state: &ProxyState,
    snap: &AisixSnapshot,
    surface: Surface,
    inbound_protocol: &'static str,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    error_class: &str,
    // Whether this failure IS a guardrail refusal, from
    // [`ProxyError::is_guardrail_block`]. Every caller but the realtime
    // connect failure — which synthesizes its class without a `ProxyError`
    // — reads it off the error it is reporting.
    guardrail_blocked: bool,
    client: &ClientContext,
    // The request's enforced guardrail hits. The failure path is where a
    // `blocked` hit lands — a guardrail refusal IS the error — so the
    // error event is the one that must not drop it. Drained by the caller
    // (`enforced_hits(&audit)`) rather than taken as a handle: the jobs
    // surface accumulates across two separately resolved chains.
    enforced: Vec<aisix_core::GuardrailEnforcedHit>,
    // The request's similarity scores (AISIX-Cloud#1467), drained the same
    // way. A refusal by a `kind: semantic` row lands HERE, so an error
    // event without them would leave the block — the one outcome an
    // operator is certain to look at — as the only execution with no
    // number attached.
    scores: Vec<aisix_core::GuardrailScore>,
    // The request's fail-open bypass tag, drained the same way. A request
    // that went upstream unscreened and THEN failed is still a request
    // that went upstream unscreened, so dropping the tag here would make
    // the field answerable only for the requests that succeeded.
    bypass: String,
) {
    let event = build_error_usage_event(
        inbound_protocol,
        request_id,
        requested_model,
        api_key_id,
        status_code,
        error_class,
        guardrail_blocked,
        client,
        enforced,
        scores,
        bypass,
    );
    // The failed request's own attribution, off the same cell
    // `request_metrics::LastTarget` reads — so the usage-event counters and
    // the request counters agree on which key a failure belongs to
    // (AISIX-Cloud#1317 / #1325).
    let attributed = crate::attribution::current().unwrap_or_default();
    let pk = ResolvedPk::resolve(snap, &attributed.provider_key_id);
    let model = usage_event_model_label(snap, requested_model);
    emit_prepared_usage_event(
        state,
        snap,
        surface,
        event,
        usage_event_labels(&model, &pk),
        client.trace.as_ref(),
    );
}

/// The [`emit_error_usage_event`] event without the emission, for a caller
/// that attributes handler-specific fields (e.g. the passthrough route name)
/// before handing it to [`emit_prepared_usage_event`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_error_usage_event(
    inbound_protocol: &'static str,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    error_class: &str,
    // See [`emit_error_usage_event`].
    guardrail_blocked: bool,
    client: &ClientContext,
    enforced: Vec<aisix_core::GuardrailEnforcedHit>,
    scores: Vec<aisix_core::GuardrailScore>,
    // See [`emit_error_usage_event`].
    bypass: String,
) -> UsageEvent {
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
        guardrail_blocked,
        guardrail_enforced_hits: enforced,
        guardrail_scores: scores,
        guardrail_bypassed_reason: bypass,
        ..Default::default()
    };
    apply_caller_identity(
        &mut event,
        client.jwt.as_ref(),
        client.caller.user_id.as_deref(),
        client.caller.user_name.as_deref(),
    );
    event
}

/// The `model` label for the usage-event counters (AISIX-Cloud#1317).
///
/// `unknown` when the request carried no model at all — the MCP, A2A and
/// passthrough tunnels, and every path that failed before resolution —
/// otherwise collapsed to the configured set exactly like the request
/// families do, because `UsageEvent::requested_model` is caller-controlled
/// text and would otherwise mint one series per made-up name (#451).
pub(crate) fn usage_event_model_label<'a>(
    snap: &AisixSnapshot,
    requested_model: &'a str,
) -> Cow<'a, str> {
    if requested_model.is_empty() {
        Cow::Borrowed(crate::request_metrics::UNKNOWN)
    } else {
        metric_model_label(snap, requested_model)
    }
}

/// Pair that label with the ProviderKey the event is attributed to.
///
/// Takes the resolved key rather than an id for the same reason
/// `request_metrics::Upstream` does: the readable name and the upstream
/// protocol can only come off the row its id names.
pub(crate) fn usage_event_labels<'a>(
    model: &'a str,
    pk: &'a ResolvedPk<'_>,
) -> UsageEventLabels<'a> {
    let labels = pk.labels();
    UsageEventLabels {
        model,
        // `ResolvedPk` reports an unresolved id verbatim, including the
        // empty string the pre-dispatch paths pass. An empty label value
        // would sit next to `unknown` as a second "nothing resolved"
        // series, so collapse it here.
        provider_key_id: if labels.id().is_empty() {
            UNKNOWN_PK
        } else {
            labels.id()
        },
        provider_key_name: labels.name(),
        // Both overwritten by `UsageSink::try_emit` from the event's own
        // member pair (AISIX-Cloud#1389, #1455) — one place, so no handler
        // in this family can build a label set that disagrees with the row.
        user_id: "unknown",
        user_name: "unknown",
        // Same `PkLabels` the request families read (AISIX-Cloud#1403), so
        // `aisix_usage_events_emitted_total` joins with them on
        // `upstream_protocol` instead of dropping out of the aggregation.
        upstream_protocol: labels.protocol(),
    }
}

pub(crate) fn emit_prepared_usage_event(
    state: &ProxyState,
    snap: &AisixSnapshot,
    surface: Surface,
    event: UsageEvent,
    labels: UsageEventLabels<'_>,
    trace: Option<&std::sync::Arc<aisix_obs::RequestTraceBundle>>,
) {
    // An error event is the request's terminal (and only) emission. Its
    // builder leaves `upstream_latency_ms` at 0, so no upstream span is
    // derivable either way — `dispatched: false` states the common case
    // (most error events never reached a provider).
    emit_usage(
        state, snap, surface, event, labels, None, trace, true, false,
    );
}

/// THE emission chokepoint (AISIX-Cloud#1279): every usage event leaves
/// through here, so the CP telemetry leg and the exporter fan-out cannot
/// drift, and the trace snapshot is taken in exactly one place.
///
/// `trace` is the request's bundle (`ClientContext::trace`); `terminal`
/// says whether this event ends the request — the terminal event carries
/// the SERVER + logical spans (ending the SERVER span at NOW, the real
/// body EOF/drop), a non-terminal one carries its own attempt span alone.
/// `dispatched` is the caller's statement that this event describes work
/// that actually reached an upstream — a cache hit and a pre-dispatch
/// error pass `false` so no fictitious upstream CLIENT span is fabricated
/// from the handler-elapsed time their events carry. The event's public
/// `trace_id` is stamped here so the CP row and the exported spans always
/// agree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_usage(
    state: &ProxyState,
    snap: &AisixSnapshot,
    surface: Surface,
    mut event: UsageEvent,
    labels: UsageEventLabels<'_>,
    content: Option<&aisix_obs::CapturedContent>,
    trace: Option<&std::sync::Arc<aisix_obs::RequestTraceBundle>>,
    terminal: bool,
    dispatched: bool,
) {
    // Stamped here rather than at each construction site: this is the one
    // point every usage event passes through, so the operation cannot be
    // present on a family's success path and missing from its error or
    // streaming one (AISIX-Cloud#1461).
    event.operation = surface.operation.to_string();
    // Request-level guardrail blocks are recorded from the terminal event,
    // not from an individual timed execution. Some fail-closed paths (for
    // example a streamed-output buffer overflow) reject before a guardrail
    // member runs, while retries and ensembles can emit several non-terminal
    // events for one request.
    if terminal && event.guardrail_blocked {
        state.metrics.record_guardrail_blocked_request();
    }
    let emission = trace.map(|bundle| {
        event.trace_id = bundle.trace_id_hex();
        bundle.emission(
            terminal,
            event.attempt_index,
            event.upstream_latency_ms,
            dispatched,
        )
    });
    state
        .usage_sink
        .try_emit(surface.handler, event.clone(), labels);
    let exporters = live_exporters(state, snap);
    state.otlp_fan_out.fan_out(
        &event,
        content,
        emission.as_ref(),
        exporters.iter().map(|e| &e.value),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_score_alone_requires_a_zero_token_event() {
        let log = Arc::new(aisix_guardrails::GuardrailAuditLog::new());
        let audit = Some(Arc::clone(&log));
        assert!(!has_guardrail_attribution(&audit, &[]));

        log.record_score(aisix_core::GuardrailScore {
            guardrail_name: "semantic-policy".into(),
            hook: "input".into(),
            direction: "deny".into(),
            score: 0.7,
            threshold: 0.8,
            matched: false,
            top_example_index: 0,
            embedding_model: "embedder".into(),
        });
        assert!(has_guardrail_attribution(&audit, &[]));
    }

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

        // The emit-chokepoint pair is COLLAPSE-ONLY: a wildcard hit folds
        // BOTH halves to the row's configured identities…
        let (m, u) = metric_model_label_pair(&snap, "openai/gpt-4o", "gpt-4o");
        assert_eq!((m.as_ref(), u.as_ref()), ("openai/*", "*"));
        // …an exact hit passes both through…
        let (m, u) = metric_model_label_pair(&snap, "openai/*", "somemodel");
        assert_eq!((m.as_ref(), u.as_ref()), ("openai/*", "somemodel"));
        // …and fixed surface sentinels and unresolvable values are NEVER
        // rewritten (rewriting `mcp`/`passthrough`/`unknown` would split
        // every series that carries them).
        for sentinel in ["passthrough", "mcp", "a2a", "unknown"] {
            let (m, u) = metric_model_label_pair(&snap, sentinel, "x");
            assert_eq!((m.as_ref(), u.as_ref()), (sentinel, "x"));
        }
        let (m, u) = metric_model_label_pair(&snap, "no-such/model", "raw-upstream");
        assert_eq!((m.as_ref(), u.as_ref()), ("no-such/model", "raw-upstream"));
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
            // AISIX-Cloud#1403: same fallback, same reason — a request
            // that reached no key reached no protocol either.
            assert_eq!(pk.labels().protocol, "unknown");
            assert_eq!(pk.telemetry_tags(), aisix_core::TelemetryTags::default());
        }
        assert_eq!(PkLabels::default().id, "unknown");
        assert_eq!(PkLabels::default().name, "unknown");
        assert_eq!(PkLabels::default().protocol, "unknown");
    }

    /// AISIX-Cloud#1403: the protocol is read off the SAME row as the id
    /// and the name, on the one lookup this type exists to make, so the
    /// three can never describe different keys.
    #[test]
    fn resolved_key_carries_its_upstream_protocol() {
        let snap = snap_with_pk("prod-openai", "");
        let pk = ResolvedPk::resolve(&snap, PK_ID);
        assert_eq!(pk.labels().id, PK_ID);
        assert_eq!(pk.labels().protocol, "openai");

        // A key whose vendor has no specialized bridge takes its
        // adapter's wire value.
        let json = r#"{"display_name":"byo-anthropic","secret":"sk","provider":"byo","adapter":"anthropic"}"#;
        let byo: ProviderKey = serde_json::from_str(json).unwrap();
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, byo, 1));
        assert_eq!(
            ResolvedPk::resolve(&snap, PK_ID).labels().protocol,
            "anthropic",
            "an open vendor string cannot stand in for the adapter"
        );
    }

    /// …and the usage-event counter's label set is built off that same
    /// resolved key, so `aisix_usage_events_emitted_total` reports the
    /// protocol of the key its `provider_key_id` names.
    ///
    /// It was the one family carrying `inbound_protocol` that had the
    /// value in hand and did not emit it (found in 0.11.0-rc.2 QA), which
    /// left it unjoinable with the families that did:
    /// `sum by (upstream_protocol)` returned an aggregate with this
    /// counter silently missing from it.
    #[test]
    fn usage_event_labels_carry_the_resolved_upstream_protocol() {
        let snap = snap_with_pk("prod-openai", "");
        let pk = ResolvedPk::resolve(&snap, PK_ID);
        let labels = usage_event_labels("gpt-4o", &pk);
        assert_eq!(labels.provider_key_id, PK_ID);
        assert_eq!(labels.provider_key_name, "prod-openai");
        // The UPSTREAM's protocol. An Anthropic-protocol caller served by
        // this key must still read `openai` here — the inbound protocol
        // travels on its own label.
        assert_eq!(labels.upstream_protocol, "openai");

        // Nothing resolved: `unknown`, matching the id/name fallback
        // beside it and the value the request families use for a
        // pre-dispatch rejection.
        let unresolved = ResolvedPk::unresolved();
        let labels = usage_event_labels("unknown", &unresolved);
        assert_eq!(labels.provider_key_id, UNKNOWN_PK);
        assert_eq!(labels.upstream_protocol, "unknown");

        // A deleted id keeps reaching the label verbatim, but there is no
        // row left to read a protocol off.
        let empty = AisixSnapshot::new();
        let gone = ResolvedPk::resolve(&empty, "pk-deleted");
        let labels = usage_event_labels("gpt-4o", &gone);
        assert_eq!(labels.provider_key_id, "pk-deleted");
        assert_eq!(labels.upstream_protocol, "unknown");
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

    /// Every `UsageEvent` this crate builds must set
    /// `guardrail_bypassed_reason`, and an emitter that genuinely has no
    /// chain behind it must say so where it is written.
    ///
    /// The field defaults to empty, so an emitter that forgets it reports
    /// "nothing was bypassed" for a request that went upstream unscreened —
    /// and no behavioural test can see the omission, because an unset field
    /// and a screened request produce the same row. That is exactly how the
    /// field spent its whole life written on `/v1/chat/completions` and
    /// nowhere else.
    ///
    /// A parse of the source rather than a list of emitters, for the reason
    /// `guardrail_coverage` parses the router instead of listing routes: a
    /// hand-written list nobody updates agrees with itself forever, and the
    /// emitter family here has sixteen members that keep gaining a
    /// seventeenth.
    #[test]
    fn every_usage_event_this_crate_builds_answers_the_bypass_question() {
        /// Written inside an emitter that has no guardrail chain behind it
        /// at all, followed by why.
        const EXEMPT: &str = "NO-GUARDRAIL-CHAIN:";

        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut blocks = 0usize;
        let mut missing = Vec::new();

        // Recursive: `src/` is flat today, and a refactor that moved a
        // handler into a subdirectory would otherwise take its emitter out
        // of scope while this still reported green.
        fn rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let entries = std::fs::read_dir(dir).expect("the crate's own src/ must be readable");
            for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
                if path.is_dir() {
                    rs_files(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let mut files = Vec::new();
        rs_files(&src_dir, &mut files);
        files.sort();

        for path in files {
            let src = std::fs::read_to_string(&path).expect("source must read");
            let name = path
                .strip_prefix(&src_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            let code = code_mask(&src);
            for (idx, _) in src.match_indices("UsageEvent {") {
                // Skip anything that is not Rust code: this file writes the
                // token it searches for in a string literal and in a
                // comment, and both would otherwise be counted as emitters
                // — a check inflating its own coverage.
                if !code[idx] {
                    continue;
                }
                // `-> UsageEvent {` (optionally path-qualified) is a
                // return type followed by a function body, not a literal.
                let before = src[..idx]
                    .trim_end()
                    .trim_end_matches(|c: char| c.is_alphanumeric() || c == '_' || c == ':')
                    .trim_end();
                if before.ends_with("->") {
                    continue;
                }
                let open = idx + "UsageEvent ".len();
                let Some(block) = braced_block(&src, &code, open) else {
                    panic!("{name}: unbalanced UsageEvent literal at byte {idx}");
                };
                blocks += 1;
                if !sets_bypass_reason(block) && !block.contains(EXEMPT) {
                    let line = src[..idx].lines().count();
                    missing.push(format!("{name}:{line}"));
                }
            }
        }

        // The parse must actually find the emitters, not silently yield an
        // empty set that makes the assertion below vacuous.
        // The crate's real count. A floor rather than an equality so
        // adding an emitter does not fail here (the `missing` check
        // already governs a new one) — but losing four to a parse that
        // quietly stopped matching is the drift this exists to catch.
        assert!(
            blocks >= 18,
            "the UsageEvent literal scan found only {blocks} — it has stopped tracking the \
             emitter family",
        );
        assert!(
            missing.is_empty(),
            "these UsageEvent emitters neither set guardrail_bypassed_reason nor carry a \
             `{EXEMPT} <why>` comment saying they have no guardrail chain: {missing:?}\n\
             An unset field reports a screened request, so an emitter that skips it makes the \
             field unusable as a negative answer.",
        );
    }

    /// A byte mask over `src` marking the bytes that are Rust CODE —
    /// false inside string literals (raw ones included), char literals,
    /// line comments and nestable block comments.
    ///
    /// A real lexer pass rather than a per-line heuristic. The heuristic
    /// this replaces judged a match by whether its line prefix held a `//`
    /// or an odd number of quotes, which gets two things wrong that occur
    /// in ordinary code: a literal after a completed string on the same
    /// line (`let u = "https://x"; ... UsageEvent {`) reads as commented
    /// out, and an escaped quote miscounts the parity.
    fn code_mask(src: &str) -> Vec<bool> {
        #[derive(Clone, Copy)]
        enum St {
            Code,
            Str,
            Raw(usize),
            Char,
            Line,
            Block(usize),
        }
        let b = src.as_bytes();
        let mut mask = vec![false; b.len()];
        let mut st = St::Code;
        let mut i = 0;
        while i < b.len() {
            match st {
                St::Code => {
                    mask[i] = true;
                    match b[i] {
                        b'/' if b.get(i + 1) == Some(&b'/') => {
                            st = St::Line;
                            mask[i] = false;
                        }
                        b'/' if b.get(i + 1) == Some(&b'*') => {
                            st = St::Block(1);
                            mask[i] = false;
                            i += 2;
                            continue;
                        }
                        b'"' => st = St::Str,
                        // A char literal is `'x'` or `'\\n'`; anything
                        // else starting with a quote is a LIFETIME
                        // (`&'a str`), and treating one as a literal
                        // swallows every byte up to the next apostrophe.
                        b'\'' if b.get(i + 1) == Some(&b'\\') || b.get(i + 2) == Some(&b'\'') => {
                            st = St::Char
                        }
                        b'r' => {
                            // `r"…"` / `r#"…"#`, but not an identifier
                            // ending in `r`, and not a lifetime.
                            let prev_ident =
                                i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
                            let mut j = i + 1;
                            while b.get(j) == Some(&b'#') {
                                j += 1;
                            }
                            if !prev_ident && b.get(j) == Some(&b'"') {
                                st = St::Raw(j - i - 1);
                                i = j + 1;
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
                St::Str => {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == b'"' {
                        st = St::Code;
                    }
                }
                St::Raw(hashes) => {
                    if b[i] == b'"' && b[i + 1..].iter().take(hashes).all(|c| *c == b'#') {
                        st = St::Code;
                        i += hashes + 1;
                        continue;
                    }
                }
                St::Char => {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == b'\'' {
                        st = St::Code;
                    }
                }
                St::Line => {
                    if b[i] == b'\n' {
                        st = St::Code;
                    }
                }
                St::Block(depth) => {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        st = St::Block(depth + 1);
                        i += 2;
                        continue;
                    }
                    if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        st = if depth == 1 {
                            St::Code
                        } else {
                            St::Block(depth - 1)
                        };
                        i += 2;
                        continue;
                    }
                }
            }
            i += 1;
        }
        mask
    }

    /// Whether an emitter block ASSIGNS the field, rather than merely
    /// mentioning it.
    ///
    /// `block.contains("guardrail_bypassed_reason")` would be satisfied by
    /// a comment, and by `guardrail_bypassed_reason: String::new()` — which
    /// is exactly the shape of a reverted fix, so the check would have been
    /// unfalsifiable in the one dimension it is here to guard.
    fn sets_bypass_reason(block: &str) -> bool {
        let Some(at) = block.find("guardrail_bypassed_reason") else {
            return false;
        };
        let rest = &block[at + "guardrail_bypassed_reason".len()..];
        // Field-init shorthand (`guardrail_bypassed_reason,`) takes its
        // value from a binding, which cannot be an inline empty literal.
        let Some(rest) = rest.strip_prefix(':') else {
            return rest.starts_with(',');
        };
        // Values run to the end of their line, except the one that wraps
        // onto continuation lines — whose first line is a receiver, which
        // is non-empty and passes for the right reason.
        let value = rest
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches(',')
            .trim();
        !matches!(
            value,
            "" | "\"\"" | "String::new()" | "String::default()" | "Default::default()"
        )
    }

    /// The `{ … }` at `open`, skipping braces the mask says are not code.
    fn braced_block<'a>(src: &'a str, code: &[bool], open: usize) -> Option<&'a str> {
        let b = src.as_bytes();
        assert_eq!(b[open], b'{');
        let mut depth = 0usize;
        for i in open..b.len() {
            if !code[i] {
                continue;
            }
            match b[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&src[open..=i]);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The mask is load-bearing for the guard above, so it is pinned
    /// directly: every one of these would be a false positive or a false
    /// negative under the line-prefix heuristic it replaces.
    #[test]
    fn the_code_mask_excludes_strings_comments_and_raw_strings() {
        let at = |src: &str, needle: &str| {
            let m = code_mask(src);
            m[src.find(needle).expect("needle must appear")]
        };
        assert!(at("let x = 1; UsageEvent {}", "UsageEvent"));
        assert!(!at("let s = \"UsageEvent {\";", "UsageEvent"));
        assert!(!at("// UsageEvent {", "UsageEvent"));
        assert!(!at("/* a /* b */ UsageEvent */", "UsageEvent"));
        assert!(!at("let s = r#\"UsageEvent {\"#;", "UsageEvent"));
        // The two the heuristic got wrong: a completed string earlier on
        // the line, and an escaped quote inside one.
        assert!(at("let u = \"https://x\"; UsageEvent {}", "UsageEvent"));
        assert!(at("let e = \"a\\\"b\"; UsageEvent {}", "UsageEvent"));
        // A char literal must not open a string, and an identifier ending
        // in `r` must not open a raw string.
        assert!(at("let c = '\"'; UsageEvent {}", "UsageEvent"));
        assert!(at("let var = 1; UsageEvent {}", "UsageEvent"));
    }
}
