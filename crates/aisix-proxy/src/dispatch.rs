//! Helpers shared by every endpoint that needs to dispatch to a Bridge.
//!
//! Every endpoint follows the same shape after Model resolution:
//!
//! 1. Take the resolved `Model` (already looked up by display_name).
//! 2. Resolve the `ProviderKey` it references (via `provider_key_id`).
//! 3. Compute the upstream base URL by combining the `Provider`'s
//!    default with the `ProviderKey`'s optional `api_base` override.
//!
//! These helpers existed inline in each endpoint as
//! `Model::base_url()`, `Model::upstream_model()`, and
//! `Model::provider_config.api_key` accessors. Phase B moved the
//! Model from "self-contained inline secret" to "ProviderKey
//! reference"; this module is the join point that recovers the old
//! ergonomics on the proxy side.
//!
//! Returns typed [`ProxyError`] variants so the caller's `?`
//! plumbing flows naturally.

use std::sync::Arc;
use std::time::Instant;

use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, ApiSurface, Model, ProviderKey};
use aisix_gateway::{Bridge, BridgeError, Hub};

/// Map a `reqwest` transport error from a raw-passthrough dispatch
/// (`/v1/responses`, `/v1/messages` Anthropic, `/v1/messages/count_tokens`)
/// into the gateway's [`BridgeError`]. A timed-out request becomes
/// [`BridgeError::Timeout`] so it surfaces as 504, classifies as `"timeout"`
/// in telemetry, and participates in routing failover exactly like the
/// Bridge-trait path (#554). Everything else stays a transport error.
///
/// `is_timeout()` is satisfied by three unrelated conditions — the
/// configured request budget expiring in hyper, the `connect_timeout`
/// expiring, and the kernel returning `ETIMEDOUT` for an unanswered SYN —
/// so the reqwest cause chain is carried onto the error. Without it all
/// three render as one sentence and an operator cannot tell a slow
/// upstream from one that was never reached (AISIX-Cloud#1093).
pub(crate) fn reqwest_error_to_bridge(e: &reqwest::Error, started: Instant) -> BridgeError {
    if e.is_timeout() {
        BridgeError::Timeout {
            elapsed_ms: started.elapsed().as_millis() as u64,
            cause: aisix_gateway::transport_error_message(e),
        }
    } else {
        BridgeError::Transport(aisix_gateway::transport_error_message(e))
    }
}

use crate::error::ProxyError;

/// Resolve the Bridge to dispatch this request through.
///
/// `Hub::dispatch_two_tier` — specialized vendor first (keyed on
/// `ProviderKey.provider`), then adapter family (keyed on
/// `ProviderKey.adapter`). Vendor identity is an open string; adapter
/// is the closed 5-value enum. Any catalog vendor cp-api admits (xai,
/// openrouter, future long-tail) resolves through the family
/// fallthrough without a DP code change.
///
/// Returns `None` when both tiers miss (the PK carries neither a
/// registered `provider` nor a registered `adapter`) — caller surfaces
/// this as 503 "no dispatch path". cp-api writes `provider` + `adapter`
/// on every PK, so a miss means a genuine misconfiguration, not a
/// migration gap.
pub(crate) fn resolve_bridge(hub: &Hub, provider_key: &ProviderKey) -> Option<Arc<dyn Bridge>> {
    hub.dispatch_two_tier(provider_key)
}

/// Look up the `ProviderKey` a given `Model` references. Returns a
/// 400 if the Model is a virtual router (those don't dispatch
/// directly — caller should walk `routing.targets` first), or if the
/// referenced ProviderKey row is missing from the snapshot.
pub(crate) fn resolve_provider_key(
    snapshot: &AisixSnapshot,
    model: &Model,
) -> Result<Arc<ResourceEntry<ProviderKey>>, ProxyError> {
    let pk_id = model.provider_key_id.as_deref().ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} has no provider_key_id (routing models can't be dispatched directly)",
            model.display_name
        ))
    })?;
    let entry = snapshot.provider_keys.get_by_id(pk_id).ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} references unknown provider_key_id {pk_id:?}",
            model.display_name
        ))
    })?;
    // The request has now committed to a concrete target. Everything that
    // can still go wrong from here — an unusable credential, the upstream
    // itself — produces an error that carries no upstream identity, so the
    // attribution is recorded at the last point that has it
    // (AISIX-Cloud#1325).
    crate::attribution::note_target(model, &entry.id);
    Ok(entry)
}

/// Required `provider` (vendor id, free-form string) for a non-routing
/// Model. 400 if absent. Dispatch routing reads
/// `ProviderKey.adapter` + `ProviderKey.provider` — this helper just
/// confirms the Model has a non-routing shape and returns the vendor
/// id for telemetry / logs.
///
/// An ensemble model is rejected here with an explicit, accurate message:
/// it has no `provider` (its panel/judge members do), so without this guard
/// the generic "routing models can't be dispatched directly" branch below
/// would fire — misleading, since the model is an *ensemble*, not a router.
/// chat.rs branches to `dispatch_ensemble` before reaching this chokepoint,
/// so this guard is what every NON-chat endpoint (embeddings, images,
/// audio, completions, …) hits for an ensemble model.
pub(crate) fn require_provider(model: &Model) -> Result<&str, ProxyError> {
    if model.is_ensemble() {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{}` is an ensemble model; only /v1/chat/completions is supported",
            model.display_name
        )));
    }
    model.provider.as_deref().ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} has no provider (routing models can't be dispatched directly)",
            model.display_name
        ))
    })
}

/// Enforce a Model's client-IP allowlist (`allowed_cidrs`, #557).
///
/// Called by every request-serving endpoint right after the requested Model
/// is resolved and before any upstream dispatch, so a disallowed source IP is
/// rejected with 403 before the provider is ever contacted (issue AC-1). For
/// routing models the check binds to the **requested** model — the access
/// decision belongs to the name the client asked for, not the chosen target.
///
/// No-op when the Model has no `allowed_cidrs` configured (the common case).
pub(crate) fn check_ip_access(model: &Model, source_ip: &str) -> Result<(), ProxyError> {
    if model.ip_allowed(source_ip) {
        return Ok(());
    }
    tracing::warn!(
        model = %model.display_name,
        source_ip = %source_ip,
        "request rejected: client IP not in model allowed_cidrs"
    );
    Err(ProxyError::ModelIpRestricted(model.display_name.clone()))
}

/// Whether this Model's upstream speaks the Anthropic wire protocol, i.e.
/// whether `/v1/messages` and `/v1/messages/count_tokens` may forward the
/// caller's Anthropic-native body verbatim instead of round-tripping it
/// through the cross-provider bridge.
///
/// Keyed on the ProviderKey's `adapter`, not on the vendor id:
/// `provider: "byo"` + `adapter: anthropic` is the documented way to front a
/// self-hosted or proxied Anthropic endpoint, and such an upstream serves
/// both routes exactly like the catalog vendor does. Gating on the vendor id
/// alone made `/v1/messages` bridge a body that needed no translation (which
/// drops caller-owned fields such as `cache_control`) and made
/// `/v1/messages/count_tokens` reject the model outright.
///
/// `model.provider` is still honoured so a ProviderKey written without an
/// adapter — cp-api's AdapterMap-absent degenerate boot — keeps dispatching
/// as before. A dangling `provider_key_id` likewise falls back to the vendor
/// id, leaving the dispatch path (not this gate) to report it.
pub(crate) fn speaks_anthropic(snapshot: &AisixSnapshot, model: &Model) -> bool {
    serves_natively(snapshot, model, ApiSurface::Messages)
}

/// Whether this Model's upstream serves `surface` natively — i.e. whether
/// the caller's body may be forwarded verbatim instead of being translated
/// through a provider bridge.
///
/// The two surfaces answer differently, because the evidence for them
/// differs. `/v1/messages` is served by any key that speaks the Anthropic
/// wire by declaration — the `anthropic` vendor id, or `provider: "byo"` +
/// `adapter: anthropic`, which is the documented way to front a
/// self-hosted or proxied Anthropic endpoint — and additionally by any key
/// that names the route in [`ProviderKey::apis`], which is how one
/// credential reaches a vendor's OpenAI path and its Anthropic path both.
///
/// `/v1/responses` has no such declaration to lean on: the Responses API
/// is a strict superset of chat completions rather than a rename, so
/// `adapter: openai` says nothing about whether the route exists. Absent
/// an `apis` map the gateway falls back to the vendor id — only OpenAI
/// itself is assumed to serve it — and once the map exists it is the
/// answer, so an operator whose OpenAI-compatible endpoint has no
/// `/v1/responses` gets the request translated rather than 404'd upstream.
pub(crate) fn serves_natively(
    snapshot: &AisixSnapshot,
    model: &Model,
    surface: ApiSurface,
) -> bool {
    let pk = resolve_provider_key(snapshot, model).ok();
    let pk = pk.as_ref().map(|e| &e.value);
    // A platform adapter reaches every surface translated — its own
    // routes are Bedrock's `/model/{id}/converse`, Vertex's
    // `:generateContent` and Azure's deployment-scoped paths, none of
    // which these entries can name — and its credential is a JSON tuple
    // the verbatim paths would send as a bearer token to a URL that does
    // not exist.
    //
    // This gates the whole answer rather than just the declaration.
    // cp-api pairs an adapter with a vendor id from its own catalog, but
    // a hand-written resources file can put `model.provider: "openai"`
    // in front of an `adapter: azure-openai` key, and the vendor-id
    // fallback below would then send that credential down the verbatim
    // path with no declaration in sight.
    // Exhaustive on purpose: a new `Adapter` variant must be a compile
    // error here rather than silently inheriting the verbatim path,
    // which is how a platform credential would end up on a route that
    // does not exist.
    let platform = pk.is_some_and(|p| match p.adapter {
        Some(aisix_core::Adapter::Bedrock)
        | Some(aisix_core::Adapter::Vertex)
        | Some(aisix_core::Adapter::AzureOpenai) => true,
        Some(aisix_core::Adapter::Openai) | Some(aisix_core::Adapter::Anthropic) | None => false,
    });
    if platform {
        return false;
    }
    let declared = pk.and_then(|p| p.apis.as_ref());
    match surface {
        ApiSurface::Messages => {
            model.provider.as_deref() == Some("anthropic")
                || pk.is_some_and(|p| p.adapter == Some(aisix_core::Adapter::Anthropic))
                || declared.is_some_and(|apis| apis.messages.is_some())
        }
        ApiSurface::Responses => match declared {
            Some(apis) => apis.responses.is_some(),
            None => model.provider.as_deref() == Some("openai"),
        },
    }
}

/// Required upstream model id (`model_name`) for a non-routing Model.
pub(crate) fn require_upstream_model(model: &Model) -> Result<&str, ProxyError> {
    model.model_name.as_deref().ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} has no model_name (routing models can't be dispatched directly)",
            model.display_name
        ))
    })
}

/// Endpoint suffixes the proxy-side handlers append themselves. If an
/// operator accidentally pasted the full upstream URL into `api_base`,
/// strip the suffix here so the later URL build does not double-append.
///
/// Only the **bare** endpoint is stripped, never a `/v1/` prefixed form:
/// the version segment belongs to the base and must survive, or a
/// pasted `https://proxy.corp/shim/v1/responses` would collapse to
/// `https://proxy.corp/shim` and rebuild as `…/shim/responses`.
/// Stripping just `/responses` leaves `…/shim/v1`, which
/// [`build_openai_url`] then preserves verbatim.
const API_BASE_ENDPOINT_SUFFIXES: &[&str] = &[
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/chat/completions",
    "/images/generations",
    "/completions",
    "/embeddings",
    "/responses",
    // Longest first: `/messages` would otherwise never match a pasted
    // count_tokens URL, and stripping it from the middle is not what a
    // suffix scan does.
    "/messages/count_tokens",
    "/messages",
    "/rerank",
];

/// Strip a known endpoint suffix from `base` and its trailing slash.
/// Idempotent. Mirrors the suffix-stripping the bridge crates do on
/// their own `resolve_base`, so handlers that bypass the bridge (audio,
/// responses, messages) get the same tolerance.
fn strip_endpoint_suffix(base: &str) -> &str {
    let trimmed = base.trim_end_matches('/');
    for suffix in API_BASE_ENDPOINT_SUFFIXES {
        if let Some(rest) = trimmed.strip_suffix(suffix) {
            return rest.trim_end_matches('/');
        }
    }
    trimmed
}

/// The upstream base URL: `provider_key.api_base` override if set,
/// otherwise the one built-in vendor default. Tolerates an operator
/// pasting the full upstream URL into `api_base` by stripping any
/// trailing endpoint suffix — see [`API_BASE_ENDPOINT_SUFFIXES`].
///
/// The `openai` vendor is the single default-base exception (#1017):
/// before it, an OpenAI key with no `api_base` worked on the
/// bridge-dispatched routes (chat, generations) and `/v1/videos` —
/// both fall back to [`aisix_provider_openai::OPENAI_DEFAULT_BASE`] —
/// but 400'd on every direct-HTTP route through this resolver (audio,
/// image edits, jobs, realtime). The fallback here converges all of
/// them. Strictly the exact vendor string: an empty or
/// OpenAI-compatible vendor still errors, because this resolver also
/// serves non-OpenAI-family paths (messages / count_tokens) and must
/// never send another vendor's credential to api.openai.com. Note the
/// family bridge is LOOSER: it refuses only a non-empty non-openai
/// vendor and still falls back for the legacy empty-provider shape —
/// so a legacy `{provider: "", adapter: "openai"}` row with no
/// `api_base` remains route-dependent (chat falls back, direct routes
/// 400); that residue is tracked in #1019 rather than widened here. A
/// key misdeclared as `provider: "openai"` on an anthropic-adapter
/// path now dispatches to api.openai.com and gets the upstream's 401
/// instead of a gateway 400 — by declaration that secret is an OpenAI
/// credential, so nothing crosses vendors. Every other catalog vendor
/// keeps requiring `api_base` — the DP does not enumerate per-vendor
/// default URLs.
///
/// Callers that cache the built URL take their fingerprint from
/// [`pk_url_fingerprint`], which carries every input this resolver
/// reads — never hand-build a partial fingerprint.
pub(crate) fn resolve_base_url(provider_key: &ProviderKey) -> Result<String, ProxyError> {
    match provider_key.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => Ok(strip_endpoint_suffix(b.trim()).to_string()),
        _ if provider_key.provider.trim().eq_ignore_ascii_case("openai") => {
            Ok(aisix_provider_openai::OPENAI_DEFAULT_BASE.to_string())
        }
        _ => {
            // Remediation detail (control-plane field names, provider
            // topology) goes to logs only — the customer-visible body
            // stays free of internal taxonomy, matching the family
            // bridge's posture.
            tracing::error!(
                pk_display_name = %provider_key.display_name,
                pk_vendor = %provider_key.provider,
                "provider_key has no api_base. Operator action: populate \
                 `api_base` on the ProviderKey resource (managed \
                 deployments: via the control plane's provider settings; \
                 standalone: directly on the resource). Only the openai \
                 vendor has a built-in default base."
            );
            Err(ProxyError::InvalidRequest(format!(
                "provider_key {:?} has no api_base configured",
                provider_key.display_name
            )))
        }
    }
}

/// True when `base` carries a path component beyond the host — i.e. the
/// operator (or the CP catalog) pinned an explicit upstream root such as
/// `…/v2`, `…/api/paas/v4` or `…/v1beta/openai`, rather than leaving the
/// bare host. Callers pass a slash-trimmed base.
fn base_has_path(base: &str) -> bool {
    base.split_once("://")
        .map_or(base, |(_, rest)| rest)
        .contains('/')
}

/// Join an OpenAI-family upstream base with a version-independent
/// endpoint path (`/responses`, `/rerank`, `/audio/speech`, `/files`, …).
///
/// In this family `api_base` **is** the versioned root, so whatever path
/// the operator or the CP catalog configured is preserved verbatim and
/// the endpoint is appended to it. Only a bare host — no path at all —
/// gets the canonical `/v1` synthesized, because several vendor defaults
/// ship that form (`https://api.deepseek.com`, `https://api.cohere.com`,
/// `https://api.dev.runwayml.com`).
///
/// Preserving the path is what makes non-`/v1` roots reachable: Baidu
/// Qianfan serves `…/v2/responses`, Zhipu `…/api/paas/v4/…`, Volcengine
/// Ark `…/api/v3/…`, Gemini `…/v1beta/openai/…`. Synthesizing `/v1`
/// there built `…/v2/v1/responses`, which every one of those upstreams
/// 404s. It also matches how `OpenAiBridge` has always built
/// `/chat/completions`, so both halves of the gateway now agree on what
/// `api_base` means — before this, chat worked while responses, rerank,
/// audio, realtime and the batch/files routes 404'd on the same key.
///
/// `path` MUST start with `/` and MUST be the version-independent route
/// (`/responses`, not `/v1/responses`).
pub(crate) fn build_openai_url(base: &str, path: &str) -> String {
    // assert!, not debug_assert! — the cost of a single bounds check
    // per upstream dispatch is negligible compared to the network
    // round-trip, and a release-mode caller passing a malformed path
    // would silently produce a wrong URL (e.g. `…/v1responses`).
    assert!(
        path.starts_with('/'),
        "build_openai_url path must start with /, got {path:?}",
    );
    let trimmed = base.trim_end_matches('/');
    if base_has_path(trimmed) {
        format!("{trimmed}{path}")
    } else {
        format!("{trimmed}/v1{path}")
    }
}

/// Reject a multipart `prompt` field that is not valid UTF-8 (#1016).
///
/// The guardrail scan and mask passes read `prompt` through
/// `std::str::from_utf8` and SKIP a field that fails to decode, while
/// the rebuilt form forwards the original bytes verbatim — so a caller
/// could smuggle text past a keyword/DLP input guardrail by prefixing
/// one invalid byte or using a non-UTF-8 encoding. Fail closed instead:
/// the upstream contracts type `prompt` as text, so a compliant caller
/// loses nothing, and the JSON endpoints already behave this way (serde
/// rejects invalid UTF-8 bodies with 400). Deliberately UNCONDITIONAL —
/// not gated on a guardrail chain being configured — so a request's
/// validity does not flip when a guardrail is attached later, matching
/// the boundary-validation posture content-inspection systems use
/// (validate encoding first, match second).
///
/// Called by every multipart surface that scans `prompt` (audio
/// transcriptions/translations, image edits) after model resolution —
/// unknown models keep answering 404 — and before the guardrail pass.
pub(crate) fn require_utf8_prompt_fields(
    fields: &[(String, Option<String>, Option<String>, bytes::Bytes)],
) -> Result<(), ProxyError> {
    for (name, _, _, data) in fields {
        if name == "prompt" && std::str::from_utf8(data).is_err() {
            return Err(ProxyError::InvalidRequest(
                "`prompt` field is not valid UTF-8".into(),
            ));
        }
    }
    Ok(())
}

/// The cache-fingerprint elements for a URL derived from
/// [`resolve_base_url`]: every raw input the resolved URL depends on —
/// `api_base` AND the vendor (the #1017 default-base fallback made the
/// output vendor-dependent). One constructor for all call sites, so a
/// future input added here reaches every cached URL at once instead of
/// relying on each site's comment discipline.
pub(crate) fn pk_url_fingerprint(provider_key: &ProviderKey) -> [&str; 2] {
    [
        provider_key.api_base.as_deref().unwrap_or(""),
        provider_key.provider.as_str(),
    ]
}

/// The base URL for one API surface: the `apis` entry's own `base` when
/// the Provider Key declares one, else [`resolve_base_url`].
///
/// This is what lets a single upstream account serve two protocols from
/// two paths — `…/v1` for the OpenAI wire and `…/anthropic` for the
/// Anthropic one — under one credential. Surfaces the `apis` map does not
/// cover keep resolving through `api_base` alone.
pub(crate) fn resolve_base_url_for(
    provider_key: &ProviderKey,
    surface: ApiSurface,
) -> Result<String, ProxyError> {
    match surface_base(provider_key, surface) {
        Some(base) => Ok(strip_endpoint_suffix(base.trim()).to_string()),
        None => resolve_base_url(provider_key),
    }
}

/// The raw per-surface `base` override, if this key declares a non-empty
/// one. Shared by [`resolve_base_url_for`] and the cache fingerprint so
/// the two can never read different inputs.
fn surface_base(provider_key: &ProviderKey, surface: ApiSurface) -> Option<&str> {
    provider_key
        .apis
        .as_ref()
        .and_then(|apis| apis.get(surface))
        .and_then(|entry| entry.base.as_deref())
        .map(str::trim)
        .filter(|base| !base.is_empty())
}

/// [`pk_url_fingerprint`] plus the per-surface override, for the callers
/// that build their URL through [`resolve_base_url_for`]. Editing an
/// `apis` entry has to invalidate the cached URL just like editing
/// `api_base` does.
pub(crate) fn pk_surface_url_fingerprint(
    provider_key: &ProviderKey,
    surface: ApiSurface,
) -> [&str; 3] {
    [
        surface_base(provider_key, surface).unwrap_or(""),
        provider_key.api_base.as_deref().unwrap_or(""),
        provider_key.provider.as_str(),
    ]
}

/// Join an Anthropic upstream base with a version-independent endpoint
/// path (`/messages`, `/messages/count_tokens`).
///
/// Anthropic's convention is the mirror image of the OpenAI family's:
/// the documented `api_base` is the bare host and the `/v1` belongs to
/// the endpoint (`POST {base}/v1/messages`). So `/v1` is synthesized for
/// every base, including one that carries a path — a self-hosted
/// Anthropic-compatible gateway mounted at `…/anthropic` serves
/// `…/anthropic/v1/messages`. A base that already ends in `/v1` is an
/// operator importing the OpenAI habit; collapse it so it can't double.
/// Mirrors `AnthropicBridge::normalize_api_base`.
pub(crate) fn build_anthropic_url(base: &str, path: &str) -> String {
    assert!(
        path.starts_with('/'),
        "build_anthropic_url path must start with /, got {path:?}",
    );
    let trimmed = base.trim_end_matches('/');
    let root = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
    format!("{root}/v1{path}")
}

/// The upstream API key — `provider_key.api_key`. Empty string is
/// treated as a config error (ProviderKey rows shouldn't be empty,
/// but a hand-edited kine row could surface one).
pub(crate) fn require_api_key<'a>(
    provider_key: &'a ProviderKey,
    model: &Model,
) -> Result<&'a str, ProxyError> {
    if provider_key.api_key.is_empty() {
        return Err(ProxyError::InvalidRequest(format!(
            "model {:?} provider_key has empty api_key",
            model.display_name
        )));
    }
    Ok(provider_key.api_key.as_str())
}

/// Build the [`BridgeContext`] for one upstream call.
///
/// The single chokepoint for wiring a Bridge dispatch: it carries the
/// snapshot ids (which a `Model` / `ProviderKey` value does not know about
/// itself) and, for a call made on behalf of a caller, that caller's
/// identity and inbound headers. Both feed
/// [`aisix_gateway::BridgeContext::header_ctx`], so a site that skipped
/// either would silently stop rendering `${...}` header templates or
/// forwarding allowlisted client headers, with no compiler signal —
/// hence one constructor rather than a chain every caller must remember.
///
/// `client` is `None` for calls with no client request behind them: the
/// semantic-router's embedding lookup and the background health prober.
/// Those forward no client header and resolve no `${request.api_key.*}`
/// variable, but still resolve the model / provider-key ones.
pub(crate) fn bridge_ctx(
    request_id: &str,
    model_id: &str,
    model: Arc<Model>,
    provider_key_id: &str,
    provider_key: Arc<ProviderKey>,
    client: Option<&crate::client_ip::ClientContext>,
) -> aisix_gateway::BridgeContext {
    let ctx = aisix_gateway::BridgeContext::new(request_id, model, provider_key)
        .with_resource_ids(model_id, provider_key_id);
    match client {
        Some(c) => ctx.with_client(
            c.caller.clone(),
            Some(c.headers.clone()),
            c.jwt.as_ref().map(|j| Arc::from(j.token())),
        ),
        None => ctx,
    }
}

/// Build the outbound-header context for a dispatch path that constructs
/// its upstream request directly instead of going through a `Bridge`
/// (`/v1/messages`, `/v1/responses`, `/v1/messages/count_tokens`, audio,
/// rerank, videos, jobs).
///
/// The Bridge paths get the same thing from
/// [`aisix_gateway::BridgeContext::header_ctx`]; both must resolve the
/// same variables from the same sources, so keep them in step.
pub(crate) fn upstream_header_ctx<'a>(
    pk: &'a ProviderKey,
    pk_id: &'a str,
    model: &'a Model,
    model_id: &'a str,
    client: &'a crate::client_ip::ClientContext,
) -> aisix_gateway::UpstreamHeaderContext<'a> {
    let caller = &client.caller;
    aisix_gateway::UpstreamHeaderContext::from_overrides(pk.request.as_ref())
        .with_vars(aisix_core::HeaderVars {
            request_id: Some(&client.request_id),
            api_key_id: Some(&caller.api_key_id),
            api_key_name: caller.api_key_name.as_deref(),
            api_key_team_id: caller.team_id.as_deref(),
            api_key_user_id: caller.user_id.as_deref(),
            model_id: Some(model_id),
            model_name: Some(&model.display_name),
            provider_key_id: Some(pk_id),
            provider_key_name: Some(&pk.display_name),
        })
        .with_client_headers(&client.headers)
        .with_caller_jwt(client.jwt.as_ref().map(|j| j.token()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::resource::ResourceEntry;

    /// AISIX-Cloud#1093: `reqwest::Error::is_timeout()` is satisfied by an
    /// expired request budget, an expired `connect_timeout`, and the
    /// kernel's `ETIMEDOUT`. The mapped error must carry the cause chain so
    /// those are distinguishable — otherwise every one of them renders as
    /// the same "timed out after Nms" sentence.
    #[tokio::test]
    async fn timeout_mapping_carries_the_transport_cause() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(std::time::Duration::from_secs(5)))
            .mount(&server)
            .await;

        let started = Instant::now();
        let err = reqwest::Client::new()
            .post(server.uri())
            .timeout(std::time::Duration::from_millis(50))
            .send()
            .await
            .expect_err("the 50ms budget must expire against a 5s upstream");
        assert!(err.is_timeout(), "precondition: reqwest reports a timeout");

        let mapped = reqwest_error_to_bridge(&err, started);
        match &mapped {
            BridgeError::Timeout { cause, .. } => {
                assert!(!cause.is_empty(), "timeout must carry its cause chain");
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
        // The rendered message keeps the elapsed budget AND names the cause.
        let rendered = mapped.to_string();
        assert!(
            rendered.contains("upstream request timed out"),
            "{rendered}"
        );
        assert!(
            rendered.len() > "upstream request timed out after 50ms".len(),
            "cause must widen the message: {rendered}"
        );
    }

    /// A non-timeout transport failure still maps to `Transport`, so the
    /// two stay distinguishable by variant as well as by message.
    #[tokio::test]
    async fn non_timeout_transport_error_stays_transport() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let err = reqwest::Client::new()
            .post(format!("http://{addr}/"))
            .send()
            .await
            .expect_err("connect to a closed port must fail");
        assert!(!err.is_timeout());

        match reqwest_error_to_bridge(&err, Instant::now()) {
            BridgeError::Transport(msg) => {
                assert!(msg.to_lowercase().contains("refused"), "{msg}");
            }
            other => panic!("expected Transport, got {other:?}"),
        }
    }

    fn snapshot_with(provider_key_id: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"openai-prod","secret":"sk-x","api_base":"https://proxy.example.com/v1","provider":"openai","adapter":"openai"}"#,
        )
        .unwrap();
        snap.provider_keys
            .insert(ResourceEntry::new(provider_key_id, pk, 1));
        snap
    }

    fn direct_model(provider_key_id: &str) -> Model {
        let cfg = format!(
            r#"{{
                "display_name": "my-gpt4",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{provider_key_id}"
            }}"#
        );
        serde_json::from_str(&cfg).unwrap()
    }

    /// Build a snapshot holding one Provider Key from raw JSON, so a test
    /// can express the exact stored document (including an `apis` map) the
    /// resolver will see.
    fn snapshot_with_pk_json(provider_key_id: &str, json: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        let pk: ProviderKey = serde_json::from_str(json).unwrap();
        snap.provider_keys
            .insert(ResourceEntry::new(provider_key_id, pk, 1));
        snap
    }

    fn model_on(provider: &str, provider_key_id: &str) -> Model {
        serde_json::from_str(&format!(
            r#"{{"display_name":"m","provider":"{provider}","model_name":"x","provider_key_id":"{provider_key_id}"}}"#
        ))
        .unwrap()
    }

    /// No `apis` map — every surface resolves exactly as it did before the
    /// field existed. This is the guard that stored configuration keeps its
    /// behavior: the vendor id decides `/v1/responses`, the vendor id or
    /// the anthropic adapter decides `/v1/messages`.
    #[test]
    fn without_apis_the_vendor_inference_is_unchanged() {
        let snap = snapshot_with_pk_json(
            "pk-1",
            r#"{"display_name":"o","secret":"k","api_base":"https://up/v1","provider":"openai","adapter":"openai"}"#,
        );
        let m = model_on("openai", "pk-1");
        assert!(serves_natively(&snap, &m, ApiSurface::Responses));
        assert!(!serves_natively(&snap, &m, ApiSurface::Messages));

        let snap = snapshot_with_pk_json(
            "pk-2",
            r#"{"display_name":"a","secret":"k","api_base":"https://up","provider":"byo","adapter":"anthropic"}"#,
        );
        let m = model_on("byo", "pk-2");
        assert!(serves_natively(&snap, &m, ApiSurface::Messages));
        assert!(!serves_natively(&snap, &m, ApiSurface::Responses));
    }

    /// AISIX-Cloud#1388: an OpenAI-compatible endpoint reached through the
    /// `openai` catalog vendor + a custom `api_base` has no `/v1/responses`
    /// route. Declaring `apis` without it makes the gateway translate
    /// instead of forwarding a request the upstream 404s.
    #[test]
    fn apis_without_responses_stops_the_verbatim_passthrough() {
        let snap = snapshot_with_pk_json(
            "pk-1",
            r#"{"display_name":"o","secret":"k","api_base":"https://vllm.internal/v1","provider":"openai","adapter":"openai","apis":{}}"#,
        );
        let m = model_on("openai", "pk-1");
        assert!(
            !serves_natively(&snap, &m, ApiSurface::Responses),
            "an apis map that omits responses is the operator saying the route is absent",
        );
    }

    /// The same key declaring the route keeps the verbatim passthrough —
    /// an OpenAI-compatible endpoint that does implement `/v1/responses`
    /// must not be downgraded just because it declared its surfaces.
    #[test]
    fn apis_listing_responses_keeps_the_verbatim_passthrough() {
        let snap = snapshot_with_pk_json(
            "pk-1",
            r#"{"display_name":"o","secret":"k","api_base":"https://vllm.internal/v1","provider":"byo","adapter":"openai","apis":{"responses":{}}}"#,
        );
        let m = model_on("byo", "pk-1");
        assert!(serves_natively(&snap, &m, ApiSurface::Responses));
    }

    /// The DeepSeek/Zhipu shape: one credential, an OpenAI-compatible path
    /// and an Anthropic-compatible one. Declaring `messages` adds the
    /// native Anthropic route to a key whose adapter is `openai`, so
    /// `/v1/messages` stops being translated (which is what drops
    /// `cache_control` and thinking blocks).
    #[test]
    fn apis_adds_a_native_anthropic_route_to_an_openai_key() {
        let snap = snapshot_with_pk_json(
            "pk-1",
            r#"{"display_name":"ds","secret":"k","api_base":"https://api.deepseek.com/v1","provider":"deepseek","adapter":"openai",
                "apis":{"messages":{"base":"https://api.deepseek.com/anthropic"}}}"#,
        );
        let m = model_on("deepseek", "pk-1");
        assert!(serves_natively(&snap, &m, ApiSurface::Messages));
        // …and it does not accidentally imply the Responses route.
        assert!(!serves_natively(&snap, &m, ApiSurface::Responses));
    }

    /// An anthropic-adapter key keeps serving `/v1/messages` at `api_base`
    /// whatever else `apis` lists — that is what the adapter declares, and
    /// an unrelated entry must not turn it off.
    #[test]
    fn an_anthropic_adapter_keeps_its_own_surface_when_apis_lists_others() {
        let snap = snapshot_with_pk_json(
            "pk-1",
            r#"{"display_name":"a","secret":"k","api_base":"https://up","provider":"byo","adapter":"anthropic","apis":{"responses":{}}}"#,
        );
        let m = model_on("byo", "pk-1");
        assert!(serves_natively(&snap, &m, ApiSurface::Messages));
    }

    /// A platform adapter never takes a native path, declaration or not:
    /// its own routes are not these paths, and its credential is a JSON
    /// tuple the verbatim dispatch would send as a bearer token.
    #[test]
    fn a_platform_adapter_never_serves_a_declared_surface() {
        for adapter in ["bedrock", "vertex", "azure-openai"] {
            let snap = snapshot_with_pk_json(
                "pk-1",
                &format!(
                    r#"{{"display_name":"p","secret":"{{}}","api_base":"https://up","provider":"byo","adapter":"{adapter}","apis":{{"responses":{{}},"messages":{{}}}}}}"#
                ),
            );
            let m = model_on("byo", "pk-1");
            assert!(
                !serves_natively(&snap, &m, ApiSurface::Responses),
                "{adapter} must not take the verbatim responses path",
            );
            assert!(
                !serves_natively(&snap, &m, ApiSurface::Messages),
                "{adapter} must not take the verbatim messages path",
            );
        }
    }

    /// The platform guard has to cover the fallback too. cp-api pairs an
    /// adapter with a vendor id from its own catalog, but a hand-written
    /// resources file can name a vendor the vendor-id inference treats as
    /// native in front of a platform key — and that path would send a
    /// JSON credential tuple as a bearer token to a route that does not
    /// exist.
    #[test]
    fn a_platform_adapter_is_not_rescued_by_the_vendor_fallback() {
        for (adapter, vendor, surface) in [
            ("azure-openai", "openai", ApiSurface::Responses),
            ("bedrock", "openai", ApiSurface::Responses),
            ("vertex", "anthropic", ApiSurface::Messages),
        ] {
            let snap = snapshot_with_pk_json(
                "pk-1",
                &format!(
                    r#"{{"display_name":"p","secret":"{{}}","api_base":"https://up","provider":"{vendor}","adapter":"{adapter}"}}"#
                ),
            );
            let m = model_on(vendor, "pk-1");
            assert!(
                !serves_natively(&snap, &m, surface),
                "{adapter} + vendor {vendor} must not reach a verbatim path",
            );
        }
    }

    /// An empty `base` resolves as "no override" rather than building a
    /// URL from an empty string. The read schema deliberately allows it
    /// through (see `ApiEndpoint::base`), so this is the only thing
    /// standing between a cleared field and a malformed upstream URL.
    #[test]
    fn an_empty_surface_base_falls_back_to_api_base() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"ds","secret":"k","api_base":"https://api.deepseek.com/v1","provider":"deepseek","apis":{"messages":{"base":"   "}}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_base_url_for(&pk, ApiSurface::Messages).unwrap(),
            "https://api.deepseek.com/v1",
        );
    }

    /// A surface entry's own `base` is what the URL is built from, and it
    /// gets the same endpoint-suffix tolerance `api_base` has. Anything the
    /// entry does not override falls back to `api_base`.
    #[test]
    fn surface_base_overrides_api_base_only_for_the_declared_surface() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"ds","secret":"k","api_base":"https://api.deepseek.com/v1","provider":"deepseek","adapter":"openai",
                "apis":{"messages":{"base":"https://api.deepseek.com/anthropic/v1/messages"}}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_base_url_for(&pk, ApiSurface::Messages).unwrap(),
            "https://api.deepseek.com/anthropic/v1",
            "a pasted full endpoint URL loses the endpoint, keeping the version segment \
             exactly as api_base does",
        );
        assert_eq!(
            build_anthropic_url(
                &resolve_base_url_for(&pk, ApiSurface::Messages).unwrap(),
                "/messages"
            ),
            "https://api.deepseek.com/anthropic/v1/messages",
        );
        assert_eq!(
            resolve_base_url_for(&pk, ApiSurface::Responses).unwrap(),
            "https://api.deepseek.com/v1",
        );
    }

    /// The cached-URL fingerprint has to carry the per-surface override, or
    /// editing an `apis` entry would keep serving the URL built from the
    /// previous one.
    #[test]
    fn surface_fingerprint_changes_when_the_entry_changes() {
        let base: ProviderKey = serde_json::from_str(
            r#"{"display_name":"ds","secret":"k","api_base":"https://api.deepseek.com/v1","provider":"deepseek","apis":{"messages":{"base":"https://a.example/anthropic"}}}"#,
        )
        .unwrap();
        let edited: ProviderKey = serde_json::from_str(
            r#"{"display_name":"ds","secret":"k","api_base":"https://api.deepseek.com/v1","provider":"deepseek","apis":{"messages":{"base":"https://b.example/anthropic"}}}"#,
        )
        .unwrap();
        assert_ne!(
            pk_surface_url_fingerprint(&base, ApiSurface::Messages),
            pk_surface_url_fingerprint(&edited, ApiSurface::Messages),
        );
        // A surface the map does not cover still fingerprints on api_base.
        assert_eq!(
            pk_surface_url_fingerprint(&base, ApiSurface::Responses)[1..],
            pk_url_fingerprint(&base)[..],
        );
    }

    fn routing_model() -> Model {
        serde_json::from_str(
            r#"{
                "display_name": "router-1",
                "routing": {
                    "strategy": "round_robin",
                    "targets": [{"model": "my-gpt4"}]
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn resolve_provider_key_happy_path() {
        let snap = snapshot_with("pk-1");
        let m = direct_model("pk-1");
        let entry = resolve_provider_key(&snap, &m).unwrap();
        assert_eq!(entry.value.display_name, "openai-prod");
    }

    #[test]
    fn resolve_provider_key_unknown_id_is_400_with_helpful_message() {
        let snap = snapshot_with("pk-1");
        let m = direct_model("pk-MISSING");
        let err = resolve_provider_key(&snap, &m).unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => {
                assert!(msg.contains("provider_key_id"), "{msg}");
                assert!(msg.contains("my-gpt4"), "{msg}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn resolve_provider_key_routing_model_is_400() {
        let snap = snapshot_with("pk-1");
        let m = routing_model();
        let err = resolve_provider_key(&snap, &m).unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));
    }

    #[test]
    fn require_provider_returns_provider_for_direct_model() {
        let m = direct_model("pk-1");
        assert_eq!(require_provider(&m).unwrap(), "openai");
    }

    #[test]
    fn require_provider_rejects_routing_model() {
        let m = routing_model();
        assert!(require_provider(&m).is_err());
    }

    #[test]
    fn resolve_base_url_uses_override_when_set() {
        let snap = snapshot_with("pk-1");
        let m = direct_model("pk-1");
        let pk_entry = resolve_provider_key(&snap, &m).unwrap();
        let base = resolve_base_url(&pk_entry.value).unwrap();
        assert_eq!(base, "https://proxy.example.com/v1");
    }

    /// Empty `api_base` on the PK is now an error — the DP no longer
    /// fabricates per-vendor defaults. cp-api populates api_base for
    /// every catalog vendor (handlers.go createProviderKey gate +
    /// featured `default_base_url`); refusing here turns any cp-api
    /// admission gap into a loud 400 instead of a silent mis-route.
    #[test]
    fn resolve_base_url_errors_when_api_base_missing() {
        // No provider (legacy row) — must NOT get the openai default:
        // sending an unknown vendor's credential to api.openai.com is
        // worse than the 400.
        let pk: ProviderKey = serde_json::from_str(r#"{"display_name":"x","secret":"k"}"#).unwrap();
        let err = resolve_base_url(&pk).unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("api_base"),
                    "error must mention api_base; got: {msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    /// #1017: the `openai` vendor is the one built-in default base — a
    /// key with no `api_base` resolves to the same base the specialized
    /// bridge and the videos route fall back to, so an OpenAI key
    /// behaves identically on every route family.
    #[test]
    fn resolve_base_url_openai_vendor_falls_back_to_default_base() {
        let pk: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"k","provider":"openai"}"#)
                .unwrap();
        assert_eq!(
            resolve_base_url(&pk).unwrap(),
            aisix_provider_openai::OPENAI_DEFAULT_BASE
        );
    }

    /// The vendor match is trim + case-insensitive (operator-typed
    /// declarative rows), and a whitespace-only `api_base` counts as
    /// unset — same emptiness rule as the error arm.
    #[test]
    fn resolve_base_url_openai_fallback_tolerates_case_and_blank_base() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"x","secret":"k","provider":" OpenAI ","api_base":"   "}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_base_url(&pk).unwrap(),
            aisix_provider_openai::OPENAI_DEFAULT_BASE
        );
    }

    /// #1016: the multipart prompt UTF-8 gate — invalid bytes in a
    /// `prompt` part reject; binary bytes in other parts (the image
    /// itself) pass through untouched.
    #[test]
    fn require_utf8_prompt_fields_rejects_only_invalid_prompt() {
        use bytes::Bytes;
        let valid = vec![
            ("model".to_string(), None, None, Bytes::from_static(b"m")),
            ("prompt".to_string(), None, None, Bytes::from_static(b"ok")),
            (
                "image".to_string(),
                Some("a.png".to_string()),
                Some("image/png".to_string()),
                Bytes::from_static(&[0xFF, 0xD8, 0xFF]),
            ),
        ];
        assert!(require_utf8_prompt_fields(&valid).is_ok());

        let invalid = vec![(
            "prompt".to_string(),
            None,
            None,
            Bytes::from_static(&[0xFF, b'h', b'i']),
        )];
        assert!(matches!(
            require_utf8_prompt_fields(&invalid).unwrap_err(),
            ProxyError::InvalidRequest(_)
        ));
    }

    /// #1017 HIGH-1 regression: the cached-URL fingerprint at every
    /// resolver-fed call site must include the vendor. Simulated here at
    /// the cache level with the exact fingerprint shape those sites use:
    /// a row cached under `provider: "openai"` (no api_base → default
    /// base) must NOT keep serving that URL after the row is repurposed
    /// to another vendor — the rebuild must run and surface the 400,
    /// or the repurposed vendor's credential would flow to
    /// api.openai.com until restart.
    #[test]
    fn cached_url_rebuilds_when_provider_changes_without_api_base() {
        let openai_pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"repurpose","secret":"k","provider":"openai"}"#,
        )
        .unwrap();
        let deepseek_pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"repurpose","secret":"k2","provider":"deepseek","adapter":"openai"}"#,
        )
        .unwrap();
        let resource_id = "test-1017-fingerprint-repurpose";

        let first = aisix_gateway::url_cache::cached_endpoint_url(
            resource_id,
            "test/1017-fingerprint",
            &pk_url_fingerprint(&openai_pk),
            || {
                let base = resolve_base_url(&openai_pk)?;
                Ok::<_, ProxyError>(build_openai_url(&base, "/responses"))
            },
        )
        .unwrap();
        assert!(format!("{first:?}").contains("api.openai.com"));

        // Same resource id + endpoint, vendor edited: the fingerprint
        // mismatch must force a rebuild, which errors — the stale
        // api.openai.com URL must not be served.
        let second = aisix_gateway::url_cache::cached_endpoint_url(
            resource_id,
            "test/1017-fingerprint",
            &pk_url_fingerprint(&deepseek_pk),
            || {
                let base = resolve_base_url(&deepseek_pk)?;
                Ok::<_, ProxyError>(build_openai_url(&base, "/responses"))
            },
        );
        assert!(
            second.is_err(),
            "repurposed vendor must rebuild and surface the 400, not the cached URL"
        );
    }

    /// An OpenAI-COMPATIBLE vendor is not OpenAI: it still requires
    /// `api_base`, mirroring the family bridge's refusal to fall back —
    /// a DeepSeek credential must never be sent to api.openai.com.
    #[test]
    fn resolve_base_url_openai_compatible_vendor_still_errors() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"x","secret":"k","provider":"deepseek","adapter":"openai"}"#,
        )
        .unwrap();
        assert!(matches!(
            resolve_base_url(&pk).unwrap_err(),
            ProxyError::InvalidRequest(_)
        ));
    }

    fn pk_with_base(api_base: &str) -> ProviderKey {
        let cfg = format!(r#"{{"display_name":"x","secret":"k","api_base":"{api_base}"}}"#);
        serde_json::from_str(&cfg).unwrap()
    }

    /// Every OpenAI-shape paste an operator might make must, when fed
    /// to `build_openai_url(base, "/<endpoint>")`, produce the canonical
    /// upstream URL. The intermediate `resolve_base_url` result may be
    /// either bare-host or `<host>/v1` — `build_openai_url` accepts both
    /// — so the assertion is on the final URL the handler dispatches to,
    /// not on the intermediate base.
    ///
    /// Without suffix stripping, pasting `…/v1/audio/transcriptions`
    /// into `api_base` produces `…/v1/audio/transcriptions/v1/audio/transcriptions`.
    #[test]
    fn resolve_base_url_strips_openai_endpoint_suffixes() {
        let cases: &[(&str, &str)] = &[
            ("https://api.openai.com/v1", "/responses"),
            ("https://api.openai.com/v1/", "/responses"),
            ("https://api.openai.com/v1/responses", "/responses"),
            (
                "https://api.openai.com/v1/audio/transcriptions",
                "/audio/transcriptions",
            ),
            (
                "https://api.openai.com/v1/audio/translations",
                "/audio/translations",
            ),
            ("https://api.openai.com/v1/audio/speech", "/audio/speech"),
            (
                "https://api.openai.com/v1/chat/completions",
                "/chat/completions",
            ),
            ("https://api.openai.com/v1/completions", "/completions"),
            ("https://api.openai.com/v1/embeddings", "/embeddings"),
            (
                "https://api.openai.com/v1/images/generations",
                "/images/generations",
            ),
            ("https://api.openai.com/v1/rerank", "/rerank"),
        ];
        for (paste, endpoint) in cases {
            let pk = pk_with_base(paste);
            let base = resolve_base_url(&pk).unwrap();
            let url = build_openai_url(&base, endpoint);
            let expected = format!("https://api.openai.com/v1{endpoint}");
            assert_eq!(
                url, expected,
                "paste {paste:?} + endpoint {endpoint:?} must build to {expected:?}",
            );
        }
    }

    /// DeepSeek serves OpenAI-compatible endpoints at the host root.
    /// Same contract: every paste must build to the canonical URL.
    #[test]
    fn resolve_base_url_strips_deepseek_endpoint_suffixes() {
        for paste in [
            "https://api.deepseek.com",
            "https://api.deepseek.com/",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/embeddings",
        ] {
            let pk = pk_with_base(paste);
            let base = resolve_base_url(&pk).unwrap();
            let url = build_openai_url(&base, "/chat/completions");
            assert_eq!(
                url, "https://api.deepseek.com/v1/chat/completions",
                "paste {paste:?} must build to the canonical chat-completions URL",
            );
        }
    }

    /// Anthropic: the messages handler builds `…/v1/messages`. A paste
    /// of the full upstream URL must strip so
    /// `build_anthropic_url("/messages")` does not produce
    /// `…/v1/messages/v1/messages`.
    #[test]
    fn resolve_base_url_strips_anthropic_messages_suffix() {
        for paste in [
            "https://api.anthropic.com",
            "https://api.anthropic.com/",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/messages/",
        ] {
            let pk = pk_with_base(paste);
            let base = resolve_base_url(&pk).unwrap();
            assert_eq!(
                build_anthropic_url(&base, "/messages"),
                "https://api.anthropic.com/v1/messages",
                "paste {paste:?} must build to the canonical messages URL",
            );
        }
    }

    /// Non-canonical hosts (corporate proxies, test mocks) pass through
    /// after suffix-stripping. The operator's path on a non-default
    /// host is trusted as-is.
    #[test]
    fn resolve_base_url_passes_non_canonical_hosts_through() {
        let pk = pk_with_base("https://proxy.example.com/openai-shim");
        assert_eq!(
            resolve_base_url(&pk).unwrap(),
            "https://proxy.example.com/openai-shim",
        );

        // Suffix stripping still applies on non-canonical hosts —
        // operator pasting the full upstream URL is still recovered.
        // Only the bare `/responses` is stripped, so the `/v1` the
        // operator wrote survives into the rebuilt URL.
        let pk = pk_with_base("https://proxy.example.com/openai-shim/v1/responses");
        let base = resolve_base_url(&pk).unwrap();
        assert_eq!(base, "https://proxy.example.com/openai-shim/v1");
        assert_eq!(
            build_openai_url(&base, "/responses"),
            "https://proxy.example.com/openai-shim/v1/responses",
        );
    }

    /// Whitespace trim must compose with suffix stripping.
    #[test]
    fn resolve_base_url_trims_whitespace_and_endpoint_suffix() {
        let pk = pk_with_base("  https://api.openai.com/v1/chat/completions/  ");
        let base = resolve_base_url(&pk).unwrap();
        assert_eq!(
            build_openai_url(&base, "/chat/completions"),
            "https://api.openai.com/v1/chat/completions",
        );
    }

    // ---------------------------------------------------------------
    // build_openai_url — the path-doubling regression fixture.
    // ---------------------------------------------------------------

    #[test]
    fn build_openai_url_appends_v1_only_for_a_bare_host() {
        // Bare-host convention: the operator pasted
        // `https://api.openai.com` without the trailing `/v1`, or the
        // vendor default ships that form (deepseek, cohere, runwayml).
        assert_eq!(
            build_openai_url("https://api.openai.com", "/responses"),
            "https://api.openai.com/v1/responses",
        );
        assert_eq!(
            build_openai_url("https://api.deepseek.com", "/chat/completions"),
            "https://api.deepseek.com/v1/chat/completions",
        );
        // Host:port with no path is still a bare host.
        assert_eq!(
            build_openai_url("http://127.0.0.1:8080", "/rerank"),
            "http://127.0.0.1:8080/v1/rerank",
        );
    }

    #[test]
    fn build_openai_url_preserves_v1_base_without_doubling() {
        // Customer follows the OpenAI SDK convention + the dashboard's
        // provider-keys form pre-fill (`https://api.openai.com/v1`).
        // A naive `format!("{base}/v1/responses")` would produce
        // `https://api.openai.com/v1/v1/responses` and 404 upstream.
        assert_eq!(
            build_openai_url("https://api.openai.com/v1", "/responses"),
            "https://api.openai.com/v1/responses",
        );
    }

    /// AISIX-Cloud#1244: an `api_base` whose root is not `/v1` must be
    /// preserved verbatim. Synthesizing `/v1` built `…/v2/v1/responses`,
    /// which the upstream 404s. Every value here is a real upstream root
    /// — three of them ship as CP catalog defaults.
    #[test]
    fn build_openai_url_preserves_non_v1_roots() {
        let cases: &[(&str, &str, &str)] = &[
            // Baidu Qianfan — the reported customer config.
            (
                "https://qianfan.baidubce.com/v2",
                "/responses",
                "https://qianfan.baidubce.com/v2/responses",
            ),
            // Zhipu — CP catalog default.
            (
                "https://open.bigmodel.cn/api/paas/v4",
                "/audio/speech",
                "https://open.bigmodel.cn/api/paas/v4/audio/speech",
            ),
            // Volcengine Ark — CP catalog default.
            (
                "https://ark.cn-beijing.volces.com/api/v3",
                "/files",
                "https://ark.cn-beijing.volces.com/api/v3/files",
            ),
            // Gemini's OpenAI-compat surface — CP catalog default, and
            // the case a "does the last segment look like a version?"
            // heuristic would miss.
            (
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "/realtime",
                "https://generativelanguage.googleapis.com/v1beta/openai/realtime",
            ),
            // Alibaba DashScope's compatible mode.
            (
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "/rerank",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank",
            ),
        ];
        for (base, path, expected) in cases {
            assert_eq!(
                &build_openai_url(base, path),
                expected,
                "base {base:?} + path {path:?}",
            );
        }
    }

    #[test]
    fn build_openai_url_strips_trailing_slash() {
        assert_eq!(
            build_openai_url("https://api.openai.com/", "/rerank"),
            "https://api.openai.com/v1/rerank",
        );
        assert_eq!(
            build_openai_url("https://api.openai.com/v1/", "/rerank"),
            "https://api.openai.com/v1/rerank",
        );
        // A trailing slash must not make a bare host look path-bearing.
        assert_eq!(
            build_openai_url("https://api.deepseek.com/", "/embeddings"),
            "https://api.deepseek.com/v1/embeddings",
        );
    }

    #[test]
    fn build_openai_url_handles_nested_paths() {
        // /audio/speech, /audio/transcriptions, /audio/translations all
        // pass nested paths — make sure the helper doesn't try to be
        // clever about them.
        assert_eq!(
            build_openai_url("https://api.openai.com", "/audio/speech"),
            "https://api.openai.com/v1/audio/speech",
        );
        assert_eq!(
            build_openai_url("https://api.openai.com/v1", "/audio/transcriptions"),
            "https://api.openai.com/v1/audio/transcriptions",
        );
    }

    #[test]
    #[should_panic(expected = "build_openai_url path must start with /")]
    fn build_openai_url_rejects_path_without_leading_slash() {
        // Misuse — handlers should always pass a `/`-prefixed path.
        let _ = build_openai_url("https://api.openai.com", "responses");
    }

    // ---------------------------------------------------------------
    // build_anthropic_url — the mirror-image convention: `/v1` belongs
    // to the endpoint, so it is synthesized for a path-bearing base too.
    // ---------------------------------------------------------------

    #[test]
    fn build_anthropic_url_synthesizes_v1_for_every_base_shape() {
        assert_eq!(
            build_anthropic_url("https://api.anthropic.com", "/messages"),
            "https://api.anthropic.com/v1/messages",
        );
        assert_eq!(
            build_anthropic_url("https://api.anthropic.com/", "/messages"),
            "https://api.anthropic.com/v1/messages",
        );
        // Operator importing the OpenAI habit — must not double.
        assert_eq!(
            build_anthropic_url("https://api.anthropic.com/v1", "/messages"),
            "https://api.anthropic.com/v1/messages",
        );
        // A self-hosted Anthropic-compatible gateway mounted under a
        // path still serves `<mount>/v1/messages` — unlike the OpenAI
        // family, the path here does NOT suppress the `/v1`.
        assert_eq!(
            build_anthropic_url("https://gw.corp.example/anthropic", "/messages"),
            "https://gw.corp.example/anthropic/v1/messages",
        );
        assert_eq!(
            build_anthropic_url(
                "https://gw.corp.example/anthropic",
                "/messages/count_tokens"
            ),
            "https://gw.corp.example/anthropic/v1/messages/count_tokens",
        );
    }

    #[test]
    #[should_panic(expected = "build_anthropic_url path must start with /")]
    fn build_anthropic_url_rejects_path_without_leading_slash() {
        let _ = build_anthropic_url("https://api.anthropic.com", "messages");
    }

    // --- resolve_bridge tests -------------------------------------
    //
    // Cover the reachable outcomes of resolve_bridge:
    //   1. specialized hit — pk.provider matches a specialized entry
    //   2. family hit       — pk.adapter matches a family entry,
    //                          specialized misses
    //   3. none miss        — neither tier matches (misconfigured PK)
    //
    // A minimal Bridge stub is used so the test doesn't need reqwest
    // or a real upstream.

    mod resolve_bridge_tests {
        use super::*;
        use aisix_core::models::Adapter;
        use aisix_gateway::{
            Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatMessage,
            ChatResponse, EmbeddingRequest, EmbeddingResponse, FinishReason, Hub, UsageStats,
        };
        use async_trait::async_trait;
        use futures::stream;

        /// Minimal Bridge that records its identity via `name()`. Lets
        /// resolve_bridge tests verify which Bridge was returned without
        /// dragging in reqwest.
        struct StubBridge {
            name: &'static str,
        }

        #[async_trait]
        impl Bridge for StubBridge {
            fn name(&self) -> &'static str {
                self.name
            }

            async fn chat(
                &self,
                req: &ChatFormat,
                _ctx: &BridgeContext,
            ) -> Result<ChatResponse, BridgeError> {
                Ok(ChatResponse {
                    id: "stub".into(),
                    model: req.model.clone(),
                    message: ChatMessage::assistant("stub"),
                    finish_reason: FinishReason::Stop,
                    usage: UsageStats::new(0, 0),
                })
            }

            async fn chat_stream(
                &self,
                _req: &ChatFormat,
                _ctx: &BridgeContext,
            ) -> Result<ChatChunkStream, BridgeError> {
                Ok(Box::pin(stream::iter(Vec::new())))
            }

            async fn embed(
                &self,
                _req: &EmbeddingRequest,
                _ctx: &BridgeContext,
            ) -> Result<EmbeddingResponse, BridgeError> {
                Err(BridgeError::Config("stub".into()))
            }
        }

        /// Build a ProviderKey JSON with the new-shape fields. `adapter`
        /// is passed as the kebab-case wire string (`"openai"` /
        /// `"azure-openai"` etc.) rather than the enum, to keep the
        /// helper independent of any `as_str()` method on `Adapter`.
        fn pk_with_provider_and_adapter(provider: &str, adapter: Option<&str>) -> ProviderKey {
            let adapter_json = match adapter {
                Some(a) => format!(", \"adapter\":\"{a}\""),
                None => String::new(),
            };
            let cfg = format!(
                r#"{{"display_name":"x","secret":"k","provider":"{provider}"{adapter_json}}}"#
            );
            serde_json::from_str(&cfg).unwrap()
        }

        #[test]
        fn specialized_hit_wins_over_family() {
            let hub = Hub::new();
            hub.register_specialized(
                "deepseek",
                Arc::new(StubBridge {
                    name: "specialized",
                }),
            );
            hub.register_family(Adapter::Openai, Arc::new(StubBridge { name: "family" }));

            let pk = pk_with_provider_and_adapter("deepseek", Some("openai"));
            let bridge = resolve_bridge(&hub, &pk).unwrap();
            assert_eq!(bridge.name(), "specialized");
        }

        #[test]
        fn family_hit_when_specialized_misses() {
            let hub = Hub::new();
            hub.register_family(Adapter::Openai, Arc::new(StubBridge { name: "family" }));

            // pk.provider = "unknown-vendor" → no specialized; pk.adapter
            // = Openai → family hit.
            let pk = pk_with_provider_and_adapter("unknown-vendor", Some("openai"));
            let bridge = resolve_bridge(&hub, &pk).unwrap();
            assert_eq!(bridge.name(), "family");
        }

        /// A PK whose `provider` matches no specialized entry and whose
        /// `adapter` matches no family entry has nothing to dispatch on
        /// — caller surfaces 503. cp-api always writes both fields, so
        /// this is a genuine misconfiguration, not a migration gap.
        #[test]
        fn none_when_neither_tier_matches() {
            let hub = Hub::new();
            hub.register_specialized("openai", Arc::new(StubBridge { name: "vendor" }));
            let pk = pk_with_provider_and_adapter("unknown-vendor", Some("anthropic"));
            assert!(resolve_bridge(&hub, &pk).is_none());
        }

        /// A PK with empty `provider` AND no `adapter` (the malformed
        /// shape the removed compat shim used to rescue) now resolves to
        /// nothing — 503.
        #[test]
        fn none_when_provider_and_adapter_both_empty() {
            let hub = Hub::new();
            hub.register_specialized("openai", Arc::new(StubBridge { name: "vendor" }));
            let pk = pk_with_provider_and_adapter("", None);
            assert!(resolve_bridge(&hub, &pk).is_none());
        }

        #[test]
        fn none_when_nothing_registered() {
            let hub = Hub::new();
            let pk = pk_with_provider_and_adapter("openai", Some("openai"));
            assert!(resolve_bridge(&hub, &pk).is_none());
        }

        /// A PK with a non-empty `provider` and an `adapter` whose
        /// family isn't registered misses both tiers authoritatively —
        /// it is NOT rescued by any fallback. If a future PR drops the
        /// `Adapter::Openai` family registration in `build_hub()`, this
        /// fires instead of silently routing elsewhere.
        #[test]
        fn none_when_adapter_family_not_registered() {
            let hub = Hub::new();
            hub.register_specialized(
                "openai",
                Arc::new(StubBridge {
                    name: "specialized-openai",
                }),
            );
            // `vendor-without-specialized` has no specialized entry;
            // `Adapter::Openai` has no family entry → None.
            let pk = pk_with_provider_and_adapter("vendor-without-specialized", Some("openai"));
            assert!(resolve_bridge(&hub, &pk).is_none());
        }
    }
}
