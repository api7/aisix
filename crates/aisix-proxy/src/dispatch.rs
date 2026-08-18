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
use aisix_core::{AisixSnapshot, Model, ProviderKey};
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
    if model.provider.as_deref() == Some("anthropic") {
        return true;
    }
    resolve_provider_key(snapshot, model)
        .is_ok_and(|pk| pk.value.adapter == Some(aisix_core::Adapter::Anthropic))
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
/// otherwise the `Provider`'s built-in default. Tolerates an operator
/// pasting the full upstream URL into `api_base` by stripping any
/// trailing endpoint suffix — see [`API_BASE_ENDPOINT_SUFFIXES`].
pub(crate) fn resolve_base_url(provider_key: &ProviderKey) -> Result<String, ProxyError> {
    match provider_key.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => Ok(strip_endpoint_suffix(b.trim()).to_string()),
        _ => Err(ProxyError::InvalidRequest(format!(
            "provider_key {:?} has no api_base — cp-api must populate api_base \
             for every catalog vendor (the DP does not enumerate per-vendor \
             default URLs)",
            provider_key.display_name
        ))),
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
        Some(c) => ctx.with_client(c.caller.clone(), Some(c.headers.clone())),
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
