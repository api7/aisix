//! Bearer-token authentication for the proxy surface.
//!
//! The extractor [`AuthenticatedKey`] reads the caller's gateway credential
//! from the request headers ([`header_credential`] has the precedence),
//! looks the key up in the current `AisixSnapshot`, and yields the
//! matching `ApiKey` entity. Handlers take `AuthenticatedKey` as an
//! argument — if parsing or lookup fails the request is short-circuited
//! with a 401 envelope before the handler runs.

use aisix_core::resource::ResourceEntry;
use aisix_core::ApiKey;
use axum::extract::{FromRef, FromRequestParts};
use axum::http::request::Parts;
use std::sync::Arc;

use crate::error::ProxyError;
use crate::state::ProxyState;

#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub entry: Arc<ResourceEntry<ApiKey>>,
    /// The external identity behind this request when it authenticated
    /// with a JWT instead of the key's plaintext; `None` on the API-key
    /// path. Carried for usage attribution — the resolved key alone
    /// cannot name the subject once claim mappings let many identities
    /// share one key (AISIX-Cloud#564).
    pub jwt: Option<Arc<JwtIdentity>>,
    /// `true` when the principal came from an entry's anonymous
    /// configuration instead of a credential the caller presented —
    /// the MCP anonymous entries (AISIX-Cloud#1313) and passthrough
    /// routes in `auth_mode: anonymous`.
    ///
    /// `entry` names the principal either way; this says whether the
    /// caller PROVED it, which `entry` alone cannot tell once a key
    /// doubles as an anonymous principal. A handler that sets it must
    /// also stamp `UsageEvent::auth_type` (see
    /// `usage_attr::apply_auth_type`), or anonymous traffic becomes
    /// indistinguishable from the key's own in the usage record.
    pub anonymous: bool,
    /// The team and member documents the key's `team_id` / `user_id`
    /// name, resolved against the snapshot the request authenticated on.
    /// Read through [`AuthenticatedKey::team_name`] and
    /// [`AuthenticatedKey::user_name`], never off the key directly — the
    /// key's own `user_name` is only the fallback.
    pub owner: aisix_core::KeyOwnerNames,
    /// The request's effective cost-attribution tags, resolved from the key
    /// and its `owner` rows ([`aisix_core::KeyOwnerNames::effective_tags`]).
    /// Read by the `tag:` rate-limit policy dimensions and stamped on every
    /// usage event through [`AuthenticatedKey::caller`].
    pub tags: Arc<aisix_core::Tags>,
}

/// The principal's effective tags, published beside its owner names so
/// `ClientContext` reuses them instead of resolving them again.
#[derive(Clone)]
pub(crate) struct EffectiveTags(pub(crate) Arc<aisix_core::Tags>);

/// The verified JWT identity a request authenticated as.
pub struct JwtIdentity {
    /// Value of the trust provider's identity claim (`sub` by default).
    pub subject: String,
    /// Name of the OIDC provider that verified the token.
    pub provider: String,
    /// Name of the claim mapping that selected the API key, or `None`
    /// when the subject was bound to the key directly via `jwt_subject`.
    pub claim_mapping: Option<String>,
    /// The bearer token itself, exactly as the trust provider verified it —
    /// what an `oauth2_token_exchange` MCP server exchanges on the caller's
    /// behalf. Never printed.
    pub(crate) token: Arc<str>,
}

impl JwtIdentity {
    pub fn new(
        subject: String,
        provider: String,
        claim_mapping: Option<String>,
        token: Arc<str>,
    ) -> Self {
        Self {
            subject,
            provider,
            claim_mapping,
            token,
        }
    }
}

/// Hand-written so this type keeps a single, reviewed print shape: it is
/// carried on the request extensions through every handler, and a derived
/// form would follow whatever fields it grows next.
impl std::fmt::Debug for JwtIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwtIdentity")
            .field("subject", &self.subject)
            .field("provider", &self.provider)
            .field("claim_mapping", &self.claim_mapping)
            .finish()
    }
}

/// Per-request context carried onto an authentication denial's
/// `aisix::auth` log line, beside the access-log line the extractor writes
/// for the same request (see [`emit_denial_access_log`]).
#[derive(Clone, Copy)]
pub(crate) struct DenialContext<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub request_id: &'a str,
    pub source_ip: LazySourceIp<'a>,
}

/// Source IP for denial logs, resolved only when a denial actually logs
/// it. The successful-auth path has no consumer for it, so it never pays
/// the forwarded-header scan + IP formatting — `ClientContext` resolves
/// the same value once for the handlers instead.
#[derive(Clone, Copy)]
pub(crate) enum LazySourceIp<'a> {
    /// Resolve from the request parts + real-ip config on first use.
    Deferred(
        &'a axum::http::request::Parts,
        &'a crate::client_ip::ResolvedRealIp,
    ),
    /// Already resolved by the caller (WebSocket subprotocol auth, which
    /// has a `ClientContext` in hand).
    Ready(&'a str),
}

impl LazySourceIp<'_> {
    pub(crate) fn resolve(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Deferred(parts, cfg) => {
                std::borrow::Cow::Owned(crate::client_ip::source_ip_from_parts(parts, cfg))
            }
            Self::Ready(s) => std::borrow::Cow::Borrowed(s),
        }
    }
}

impl AuthenticatedKey {
    /// A principal resolved from `snapshot`, with its owner names read
    /// from the same snapshot.
    pub(crate) fn new(
        snapshot: &aisix_core::AisixSnapshot,
        entry: Arc<ResourceEntry<ApiKey>>,
        jwt: Option<Arc<JwtIdentity>>,
        anonymous: bool,
    ) -> Self {
        let owner = aisix_core::KeyOwnerNames::resolve(snapshot, &entry.value);
        Self {
            tags: Arc::new(owner.effective_tags(&entry.value)),
            owner,
            entry,
            jwt,
            anonymous,
        }
    }

    pub fn key(&self) -> &ApiKey {
        &self.entry.value
    }

    /// Publish the principal to the request's extensions, where
    /// `ClientContext` reads the key, its resolved owner names and tags.
    pub(crate) fn publish(&self, parts: &mut Parts) {
        parts.extensions.insert(self.entry.clone());
        parts.extensions.insert(self.owner.clone());
        parts
            .extensions
            .insert(EffectiveTags(Arc::clone(&self.tags)));
    }

    /// The key's display name, for the `api_key_name` metric label.
    pub(crate) fn api_key_name(&self) -> Option<&str> {
        self.key().telemetry_name()
    }

    /// See [`aisix_core::KeyOwnerNames::team_name`].
    pub(crate) fn team_name(&self) -> Option<&str> {
        self.owner.team_name()
    }

    /// See [`aisix_core::KeyOwnerNames::user_name`].
    pub(crate) fn user_name(&self) -> Option<&str> {
        self.owner.user_name(self.key())
    }

    /// The caller identity a usage event is attributed to, resolved from
    /// the same owner rows as [`Self::user_name`].
    pub(crate) fn caller(&self) -> aisix_gateway::CallerIdentity {
        aisix_gateway::CallerIdentity::from_entry(&self.entry, &self.owner, Arc::clone(&self.tags))
    }
}

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthenticatedKey
where
    S: Send + Sync,
    ProxyState: FromRef<S>,
{
    type Rejection = ProxyError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let started = std::time::Instant::now();
        let result = extract_authenticated_key(parts, state).await;
        if let Err(err) = &result {
            emit_denial_access_log(parts, started, err);
            record_denial_metrics(&ProxyState::from_ref(state), parts, started, err);
        }
        result
    }
}

/// Count a request the extractor refused in the request families, as every
/// other refusal is: without it a 401 / 403 on a typed route existed in the
/// access log and nowhere in `aisix_proxy_*` / `aisix_llm_*`.
fn record_denial_metrics(
    state: &ProxyState,
    parts: &Parts,
    started: std::time::Instant,
    err: &ProxyError,
) {
    crate::request_metrics::record_undispatched(
        state,
        // The raw path is caller-controlled; only the route template may
        // become a label (#451).
        crate::normalize_endpoint_label(parts.uri.path()),
        crate::request_metrics::Caller::unattributed(None),
        crate::request_metrics::Upstream {
            provider: crate::request_metrics::UNKNOWN,
            model: crate::usage_attr::UNRESOLVED_MODEL_LABEL,
            ..Default::default()
        },
        err.status().as_u16(),
        started.elapsed(),
    );
}

/// Write the access-log line for a request the extractor refused.
///
/// The refusal short-circuits ahead of the handler, which is where every
/// other line is written, so without this an authentication denial left no
/// access-log record at all. Nothing about the caller is resolved yet: the
/// line carries no key id (a disabled or expired key's id is on the
/// `aisix::auth` warn line), and never any part of the presented credential.
fn emit_denial_access_log(parts: &Parts, started: std::time::Instant, err: &ProxyError) {
    let request_id = parts
        .extensions
        .get::<crate::request_id::RequestId>()
        .map(|r| r.0.as_str())
        .unwrap_or_default();
    let elapsed = started.elapsed();
    let (error_kind, error) = crate::attempt::access_log_error(err);
    crate::attribution::emit_access_log(aisix_obs::AccessLog {
        method: parts.method.as_str(),
        path: parts.uri.path(),
        status: err.status().as_u16(),
        latency: elapsed,
        duration: elapsed,
        provider: None,
        model: None,
        upstream_model: None,
        provider_key_id: None,
        api_key_id: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        provider_request_id: None,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind: Some(error_kind),
        error: Some(&error),
        mcp: None,
        cache: None,
        request_body_bytes: None,
        response_body_bytes: None,
    });
}

async fn extract_authenticated_key<S>(
    parts: &mut Parts,
    state: &S,
) -> Result<AuthenticatedKey, ProxyError>
where
    S: Send + Sync,
    ProxyState: FromRef<S>,
{
    let proxy_state = ProxyState::from_ref(state);
    let request_id = parts
        .extensions
        .get::<crate::request_id::RequestId>()
        .map(|r| r.0.as_str())
        .unwrap_or_default();
    let ctx = DenialContext {
        method: parts.method.as_str(),
        path: parts.uri.path(),
        request_id,
        // Deferred: only a denial resolves the source IP.
        source_ip: LazySourceIp::Deferred(&*parts, proxy_state.real_ip.as_ref()),
    };
    let token = match extract_bearer(parts) {
        Ok(t) => t,
        Err(e) => {
            // No credential at all: the scanner-probe shape, so the
            // `aisix::auth` line stays at debug.
            proxy_state
                .metrics
                .record_auth_decision("none", false, "missing_credentials");
            tracing::debug!(
                target: "aisix::auth",
                method = "none",
                reason = "missing_credentials",
                http_method = %ctx.method,
                path = %ctx.path,
                request_id = %ctx.request_id,
                source_ip = %ctx.source_ip.resolve(),
                "rejected inbound request without a credential",
            );
            return Err(e);
        }
    };
    let authed = authenticate_token(&proxy_state, &token, ctx).await?;
    // Publish the resolved key so extractors that run after this one can
    // see who is calling without re-authenticating. `ClientContext` reads
    // it for the `${request.api_key.*}` header templates
    // (AISIX-Cloud#1112) — which is why every handler declares
    // `auth: AuthenticatedKey` before `client: ClientContext`.
    authed.publish(parts);
    // Same for the JWT identity: `ClientContext` carries it to each
    // handler's usage-event emitter for attribution.
    if let Some(jwt) = &authed.jwt {
        parts.extensions.insert(jwt.clone());
    }
    Ok(authed)
}

/// Authenticate a plaintext bearer and enforce key lifecycle. The single
/// auth choke point behind the [`AuthenticatedKey`] extractor; also
/// called directly by surfaces whose credentials arrive outside the
/// standard headers (WebSocket subprotocol auth on `/v1/realtime`).
///
/// A JWT-shaped bearer takes the OIDC path (`crate::jwt`) whenever the
/// environment has an enabled trust provider — with no fallback to the
/// key lookup on failure, so a rejected JWT can never be retried as a
/// key. Everything else is looked up as an API key by hash.
pub(crate) async fn authenticate_token(
    state: &ProxyState,
    token: &str,
    ctx: DenialContext<'_>,
) -> Result<AuthenticatedKey, ProxyError> {
    let authed = authenticate_token_inner(state, token, ctx).await?;
    // Hand the principal to the request's attribution cell, so a caller
    // that hangs up still files an ATTRIBUTABLE usage row
    // (AISIX-Cloud#1571). Here rather than in the extractor above because
    // three surfaces authenticate without it — a passthrough route's own
    // `auth_mode`, `/v1/realtime`'s WebSocket subprotocol, and `/mcp` —
    // and two of them build no `ClientContext` either. The two places that
    // mint an ANONYMOUS principal instead of verifying a credential note it
    // themselves (`mcp::resolve_caller`, `passthrough_route::authenticate`).
    crate::attribution::note_authenticated(&authed);
    Ok(authed)
}

async fn authenticate_token_inner(
    state: &ProxyState,
    token: &str,
    ctx: DenialContext<'_>,
) -> Result<AuthenticatedKey, ProxyError> {
    let snapshot = state.snapshot.load();
    // Provider gate first: it is O(1) when no trust provider is
    // configured, so key-only deployments never pay the structural JWT
    // probe (base64 + JSON decode of the header segment).
    if crate::jwt::any_enabled_provider(&snapshot) && crate::jwt::looks_like_jwt(token) {
        return crate::jwt::authenticate_jwt(state, &snapshot, token, ctx).await;
        // No enabled trust provider: fall through to the key path so a
        // custom-imported key that happens to look like a JWT keeps
        // authenticating on deployments that never enable JWT auth.
    }
    // The control plane (prd-09a §9A.7B.4) stores
    // SHA-256 hashes of the plaintext bearer, never the plaintext
    // itself. Hash the incoming token via the canonical helper
    // and look up by the hex digest. cp-api hashes with the same
    // function before persistence, so the two sides agree byte
    // for byte.
    let Some(entry) = snapshot.apikeys.get_by_name(&ApiKey::hash_bearer(token)) else {
        return Err(deny_key(
            state,
            "unknown_key",
            None,
            ctx,
            ProxyError::InvalidApiKey,
        ));
    };
    // Lifecycle enforcement (#933): a known key that is disabled or
    // past its expiry deadline must be rejected here, at the single
    // auth choke point, so every proxy surface (chat, messages,
    // responses, embeddings, audio, passthrough, MCP, …) inherits
    // the same 401 without per-handler checks.
    if entry.value.disabled {
        return Err(deny_key(
            state,
            "key_disabled",
            Some(&entry.id),
            ctx,
            ProxyError::ApiKeyDisabled,
        ));
    }
    // Read the wall clock only for keys that actually carry a deadline;
    // enforcement is unchanged (`is_expired_at` is false for `None`).
    if entry.value.expires_at.is_some() && entry.value.is_expired_at(chrono::Utc::now()) {
        return Err(deny_key(
            state,
            "key_expired",
            Some(&entry.id),
            ctx,
            ProxyError::ApiKeyExpired,
        ));
    }
    state.metrics.record_auth_decision("api_key", true, "");
    Ok(AuthenticatedKey::new(&snapshot, entry, None, false))
}

/// Record an API-key denial on the decision metric + log. The token itself
/// is never logged.
///
/// `unknown_key` is the scanner-probe shape (`Bearer <junk>`), so it logs
/// at `debug` and an internet-facing DP is not flooded at the default
/// level. `key_disabled` / `key_expired` name a real key an operator
/// provisioned, so they stay at `warn` and carry `api_key_id`.
///
/// Every line carries the [`DenialContext`] — the caller's address, the
/// route it hit, and the request id — which the metric has no dimensions
/// for and the access-log line does not carry (it has no source address).
fn deny_key(
    state: &ProxyState,
    reason: &'static str,
    api_key_id: Option<&str>,
    ctx: DenialContext<'_>,
    err: ProxyError,
) -> ProxyError {
    state.metrics.record_auth_decision("api_key", false, reason);
    if reason == "unknown_key" {
        tracing::debug!(
            target: "aisix::auth",
            method = "api_key",
            reason = %reason,
            http_method = %ctx.method,
            path = %ctx.path,
            request_id = %ctx.request_id,
            source_ip = %ctx.source_ip.resolve(),
            "rejected inbound credential",
        );
    } else {
        tracing::warn!(
            target: "aisix::auth",
            method = "api_key",
            reason = %reason,
            api_key_id = ?api_key_id,
            http_method = %ctx.method,
            path = %ctx.path,
            request_id = %ctx.request_id,
            source_ip = %ctx.source_ip.resolve(),
            "rejected inbound credential",
        );
    }
    err
}

/// Dedicated header for the caller's gateway credential. When present it
/// is the ONLY header read for gateway authentication, which leaves
/// `Authorization` free to carry a credential meant for an upstream (an MCP
/// server's own OAuth token, for instance). Its `x-aisix-` prefix keeps it
/// off every upstream request (`aisix_core::header_forward_blocked`).
pub(crate) const GATEWAY_KEY_HEADER: &str =
    aisix_core::forwarded_headers::GATEWAY_CREDENTIAL_HEADER;

/// How a `MissingAuth` message spells `Authorization` (display text, not
/// the lowercase wire name) — also what it names when the request carries
/// no credential header at all.
pub(crate) const AUTHORIZATION_LABEL: &str = "Authorization";
pub(crate) const X_API_KEY_HEADER: &str = "x-api-key";

/// Whether the request OFFERS an inbound gateway credential at all.
///
/// Deliberately wider than [`header_credential`]: a present-but-unusable
/// credential header — wrong scheme, empty value, bytes that are not
/// ASCII — counts as offered. Anonymous entry (AISIX-Cloud#1313) is the
/// no-credential path, so a caller that tried to authenticate and got
/// it wrong must be rejected rather than quietly succeed as the
/// anonymous principal with a different grant than it asked for.
pub(crate) fn credential_offered(parts: &Parts) -> bool {
    [
        GATEWAY_KEY_HEADER,
        axum::http::header::AUTHORIZATION.as_str(),
        X_API_KEY_HEADER,
    ]
    .iter()
    .any(|name| parts.headers.contains_key(*name))
}

/// Whether the caller's gateway credential came from [`GATEWAY_KEY_HEADER`]
/// rather than `Authorization` / `x-api-key` — i.e. whether an
/// `Authorization` on the request was NOT what the gateway admitted it with.
pub(crate) fn admitted_via_gateway_key_header(headers: &axum::http::HeaderMap) -> bool {
    headers.contains_key(GATEWAY_KEY_HEADER)
}

/// The caller's gateway credential, read from the request headers by
/// precedence: [`GATEWAY_KEY_HEADER`] when the header is present, else
/// `Authorization: Bearer`, else `x-api-key`. The first header present
/// decides — an empty or malformed one is `MissingAuth` naming that
/// header, never a fall-through to the next. `None` when none of the three
/// is present.
pub(crate) fn header_credential(
    headers: &axum::http::HeaderMap,
) -> Option<Result<String, ProxyError>> {
    if let Some(raw) = headers.get(GATEWAY_KEY_HEADER) {
        return Some(gateway_key_header_credential(raw));
    }
    if let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) {
        let Ok(s) = auth.to_str() else {
            return Some(Err(ProxyError::MissingAuth(AUTHORIZATION_LABEL)));
        };
        return Some(match s.strip_prefix("Bearer ").map(str::trim) {
            Some(rest) if !rest.is_empty() => Ok(rest.to_string()),
            _ => Err(ProxyError::MissingAuth(AUTHORIZATION_LABEL)),
        });
    }
    headers
        .get(X_API_KEY_HEADER)
        .map(|raw| raw_credential(raw, X_API_KEY_HEADER))
}

/// The credential in [`GATEWAY_KEY_HEADER`]: the key itself, or the key
/// behind a `Bearer ` scheme for clients that only know how to send one.
pub(crate) fn gateway_key_header_credential(
    raw: &axum::http::HeaderValue,
) -> Result<String, ProxyError> {
    let value = raw_credential(raw, GATEWAY_KEY_HEADER)?;
    match value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .map(str::trim)
    {
        Some("") => Err(ProxyError::MissingAuth(GATEWAY_KEY_HEADER)),
        Some(key) => Ok(key.to_string()),
        None => Ok(value),
    }
}

fn raw_credential(
    raw: &axum::http::HeaderValue,
    header: &'static str,
) -> Result<String, ProxyError> {
    let s = raw
        .to_str()
        .map_err(|_| ProxyError::MissingAuth(header))?
        .trim();
    if s.is_empty() {
        return Err(ProxyError::MissingAuth(header));
    }
    Ok(s.to_string())
}

fn extract_bearer(parts: &Parts) -> Result<String, ProxyError> {
    header_credential(&parts.headers).unwrap_or(Err(ProxyError::MissingAuth(AUTHORIZATION_LABEL)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue, Request};

    /// A tracing writer that appends every emitted byte into a shared buffer
    /// so a test can assert what a log line carried.
    #[derive(Clone)]
    struct BufWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl tracing_subscriber::fmt::MakeWriter<'_> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    /// Only one test at a time may install a process-wide subscriber.
    static CAPTURE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Keep every callsite emittable for the whole test binary.
    ///
    /// `set_default` below is thread-local, but a callsite's `Interest` is
    /// cached *process-wide* the first time that callsite is hit, from
    /// whichever dispatcher the hitting thread has. With no global default
    /// that is `NoSubscriber`, so a sibling test reaching the denial path on
    /// another thread caches `Interest::never()` and the event is then
    /// skipped everywhere — including inside the capture, which sees an
    /// empty buffer. Dozens of router tests hit the unauthenticated path, so
    /// under the parallel harness that race is routine, not exotic.
    ///
    /// A permissive global default removes the outcome: no thread ever falls
    /// back to `NoSubscriber`. Registering it also re-evaluates the
    /// callsites seen so far, so installing it lazily still repairs a cache
    /// poisoned earlier in the run.
    fn keep_callsites_enabled() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // A bare registry writes nothing and formats nothing; it is here
            // only so that callsites register as enabled. The captured events
            // are rendered by the scoped subscriber below.
            let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
        });
    }

    /// Run `f` with a log-capturing subscriber installed and return
    /// everything it emitted. `unknown_key` / `missing_credentials` are the
    /// scanner-probe shapes and log at DEBUG on purpose, so widen past the
    /// builder's INFO default.
    async fn capture_logs<F, Fut>(f: F) -> String
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        /// One capture pass: run `emit` with a fresh buffer-backed scoped
        /// subscriber installed and return what it wrote.
        async fn scoped_capture<Fut: std::future::Future<Output = ()>>(
            emit: impl FnOnce() -> Fut,
        ) -> String {
            let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(false)
                .with_max_level(tracing::Level::DEBUG)
                .with_writer(BufWriter(buf.clone()))
                .finish();
            {
                let _guard = tracing::subscriber::set_default(subscriber);
                emit().await;
            }
            let bytes = buf.lock().unwrap().clone();
            String::from_utf8(bytes).unwrap()
        }

        keep_callsites_enabled();
        let _capture_guard = CAPTURE_LOCK.lock().await;
        let text = scoped_capture(&f).await;
        if text.is_empty() {
            // An empty capture has flaked exactly once on main (run
            // 31137791884, `got: ` with nothing behind it) and did not
            // reproduce in ~390 stressed suite executions, so it cannot be
            // studied on demand. Self-diagnose instead, splitting the
            // mechanism space three ways for the next natural occurrence:
            // re-run `f` — driving the SAME production callsites — under a
            // fresh scope, and emit a probe through a fresh callsite. The
            // rerun capturing means the original callsites work now (the
            // emptiness was confined to the first scope); only the probe
            // capturing means those specific callsites stay suppressed
            // while the machinery works (per-callsite Interest poisoning);
            // neither capturing means global suppression (max-level hint /
            // dispatcher state).
            eprintln!(
                "capture_logs: EMPTY capture; LevelFilter::current()={:?}",
                tracing::level_filters::LevelFilter::current(),
            );
            let rerun_captured = !scoped_capture(&f).await.is_empty();
            let probe_captured = !scoped_capture(|| async {
                tracing::debug!(target: "aisix::auth", probe = true, "capture-probe");
            })
            .await
            .is_empty();
            let verdict = match (rerun_captured, probe_captured) {
                (true, _) => "rerun captured — emptiness was confined to the first scope",
                (false, true) => {
                    "rerun EMPTY, fresh-callsite probe captured — original callsites suppressed"
                }
                (false, false) => "rerun and probe both EMPTY — global suppression",
            };
            eprintln!("capture_logs: {verdict}");
        }
        text
    }

    const GOOD_TOKEN: &str = "sk-auth-ctx-good";

    fn router_with_key(disabled: bool) -> axum::Router {
        use aisix_core::snapshot::SnapshotHandle;
        use aisix_core::{AisixSnapshot, ProxyConfig};

        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": ApiKey::hash_bearer(GOOD_TOKEN),
            "allowed_models": ["*"],
            "disabled": disabled,
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-ctx-1", apikey, 1));
        let cfg = ProxyConfig {
            thread_per_core: None,
            workers: None,
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1 << 20,
            tls: None,
            listeners: Vec::new(),
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            expose_debug_headers: false,
        };
        crate::build_router(
            ProxyState::new(
                SnapshotHandle::new(snapshot),
                Arc::new(aisix_gateway::Hub::new()),
                &cfg,
            )
            .without_cache(),
        )
    }

    async fn call_with_bearer(router: axum::Router, bearer: Option<&str>) {
        use tower::ServiceExt;
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json");
        if let Some(b) = bearer {
            builder = builder.header("authorization", format!("Bearer {b}"));
        }
        let mut request = builder
            .body(axum::body::Body::from(r#"{"model":"m","messages":[]}"#))
            .unwrap();
        // The peer address the real server injects; without it the
        // extractor has nothing to resolve the caller's address from.
        request
            .extensions_mut()
            .insert(axum::extract::ConnectInfo(std::net::SocketAddr::from((
                [203, 0, 113, 7],
                5555,
            ))));
        let _ = router.oneshot(request).await.expect("router responds");
    }

    /// A 401 short-circuits ahead of every handler, so it never reaches an
    /// access-log emit and `aisix_auth_decisions_total` is the only other
    /// record — and that metric has no caller, route, or request id. The
    /// denial line is therefore the one place an operator can learn who is
    /// hammering the gateway with a bad key; it must name them.
    #[tokio::test]
    async fn an_unknown_key_denial_names_the_caller_and_the_route() {
        let logs = capture_logs(|| call_with_bearer(router_with_key(false), Some("sk-nope"))).await;
        assert!(logs.contains("reason=unknown_key"), "got: {logs}");
        assert!(logs.contains("http_method=POST"), "got: {logs}");
        assert!(logs.contains("path=/v1/chat/completions"), "got: {logs}");
        assert!(logs.contains("source_ip=203.0.113.7"), "got: {logs}");
        // Correlates the denial with everything else stamped with this id.
        assert!(logs.contains("request_id="), "got: {logs}");
    }

    /// Same for a request that carries no credential at all.
    #[tokio::test]
    async fn a_missing_credential_denial_names_the_caller_and_the_route() {
        let logs = capture_logs(|| call_with_bearer(router_with_key(false), None)).await;
        assert!(
            logs.contains(r#"reason="missing_credentials""#),
            "got: {logs}"
        );
        assert!(logs.contains("path=/v1/chat/completions"), "got: {logs}");
        assert!(logs.contains("source_ip=203.0.113.7"), "got: {logs}");
    }

    /// A disabled key names a real key an operator provisioned, so the line
    /// additionally identifies WHICH key — the operator's next action is to
    /// look it up.
    #[tokio::test]
    async fn a_disabled_key_denial_also_names_the_key() {
        let logs = capture_logs(|| call_with_bearer(router_with_key(true), Some(GOOD_TOKEN))).await;
        assert!(logs.contains("reason=key_disabled"), "got: {logs}");
        assert!(logs.contains("ak-ctx-1"), "got: {logs}");
        assert!(logs.contains("source_ip=203.0.113.7"), "got: {logs}");
    }

    fn parts_with(headers: HeaderMap) -> Parts {
        let mut req = Request::builder().uri("/").body(()).unwrap();
        *req.headers_mut() = headers;
        req.into_parts().0
    }

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (name, value) in pairs {
            h.insert(*name, HeaderValue::from_static(value));
        }
        h
    }

    #[test]
    fn extract_bearer_happy_path() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-abc"),
        );
        let parts = parts_with(h);
        assert_eq!(extract_bearer(&parts).unwrap(), "sk-abc");
    }

    #[test]
    fn extract_bearer_accepts_x_api_key_as_alternative() {
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("sk-abc"));
        let parts = parts_with(h);
        assert_eq!(extract_bearer(&parts).unwrap(), "sk-abc");
    }

    #[test]
    fn the_gateway_key_header_wins_and_never_falls_through() {
        let both = headers(&[
            ("x-aisix-api-key", "sk-gateway"),
            ("authorization", "Bearer upstream-token"),
            ("x-api-key", "sk-other"),
        ]);
        assert_eq!(extract_bearer(&parts_with(both)).unwrap(), "sk-gateway");
        let with_scheme = headers(&[("x-aisix-api-key", "Bearer sk-gateway")]);
        assert_eq!(
            extract_bearer(&parts_with(with_scheme)).unwrap(),
            "sk-gateway"
        );
        // Present but empty: the missing-auth error, not the next header.
        let empty = headers(&[
            ("x-aisix-api-key", "  "),
            ("authorization", "Bearer sk-valid"),
        ]);
        assert!(matches!(
            extract_bearer(&parts_with(empty.clone())),
            Err(ProxyError::MissingAuth(_))
        ));
        assert!(credential_offered(&parts_with(headers(&[(
            "x-aisix-api-key",
            "sk-gateway"
        )]))));
        // Without it, Authorization still precedes x-api-key.
        let legacy = headers(&[
            ("authorization", "Bearer sk-auth"),
            ("x-api-key", "sk-xapi"),
        ]);
        assert_eq!(extract_bearer(&parts_with(legacy)).unwrap(), "sk-auth");
    }

    #[test]
    fn extract_bearer_rejects_missing_header() {
        let parts = parts_with(HeaderMap::new());
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth(_))
        ));
    }

    #[test]
    fn extract_bearer_rejects_wrong_scheme() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwdw=="),
        );
        let parts = parts_with(h);
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth(_))
        ));
    }

    #[test]
    fn credential_offered_covers_unusable_credentials() {
        // Absent → not offered: the anonymous path may serve these.
        assert!(!credential_offered(&parts_with(HeaderMap::new())));

        // Present but unusable → offered: must fail, never downgrade. An
        // empty or whitespace-only value is one of those.
        for name in [
            GATEWAY_KEY_HEADER,
            axum::http::header::AUTHORIZATION.as_str(),
            X_API_KEY_HEADER,
        ] {
            for value in ["", "   ", "Bearer   ", "Basic dXNlcjpwdw==", "sk-abc"] {
                let mut h = HeaderMap::new();
                h.insert(name, HeaderValue::from_str(value).unwrap());
                assert!(
                    credential_offered(&parts_with(h)),
                    "{name}: {value:?} must count as an offered credential"
                );
            }
        }
    }

    #[test]
    fn missing_auth_names_the_header_that_failed() {
        let message = |pairs: &[(&'static str, &'static str)]| {
            extract_bearer(&parts_with(headers(pairs)))
                .unwrap_err()
                .to_string()
        };
        let authorization = "missing or malformed Authorization header";
        assert_eq!(message(&[]), authorization);
        assert_eq!(message(&[("authorization", "")]), authorization);
        assert_eq!(message(&[("authorization", "Basic abc")]), authorization);
        assert_eq!(
            message(&[
                ("x-aisix-api-key", ""),
                ("authorization", "Bearer sk-valid")
            ]),
            "missing or malformed x-aisix-api-key header"
        );
        assert_eq!(
            message(&[("x-api-key", "  ")]),
            "missing or malformed x-api-key header"
        );
    }

    #[test]
    fn extract_bearer_rejects_empty_bearer() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer   "),
        );
        let parts = parts_with(h);
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth(_))
        ));
    }
}
