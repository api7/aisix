//! Bearer-token authentication for the proxy surface.
//!
//! The extractor [`AuthenticatedKey`] parses `Authorization: Bearer <key>`
//! (or `x-api-key: <key>` as a convenience alternative), looks the key
//! up in the current `AisixSnapshot`, and yields the matching `ApiKey`
//! entity. Handlers take `AuthenticatedKey` as an argument — if parsing
//! or lookup fails the request is short-circuited with a 401 envelope
//! before the handler runs.

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
}

/// The verified JWT identity a request authenticated as.
pub struct JwtIdentity {
    /// Value of the trust provider's identity claim (`sub` by default).
    pub subject: String,
    /// Name of the OIDC provider that verified the token.
    pub provider: String,
    /// Name of the claim mapping that selected the API key, or `None`
    /// when the subject was bound to the key directly via `jwt_subject`.
    pub claim_mapping: Option<String>,
}

impl JwtIdentity {
    pub fn new(subject: String, provider: String, claim_mapping: Option<String>) -> Self {
        Self {
            subject,
            provider,
            claim_mapping,
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

/// Per-request context carried onto an authentication denial.
///
/// A 401 short-circuits ahead of every handler, so the request never
/// reaches an access-log emit; AISIX-Cloud#1081 deliberately kept these out
/// of the access log because an internet-facing DP would drown in scanner
/// probes, leaving `aisix_auth_decisions_total` as the only record. That
/// metric answers "how many" but not "who, when, against what" — so the
/// denial log line, which an operator turns on precisely when investigating,
/// carries the identifying detail instead.
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
    pub fn key(&self) -> &ApiKey {
        &self.entry.value
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
                // No credential at all: metric + a debug line, never the
                // access log — logging every scanner probe at the default
                // level would be noise (AISIX-Cloud#1081).
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
        parts.extensions.insert(authed.entry.clone());
        // Same for the JWT identity: `ClientContext` carries it to each
        // handler's usage-event emitter for attribution.
        if let Some(jwt) = &authed.jwt {
            parts.extensions.insert(jwt.clone());
        }
        Ok(authed)
    }
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
    // Self-hosted CP (prd-09a §9A.7B.4): the snapshot stores
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
    Ok(AuthenticatedKey {
        entry,
        jwt: None,
        anonymous: false,
    })
}

/// Record an API-key denial on the decision metric + log
/// (AISIX-Cloud#1081 — before this, a 401 was invisible: the extractor
/// short-circuits ahead of every handler, so neither the request
/// counter nor the access log ever fired). The token itself is never
/// logged.
///
/// `unknown_key` is the scanner-probe shape (`Bearer <junk>`), so it logs
/// at `debug` and an internet-facing DP is not flooded at the default
/// level. `key_disabled` / `key_expired` name a real key an operator
/// provisioned, so they stay at `warn` and carry `api_key_id`.
///
/// Every line carries the [`DenialContext`] — the caller's address, the
/// route it hit, and the request id. Without it the record was
/// unactionable: the metric has no such dimensions, and the 401 never
/// reaches the access log, so "who is hammering us with a bad key" had no
/// answer anywhere.
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

/// Whether the request OFFERS an inbound gateway credential at all.
///
/// Deliberately wider than [`extract_bearer`]: a present-but-unusable
/// `Authorization` — wrong scheme, empty value, bytes that are not
/// ASCII — counts as offered. Anonymous entry (AISIX-Cloud#1313) is the
/// no-credential path, so a caller that tried to authenticate and got
/// it wrong must be rejected rather than quietly succeed as the
/// anonymous principal with a different grant than it asked for.
pub(crate) fn credential_offered(parts: &Parts) -> bool {
    [axum::http::header::AUTHORIZATION.as_str(), "x-api-key"]
        .iter()
        .any(|name| {
            parts
                .headers
                .get(*name)
                .is_some_and(|v| v.to_str().map_or(true, |s| !s.trim().is_empty()))
        })
}

fn extract_bearer(parts: &Parts) -> Result<String, ProxyError> {
    if let Some(auth) = parts.headers.get(axum::http::header::AUTHORIZATION) {
        let s = auth.to_str().map_err(|_| ProxyError::MissingAuth)?;
        if let Some(rest) = s.strip_prefix("Bearer ") {
            let rest = rest.trim();
            if rest.is_empty() {
                return Err(ProxyError::MissingAuth);
            }
            return Ok(rest.to_string());
        }
        return Err(ProxyError::MissingAuth);
    }
    if let Some(raw) = parts.headers.get("x-api-key") {
        let s = raw.to_str().map_err(|_| ProxyError::MissingAuth)?;
        let s = s.trim();
        if s.is_empty() {
            return Err(ProxyError::MissingAuth);
        }
        return Ok(s.to_string());
    }
    Err(ProxyError::MissingAuth)
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
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
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
    fn extract_bearer_rejects_missing_header() {
        let parts = parts_with(HeaderMap::new());
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth)
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
            Err(ProxyError::MissingAuth)
        ));
    }

    #[test]
    fn credential_offered_covers_unusable_credentials() {
        // Absent → not offered: the anonymous path may serve these.
        assert!(!credential_offered(&parts_with(HeaderMap::new())));
        let mut empty = HeaderMap::new();
        empty.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("   "),
        );
        assert!(!credential_offered(&parts_with(empty)));

        // Present but unusable → offered: must fail, never downgrade.
        for (name, value) in [
            (axum::http::header::AUTHORIZATION.as_str(), "Bearer sk-abc"),
            (axum::http::header::AUTHORIZATION.as_str(), "Bearer   "),
            (
                axum::http::header::AUTHORIZATION.as_str(),
                "Basic dXNlcjpwdw==",
            ),
            ("x-api-key", "sk-abc"),
        ] {
            let mut h = HeaderMap::new();
            h.insert(name, HeaderValue::from_str(value).unwrap());
            assert!(
                credential_offered(&parts_with(h)),
                "{name}: {value:?} must count as an offered credential"
            );
        }
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
            Err(ProxyError::MissingAuth)
        ));
    }
}
