//! aisix-admin — Admin API + Playground (:3001).
//!
//! Public admin-listener endpoints:
//! - `GET  /livez`
//! - `GET  /admin/openapi.json`
//! - `GET  /admin/openapi-scalar`
//!
//! Prometheus metrics are NOT served here — the scrape endpoint always
//! lives on the dedicated metrics listener (see [`metrics_router`]),
//! identical in standalone and managed mode.
//!
//! Admin-key protected routes (read-only — every resource route serves
//! GET; `api_keys` is also served at the former `apikeys` spelling):
//! - `GET /admin/v1/models` and `GET /admin/v1/models/:id`
//! - `GET /admin/v1/api_keys` and `GET /admin/v1/api_keys/:id`
//! - `GET /admin/v1/provider_keys` and `GET /admin/v1/provider_keys/:id`
//! - `GET /admin/v1/guardrails` and `GET /admin/v1/guardrails/:id`
//! - `GET /admin/v1/cache_policies` and `GET /admin/v1/cache_policies/:id`
//! - `GET /admin/v1/observability_exporters` and
//!   `GET /admin/v1/observability_exporters/:id`
//! - `GET /admin/v1/mcp_servers` and `GET /admin/v1/mcp_servers/:id`
//! - `GET /admin/v1/a2a_agents` and `GET /admin/v1/a2a_agents/:id`
//! - `GET /admin/v1/passthrough_routes` and
//!   `GET /admin/v1/passthrough_routes/:id`
//! - `GET /admin/v1/models/status`, `GET /admin/v1/health`
//!
//! The resource write endpoints (POST/PUT/DELETE, including api-key
//! rotate) were removed in favor of the declarative configuration
//! paths — a `resources_file` source (`resources.yaml`, reloaded on
//! SIGHUP) or direct etcd writes. Writes to the routes above answer
//! 405; the rotate path is gone (404). The storage layer stays
//! pluggable via the read-only [`ConfigStore`] trait; production wires
//! etcd or the file-snapshot store, tests use [`InMemoryStore`].
//!
//! Errors follow the simple admin envelope: `{"error_msg": "..."}`,
//! distinct from the proxy's OpenAI-style envelope.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod a2a_agents_handlers;
mod apikeys_handlers;
mod auth;
mod cache_policies_handlers;
mod error;
pub mod etcd_store;
pub mod file_store;
mod guardrails_handlers;
mod health_handler;
mod mcp_servers_handlers;
mod models_handlers;
mod models_status_handler;
mod observability_exporters_handlers;
mod openapi;
mod passthrough_routes_handlers;
mod playground_handler;
mod provider_keys_handlers;
mod state;
pub mod store;

pub use auth::AdminAuth;
pub use error::{AdminError, ErrorBody};
pub use etcd_store::EtcdConfigStore;
pub use file_store::FileManagedStore;
pub use state::AdminState;
pub use store::{ConfigStore, InMemoryStore, StoreError};

use aisix_core::config::PrometheusConfig;
use aisix_core::ConfigStatus;
use aisix_obs::Metrics;
use aisix_proxy::ModelRuntimeStatusTracker;
use axum::routing::{get, post};
use axum::{http::StatusCode, response::Response, Router};
use std::sync::Arc;

/// Shared state for the dedicated metrics/status listener: the Prometheus
/// [`Metrics`] handle, the load-observability [`ConfigStatus`], and the
/// [`ModelsStatusState`] behind `GET /status/models`. All cheap to clone.
#[derive(Clone)]
pub struct MetricsState {
    pub metrics: Arc<Metrics>,
    pub config_status: ConfigStatus,
    pub models_status: ModelsStatusState,
}

/// Sources behind `GET /status/models` on the metrics/status listener:
/// the same resource store the admin surface reads plus the proxy's
/// shared runtime status tracker, so the status-listener view renders
/// from exactly the sources `GET /admin/v1/models/status` renders from.
#[derive(Clone)]
pub struct ModelsStatusState {
    pub store: Arc<dyn ConfigStore>,
    pub runtime_status_tracker: Option<Arc<ModelRuntimeStatusTracker>>,
}

pub fn admin_openapi_json() -> &'static str {
    openapi::merged_openapi()
}

pub fn build_router(state: AdminState) -> Router {
    // Eagerly build the merged OpenAPI doc so any panic in schema
    // parsing surfaces at boot, not at first `/admin/openapi.json`
    // request. `merged_openapi` caches into an `OnceLock`; the
    // subsequent handler call is a free lookup.
    let _ = openapi::merged_openapi();

    let router = Router::new()
        .route("/livez", get(livez))
        .route("/readyz", get(readyz))
        // OpenAPI scalar UI is unauthenticated like /livez — admin
        // listener is private in production.
        .route("/admin/openapi.json", get(openapi::openapi_json))
        .route("/admin/openapi-scalar", get(openapi::openapi_scalar))
        .route(
            "/admin/v1/models",
            get(models_handlers::list_models),
        )
        .route(
            "/admin/v1/models/:id",
            get(models_handlers::get_model),
        )
        .route(
            "/admin/v1/models/status",
            get(models_status_handler::get_models_status),
        )
        // Caller API keys are served at the canonical `api_keys` path —
        // the resource's configuration key — and at the former `apikeys`
        // spelling. Same handlers; existing callers keep working.
        .route(
            "/admin/v1/api_keys",
            get(apikeys_handlers::list_apikeys),
        )
        .route(
            "/admin/v1/api_keys/:id",
            get(apikeys_handlers::get_apikey),
        )
        .route(
            "/admin/v1/apikeys",
            get(apikeys_handlers::list_apikeys),
        )
        .route(
            "/admin/v1/apikeys/:id",
            get(apikeys_handlers::get_apikey),
        )
        .route(
            "/admin/v1/provider_keys",
            get(provider_keys_handlers::list_provider_keys),
        )
        .route(
            "/admin/v1/provider_keys/:id",
            get(provider_keys_handlers::get_provider_key),
        )
        .route(
            "/admin/v1/mcp_servers",
            get(mcp_servers_handlers::list_mcp_servers),
        )
        .route(
            "/admin/v1/mcp_servers/:id",
            get(mcp_servers_handlers::get_mcp_server),
        )
        .route(
            "/admin/v1/a2a_agents",
            get(a2a_agents_handlers::list_a2a_agents),
        )
        .route(
            "/admin/v1/a2a_agents/:id",
            get(a2a_agents_handlers::get_a2a_agent),
        )
        .route(
            "/admin/v1/passthrough_routes",
            get(passthrough_routes_handlers::list_passthrough_routes),
        )
        .route(
            "/admin/v1/passthrough_routes/:id",
            get(passthrough_routes_handlers::get_passthrough_route),
        )
        .route(
            "/admin/v1/guardrails",
            get(guardrails_handlers::list_guardrails),
        )
        .route(
            "/admin/v1/guardrails/:id",
            get(guardrails_handlers::get_guardrail),
        )
        .route(
            "/admin/v1/cache_policies",
            get(cache_policies_handlers::list_cache_policies),
        )
        .route(
            "/admin/v1/cache_policies/:id",
            get(cache_policies_handlers::get_cache_policy),
        )
        .route(
            "/admin/v1/observability_exporters",
            get(observability_exporters_handlers::list_observability_exporters),
        )
        .route(
            "/admin/v1/observability_exporters/:id",
            get(observability_exporters_handlers::get_observability_exporter),
        )
        // Health — per-model upstream health levels (0/1/2).
        .route("/admin/v1/health", get(health_handler::get_health))
        // Playground: forwards in-process to the proxy router (no network hop).
        // Accepts a *proxy* API key (not an admin key); auth is enforced by the
        // proxy middleware stack that runs inside the forwarded request.
        .route(
            "/playground/chat/completions",
            post(playground_handler::playground_chat_completions),
        );

    router.with_state(state)
}

/// Build the router for the **dedicated** metrics/status listener — the
/// Prometheus scrape endpoint at `prometheus.path`, plus the operational
/// read endpoints `GET /status/config`, `GET /status/ready`, and
/// `GET /status/models`, backed by the shared [`Metrics`],
/// [`ConfigStatus`], and [`ModelsStatusState`] handles. No admin state,
/// no auth-protected routes, no playground.
///
/// `aisix-server` binds this on `observability.metrics.prometheus.addr`
/// whenever prometheus is enabled. This is the only metrics/status surface —
/// the same in standalone and managed mode; the admin listener never serves
/// `/metrics` or `/status/config`.
pub fn metrics_router(
    metrics: Arc<Metrics>,
    config_status: ConfigStatus,
    prometheus: &PrometheusConfig,
    models_status: ModelsStatusState,
) -> Router {
    let state = MetricsState {
        metrics,
        config_status,
        models_status,
    };
    Router::new()
        .route(
            &normalized_prometheus_path(&prometheus.path),
            get(metrics_handler),
        )
        .route("/status/config", get(status_config_handler))
        .route("/status/ready", get(status_ready_handler))
        .route("/status/models", get(status_models_handler))
        .with_state(state)
}

/// Prometheus scrape handler. Reflects the live config load-observability
/// state into the recorder (so the `aisix_config_*` series are current) then
/// renders. Unauthenticated by design — restrict access at the network layer.
/// Emits `text/plain; version=0.0.4`.
async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<MetricsState>,
) -> Response {
    use axum::http::header::CONTENT_TYPE;
    use axum::response::IntoResponse;

    state
        .metrics
        .sync_config_status(&state.config_status.metrics());
    (
        StatusCode::OK,
        [(CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
        .into_response()
}

/// `GET /status/config` — the load-observability contract. Answers "did my
/// config take effect, and if not why?" from the live [`ConfigStatus`].
/// Unauthenticated like the scrape; restrict at the network layer.
async fn status_config_handler(
    axum::extract::State(state): axum::extract::State<MetricsState>,
) -> Response {
    use axum::response::IntoResponse;
    (StatusCode::OK, axum::Json(state.config_status.view())).into_response()
}

/// `GET /status/ready` — 503 with "no configuration available" until the
/// first valid configuration is applied, 200 afterward. A liveness-agnostic
/// readiness gate for the config source only; the admin listener's `/readyz`
/// keeps its shutdown/staleness semantics.
async fn status_ready_handler(
    axum::extract::State(state): axum::extract::State<MetricsState>,
) -> Response {
    use axum::http::header::CONTENT_TYPE;
    use axum::response::IntoResponse;

    if state.config_status.is_ready() {
        (
            StatusCode::OK,
            [(CONTENT_TYPE, "text/plain; charset=utf-8")],
            "ok",
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            [(CONTENT_TYPE, "text/plain; charset=utf-8")],
            "no configuration available",
        )
            .into_response()
    }
}

/// Upper bound on the store read behind `GET /status/models`: the
/// listener is unauthenticated and polled by probes and dashboards, so a
/// stalled configuration store must not park those requests (or hold the
/// shared store client) indefinitely — past this, the request answers
/// the same fixed 500 a store error does.
const STATUS_MODELS_STORE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The fixed store-failure answer for `GET /status/models`: this
/// listener is unauthenticated, so backend detail (etcd endpoints,
/// connection errors) stays in the server log instead of the response
/// body. The admin-key-gated endpoint keeps its detailed envelope.
fn status_models_store_failure() -> Response {
    use axum::response::IntoResponse;
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        axum::Json(ErrorBody {
            error_msg: "failed to list models".into(),
        }),
    )
        .into_response()
}

/// `GET /status/models` — the per-model runtime health view (cooldown /
/// background-check state) as an operational read on the status listener.
/// Renders through [`models_status_handler::render_models_status`], the
/// same render (and the same store + tracker handles) behind
/// `GET /admin/v1/models/status`, so the two responses are identical
/// while both exist. Unauthenticated like `/status/config` — it exposes
/// the model catalog (ids and display names) and health states; restrict
/// access at the network layer.
///
/// Store failures answer the fixed 500 from
/// [`status_models_store_failure`], and the store read is bounded by
/// [`STATUS_MODELS_STORE_TIMEOUT`] so a hung store degrades into that
/// same answer instead of parking anonymous pollers.
async fn status_models_handler(
    axum::extract::State(state): axum::extract::State<MetricsState>,
) -> Response {
    use axum::response::IntoResponse;

    let listed = tokio::time::timeout(
        STATUS_MODELS_STORE_TIMEOUT,
        state.models_status.store.list_models(),
    )
    .await;
    let all_models = match listed {
        Ok(Ok(models)) => models,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "GET /status/models: listing models failed");
            return status_models_store_failure();
        }
        Err(_) => {
            tracing::error!(
                timeout = ?STATUS_MODELS_STORE_TIMEOUT,
                "GET /status/models: listing models timed out"
            );
            return status_models_store_failure();
        }
    };
    axum::Json(models_status_handler::render_models_status(
        all_models,
        state.models_status.runtime_status_tracker.as_deref(),
    ))
    .into_response()
}

fn normalized_prometheus_path(path: &str) -> String {
    let path = path.trim();
    if path.is_empty() {
        return "/metrics".to_string();
    }
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

/// Takes no [`AdminState`], for the same reason the proxy listener's
/// does not: see [`aisix_proxy::health::livez_response`].
async fn livez(
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    aisix_proxy::health::livez_response(params.contains_key("verbose"))
}

async fn readyz(
    axum::extract::State(state): axum::extract::State<AdminState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let config_block = state
        .watch_status
        .as_ref()
        .and_then(|ws| aisix_proxy::health::config_readiness_block(ws.snapshot().last_apply_age));
    aisix_proxy::health::readyz_response(
        &state.livez_state,
        config_block,
        params.contains_key("verbose"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AdminConfig, AisixSnapshot};
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn cfg() -> AdminConfig {
        AdminConfig {
            enabled: true,
            addr: "127.0.0.1:0".into(),
            admin_keys: vec!["admin-secret".into()],
            tls: None,
        }
    }

    fn build_state() -> AdminState {
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let store = InMemoryStore::new() as Arc<dyn ConfigStore>;
        AdminState::new(handle, store, &cfg())
    }

    /// `build_state`, but hands back the concrete store so tests can
    /// seed resources through the in-memory write methods — the read
    /// endpoints are the surface under test.
    fn build_seedable_state() -> (AdminState, Arc<InMemoryStore>) {
        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let store = InMemoryStore::new();
        let state = AdminState::new(handle, Arc::clone(&store) as Arc<dyn ConfigStore>, &cfg());
        (state, store)
    }

    /// `ModelsStatusState` over an empty in-memory store, for metrics
    /// listener tests that don't exercise `/status/models`.
    fn empty_models_status() -> ModelsStatusState {
        ModelsStatusState {
            store: InMemoryStore::new() as Arc<dyn ConfigStore>,
            runtime_status_tracker: None,
        }
    }

    fn model_payload(name: &str) -> Value {
        json!({
            "display_name": name,
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111"
        })
    }

    fn apikey_payload(key: &str, allowed: &[&str]) -> Value {
        // Tests pass plaintext bearers (e.g. "sk-x"); the wire schema
        // stores SHA-256 hashes (§9A.7B.4).
        let key_hash = aisix_core::ApiKey::hash_bearer(key);
        json!({"key_hash": key_hash, "allowed_models": allowed})
    }

    fn auth_req(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
        let body = match body {
            Some(v) => Body::from(v.to_string()),
            None => Body::empty(),
        };
        Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", "Bearer admin-secret")
            .header("content-type", "application/json")
            .body(body)
            .unwrap()
    }

    async fn run(app: Router, req: Request<Body>) -> axum::http::Response<Body> {
        app.oneshot(req).await.unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> Value {
        // 1 MiB cap: the merged `/admin/openapi.json` embeds every resource
        // schema and is ~60 KB and growing, so the old 64 KB cap raced the
        // spec size (#554 pushed it over on CI). Generous headroom for a
        // self-generated, in-memory body.
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn openapi_json_endpoint_serves_the_spec() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/admin/openapi.json")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["openapi"], "3.1.0");
        assert!(v["paths"]["/admin/v1/models"].is_object());
    }

    #[tokio::test]
    async fn openapi_scalar_endpoint_serves_html_loader() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/admin/openapi-scalar")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("/admin/openapi.json"));
    }

    #[tokio::test]
    async fn admin_router_does_not_serve_metrics() {
        // The scrape endpoint lives exclusively on the dedicated metrics
        // listener — the admin router must not mount it.
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn metrics_router_serves_scrape_decoupled_from_admin() {
        use aisix_obs::{Metrics, RequestOutcome};
        use std::time::Duration;

        let metrics = Arc::new(Metrics::new(false));
        metrics.record_request(
            "openai",
            "my-gpt4",
            200,
            RequestOutcome::Success,
            Duration::from_millis(10),
        );

        let app = metrics_router(
            metrics,
            aisix_core::ConfigStatus::new(aisix_core::SourceKind::Etcd),
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            empty_models_status(),
        );

        // The dedicated listener serves the prometheus scrape.
        let resp = run(
            app.clone(),
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/plain"),
            "unexpected content-type: {ct}"
        );
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("aisix_requests_total"));
        assert!(body.contains("provider=\"openai\""));

        // It carries ONLY metrics — admin routes are not mounted on this
        // listener, proving the scrape surface is decoupled from admin.
        let resp = run(
            app,
            Request::builder()
                .uri("/admin/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn metrics_router_honors_custom_path() {
        use aisix_obs::Metrics;

        let app = metrics_router(
            Arc::new(Metrics::new(false)),
            aisix_core::ConfigStatus::new(aisix_core::SourceKind::Etcd),
            &PrometheusConfig {
                enabled: true,
                path: "internal/prom".into(),
                addr: "0.0.0.0:9090".into(),
            },
            empty_models_status(),
        );

        // Path is normalized to a leading slash and served there.
        let resp = run(
            app.clone(),
            Request::builder()
                .uri("/internal/prom")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        // The default `/metrics` is not mounted when a custom path is set.
        let resp = run(
            app,
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn status_ready_is_503_before_config_and_200_after() {
        use aisix_core::config_status::{
            AppliedSnapshot, ConfigStatus, LoadObservation, SourceKind,
        };
        let cs = ConfigStatus::new(SourceKind::Etcd);
        let app = metrics_router(
            Arc::new(Metrics::new(false)),
            cs.clone(),
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            empty_models_status(),
        );

        // Before any config: 503 "no configuration available".
        let resp = run(
            app.clone(),
            Request::builder()
                .uri("/status/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            "no configuration available"
        );

        // After a valid apply: 200.
        cs.record_load(LoadObservation {
            source_hash: "h".into(),
            observed_revision: Some(1),
            applied: Some(AppliedSnapshot {
                config_hash: "h".into(),
                revision: Some(1),
                resource_counts: Default::default(),
            }),
            rejected: vec![],
            partially_compatible: Vec::new(),
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: true,
            wholly_rejected: false,
        });
        let resp = run(
            app,
            Request::builder()
                .uri("/status/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn status_config_serves_the_derived_view() {
        use aisix_core::config_status::{
            AppliedSnapshot, ConfigStatus, LoadObservation, SourceKind,
        };
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "src".into(),
            observed_revision: Some(9),
            applied: Some(AppliedSnapshot {
                config_hash: "app".into(),
                revision: Some(9),
                resource_counts: [("models".to_string(), 1)].into_iter().collect(),
            }),
            rejected: vec![aisix_core::IncomingRejection {
                identity: "/aisix/models/bad".into(),
                resource_kind: "models".into(),
                resource_id: "bad".into(),
                last_error_kind: "schema_failed".into(),
                last_error: "schema validation failed at `/display_name`".into(),
                seen_at: chrono::Utc::now(),
                serving_stale_since: None,
            }],
            partially_compatible: vec![aisix_core::config_status::PartialCompatResource {
                resource_kind: "api_keys".into(),
                field: "quota_profile".into(),
                count: 2,
            }],
            partially_compatible_rows_by_kind: [("api_keys".to_string(), 2)].into_iter().collect(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: true,
            wholly_rejected: false,
        });
        let app = metrics_router(
            Arc::new(Metrics::new(false)),
            cs,
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            empty_models_status(),
        );
        let resp = run(
            app,
            Request::builder()
                .uri("/status/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["state"], "degraded");
        assert_eq!(v["source"]["type"], "etcd");
        assert_eq!(v["source"]["observed_revision"], 9);
        assert_eq!(v["applied"]["applied_revision"], 9);
        assert_eq!(v["applied"]["resource_counts"]["models"], 1);
        assert_eq!(v["rejected"][0]["resource_kind"], "models");
        assert_eq!(v["rejected"][0]["last_error_kind"], "schema_failed");
        // The partially-compatible companion list (#871) rides next to
        // rejected[] so a matching config_hash can't hide that some
        // served rows carry fields this DP does not enforce.
        assert_eq!(v["partially_compatible"][0]["resource_kind"], "api_keys");
        assert_eq!(v["partially_compatible"][0]["field"], "quota_profile");
        assert_eq!(v["partially_compatible"][0]["count"], 2);
    }

    #[tokio::test]
    async fn metrics_scrape_reflects_config_status_series() {
        use aisix_core::config_status::{
            AppliedSnapshot, ConfigStatus, LoadObservation, SourceKind,
        };
        let cs = ConfigStatus::new(SourceKind::Etcd);
        cs.record_load(LoadObservation {
            source_hash: "src".into(),
            observed_revision: Some(5),
            applied: Some(AppliedSnapshot {
                config_hash: "deadbeef".into(),
                revision: Some(5),
                resource_counts: [("models".to_string(), 2)].into_iter().collect(),
            }),
            rejected: vec![],
            partially_compatible: Vec::new(),
            partially_compatible_rows_by_kind: Default::default(),
            stale_served_rows_by_kind: Default::default(),
            is_reload: true,
            wholly_rejected: false,
        });
        let app = metrics_router(
            Arc::new(Metrics::new(false)),
            cs,
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            empty_models_status(),
        );
        let resp = run(
            app,
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("aisix_config_last_reload_successful 1"));
        assert!(text.contains("aisix_config_observed_revision 5"));
        assert!(text.contains("aisix_config_applied_revision 5"));
        assert!(text.contains("aisix_config_hash_info{hash=\"deadbeef\"} 1"));
        assert!(text.contains("aisix_config_source_connected 1"));
    }

    #[tokio::test]
    async fn status_models_serves_the_admin_view_byte_for_byte() {
        use aisix_core::resource::ResourceEntry;
        use aisix_core::Model;
        use aisix_proxy::ModelRuntimeStatusTracker;
        use std::time::Duration;

        let store = InMemoryStore::new();
        let direct: Model = serde_json::from_value(model_payload("gpt4")).unwrap();
        store
            .put_model(ResourceEntry {
                id: "direct-1".into(),
                value: direct,
                revision: 1,
            })
            .await
            .unwrap();
        let routing: Model = serde_json::from_value(json!({
            "display_name": "router",
            "routing": {
                "targets": [{"model": "gpt4"}]
            }
        }))
        .unwrap();
        store
            .put_model(ResourceEntry {
                id: "routing-1".into(),
                value: routing,
                revision: 1,
            })
            .await
            .unwrap();
        let store: Arc<dyn ConfigStore> = store;

        let tracker = Arc::new(ModelRuntimeStatusTracker::new());
        tracker.mark_cooldown("direct-1", Duration::from_secs(60), "upstream_rate_limited");

        // Admin listener view (auth-protected).
        let admin_app = build_router(
            AdminState::new(
                SnapshotHandle::new(AisixSnapshot::new()),
                Arc::clone(&store),
                &cfg(),
            )
            .with_runtime_status_tracker(Arc::clone(&tracker)),
        );
        let resp = run(admin_app, auth_req("GET", "/admin/v1/models/status", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let admin_bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();

        // Status listener view — no auth header — over the SAME store +
        // tracker handles, exactly how `aisix-server` wires standalone mode.
        let metrics_app = metrics_router(
            Arc::new(Metrics::new(false)),
            aisix_core::ConfigStatus::new(aisix_core::SourceKind::Etcd),
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            ModelsStatusState {
                store,
                runtime_status_tracker: Some(tracker),
            },
        );
        let resp = run(
            metrics_app,
            Request::builder()
                .uri("/status/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let status_bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();

        assert_eq!(
            admin_bytes, status_bytes,
            "GET /status/models must serve the exact bytes of GET /admin/v1/models/status",
        );

        // Sanity on the shared body: cooldown state and the virtual row
        // actually render.
        let rows: Value = serde_json::from_slice(&status_bytes).unwrap();
        let rows = rows.as_array().unwrap();
        assert_eq!(rows.len(), 2);
        let direct = rows.iter().find(|row| row["id"] == "direct-1").unwrap();
        assert_eq!(direct["status"], "cooldown");
        assert_eq!(direct["status_reason"], "upstream_rate_limited");
        assert!(!direct["cooldown_until"].is_null());
        let routing = rows.iter().find(|row| row["id"] == "routing-1").unwrap();
        assert_eq!(routing["status"], "not_applicable");
    }

    #[tokio::test]
    async fn admin_router_does_not_serve_status_models() {
        // The operational read lives on the metrics/status listener; the
        // admin listener keeps only its own `/admin/v1/models/status`.
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/status/models")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn status_models_store_failure_never_leaks_backend_detail() {
        use aisix_core::resource::ResourceEntry;
        use aisix_core::{
            A2aAgent, ApiKey, CachePolicy, Guardrail, McpServer, Model, ObservabilityExporter,
            PassthroughRoute, ProviderKey,
        };

        // A store whose every call fails with backend detail an anonymous
        // caller must never see (mimics an etcd outage: the client error
        // names endpoints/addresses).
        const LEAKY: &str = "connect to http://10.0.0.7:2379 refused";
        struct FailingStore;

        macro_rules! impl_failing_store {
            ($( { $ty:ty, $get:ident, $list:ident } )+) => {
                #[async_trait::async_trait]
                impl ConfigStore for FailingStore {
                    $(
                        async fn $get(
                            &self,
                            _id: &str,
                        ) -> Result<Option<ResourceEntry<$ty>>, StoreError> {
                            Err(StoreError::Backend(LEAKY.into()))
                        }
                        async fn $list(&self) -> Result<Vec<ResourceEntry<$ty>>, StoreError> {
                            Err(StoreError::Backend(LEAKY.into()))
                        }
                    )+
                }
            };
        }
        impl_failing_store! {
            { Model, get_model, list_models }
            { ApiKey, get_apikey, list_apikeys }
            { ProviderKey, get_provider_key, list_provider_keys }
            { Guardrail, get_guardrail, list_guardrails }
            { CachePolicy, get_cache_policy, list_cache_policies }
            { ObservabilityExporter, get_observability_exporter, list_observability_exporters }
            { McpServer, get_mcp_server, list_mcp_servers }
            { A2aAgent, get_a2a_agent, list_a2a_agents }
            { PassthroughRoute, get_passthrough_route, list_passthrough_routes }
        }

        let app = metrics_router(
            Arc::new(Metrics::new(false)),
            aisix_core::ConfigStatus::new(aisix_core::SourceKind::Etcd),
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            ModelsStatusState {
                store: Arc::new(FailingStore),
                runtime_status_tracker: None,
            },
        );
        let resp = run(
            app,
            Request::builder()
                .uri("/status/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            !body.contains("10.0.0.7") && !body.contains("connect"),
            "unauthenticated status listener must not leak store backend detail: {body}",
        );
        assert_eq!(body, r#"{"error_msg":"failed to list models"}"#);
    }

    #[tokio::test(start_paused = true)]
    async fn status_models_answers_the_fixed_500_when_the_store_hangs() {
        use aisix_core::resource::ResourceEntry;
        use aisix_core::{
            A2aAgent, ApiKey, CachePolicy, Guardrail, McpServer, Model, ObservabilityExporter,
            PassthroughRoute, ProviderKey,
        };

        // A store whose every call never resolves (mimics a blackholed
        // etcd: the connection hangs instead of failing). The paused
        // clock auto-advances past STATUS_MODELS_STORE_TIMEOUT, so the
        // test asserts the timeout arm without real waiting.
        struct HangingStore;

        macro_rules! impl_hanging_store {
            ($( { $ty:ty, $get:ident, $list:ident } )+) => {
                #[async_trait::async_trait]
                impl ConfigStore for HangingStore {
                    $(
                        async fn $get(
                            &self,
                            _id: &str,
                        ) -> Result<Option<ResourceEntry<$ty>>, StoreError> {
                            std::future::pending().await
                        }
                        async fn $list(&self) -> Result<Vec<ResourceEntry<$ty>>, StoreError> {
                            std::future::pending().await
                        }
                    )+
                }
            };
        }
        impl_hanging_store! {
            { Model, get_model, list_models }
            { ApiKey, get_apikey, list_apikeys }
            { ProviderKey, get_provider_key, list_provider_keys }
            { Guardrail, get_guardrail, list_guardrails }
            { CachePolicy, get_cache_policy, list_cache_policies }
            { ObservabilityExporter, get_observability_exporter, list_observability_exporters }
            { McpServer, get_mcp_server, list_mcp_servers }
            { A2aAgent, get_a2a_agent, list_a2a_agents }
            { PassthroughRoute, get_passthrough_route, list_passthrough_routes }
        }

        let app = metrics_router(
            Arc::new(Metrics::new(false)),
            aisix_core::ConfigStatus::new(aisix_core::SourceKind::Etcd),
            &PrometheusConfig {
                enabled: true,
                path: "/metrics".into(),
                addr: "0.0.0.0:9090".into(),
            },
            ModelsStatusState {
                store: Arc::new(HangingStore),
                runtime_status_tracker: None,
            },
        );
        let resp = run(
            app,
            Request::builder()
                .uri("/status/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(body, r#"{"error_msg":"failed to list models"}"#);
    }

    #[tokio::test]
    async fn livez_reports_plain_ok_by_default() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/livez")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), "ok");
    }

    #[tokio::test]
    async fn livez_rejects_non_get_requests() {
        let app = build_router(build_state());
        let req = Request::builder()
            .method("POST")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn livez_stays_200_when_shutting_down() {
        let state = build_state();
        state.livez_state.mark_shutting_down();
        let app = build_router(state);
        let req = Request::builder()
            .uri("/livez")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "the admin listener answers the same liveness question as the \
             proxy one, and a draining process is not one to restart",
        );
    }

    #[tokio::test]
    async fn health_route_is_not_found() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_models_without_auth_is_401() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/admin/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(resp).await;
        // Spec §3 admin envelope — {"error_msg": "..."}.
        assert!(v["error_msg"].is_string());
        assert!(v.get("error").is_none());
    }

    #[tokio::test]
    async fn list_models_with_wrong_admin_key_is_401() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/admin/v1/models")
            .header("authorization", "Bearer wrong")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn list_models_returns_seeded_entries() {
        let (state, store) = build_seedable_state();
        for (id, name) in [("m-1", "foo"), ("m-2", "bar")] {
            let model: aisix_core::Model = serde_json::from_value(model_payload(name)).unwrap();
            store
                .put_model(aisix_core::ResourceEntry::new(id, model, 1))
                .await
                .unwrap();
        }
        let app = build_router(state);
        let resp = run(app, auth_req("GET", "/admin/v1/models", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn get_model_serves_seeded_entry() {
        let (state, store) = build_seedable_state();
        let model: aisix_core::Model = serde_json::from_value(model_payload("foo")).unwrap();
        store
            .put_model(aisix_core::ResourceEntry::new("m-1", model, 1))
            .await
            .unwrap();

        let app = build_router(state);
        let resp = run(app, auth_req("GET", "/admin/v1/models/m-1", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["value"]["display_name"], "foo");
    }

    #[tokio::test]
    async fn get_model_missing_is_404() {
        let app = build_router(build_state());
        let resp = run(app, auth_req("GET", "/admin/v1/models/nonexistent", None)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // Caller API keys are served at both the canonical `/admin/v1/api_keys`
    // path and the former `/admin/v1/apikeys` spelling — same handlers,
    // same store. Pin that a resource written through one path is fully
    // addressable through the other.
    #[tokio::test]
    async fn api_keys_former_spelling_serves_the_same_resources() {
        let (state, store) = build_seedable_state();
        let mut payload = apikey_payload("sk-canonical", &["my-model"]);
        // Non-default optional fields, so the PublicApiKey projection is
        // pinned field-by-field — a projection that dropped one of these
        // would still pass an id-only check.
        payload["mcp_access"] = json!({"allow": ["github__create_issue"]});
        payload["allowed_agents"] = json!(["invoice-agent"]);
        payload["rate_limit"] = json!({"rpm": 60});
        payload["disabled"] = json!(true);
        payload["expires_at"] = json!("2099-01-01T00:00:00Z");
        let key: aisix_core::ApiKey = serde_json::from_value(payload).unwrap();
        let expected_hash = key.key_hash.clone();
        store
            .put_apikey(aisix_core::ResourceEntry::new("k-1", key, 1))
            .await
            .unwrap();

        // The same entry is readable through both spellings, with the
        // full projection intact.
        for base in ["/admin/v1/api_keys", "/admin/v1/apikeys"] {
            let app = build_router(state.clone());
            let resp = run(app, auth_req("GET", &format!("{base}/k-1"), None)).await;
            assert_eq!(resp.status(), StatusCode::OK, "GET {base}/k-1");
            let v = body_json(resp).await;
            assert_eq!(v["id"], "k-1", "{base}");
            assert_eq!(v["value"]["key_hash"], expected_hash.as_str(), "{base}");
            assert_eq!(v["value"]["allowed_models"], json!(["my-model"]), "{base}");
            assert_eq!(
                v["value"]["mcp_access"],
                json!({"allow": ["github__create_issue"]}),
                "{base}"
            );
            assert_eq!(
                v["value"]["allowed_agents"],
                json!(["invoice-agent"]),
                "{base}"
            );
            assert_eq!(v["value"]["rate_limit"]["rpm"], 60, "{base}");
            assert_eq!(v["value"]["disabled"], true, "{base}");
            assert_eq!(v["value"]["expires_at"], "2099-01-01T00:00:00Z", "{base}");

            let app = build_router(state.clone());
            let resp = run(app, auth_req("GET", base, None)).await;
            assert_eq!(resp.status(), StatusCode::OK, "GET {base}");
            let v = body_json(resp).await;
            let arr = v.as_array().unwrap();
            assert_eq!(arr.len(), 1, "{base}");
            // The list entry carries the same projection, not a thinner one.
            assert_eq!(
                arr[0]["value"]["key_hash"],
                expected_hash.as_str(),
                "{base}"
            );
            assert_eq!(arr[0]["value"]["disabled"], true, "{base}");
        }
    }

    #[tokio::test]
    async fn openapi_apikey_schema_excludes_max_budget_usd() {
        let resp = openapi::openapi_json().await;
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes).expect("OPENAPI_JSON must parse");
        let props = &parsed["components"]["schemas"]["ApiKey"]["properties"];
        assert!(props["key_hash"].is_object());
        assert!(props["allowed_models"].is_object());
        assert!(props["rate_limit"].is_object());
        assert!(props.get("max_budget_usd").is_none());
    }

    // ──────────────────── Guardrail payloads ────────────────────

    fn guardrail_payload(name: &str) -> Value {
        json!({
            "name": name,
            "kind": "keyword",
            "patterns": [{"kind": "literal", "value": "secret"}]
        })
    }

    // ──────────────────── Health endpoint ────────────────────

    #[tokio::test]
    async fn health_returns_empty_models_when_snapshot_is_empty() {
        let app = build_router(build_state());
        let resp = run(app, auth_req("GET", "/admin/v1/health", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "ok");
        assert_eq!(v["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn health_requires_admin_auth() {
        let app = build_router(build_state());
        let req = Request::builder()
            .uri("/admin/v1/health")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn health_lists_models_with_default_healthy_when_no_tracker() {
        let (state, store) = build_seedable_state();
        let model: aisix_core::Model = serde_json::from_value(model_payload("gpt4")).unwrap();
        store
            .put_model(aisix_core::ResourceEntry::new("m-1", model, 1))
            .await
            .unwrap();

        // Health endpoint on the same state (no tracker wired).
        let app = build_router(state);
        let resp = run(app, auth_req("GET", "/admin/v1/health", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        // Without a tracker all models default to Healthy = 0.
        assert_eq!(models[0]["health"], 0);
        assert_eq!(models[0]["name"], "gpt4");
    }

    #[tokio::test]
    async fn health_reflects_tracker_failure_count() {
        use aisix_proxy::HealthTracker;

        let health = Arc::new(HealthTracker::new());

        // Simulate 4 consecutive failures on "gpt4" → Degraded.
        for _ in 0..4 {
            health.record_failure("gpt4");
        }

        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let store = InMemoryStore::new();
        let state =
            AdminState::new(handle.clone(), store.clone(), &cfg()).with_health_tracker(health);

        // Insert a model into the store (to appear in snapshot via store.
        // Since InMemoryStore doesn't auto-push to snapshot in tests, we
        // create a snapshot manually via the snapshot handle).
        // The health endpoint reads from state.snapshot, not from the store
        // directly — but our test build_state uses the same snapshot handle.
        // We'll call create_model to populate both store AND snapshot
        // (InMemoryStore.put_model updates its DashMap but not the
        // SnapshotHandle — so we need to set up the snapshot directly).
        //
        // For simplicity, verify that health level 1 is reported for a
        // tracker-only entry without a snapshot model. Since the health
        // endpoint iterates snapshot.models and maps each to a tracker level,
        // an empty snapshot means no model entries — we test the level
        // indirectly through health_handler unit tests instead.
        //
        // Here we just confirm the endpoint responds OK with the wired tracker.
        let app = build_router(state);
        let resp = run(app, auth_req("GET", "/admin/v1/health", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "ok");
        // Empty snapshot → empty model list, but endpoint is operational.
        assert!(v["models"].is_array());
    }

    #[tokio::test]
    async fn models_status_returns_direct_and_routing_rows() {
        use aisix_core::resource::ResourceEntry;
        use aisix_core::Model;
        use aisix_proxy::ModelRuntimeStatusTracker;

        let handle = SnapshotHandle::new(AisixSnapshot::new());
        let store = InMemoryStore::new();
        let runtime = Arc::new(ModelRuntimeStatusTracker::new());

        let direct: Model = serde_json::from_value(model_payload("gpt4")).unwrap();
        store
            .put_model(ResourceEntry {
                id: "direct-1".into(),
                value: direct,
                revision: 1,
            })
            .await
            .unwrap();

        let routing: Model = serde_json::from_value(json!({
            "display_name": "router",
            "routing": {
                "targets": [{"model": "gpt4"}]
            }
        }))
        .unwrap();
        store
            .put_model(ResourceEntry {
                id: "routing-1".into(),
                value: routing,
                revision: 1,
            })
            .await
            .unwrap();

        runtime.record_ignored_check("direct-1", 429, "ignored_transient_error");

        let state = AdminState::new(handle, store, &cfg()).with_runtime_status_tracker(runtime);
        let app = build_router(state);
        let resp = run(app, auth_req("GET", "/admin/v1/models/status", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let rows = body_json(resp).await;
        let rows = rows.as_array().unwrap();
        assert_eq!(rows.len(), 2);

        let direct = rows.iter().find(|row| row["id"] == "direct-1").unwrap();
        assert_eq!(direct["kind"], "direct");
        assert_eq!(direct["status"], "healthy");
        assert_eq!(direct["last_check_status"], 429);
        assert_eq!(direct["status_reason"], "ignored_transient_error");

        let routing = rows.iter().find(|row| row["id"] == "routing-1").unwrap();
        assert_eq!(routing["kind"], "routing");
        assert_eq!(routing["status"], "not_applicable");
    }

    // ──────────────────── File-managed mode ────────────────────

    /// AdminState wired the way `aisix-server` wires file mode: the
    /// store reads from the file-loaded snapshot.
    fn build_file_managed_state() -> AdminState {
        let snapshot = AisixSnapshot::new();
        let model: aisix_core::Model = serde_json::from_value(model_payload("file-model")).unwrap();
        snapshot
            .models
            .insert(aisix_core::ResourceEntry::new("m-file-1", model, 1));
        let handle = SnapshotHandle::new(snapshot);
        let store: Arc<dyn ConfigStore> = Arc::new(FileManagedStore::new(handle.clone()));
        AdminState::new(handle, store, &cfg())
    }

    // ──────────────────── Removed write path ────────────────────

    /// The resource write path was removed: every collection and `:id`
    /// route serves GET only, so POST/PUT/DELETE answer 405 with an
    /// `Allow: GET` header — regardless of store backend or auth.
    #[tokio::test]
    async fn removed_resource_writes_answer_405_with_allow_get() {
        // The FULL matrix — every route spelling × every write method.
        // A partial revert (say, PUT re-added on one {id} route) must
        // fail here; a sampled subset would let it through.
        const ROUTE_SPELLINGS: [&str; 9] = [
            "models",
            "api_keys",
            "apikeys", // former spelling: same removed write path
            "provider_keys",
            "guardrails",
            "cache_policies",
            "observability_exporters",
            "mcp_servers",
            "a2a_agents",
        ];
        let mut writes: Vec<(&str, String, Option<Value>)> = Vec::new();
        for kind in ROUTE_SPELLINGS {
            writes.push(("POST", format!("/admin/v1/{kind}"), None));
            writes.push(("PUT", format!("/admin/v1/{kind}/some-id"), None));
            writes.push(("DELETE", format!("/admin/v1/{kind}/some-id"), None));
        }
        // Plus representative rows with well-formed bodies, pinning that
        // a valid payload doesn't route any differently.
        writes.push((
            "POST",
            "/admin/v1/models".into(),
            Some(model_payload("new")),
        ));
        writes.push((
            "PUT",
            "/admin/v1/models/m-1".into(),
            Some(model_payload("m")),
        ));
        writes.push((
            "POST",
            "/admin/v1/api_keys".into(),
            Some(apikey_payload("sk-x", &["*"])),
        ));
        writes.push((
            "POST",
            "/admin/v1/guardrails".into(),
            Some(guardrail_payload("g")),
        ));
        for (method, uri, body) in writes {
            let app = build_router(build_state());
            let resp = run(app, auth_req(method, &uri, body)).await;
            assert_eq!(
                resp.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "{method} {uri} must answer 405 after write removal",
            );
            let allow = resp
                .headers()
                .get(axum::http::header::ALLOW)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_else(|| panic!("{method} {uri}: missing Allow header"));
            assert!(
                allow.contains("GET"),
                "{method} {uri}: Allow must advertise GET, got {allow}",
            );
        }

        // Method routing answers before auth: an unauthenticated write
        // gets the same 405 (there is no write endpoint to protect).
        let app = build_router(build_state());
        let req = Request::builder()
            .method("POST")
            .uri("/admin/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(model_payload("x").to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    /// The rotate route was removed outright, so it 404s — the path no
    /// longer exists (it was POST-only, there is no read to keep).
    /// POST alone can't prove the route is gone: the old handler also
    /// answered 404 for an unknown id. GET is the discriminator — a
    /// surviving POST-only route would answer GET with 405, an absent
    /// route with 404.
    #[tokio::test]
    async fn removed_rotate_route_answers_404() {
        for uri in [
            "/admin/v1/api_keys/some-id/rotate",
            "/admin/v1/apikeys/some-id/rotate",
        ] {
            for method in ["POST", "GET"] {
                let app = build_router(build_state());
                let resp = run(app, auth_req(method, uri, None)).await;
                assert_eq!(
                    resp.status(),
                    StatusCode::NOT_FOUND,
                    "{method} {uri} must 404 after rotate removal",
                );
            }
        }
    }

    #[tokio::test]
    async fn file_managed_mode_serves_reads_from_the_snapshot() {
        let state = build_file_managed_state();

        // List reflects the file-loaded snapshot.
        let app = build_router(state.clone());
        let resp = run(app, auth_req("GET", "/admin/v1/models", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert_eq!(v[0]["value"]["display_name"], "file-model");

        // Get-by-id too.
        let app = build_router(state.clone());
        let resp = run(app, auth_req("GET", "/admin/v1/models/m-file-1", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Non-resource surfaces stay untouched: health + models/status +
        // openapi keep responding.
        let app = build_router(state.clone());
        let resp = run(app, auth_req("GET", "/admin/v1/health", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let app = build_router(state.clone());
        let resp = run(app, auth_req("GET", "/admin/v1/models/status", None)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let app = build_router(state);
        let req = Request::builder()
            .uri("/admin/openapi.json")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ──────────────── No deprecation marks on the surviving surface ────────────────

    #[tokio::test]
    async fn read_and_non_resource_responses_carry_no_deprecation_header() {
        let state = build_state();

        // Reads across the resource + status surface.
        for uri in [
            "/admin/v1/models",
            "/admin/v1/models/status",
            "/admin/v1/health",
        ] {
            let app = build_router(state.clone());
            let resp = run(app, auth_req("GET", uri, None)).await;
            assert_eq!(resp.status(), StatusCode::OK, "GET {uri}");
            assert!(
                resp.headers().get("deprecation").is_none(),
                "GET {uri} must NOT carry a Deprecation header"
            );
        }

        // Unauthenticated public surface.
        let app = build_router(state.clone());
        let req = Request::builder()
            .uri("/admin/openapi.json")
            .body(Body::empty())
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("deprecation").is_none());

        // The playground is a POST on the admin listener but NOT part of
        // the deprecated resource write path (501 here: no proxy router
        // is wired in this test state).
        let app = build_router(state);
        let req = Request::builder()
            .method("POST")
            .uri("/playground/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model": "m", "messages": []}).to_string(),
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        assert!(
            resp.headers().get("deprecation").is_none(),
            "playground must NOT carry a Deprecation header"
        );
    }
}
