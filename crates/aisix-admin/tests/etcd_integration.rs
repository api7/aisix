//! End-to-end declarative-write → etcd → admin-read/loader tests.
//!
//! Bracketed by `ADMIN_TEST_ETCD_URL` (mirrors the
//! `CACHE_TEST_REDIS_URL` pattern in `aisix-cache/tests/redis_integration.rs`):
//! tests no-op when unset so local `cargo test` without docker still
//! passes; CI sets the env var via the `etcd` service in
//! `.github/workflows/ci.yml`.
//!
//! Resources reach etcd through direct writes (the declarative path —
//! the same front door the control plane uses); the admin listener and
//! `aisix-etcd::loader` are the read sides. Why a real etcd instead of
//! `InMemoryStore`:
//!
//! 1. Verifies the byte shape operators write (entity-value JSON at
//!    `{prefix}/{kind}/{id}`) against the shape `EtcdConfigStore`
//!    reads — the subkey constants on the two sides have drifted
//!    before, and unit tests against the in-memory store wouldn't
//!    catch it.
//! 2. Catches the `EtcdConfigStore` read impls themselves (the
//!    in-memory store doesn't exercise serde + grpc + revision
//!    plumbing).
//! 3. Exercises the full etcd → ConfigStore → Admin handler read path,
//!    and separately etcd → `aisix-etcd::loader` — the two consumers
//!    of the same key layout.

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aisix_admin::{build_router, AdminState, ConfigStore, EtcdConfigStore};
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AdminConfig, AisixSnapshot};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

const ADMIN_KEY: &str = "admin-it-secret";

fn etcd_url() -> Option<String> {
    std::env::var("ADMIN_TEST_ETCD_URL").ok()
}

/// Per-test prefix so concurrent tests in this binary don't collide.
fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!(
        "/aisix-admin-it/{nanos:x}-{:?}",
        std::thread::current().id()
    )
    .replace(['(', ')', ' '], "")
}

async fn etcd_client_for(url: &str) -> etcd_client::Client {
    etcd_client::Client::connect([url], None)
        .await
        .expect("etcd connect")
}

async fn build_state(client: etcd_client::Client, prefix: &str) -> AdminState {
    let store: Arc<dyn ConfigStore> = Arc::new(EtcdConfigStore::new(client, prefix));
    let handle = SnapshotHandle::new(AisixSnapshot::new());
    let cfg = AdminConfig {
        enabled: true,
        addr: "127.0.0.1:0".into(),
        admin_keys: vec![ADMIN_KEY.into()],
        tls: None,
    };
    AdminState::new(handle, store, &cfg)
}

/// Direct declarative write: entity-value JSON at `{prefix}/{kind}/{id}`.
async fn seed(client: &mut etcd_client::Client, prefix: &str, kind: &str, id: &str, doc: &Value) {
    client
        .put(
            format!("{prefix}/{kind}/{id}"),
            serde_json::to_vec(doc).expect("serialize"),
            None,
        )
        .await
        .expect("direct etcd put");
}

fn auth_get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("authorization", format!("Bearer {ADMIN_KEY}"))
        .body(Body::empty())
        .unwrap()
}

fn auth_req(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
    let body = match body {
        Some(v) => Body::from(v.to_string()),
        None => Body::empty(),
    };
    Request::builder()
        .method(method)
        .uri(uri)
        .header("authorization", format!("Bearer {ADMIN_KEY}"))
        .header("content-type", "application/json")
        .body(body)
        .unwrap()
}

async fn body_json(resp: axum::http::Response<Body>) -> Value {
    let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Seed a document directly, read it back through the Admin HTTP layer
/// (list + get), then delete it directly and assert the list empties.
async fn direct_write_read_round_trip(
    url: &str,
    kind: &str,
    list_uri: &str,
    id: &str,
    payload: Value,
) {
    let prefix = unique_prefix();
    let mut client = etcd_client_for(url).await;
    let state = build_state(client.clone(), &prefix).await;

    seed(&mut client, &prefix, kind, id, &payload).await;

    // LIST serves the directly-written entry.
    let app = build_router(state.clone());
    let resp = app.oneshot(auth_get(list_uri)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET {list_uri}");
    let listed = body_json(resp).await;
    let arr = listed.as_array().expect("list array");
    assert_eq!(arr.len(), 1, "list for {list_uri}: {listed}");
    assert_eq!(arr[0]["id"], id);
    assert!(arr[0]["revision"].as_i64().unwrap_or(0) >= 1);

    // GET by id too.
    let app = build_router(state.clone());
    let item_uri = format!("{list_uri}/{id}");
    let resp = app.oneshot(auth_get(&item_uri)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET {item_uri}");

    // Direct delete → the read surface empties.
    client
        .delete(format!("{prefix}/{kind}/{id}"), None)
        .await
        .expect("direct etcd delete");
    let app = build_router(state);
    let resp = app.oneshot(auth_get(list_uri)).await.unwrap();
    let listed = body_json(resp).await;
    assert!(listed.as_array().unwrap().is_empty());
}

// ─────────────────────────── Per-resource round-trips ───────────────────────────

#[tokio::test]
async fn models_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    direct_write_read_round_trip(
        &url,
        "models",
        "/admin/v1/models",
        "m-it-1",
        json!({
            "display_name": "it-gpt4",
            "provider": "openai",
            "model_name": "gpt-4o",
            "provider_key_id": "11111111-1111-1111-1111-111111111111"
        }),
    )
    .await;
}

#[tokio::test]
async fn apikeys_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    let key_hash = aisix_core::ApiKey::hash_bearer("sk-it-bearer");
    // Canonical kind is `api_keys`; the former `apikeys` route spelling
    // reads the same store.
    direct_write_read_round_trip(
        &url,
        "api_keys",
        "/admin/v1/apikeys",
        "k-it-1",
        json!({
            "key_hash": key_hash,
            "allowed_models": ["it-gpt4"],
            "allowed_tools": ["github__create_issue", "*"]
        }),
    )
    .await;
}

#[tokio::test]
async fn provider_keys_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    direct_write_read_round_trip(
        &url,
        "provider_keys",
        "/admin/v1/provider_keys",
        "pk-it-1",
        json!({
            "display_name": "openai-it",
            "provider": "openai",
            "adapter": "openai",
            "secret": "sk-it"
        }),
    )
    .await;
}

#[tokio::test]
async fn mcp_servers_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    direct_write_read_round_trip(
        &url,
        "mcp_servers",
        "/admin/v1/mcp_servers",
        "mcp-it-1",
        json!({
            "name": "github-it",
            "url": "https://api.example.com/mcp",
            "auth_type": "bearer",
            "secret": "tok-it"
        }),
    )
    .await;
}

#[tokio::test]
async fn guardrails_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    direct_write_read_round_trip(
        &url,
        "guardrails",
        "/admin/v1/guardrails",
        "g-it-1",
        json!({
            "name": "it-block",
            "kind": "keyword",
            "patterns": [{"kind": "literal", "value": "secret"}]
        }),
    )
    .await;
}

#[tokio::test]
async fn cache_policies_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    direct_write_read_round_trip(
        &url,
        "cache_policies",
        "/admin/v1/cache_policies",
        "cp-it-1",
        json!({"name": "it-cache", "enabled": true, "ttl_seconds": 600}),
    )
    .await;
}

#[tokio::test]
async fn observability_exporters_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    direct_write_read_round_trip(
        &url,
        "observability_exporters",
        "/admin/v1/observability_exporters",
        "oe-it-1",
        json!({
            "name": "it-otel",
            "kind": "otlp_http",
            "endpoint": "https://otel.example.com/v1/traces"
        }),
    )
    .await;
}

// ─────────────────────────── Removed write path ───────────────────────────

/// The write contract holds against the real etcd-backed store too:
/// resource writes answer 405 (`Allow: GET`), rotate is 404, and
/// nothing lands in etcd.
#[tokio::test]
async fn admin_writes_are_refused_against_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    let prefix = unique_prefix();
    let mut client = etcd_client_for(&url).await;
    let state = build_state(client.clone(), &prefix).await;

    let app = build_router(state.clone());
    let resp = app
        .oneshot(auth_req(
            "POST",
            "/admin/v1/models",
            Some(json!({
                "display_name": "sneaky",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    let allow = resp
        .headers()
        .get(axum::http::header::ALLOW)
        .and_then(|v| v.to_str().ok())
        .expect("Allow header");
    assert!(allow.contains("GET"), "Allow: {allow}");

    let app = build_router(state);
    let resp = app
        .oneshot(auth_req("POST", "/admin/v1/api_keys/any-id/rotate", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // Nothing was written.
    let resp = client
        .get(
            prefix.as_bytes().to_vec(),
            Some(etcd_client::GetOptions::new().with_prefix()),
        )
        .await
        .expect("range get");
    assert!(resp.kvs().is_empty(), "refused writes must not touch etcd");
}

// ─────────────────────────── Loader compatibility ───────────────────────────
//
// The most load-bearing assertion in this file: after direct writes of
// one entry of every resource kind (the canonical documents operators
// and the control plane write), build a fresh snapshot via
// `aisix-etcd::loader` from the SAME etcd prefix and verify every
// resource table is populated. This catches:
//
//   - subkey constant drift between the documented key layout and the
//     match arms in `aisix_etcd::loader::build_snapshot`
//   - JSON shape drift between the canonical documents and the
//     loader's serde parse (e.g. a field rename that misses one side)
//   - schema validation drift — the loader re-validates on read; if a
//     canonical document stops validating, the row gets logged +
//     skipped silently in production. This test fails loudly instead.

#[tokio::test]
async fn loader_picks_up_every_direct_write() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    let prefix = unique_prefix();
    let mut client = etcd_client_for(&url).await;

    let key_hash = aisix_core::ApiKey::hash_bearer("sk-loader-it");
    let writes = [
        (
            "models",
            "m-loader-1",
            json!({
                "display_name": "loader-gpt4",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }),
        ),
        (
            "api_keys",
            "k-loader-1",
            json!({"key_hash": key_hash, "allowed_models": ["loader-gpt4"]}),
        ),
        (
            "provider_keys",
            "pk-loader-1",
            json!({
                "display_name": "loader-pk",
                "provider": "openai",
                "adapter": "openai",
                "secret": "sk-loader"
            }),
        ),
        (
            "guardrails",
            "g-loader-1",
            json!({
                "name": "loader-block",
                "kind": "keyword",
                "patterns": [{"kind": "literal", "value": "x"}]
            }),
        ),
        (
            "cache_policies",
            "cp-loader-1",
            json!({"name": "loader-cache", "enabled": true}),
        ),
        (
            "observability_exporters",
            "oe-loader-1",
            json!({
                "name": "loader-otel",
                "kind": "otlp_http",
                "endpoint": "https://otel.example.com/v1/traces"
            }),
        ),
        (
            "mcp_servers",
            "mcp-loader-1",
            json!({"name": "loader-mcp", "url": "https://api.example.com/mcp"}),
        ),
    ];
    for (kind, id, doc) in &writes {
        seed(&mut client, &prefix, kind, id, doc).await;
    }

    // Read the raw etcd entries back via a fresh client and run them
    // through the loader.
    let client = etcd_client_for(&url).await;
    let mut kv = client.kv_client();
    let resp = kv
        .get(
            prefix.as_bytes().to_vec(),
            Some(etcd_client::GetOptions::new().with_prefix()),
        )
        .await
        .expect("range get");

    let raw_entries: Vec<aisix_etcd::RawEntry> = resp
        .kvs()
        .iter()
        .map(|kv| aisix_etcd::RawEntry {
            key: String::from_utf8_lossy(kv.key()).into_owned(),
            value: kv.value().to_vec(),
            revision: kv.mod_revision(),
        })
        .collect();

    let (snap, stats) = aisix_etcd::build_snapshot(&prefix, &raw_entries);
    assert_eq!(
        stats.schema_rejected, 0,
        "loader rejected a canonical document: {stats:?}"
    );
    assert_eq!(
        stats.parse_rejected, 0,
        "loader couldn't parse a canonical document: {stats:?}"
    );
    assert_eq!(
        stats.unknown_kind, 0,
        "loader didn't recognise a kind in the documented key layout — \
         likely a subkey drift against the match arms in \
         aisix_etcd::loader: {stats:?}"
    );
    assert_eq!(stats.accepted, 7, "expected 7 entries; got {stats:?}");

    // Each resource table should now have exactly one entry.
    assert_eq!(snap.models.len(), 1);
    assert_eq!(snap.apikeys.len(), 1);
    assert_eq!(snap.provider_keys.len(), 1);
    assert_eq!(snap.guardrails.len(), 1);
    assert_eq!(snap.cache_policies.len(), 1);
    assert_eq!(snap.observability_exporters.len(), 1);
    assert_eq!(snap.mcp_servers.len(), 1);
}
