//! End-to-end Admin → etcd → loader round-trip tests.
//!
//! Bracketed by `ADMIN_TEST_ETCD_URL` (mirrors the
//! `CACHE_TEST_REDIS_URL` pattern in `aisix-cache/tests/redis_integration.rs`):
//! tests no-op when unset so local `cargo test` without docker still
//! passes; CI sets the env var via the `etcd` service in
//! `.github/workflows/ci.yml`.
//!
//! Why a real etcd instead of `InMemoryStore`:
//!
//! 1. Verifies the byte shape `EtcdConfigStore` writes against the
//!    shape `aisix-etcd::loader` reads — the subkey constants on the
//!    two sides have drifted before, and unit tests against the
//!    in-memory store wouldn't catch it.
//! 2. Catches the `EtcdConfigStore` impls themselves (the in-memory
//!    store doesn't exercise serde + grpc + revision plumbing).
//! 3. Exercises the full Admin handler → ConfigStore → etcd path so a
//!    handler that forgets to call `state.store.put_X` gets caught
//!    here even if the in-memory tests don't.

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

/// Separate from `ADMIN_TEST_ETCD_URL`: CI's etcd container has no auth.
fn auth_etcd_url() -> Option<String> {
    std::env::var("ADMIN_TEST_ETCD_AUTH_URL").ok()
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

async fn build_state_with_real_etcd(url: &str, prefix: &str) -> AdminState {
    let client = etcd_client::Client::connect([url], None)
        .await
        .expect("etcd connect");
    let store: Arc<dyn ConfigStore> = Arc::new(EtcdConfigStore::new(client, prefix, None));
    let handle = SnapshotHandle::new(AisixSnapshot::new());
    let cfg = AdminConfig {
        addr: "127.0.0.1:0".into(),
        admin_keys: vec![ADMIN_KEY.into()],
        tls: None,
    };
    AdminState::new(handle, store, &cfg)
}

fn auth_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("authorization", format!("Bearer {ADMIN_KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn auth_get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("authorization", format!("Bearer {ADMIN_KEY}"))
        .body(Body::empty())
        .unwrap()
}

fn auth_delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .header("authorization", format!("Bearer {ADMIN_KEY}"))
        .body(Body::empty())
        .unwrap()
}

async fn body_json(resp: axum::http::Response<Body>) -> Value {
    let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Drive `(POST → list → DELETE → list)` through the Admin HTTP layer
/// and assert the round-trip lands real entries in etcd. Returns `true`
/// on success so the per-resource macro can assert.
async fn admin_crud_round_trip(state: AdminState, list_uri: &str, payload: Value) {
    let app = build_router(state.clone());

    // POST
    let resp = app
        .oneshot(auth_post(list_uri, payload.clone()))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "POST {list_uri}");
    let created = body_json(resp).await;
    let id = created["id"].as_str().expect("created.id").to_string();
    assert!(created["revision"].as_i64().unwrap_or(0) >= 1);

    // LIST
    let app = build_router(state.clone());
    let resp = app.oneshot(auth_get(list_uri)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let listed = body_json(resp).await;
    let arr = listed.as_array().expect("list array");
    assert_eq!(
        arr.len(),
        1,
        "list returned {} entries for {list_uri}",
        arr.len()
    );
    assert_eq!(arr[0]["id"], id);

    // DELETE
    let app = build_router(state.clone());
    let item_uri = format!("{list_uri}/{id}");
    let resp = app.oneshot(auth_delete(&item_uri)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // LIST again — empty
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
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    admin_crud_round_trip(
        state,
        "/admin/v1/models",
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
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    let key_hash = aisix_core::ApiKey::hash_bearer("sk-it-bearer");
    admin_crud_round_trip(
        state,
        "/admin/v1/apikeys",
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
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    admin_crud_round_trip(
        state,
        "/admin/v1/provider_keys",
        json!({"display_name": "openai-it", "secret": "sk-it"}),
    )
    .await;
}

#[tokio::test]
async fn mcp_servers_round_trip_through_real_etcd() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    admin_crud_round_trip(
        state,
        "/admin/v1/mcp_servers",
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
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    admin_crud_round_trip(
        state,
        "/admin/v1/guardrails",
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
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    admin_crud_round_trip(
        state,
        "/admin/v1/cache_policies",
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
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;
    admin_crud_round_trip(
        state,
        "/admin/v1/observability_exporters",
        json!({
            "name": "it-otel",
            "kind": "otlp_http",
            "endpoint": "https://otel.example.com/v1/traces"
        }),
    )
    .await;
}

// ─────────────────────────── Loader compatibility ───────────────────────────
//
// The most load-bearing assertion in this file: after the Admin path
// writes one entry of every resource type, build a fresh snapshot via
// `aisix-etcd::loader` from the SAME etcd prefix and verify every
// resource table is populated. This catches:
//
//   - subkey constant drift between `EtcdConfigStore::*_SUBKEY` and the
//     match arms in `aisix_etcd::loader::build_snapshot`
//   - JSON shape drift between the admin write and the loader's serde
//     parse (e.g. a field rename that misses one side)
//   - schema validation drift — the loader re-validates on read; if
//     the admin path persists a value the loader's schema rejects, the
//     row gets logged + skipped silently in production. This test
//     fails loudly instead.

#[tokio::test]
async fn loader_picks_up_every_admin_write() {
    let Some(url) = etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_URL not set");
        return;
    };
    let prefix = unique_prefix();
    let state = build_state_with_real_etcd(&url, &prefix).await;

    // Seed one of every resource through the Admin HTTP path.
    let key_hash = aisix_core::ApiKey::hash_bearer("sk-loader-it");
    let writes = [
        (
            "/admin/v1/models",
            json!({
                "display_name": "loader-gpt4",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }),
        ),
        (
            "/admin/v1/apikeys",
            json!({"key_hash": key_hash, "allowed_models": ["loader-gpt4"]}),
        ),
        (
            "/admin/v1/provider_keys",
            json!({"display_name": "loader-pk", "secret": "sk-loader"}),
        ),
        (
            "/admin/v1/guardrails",
            json!({
                "name": "loader-block",
                "kind": "keyword",
                "patterns": [{"kind": "literal", "value": "x"}]
            }),
        ),
        (
            "/admin/v1/cache_policies",
            json!({"name": "loader-cache", "enabled": true}),
        ),
        (
            "/admin/v1/observability_exporters",
            json!({
                "name": "loader-otel",
                "kind": "otlp_http",
                "endpoint": "https://otel.example.com/v1/traces"
            }),
        ),
        (
            "/admin/v1/mcp_servers",
            json!({"display_name": "loader-mcp", "url": "https://api.example.com/mcp"}),
        ),
    ];
    for (uri, body) in writes {
        let app = build_router(state.clone());
        let resp = app.oneshot(auth_post(uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "POST {uri}");
    }

    // Read the raw etcd entries back via a fresh client and run them
    // through the loader.
    let client = etcd_client::Client::connect([url.as_str()], None)
        .await
        .expect("loader-side etcd client");
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
        "loader rejected an admin-written row: {stats:?}"
    );
    assert_eq!(
        stats.parse_rejected, 0,
        "loader couldn't parse an admin-written row: {stats:?}"
    );
    assert_eq!(
        stats.unknown_kind, 0,
        "loader didn't recognise a kind written by the admin path — \
         likely a subkey-constant drift between EtcdConfigStore::*_SUBKEY \
         and the match arms in aisix_etcd::loader: {stats:?}"
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

// ─────────────────────────── Auth token refresh ───────────────────────────
//
// Regression test for a production incident: Admin API calls made long
// after boot failed with "etcdserver: invalid auth token". Root cause:
// `EtcdConfigStore`'s write-path client authenticated once at connect
// time and never refreshed, so any write issued past
// `--auth-token-ttl` (default 300s) hit UNAUTHENTICATED.
// `EtcdConfigStore::new` now spawns `aisix_etcd::start_token_refresh_task`
// to fix this.
//
// Requires `ADMIN_TEST_ETCD_AUTH_URL` (auth-enabled etcd, short TTL) +
// `ADMIN_TEST_ETCD_USER` / `ADMIN_TEST_ETCD_PASSWORD`; no-ops otherwise.
//
// Local run:
//   docker run --rm -p 2379:2379 quay.io/coreos/etcd:v3.5.18 \
//     --listen-client-urls=http://0.0.0.0:2379 \
//     --advertise-client-urls=http://0.0.0.0:2379 --auth-token-ttl=5
//   etcdctl user add root:rootpw && etcdctl user grant-role root root \
//     && etcdctl user add test:testpw && etcdctl user grant-role test root \
//     && etcdctl auth enable
//   ADMIN_TEST_ETCD_AUTH_URL=http://127.0.0.1:2379 ADMIN_TEST_ETCD_USER=test \
//   ADMIN_TEST_ETCD_PASSWORD=testpw \
//   cargo test -p aisix-admin --test etcd_integration \
//     admin_write_survives_token_expiry -- --nocapture
#[tokio::test]
async fn admin_write_survives_token_expiry() {
    let Some(url) = auth_etcd_url() else {
        eprintln!("skipping: ADMIN_TEST_ETCD_AUTH_URL not set");
        return;
    };
    let user = std::env::var("ADMIN_TEST_ETCD_USER").expect("ADMIN_TEST_ETCD_USER required");
    let password =
        std::env::var("ADMIN_TEST_ETCD_PASSWORD").expect("ADMIN_TEST_ETCD_PASSWORD required");

    let prefix = unique_prefix();
    let options = etcd_client::ConnectOptions::new().with_user(user, password);
    let client = etcd_client::Client::connect([url.as_str()], Some(options))
        .await
        .expect("etcd connect with auth");
    // Mirrors main.rs's production bootstrap call, but with a 2s refresh
    // interval — well under the 5s --auth-token-ttl the local-run etcd is
    // started with above, so the loop refreshes before expiry.
    let store: Arc<dyn ConfigStore> = Arc::new(EtcdConfigStore::new(client, prefix, Some(2)));
    let handle = SnapshotHandle::new(AisixSnapshot::new());
    let cfg = AdminConfig {
        addr: "127.0.0.1:0".into(),
        admin_keys: vec![ADMIN_KEY.into()],
        tls: None,
    };
    let state = AdminState::new(handle, store, &cfg);

    // Wait past the 5s TTL window (1s margin absorbs scheduler jitter);
    // without refresh this POST would 401.
    tokio::time::sleep(std::time::Duration::from_secs(6)).await;

    let app = build_router(state);
    let resp = app
        .oneshot(auth_post(
            "/admin/v1/models",
            json!({
                "display_name": "post-expiry",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }),
        ))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "admin write after token-TTL window should succeed; a non-OK \
         status here means the auth-token refresh loop isn't running",
    );
}
