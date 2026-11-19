//! The token-estimate pool takes its thread count from the proxy's worker
//! count, not from the CPUs: each pool thread holds its own tokenizer
//! instances, so `proxy.workers` is what bounds the pool's memory. Its own
//! test binary, because the pool is created once per process.
#![cfg(target_os = "linux")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use aisix_core::resource::ResourceEntry;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ApiKey, Model, ProviderKey, ProxyConfig};
use aisix_gateway::Hub;
use aisix_provider_openai::OpenAiBridge;
use aisix_proxy::{build_router, ProxyState};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

fn pool_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
        .filter(|comm| comm.starts_with("token-estimate"))
        .count()
}

#[tokio::test]
async fn pool_has_one_thread_per_configured_worker() {
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let workers = cpus + 1;
    let config = ProxyConfig {
        addr: "127.0.0.1:0".into(),
        request_body_limit_bytes: 1024 * 1024,
        real_ip: Default::default(),
        request_id: Default::default(),
        url_rewrites: Vec::new(),
        expose_debug_headers: false,
        tls: None,
        listeners: Vec::new(),
        thread_per_core: None,
        workers: Some(workers),
    };
    aisix_proxy::configure_estimate_pool(&config);

    // A response with no usage and more text than is counted inline, so
    // the gateway estimates it on the pool.
    let upstream = MockServer::builder().start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test", "object": "chat.completion", "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "word ".repeat(20_000)}}]
        })))
        .mount(&upstream)
        .await;

    let snapshot = AisixSnapshot::new();
    let pk: ProviderKey = serde_json::from_value(json!({
        "display_name": "upstream", "api_key": "test-upstream-credential",
        "api_base": upstream.uri(), "provider": "openai", "adapter": "openai"
    }))
    .unwrap();
    snapshot
        .provider_keys
        .insert(ResourceEntry::new(PK_ID, pk, 1));
    let model: Model = serde_json::from_value(json!({
        "display_name": "client-model", "provider": "openai",
        "model_name": "gpt-4o", "provider_key_id": PK_ID
    }))
    .unwrap();
    snapshot
        .models
        .insert(ResourceEntry::new("model", model, 1));
    let key: ApiKey = serde_json::from_value(json!({
        "key_hash": ApiKey::hash_bearer("test-caller"), "allowed_models": ["client-model"]
    }))
    .unwrap();
    snapshot
        .apikeys
        .insert(ResourceEntry::new("caller", key, 1));
    let hub = Arc::new(Hub::new());
    hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
    let app =
        build_router(ProxyState::new(SnapshotHandle::new(snapshot), hub, &config).without_cache());

    assert_eq!(pool_threads(), 0, "the pool starts on its first estimate");
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer test-caller")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"client-model","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    // A pool thread names itself once it runs, so wait for every name.
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool_threads() < workers && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        pool_threads(),
        workers,
        "one pool thread per configured worker ({workers}), not per CPU ({cpus})"
    );
}
