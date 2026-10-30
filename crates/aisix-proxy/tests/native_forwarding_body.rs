//! Native forwarding must preserve the model-only wire body and still apply rewrites.
//! Each case drives the real router and inspects the bytes received by its upstream.

use std::sync::Arc;
use std::time::Duration;

use aisix_core::resource::ResourceEntry;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, ApiKey, Model, ProviderKey, ProxyConfig};
use aisix_gateway::Hub;
use aisix_provider_anthropic::AnthropicBridge;
use aisix_provider_openai::OpenAiBridge;
use aisix_proxy::{build_router, ProxyState};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MODEL_ALIAS: &str = "client-model";
const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

struct Endpoint {
    path: &'static str,
    provider: &'static str,
    upstream_model: &'static str,
    body: &'static str,
    response: Value,
}

fn messages() -> Endpoint {
    Endpoint {
        path: "/v1/messages",
        provider: "anthropic",
        upstream_model: "claude-sonnet-4-5",
        body: r#"{"max_tokens":64,"messages":[{"content":"quote: \"; backslash: \\; newline: \n; Unicode: 猫","role":"user"}],"metadata":{"model":"nested-model"},"model":"client-model","stream":false}"#,
        response: json!({
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-5", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 1}
        }),
    }
}

fn count_tokens() -> Endpoint {
    Endpoint {
        path: "/v1/messages/count_tokens",
        response: json!({"input_tokens": 2}),
        ..messages()
    }
}

fn rerank() -> Endpoint {
    Endpoint {
        path: "/v1/rerank",
        provider: "openai",
        upstream_model: "rerank-v3",
        body: r#"{"documents":["quote: \"; backslash: \\; newline: \n; Unicode: 猫"],"metadata":{"model":"nested-model"},"model":"client-model","query":"hello"}"#,
        response: json!({"results": [{"index": 0, "relevance_score": 0.9}]}),
    }
}

fn speech() -> Endpoint {
    Endpoint {
        path: "/v1/audio/speech",
        provider: "openai",
        upstream_model: "tts-1",
        body: r#"{"input":"quote: \"; backslash: \\; newline: \n; Unicode: 猫","metadata":{"model":"nested-model"},"model":"client-model","voice":"alloy"}"#,
        response: Value::Null,
    }
}

fn responses() -> Endpoint {
    Endpoint {
        path: "/v1/responses",
        provider: "openai",
        upstream_model: "gpt-4o",
        body: r#"{"input":"quote: \"; backslash: \\; newline: \n; Unicode: 猫","metadata":{"model":"nested-model"},"model":"client-model","stream":false}"#,
        response: json!({
            "id": "resp_test", "object": "response", "status": "completed",
            "model": "gpt-4o", "output": [],
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        }),
    }
}

/// The exact body the upstream received, as text so a failed comparison is readable.
async fn forward(endpoint: &Endpoint, body: &str, overrides: Value) -> String {
    let upstream = MockServer::builder().start().await;
    let response = if endpoint.path == "/v1/audio/speech" {
        ResponseTemplate::new(200)
            .insert_header("content-type", "audio/mpeg")
            .set_body_bytes(b"AUDIO".to_vec())
    } else {
        ResponseTemplate::new(200).set_body_json(&endpoint.response)
    };
    Mock::given(method("POST"))
        .and(path(endpoint.path))
        .respond_with(response)
        .expect(1)
        .mount(&upstream)
        .await;

    let snapshot = AisixSnapshot::new();
    let pk: ProviderKey = serde_json::from_value(json!({
        "display_name": "native-upstream", "secret": "test-upstream-credential",
        "api_base": upstream.uri(), "provider": endpoint.provider,
        "adapter": endpoint.provider, "request": overrides
    }))
    .unwrap();
    snapshot
        .provider_keys
        .insert(ResourceEntry::new(PK_ID, pk, 1));
    let model: Model = serde_json::from_value(json!({
        "display_name": MODEL_ALIAS, "provider": endpoint.provider,
        "model_name": endpoint.upstream_model, "provider_key_id": PK_ID
    }))
    .unwrap();
    snapshot
        .models
        .insert(ResourceEntry::new("model-native", model, 1));
    let key: ApiKey = serde_json::from_value(json!({
        "key_hash": ApiKey::hash_bearer("test-caller"), "allowed_models": [MODEL_ALIAS]
    }))
    .unwrap();
    snapshot
        .apikeys
        .insert(ResourceEntry::new("caller", key, 1));
    let hub = Arc::new(Hub::new());
    hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
    hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
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
        workers: None,
    };
    let app =
        build_router(ProxyState::new(SnapshotHandle::new(snapshot), hub, &config).without_cache());
    let request = Request::builder()
        .method("POST")
        .uri(endpoint.path)
        .header("authorization", "Bearer test-caller")
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = tokio::time::timeout(Duration::from_secs(30), app.oneshot(request))
        .await
        .expect("native request timed out")
        .unwrap();
    let status = response.status();
    let response_body = tokio::time::timeout(
        Duration::from_secs(30),
        to_bytes(response.into_body(), 1024 * 1024),
    )
    .await
    .expect("native response body timed out")
    .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}: {}",
        endpoint.path,
        String::from_utf8_lossy(&response_body)
    );
    if endpoint.path == "/v1/audio/speech" {
        assert_eq!(response_body.as_ref(), b"AUDIO");
    }
    let received = upstream.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        1,
        "{}: one upstream request, without retries",
        endpoint.path
    );
    String::from_utf8(received[0].body.clone()).expect("upstream body is UTF-8")
}

async fn preserves_wire_and_overrides(endpoint: Endpoint) {
    let expected = endpoint.body.replacen(
        r#""model":"client-model""#,
        &format!(r#""model":"{}""#, endpoint.upstream_model),
        1,
    );
    let received = forward(&endpoint, endpoint.body, json!({})).await;
    // Deliberately compare bytes, not parsed JSON: nested model, order and escaping matter.
    assert_eq!(received, expected, "{} model-only", endpoint.path);

    let mut expected: Value = serde_json::from_str(&expected).unwrap();
    expected["required_option"] = json!("applied");
    let received = forward(
        &endpoint,
        endpoint.body,
        json!({"default_body_fields": {"required_option": "applied"}}),
    )
    .await;
    assert_eq!(
        received,
        serde_json::to_string(&expected).unwrap(),
        "{} must retain provider body rewrites",
        endpoint.path
    );
}

#[tokio::test]
async fn messages_preserves_model_only_wire_and_body_overrides() {
    preserves_wire_and_overrides(messages()).await;
}

#[tokio::test]
async fn count_tokens_preserves_model_only_wire_and_body_overrides() {
    preserves_wire_and_overrides(count_tokens()).await;
}

#[tokio::test]
async fn rerank_preserves_model_only_wire_and_body_overrides() {
    preserves_wire_and_overrides(rerank()).await;
}

#[tokio::test]
async fn speech_preserves_model_only_wire_and_body_overrides() {
    preserves_wire_and_overrides(speech()).await;
}

#[tokio::test]
async fn responses_preserves_model_only_wire_and_body_overrides() {
    preserves_wire_and_overrides(responses()).await;
}

async fn assert_signed_tool_ids_stripped(endpoint: Endpoint) {
    let body = r#"{"max_tokens":64,"messages":[{"content":[{"id":"call_1__thought__eHl6","input":{"id":"argument__thought__keep"},"name":"lookup","type":"tool_use"}],"role":"assistant"},{"content":[{"content":"done","tool_use_id":"call_1__thought__eHl6","type":"tool_result"}],"role":"user"}],"model":"client-model"}"#;
    let received = forward(&endpoint, body, json!({})).await;
    let expected = body
        .replace("call_1__thought__eHl6", "call_1")
        .replace(MODEL_ALIAS, endpoint.upstream_model);
    assert_eq!(received, expected, "{} signed tool ids", endpoint.path);
}

#[tokio::test]
async fn messages_strips_signed_tool_ids_without_touching_tool_arguments() {
    assert_signed_tool_ids_stripped(messages()).await;
}

#[tokio::test]
async fn count_tokens_strips_signed_tool_ids_without_touching_tool_arguments() {
    assert_signed_tool_ids_stripped(count_tokens()).await;
}
