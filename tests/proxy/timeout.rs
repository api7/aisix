use std::time::Duration;

use aisix::admin::PATH_PREFIX;
use axum::{Router, body::Body};
use http::{Method, StatusCode};

use crate::{
    proxy::{TEST_ADMIN_KEY, create_routers},
    utils::http::{build_req, oneshot_json},
};

const TEST_PROXY_KEY: &str = "sk-proxy-timeout";

fn admin_req(method: Method, path: &str, body: Option<serde_json::Value>) -> http::Request<Body> {
    build_req(
        method,
        &format!("{PATH_PREFIX}{path}"),
        body,
        TEST_ADMIN_KEY,
    )
}

fn proxy_req(method: Method, path: &str, body: Option<serde_json::Value>) -> http::Request<Body> {
    build_req(method, path, body, TEST_PROXY_KEY)
}

async fn setup(timeout: Option<u64>) -> Router {
    let (admin_router, proxy_router) = create_routers(None).await;

    let mut model_body = serde_json::json!({
        "name": "timeout-model",
        "model": "mock/mock",
        "provider_config": {},
    });
    if let Some(timeout) = timeout {
        model_body["timeout"] = serde_json::json!(timeout);
    }

    let (status, _) = oneshot_json(
        &admin_router,
        admin_req(Method::POST, "/models", Some(model_body)),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, _) = oneshot_json(
        &admin_router,
        admin_req(
            Method::POST,
            "/apikeys",
            Some(serde_json::json!({
                "key": TEST_PROXY_KEY,
                "allowed_models": ["timeout-model"],
            })),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    tokio::time::sleep(Duration::from_millis(50)).await;

    proxy_router
}

#[tokio::test]
async fn chat_completion_returns_gateway_timeout_when_upstream_exceeds_model_timeout() {
    let proxy_router = setup(Some(50)).await;

    let (status, body) = oneshot_json(
        &proxy_router,
        proxy_req(
            Method::POST,
            "/v1/chat/completions",
            Some(serde_json::json!({
                "model": "timeout-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "hello"
                    }
                ]
            })),
        ),
    )
    .await;

    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(body["error"]["code"], "request_timeout");
}
