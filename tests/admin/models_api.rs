use ai_gateway::admin::PATH_PREFIX;
use axum::body::Body;
use http::{Method, StatusCode};

use crate::{
    admin::{TEST_ADMIN_KEY, create_router},
    utils::http::{build_req, oneshot_json},
};

fn models_url(path: &str) -> String {
    format!("{PATH_PREFIX}/models{path}")
}

fn req(method: Method, path: &str, body: Option<serde_json::Value>) -> http::Request<Body> {
    build_req(method, &models_url(path), body, TEST_ADMIN_KEY)
}

#[tokio::test]
async fn test_crud() {
    let router = create_router(None).await;

    let model_body = serde_json::json!({
        "name": "test_model",
        "model": "mock/mock",
        "provider_config": {}
    });

    // 1. list — expect empty
    let (status, body) = oneshot_json(&router, req(Method::GET, "", None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 0, "list should be empty initially");

    // 2. create (POST) — expect 201, extract id
    let (status, body) =
        oneshot_json(&router, req(Method::POST, "", Some(model_body.clone()))).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = body["key"]
        .as_str()
        .expect("response should contain key")
        .split('/')
        .next_back()
        .expect("key should contain id")
        .to_string();

    // 3. list — expect 1 item
    let (status, body) = oneshot_json(&router, req(Method::GET, "", None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 1, "list should contain 1 item after create");

    // 4. update (PUT) — expect 200
    let updated_body = serde_json::json!({
        "name": "updated_model",
        "model": "mock/mock",
        "provider_config": {}
    });
    let (status, body) = oneshot_json(
        &router,
        req(Method::PUT, &format!("/{id}"), Some(updated_body)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["value"]["name"], "updated_model");

    // 5. get — expect updated name
    let (status, body) = oneshot_json(&router, req(Method::GET, &format!("/{id}"), None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["value"]["name"], "updated_model");

    // 6. delete — expect 200 with deleted=1
    let (status, body) = oneshot_json(&router, req(Method::DELETE, &format!("/{id}"), None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["deleted"], 1);

    // 7. list — expect empty again
    let (status, body) = oneshot_json(&router, req(Method::GET, "", None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 0, "list should be empty after delete");
}

#[tokio::test]
async fn test_put_status_codes() {
    let router = create_router(None).await;
    let model_body = serde_json::json!({
        "name": "put_model",
        "model": "mock/mock",
        "provider_config": {}
    });
    // first PUT on non-existent key — expect 201 Created
    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-test-fixed-id", Some(model_body.clone())),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "first PUT should return 201 when key does not exist"
    );

    // second PUT on existing key — expect 200 OK
    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-test-fixed-id", Some(model_body)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "second PUT should return 200 when key already exists"
    );
}

#[tokio::test]
async fn test_put_duplicate_name_rejected() {
    let router = create_router(None).await;

    let first_model = serde_json::json!({
        "name": "put-dup-name-a",
        "model": "mock/mock",
        "provider_config": {}
    });
    let second_model = serde_json::json!({
        "name": "put-dup-name-b",
        "model": "mock/mock",
        "provider_config": {}
    });

    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-dup-model-a", Some(first_model.clone())),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-dup-model-b", Some(second_model)),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // update model-b to reuse model-a's name should be rejected
    let (status, body) = oneshot_json(
        &router,
        req(Method::PUT, "/put-dup-model-b", Some(first_model)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "PUT should reject duplicate model name"
    );
    assert_eq!(body["error_msg"], "Model name already exists");
}

#[tokio::test]
async fn test_duplicate_name_rejected() {
    let router = create_router(None).await;
    let model_body = serde_json::json!({
        "name": "duplicate_model_name",
        "model": "mock/mock",
        "provider_config": {}
    });

    // first POST — expect 201
    let (status, body) =
        oneshot_json(&router, req(Method::POST, "", Some(model_body.clone()))).await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(
        body["key"].as_str().is_some(),
        "response should contain key"
    );

    // second POST with same name field — expect 400 Bad Request
    let (status, body) = oneshot_json(&router, req(Method::POST, "", Some(model_body))).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "duplicate model name should be rejected with 400"
    );
    assert_eq!(body["error_msg"], "Model name already exists");

    // no cleanup required: each test uses an isolated etcd prefix
}
