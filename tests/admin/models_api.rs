use ai_gateway::admin::PATH_PREFIX;
use axum::body::Body;
use http::{Method, StatusCode};

use crate::admin::{TEST_ADMIN_KEY, create_router};
use crate::utils::http::{build_req, oneshot_json};

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
        .last()
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
