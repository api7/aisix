use ai_gateway::admin::PATH_PREFIX;
use axum::body::Body;
use http::{Method, StatusCode};

use crate::{
    admin::{TEST_ADMIN_KEY, create_router},
    utils::http::{build_req, oneshot_json},
};

fn apikeys_url(path: &str) -> String {
    format!("{PATH_PREFIX}/apikeys{path}")
}

fn req(method: Method, path: &str, body: Option<serde_json::Value>) -> http::Request<Body> {
    build_req(method, &apikeys_url(path), body, TEST_ADMIN_KEY)
}

#[tokio::test]
async fn test_crud() {
    let router = create_router(None).await;

    let apikey_body = serde_json::json!({
        "key": "sk-test-crud",
        "allowed_models": ["mock/mock"]
    });

    // 1. list — expect empty
    let (status, body) = oneshot_json(&router, req(Method::GET, "", None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 0, "list should be empty initially");

    // 2. create (POST) — expect 201, extract id
    let (status, body) =
        oneshot_json(&router, req(Method::POST, "", Some(apikey_body.clone()))).await;
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

    // 4. update (PUT) — expect 200, change allowed_models
    let updated_body = serde_json::json!({
        "key": "sk-test-crud",
        "allowed_models": ["mock/mock", "openai/gpt-4"]
    });
    let (status, body) = oneshot_json(
        &router,
        req(Method::PUT, &format!("/{id}"), Some(updated_body)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["value"]["allowed_models"],
        serde_json::json!(["mock/mock", "openai/gpt-4"])
    );

    // 5. get — expect updated allowed_models
    let (status, body) = oneshot_json(&router, req(Method::GET, &format!("/{id}"), None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["value"]["allowed_models"],
        serde_json::json!(["mock/mock", "openai/gpt-4"])
    );

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
    let apikey_body = serde_json::json!({
        "key": "sk-test-put-status",
        "allowed_models": []
    });

    // first PUT on non-existent id — expect 201 Created
    let (status, _) = oneshot_json(
        &router,
        req(
            Method::PUT,
            "/put-status-fixed-id",
            Some(apikey_body.clone()),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "first PUT should return 201 when key does not exist"
    );

    // second PUT on existing id — expect 200 OK
    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-status-fixed-id", Some(apikey_body)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "second PUT should return 200 when key already exists"
    );
}

#[tokio::test]
async fn test_put_duplicate_key_rejected() {
    let router = create_router(None).await;

    let first_apikey = serde_json::json!({
        "key": "sk-put-dup-a",
        "allowed_models": []
    });
    let second_apikey = serde_json::json!({
        "key": "sk-put-dup-b",
        "allowed_models": []
    });

    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-dup-apikey-a", Some(first_apikey.clone())),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, _) = oneshot_json(
        &router,
        req(Method::PUT, "/put-dup-apikey-b", Some(second_apikey)),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // update apikey-b to reuse apikey-a's key should be rejected
    let (status, body) = oneshot_json(
        &router,
        req(Method::PUT, "/put-dup-apikey-b", Some(first_apikey)),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "PUT should reject duplicate API key"
    );
    assert_eq!(body["error_msg"], "API key already exists");
}

#[tokio::test]
async fn test_duplicate_key_rejected() {
    let router = create_router(None).await;
    let apikey_body = serde_json::json!({
        "key": "sk-duplicate",
        "allowed_models": []
    });

    // first POST — expect 201, save storage id for cleanup
    let (status, body) =
        oneshot_json(&router, req(Method::POST, "", Some(apikey_body.clone()))).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = body["key"]
        .as_str()
        .expect("response should contain key")
        .split('/')
        .next_back()
        .expect("key should contain id")
        .to_string();

    // second POST with same key field — expect 400 Bad Request
    let (status, body) = oneshot_json(&router, req(Method::POST, "", Some(apikey_body))).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "duplicate key should be rejected with 400"
    );
    assert_eq!(body["error_msg"], "API key already exists");

    // cleanup
    let (status, _) = oneshot_json(&router, req(Method::DELETE, &format!("/{id}"), None)).await;
    assert_eq!(status, StatusCode::OK);
}
