use aisix::admin::PATH_PREFIX;
use axum::body::Body;
use tower::ServiceExt;

use crate::{
    admin::{TEST_ADMIN_KEY, create_router},
    utils::http::to_string,
};

#[tokio::test]
async fn auth_bearer_token_ok() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .uri(format!("{PATH_PREFIX}/models"))
        .header(
            http::header::AUTHORIZATION,
            format!("Bearer {TEST_ADMIN_KEY}"),
        )
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
}

#[tokio::test]
async fn auth_x_api_key_ok() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .uri(format!("{PATH_PREFIX}/models"))
        .header("x-api-key", TEST_ADMIN_KEY)
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
}

#[tokio::test]
async fn auth_prefer_bearer_token() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .uri(format!("{PATH_PREFIX}/models"))
        .header(
            http::header::AUTHORIZATION,
            format!("Bearer {TEST_ADMIN_KEY}"),
        )
        .header("x-api-key", "invalid_key")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
}

#[tokio::test]
async fn no_auth_header() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .uri(format!("{PATH_PREFIX}/models"))
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::UNAUTHORIZED);
    assert_eq!(
        to_string(resp.into_body()).await,
        "{\"error_msg\":\"Missing API key\"}"
    );
}

#[tokio::test]
async fn invalid_auth_header() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .uri(format!("{PATH_PREFIX}/models"))
        .header(http::header::AUTHORIZATION, "Bearer invalid_token")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::UNAUTHORIZED);
    assert_eq!(
        to_string(resp.into_body()).await,
        "{\"error_msg\":\"Invalid API key\"}"
    );
}
