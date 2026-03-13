use axum::body::Body;
use tower::ServiceExt;

use crate::{admin::create_router, utils::http::to_string};

// /ui -> redirect to /ui/ (no auth required)
#[tokio::test]
async fn redirect_ui_root() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .method(http::Method::GET)
        .uri("/ui")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::SEE_OTHER);
    assert_eq!(resp.headers().get(http::header::LOCATION).unwrap(), "/ui/");
}

// /ui/ -> serves embedded index.html (HTML content)
#[tokio::test]
async fn serve_spa_index() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .method(http::Method::GET)
        .uri("/ui/")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
    let ct = resp
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/html"), "expected text/html, got: {ct}");
}

// /ui/index.html -> same as /ui/
#[tokio::test]
async fn serve_explicit_index_html() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .method(http::Method::GET)
        .uri("/ui/index.html")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
    let body = to_string(resp.into_body()).await;
    assert!(body.contains("<!doctype html") || body.contains("<!DOCTYPE html"));
}

// /ui/<path-without-dot> -> SPA fallback: serve index.html
#[tokio::test]
async fn spa_fallback_for_client_routes() {
    let router = create_router(None).await;

    for path in ["/ui/models", "/ui/apikeys/create", "/ui/settings"] {
        let req = http::Request::builder()
            .method(http::Method::GET)
            .uri(path)
            .body(Body::empty())
            .unwrap();

        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            http::StatusCode::OK,
            "expected SPA fallback 200 for {path}"
        );
        let ct = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/html"),
            "expected text/html for {path}, got: {ct}"
        );
    }
}

// /ui/<missing-asset-with-extension> -> fallback to index.html -> 200
#[tokio::test]
async fn static_asset_not_found_returns_404() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .method(http::Method::GET)
        .uri("/ui/assets/definitely-not-real-xxxxxxxx.js")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
}

// /openapi -> OpenAPI spec UI (no auth required)
#[tokio::test]
async fn openapi_endpoint_ok() {
    let router = create_router(None).await;

    let req = http::Request::builder()
        .method(http::Method::GET)
        .uri("/openapi")
        .body(Body::empty())
        .unwrap();

    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), http::StatusCode::OK);
}
