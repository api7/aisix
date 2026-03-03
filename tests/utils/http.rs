use axum::{
    Router,
    body::{Body, to_bytes},
};
use http::{Method, Request, StatusCode};
use tower::ServiceExt;

pub async fn to_string(body: Body) -> String {
    String::from_utf8(to_bytes(body, usize::MAX).await.unwrap().to_vec()).unwrap()
}

pub fn build_req(
    method: Method,
    uri: &str,
    body: Option<serde_json::Value>,
    auth_key: &str,
) -> Request<Body> {
    let (content_type, body_bytes) = match body {
        Some(v) => (
            Some("application/json"),
            Body::from(serde_json::to_vec(&v).unwrap()),
        ),
        None => (None, Body::empty()),
    };
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header(http::header::AUTHORIZATION, format!("Bearer {auth_key}"));
    if let Some(ct) = content_type {
        builder = builder.header(http::header::CONTENT_TYPE, ct);
    }
    builder.body(body_bytes).unwrap()
}

pub async fn oneshot_json(router: &Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = serde_json::from_str(&to_string(resp.into_body()).await).unwrap();
    (status, body)
}
