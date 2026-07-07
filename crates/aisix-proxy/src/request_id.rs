use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

/// Response header carrying the gateway request id so a client can
/// correlate a response to its usage event (both key on this id).
pub(crate) const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-aisix-request-id");

/// Gateway correlation IDs may be written to telemetry fields backed by UUID
/// columns, so keep handler request IDs as plain UUID strings.
pub(crate) fn new_request_id() -> String {
    Uuid::new_v4().to_string()
}

/// The per-request correlation id, stashed in the request extensions by
/// [`ensure_request_id`] so every handler resolves the SAME id for both
/// its usage event and the response header. Handlers with a
/// [`ClientContext`](crate::client_ip::ClientContext) read it from there;
/// the few that don't take one use an `Extension<RequestId>` extractor.
#[derive(Debug, Clone)]
pub(crate) struct RequestId(pub String);

/// Ingress+egress middleware that gives every proxied response an
/// `x-aisix-request-id` header derived from the same id the handler
/// attributes its usage event to.
///
/// One shared mechanism instead of a per-handler header insert: the
/// family had drifted (some handlers set it, some didn't — chat /
/// completions / embeddings / responses / messages all shipped without
/// it in v0.3.0), which is exactly the kind of gap the
/// fix-the-whole-class rule exists to prevent. Minting here and reading
/// it back through `ClientContext` keeps the header equal to the
/// telemetry `request_id`, so the header is actually usable for
/// correlation rather than a second, unrelated id.
pub(crate) async fn ensure_request_id(mut request: Request, next: Next) -> Response {
    let id = request
        .extensions()
        .get::<RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(new_request_id);
    request.extensions_mut().insert(RequestId(id.clone()));

    let mut response = next.run(request).await;

    // If the handler already stamped the header (from the same id), keep
    // it; otherwise stamp it here so no response is ever without one.
    if !response.headers().contains_key(&REQUEST_ID_HEADER) {
        if let Ok(hv) = HeaderValue::from_str(&id) {
            response.headers_mut().insert(REQUEST_ID_HEADER, hv);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    // A handler that echoes the RequestId it sees in the extensions, so
    // the test can prove the middleware exposes the SAME id it stamps on
    // the response header (header == telemetry id).
    async fn echo_extension_id(request: Request) -> Response {
        let seen = request
            .extensions()
            .get::<RequestId>()
            .map(|r| r.0.clone())
            .unwrap_or_default();
        seen.into_response()
    }

    async fn sets_own_header() -> Response {
        let mut resp = "ok".into_response();
        resp.headers_mut()
            .insert(REQUEST_ID_HEADER, HeaderValue::from_static("handler-set"));
        resp
    }

    #[tokio::test]
    async fn stamps_header_and_matches_the_extension_id() {
        let app = Router::new()
            .route("/", get(echo_extension_id))
            .layer(axum::middleware::from_fn(ensure_request_id));

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let header = resp
            .headers()
            .get(&REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .expect("response must carry x-aisix-request-id");
        assert!(
            uuid::Uuid::parse_str(&header).is_ok(),
            "stamped id must be a UUID, got {header:?}"
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let seen_by_handler = String::from_utf8(body.to_vec()).unwrap();
        assert_eq!(
            header, seen_by_handler,
            "response header must equal the id the handler saw (correlation contract)"
        );
    }

    #[tokio::test]
    async fn preserves_a_handler_set_header() {
        let app = Router::new()
            .route("/", get(sets_own_header))
            .layer(axum::middleware::from_fn(ensure_request_id));

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            resp.headers().get(&REQUEST_ID_HEADER).unwrap(),
            "handler-set",
            "middleware must not clobber a header the handler already set"
        );
    }
}
