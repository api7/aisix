//! Trace middleware
//! Derived from [fastrace-axum](https://github.com/fast/fastrace-axum)
//! However, it enables the generation of root spans on its own,
//! rather than skipping tracing when a trace context is absent.

use std::net::SocketAddr;

use axum::extract::ConnectInfo;
use axum::extract::MatchedPath;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use fastrace::local::LocalSpan;
use fastrace::prelude::*;
use log::info;
use opentelemetry_semantic_conventions::trace::{
    HTTP_REQUEST_METHOD, HTTP_RESPONSE_STATUS_CODE, HTTP_ROUTE, URL_PATH,
};

pub const TRACEPARENT_HEADER: &str = "traceparent";

pub async fn trace(req: Request, next: Next) -> Response {
    let headers = req.headers();
    let conn_info = req.extensions().get::<ConnectInfo<SocketAddr>>().cloned();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let http_version = req.version();
    let matched_path = req.extensions().get::<MatchedPath>().cloned();
    let user_agent = headers
        .get(http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    let path = req.uri().path().to_string();

    let span = generate_span(&req);

    span.add_properties(|| {
        [
            (HTTP_REQUEST_METHOD, method.to_string()),
            (URL_PATH, path.clone()),
        ]
    });

    if let Some(ref route) = matched_path {
        span.add_property(|| (HTTP_ROUTE, route.as_str().to_string()));
    }

    let response = async {
        let response = next.run(req).await;
        LocalSpan::add_property(|| {
            (
                HTTP_RESPONSE_STATUS_CODE,
                response.status().as_u16().to_string(),
            )
        });
        response
    }
    .in_span(span)
    .await;

    info!(
        target: "access_log",
        "{} - \"{} {} {:?}\" {} \"{}\"",
        conn_info
            .as_ref()
            .map(|c| c.0.to_string())
            .unwrap_or_else(|| "-".to_string()),
        method,
        uri.path(),
        http_version,
        response.status(),
        user_agent
    );

    response
}

fn generate_span(req: &Request) -> Span {
    let name = if let Some(target) = req.extensions().get::<MatchedPath>() {
        format!("{} {}", req.method(), target.as_str())
    } else {
        req.method().to_string()
    };

    let parent = req
        .headers()
        .get(TRACEPARENT_HEADER)
        .and_then(|traceparent| SpanContext::decode_w3c_traceparent(traceparent.to_str().ok()?))
        .unwrap_or_else(|| SpanContext::random());

    Span::root(name, parent)
}
