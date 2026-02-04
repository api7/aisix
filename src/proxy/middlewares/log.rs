use axum::{
    extract::{ConnectInfo, Request},
    middleware::Next,
    response::Response,
};
use log::info;
use std::net::SocketAddr;

pub async fn log(request: Request, next: Next) -> Response {
    let extensions = request.extensions().clone();
    let conn_info = extensions.get::<ConnectInfo<SocketAddr>>();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let http_version = request.version().clone();

    let headers = request.headers().clone();
    let user_agent = headers
        .get(http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");

    // call handler
    let response = next.run(request).await;

    let status = response.status();

    info!(
        target: "access_log",
        "{} - \"{} {} {:?}\" {} \"{}\"",
        conn_info
            .map(|c| c.0.to_string())
            .unwrap_or_else(|| "-".to_string()),
        method,
        uri.path(),
        http_version,
        status,
        user_agent
    );

    response
}
