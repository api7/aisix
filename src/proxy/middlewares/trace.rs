//! Trace middleware
//! 1. Generate root spans for tracing
//! 2. Generate access logs
//! 3. Record request count and latency metrics

use std::{
    net::SocketAddr,
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    Error,
    body::{Body, Bytes},
    extract::{ConnectInfo, MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use fastrace::prelude::*;
use http_body::Frame;
use log::info;
use metrics::{counter, histogram};
use opentelemetry_semantic_conventions::trace::{
    HTTP_REQUEST_METHOD, HTTP_RESPONSE_STATUS_CODE, HTTP_ROUTE, URL_PATH,
};

pub const TRACEPARENT_HEADER: &str = "traceparent";

pub struct TimedBody {
    start_time: Instant,
    inner: Body,
    metric_method: String,
    metric_endpoint: String,
    metric_status_code: u16,
    latency_recorded: bool,
}

impl http_body::Body for TimedBody {
    type Data = Bytes;
    type Error = Error;

    #[inline]
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let poll = Pin::new(&mut self.inner).poll_frame(cx);

        if let Poll::Ready(None) = &poll {
            // At this moment, all frames have been consumed by hyper, but it remains uncertain whether
            // the data has been written to the kernel TCP buffer or sent to the client.
            self.record_latency();
        }

        poll
    }

    #[inline]
    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }

    #[inline]
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
}

impl Drop for TimedBody {
    fn drop(&mut self) {
        //TODO: consider record or not record latency in drop, since in some cases the body might not be fully consumed, the latency might be inaccurate.
        self.record_latency();
    }
}

impl TimedBody {
    fn record_latency(&mut self) {
        if self.latency_recorded {
            return;
        }
        let latency = self.start_time.elapsed().as_millis() as f64;
        histogram!(
            crate::utils::metrics::REQUEST_LATENCY_KEY,
            "method" => self.metric_method.clone(),
            "endpoint" => self.metric_endpoint.clone(),
            "status_code" => self.metric_status_code.to_string(),
        )
        .record(latency);
        self.latency_recorded = true;
    }
}

pub async fn trace(mut req: Request, next: Next) -> Response<TimedBody> {
    let start_time = Instant::now();

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

    let (span, span_ctx) = generate_span(&req);

    // Inject span context to facilitate generate new root spans throughout the project.
    // A typical use case is span recording for SSE streams.
    req.extensions_mut().insert(span_ctx);

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

    let (parts, body) = response.into_parts();
    let status = parts.status;

    let metric_method = method.to_string();
    let metric_endpoint = matched_path
        .clone()
        .map(|p| p.as_str().to_string())
        .unwrap_or("-".to_string());
    let metric_status_code = status.as_u16();

    let response = Response::from_parts(
        parts,
        TimedBody {
            start_time,
            inner: body,
            metric_method: metric_method.clone(),
            metric_endpoint: metric_endpoint.clone(),
            metric_status_code,
            latency_recorded: false,
        },
    );

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

    counter!(
        crate::utils::metrics::REQUEST_COUNT_KEY,
        "method" => metric_method,
        "endpoint" => metric_endpoint,
        "status_code" => metric_status_code.to_string(),
    )
    .increment(1);

    response
}

fn generate_span(req: &Request) -> (Span, SpanContext) {
    let name = if let Some(target) = req.extensions().get::<MatchedPath>() {
        format!("{} {}", req.method(), target.as_str())
    } else {
        req.method().to_string()
    };

    let parent = req
        .headers()
        .get(TRACEPARENT_HEADER)
        .and_then(|traceparent| SpanContext::decode_w3c_traceparent(traceparent.to_str().ok()?))
        .unwrap_or_else(SpanContext::random);

    (Span::root(name, parent), parent)
}
