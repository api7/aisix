//! Trace middleware
//! Derived from [fastrace-axum](https://github.com/fast/fastrace-axum)
//! However, it enables the generation of root spans on its own,
//! rather than skipping tracing when a trace context is absent.

use std::future::Future;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use axum::extract::MatchedPath;
use axum::response::Response;
use fastrace::future::InSpan;
use fastrace::local::LocalSpan;
use fastrace::prelude::*;
use opentelemetry_semantic_conventions::trace::HTTP_REQUEST_METHOD;
use opentelemetry_semantic_conventions::trace::HTTP_RESPONSE_STATUS_CODE;
use opentelemetry_semantic_conventions::trace::HTTP_ROUTE;
use opentelemetry_semantic_conventions::trace::URL_PATH;
use tower::{Layer, Service};

pub const TRACEPARENT_HEADER: &str = "traceparent";

#[derive(Clone)]
pub struct TraceLayer;

impl<S> Layer<S> for TraceLayer {
    type Service = TraceService<S>;

    fn layer(&self, service: S) -> Self::Service {
        TraceService { service }
    }
}

#[derive(Clone)]
pub struct TraceService<S> {
    service: S,
}

use axum::extract::Request;

impl<S> Service<Request> for TraceService<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = InSpan<InspectHttpResponse<S::Future>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let headers = req.headers();
        let parent = headers
            .get(TRACEPARENT_HEADER)
            .and_then(|traceparent| SpanContext::decode_w3c_traceparent(traceparent.to_str().ok()?))
            .unwrap_or_else(|| SpanContext::random());

        // https://opentelemetry.io/docs/specs/semconv/http/http-spans/#name
        let name = if let Some(target) = req.extensions().get::<MatchedPath>() {
            format!("{} {}", req.method(), target.as_str())
        } else {
            req.method().to_string()
        };

        let span = Span::root(name, parent);

        span.add_properties(|| {
            [
                (HTTP_REQUEST_METHOD, req.method().to_string()),
                (URL_PATH, req.uri().path().to_string()),
            ]
        });

        if let Some(route) = req.extensions().get::<MatchedPath>() {
            span.add_property(|| (HTTP_ROUTE, route.as_str().to_string()));
        }

        let fut = self.service.call(req);
        let fut = InspectHttpResponse { inner: fut };
        fut.in_span(span)
    }
}

#[pin_project::pin_project]
pub struct InspectHttpResponse<F> {
    #[pin]
    inner: F,
}

impl<F, E> Future for InspectHttpResponse<F>
where
    F: Future<Output = Result<Response, E>>,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let poll = this.inner.poll(cx);

        if let Poll::Ready(Ok(response)) = &poll {
            LocalSpan::add_property(|| {
                (
                    HTTP_RESPONSE_STATUS_CODE,
                    response.status().as_u16().to_string(),
                )
            });
        }

        poll
    }
}
