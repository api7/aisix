use std::{convert::Infallible, time::Duration};

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
};
use fastrace::prelude::{Event as TraceEvent, Span};
use log::error;
use reqwest::Url;
use serde::Serialize;
use tokio::{
    sync::{oneshot, oneshot::error::TryRecvError},
    time::error::Elapsed,
};

use crate::{
    config::entities::{Model, ResourceEntry},
    gateway::{
        error::GatewayError,
        traits::{ChatFormat, ProviderCapabilities},
        types::{
            common::Usage,
            response::{ChatResponse, ChatResponseStream},
        },
    },
    proxy::{
        AppState,
        hooks::{
            self, RequestContext, authorization::AuthorizationError, rate_limit::RateLimitError,
        },
        provider::create_provider_instance,
        utils::trace::span_attributes::{apply_span_properties, usage_span_properties},
    },
    utils::future::{WithSpan, maybe_timeout},
};

type AdapterFormat<A> = <A as FormatHandlerAdapter>::Format;
type AdapterRequest<A> = <A as FormatHandlerAdapter>::Request;
type AdapterResponse<A> = <A as FormatHandlerAdapter>::Response;
type AdapterCollector<A> = <A as FormatHandlerAdapter>::Collector;

pub(crate) trait FormatHandlerAdapter: Send + Sync + 'static {
    type Format: ChatFormat<
            Request = Self::Request,
            Response = Self::Response,
            StreamChunk = Self::StreamChunk,
        >;
    type Request;
    type Response: Serialize;
    type StreamChunk: Serialize + Send + 'static;
    type Error: IntoResponse
        + From<AuthorizationError>
        + From<RateLimitError>
        + From<GatewayError>
        + From<Elapsed>;
    type Collector: Default + Send + 'static;

    fn span_name() -> &'static str;

    fn missing_model_error() -> Self::Error;

    fn set_model(request: &mut Self::Request, model: String);

    fn request_span_properties(
        request: &Self::Request,
        provider: &dyn ProviderCapabilities,
        base_url: Option<&Url>,
    ) -> Vec<(String, String)>;

    fn response_span_properties(response: &Self::Response, usage: &Usage) -> Vec<(String, String)>;

    fn apply_chunk_span_properties(span: &Span, chunk: &Self::StreamChunk, is_first_item: bool);

    fn starts_output(chunk: &Self::StreamChunk) -> bool;

    fn record_stream_item(collector: &mut Self::Collector, chunk: &Self::StreamChunk);

    fn output_message_span_properties(collector: &Self::Collector) -> Vec<(String, String)>;

    fn serialize_stream_item(chunk: &Self::StreamChunk) -> SseEvent {
        let mut event =
            SseEvent::default().data(<Self::Format as ChatFormat>::serialize_chunk_payload(chunk));

        if let Some(event_type) = <Self::Format as ChatFormat>::sse_event_type(chunk) {
            event = event.event(event_type);
        }

        event
    }

    fn stream_error_event(_error: &GatewayError) -> Option<SseEvent> {
        None
    }

    fn end_of_stream_event(_saw_item: bool) -> Option<SseEvent> {
        None
    }
}

pub(crate) async fn format_handler<A>(
    State(state): State<AppState>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<AdapterRequest<A>>,
) -> Result<Response, A::Error>
where
    A: FormatHandlerAdapter,
{
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(
        &mut request_ctx,
        <AdapterFormat<A> as ChatFormat>::extract_model(&request_data).to_owned(),
    )
    .await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .ok_or_else(A::missing_model_error)?;

    A::set_model(&mut request_data, model.model.clone());
    let timeout = model.timeout.map(Duration::from_millis);

    let gateway = state.gateway();
    let resources = state.resources();
    let provider = model.provider(resources.as_ref()).ok_or_else(|| {
        GatewayError::Internal(format!("provider {} not found", model.provider_id))
    })?;
    let provider_instance = create_provider_instance(gateway.as_ref(), &provider)?;
    let provider_base_url = provider_instance.effective_base_url().ok();

    let span = Span::enter_with_local_parent(A::span_name());
    apply_span_properties(
        &span,
        A::request_span_properties(
            &request_data,
            provider_instance.def.as_ref(),
            provider_base_url.as_ref(),
        ),
    );

    let (response, span) = (WithSpan {
        inner: maybe_timeout(
            timeout,
            gateway.chat::<AdapterFormat<A>>(&request_data, &provider_instance),
        ),
        span: Some(span),
    })
    .await;

    match response {
        Ok(Ok(ChatResponse::Complete { response, usage })) => {
            span.add_properties(|| A::response_span_properties(&response, &usage));
            handle_regular_response::<A>(response, usage, &mut request_ctx).await
        }
        Ok(Ok(ChatResponse::Stream { stream, usage_rx })) => {
            handle_stream_response::<A>(stream, usage_rx, &mut request_ctx, span).await
        }
        Ok(Err(err)) => {
            span.add_property(|| ("error.type", "gateway_error"));
            Err(err.into())
        }
        Err(err) => {
            span.add_property(|| ("error.type", "timeout"));
            Err(err.into())
        }
    }
}

async fn handle_regular_response<A>(
    response: AdapterResponse<A>,
    usage: Usage,
    request_ctx: &mut RequestContext,
) -> Result<Response, A::Error>
where
    A: FormatHandlerAdapter,
{
    if let Err(err) = hooks::rate_limit::post_check(request_ctx, &usage).await {
        error!("Rate limit post_check error: {}", err);
    }

    let mut response = Json(response).into_response();
    hooks::rate_limit::inject_response_headers(request_ctx, response.headers_mut()).await;
    hooks::observability::record_usage(request_ctx, &usage).await;

    Ok(response)
}

fn spawn_stream_usage_observer(request_ctx: RequestContext, usage_rx: oneshot::Receiver<Usage>) {
    tokio::spawn(async move {
        let mut request_ctx = request_ctx;

        match usage_rx.await {
            Ok(usage) => {
                if let Err(err) =
                    hooks::rate_limit::post_check_streaming(&mut request_ctx, &usage).await
                {
                    error!("Rate limit post_check_streaming error: {}", err);
                }
                hooks::observability::record_streaming_usage(&mut request_ctx, &usage).await;
            }
            Err(err) => {
                error!("Failed to receive streaming usage from gateway: {}", err);
            }
        }
    });
}

async fn finalize_stream_usage(
    request_ctx: &mut RequestContext,
    usage_rx: &mut Option<oneshot::Receiver<Usage>>,
    span: &Span,
    output_message_properties: Vec<(String, String)>,
) {
    span.add_properties(|| output_message_properties);

    if let Some(mut usage_rx) = usage_rx.take() {
        match usage_rx.try_recv() {
            Ok(usage) => {
                if let Err(err) = hooks::rate_limit::post_check_streaming(request_ctx, &usage).await
                {
                    error!("Rate limit post_check_streaming error: {}", err);
                }
                hooks::observability::record_streaming_usage(request_ctx, &usage).await;
                span.add_properties(|| usage_span_properties(&usage));
            }
            Err(TryRecvError::Empty) => {
                spawn_stream_usage_observer(request_ctx.clone(), usage_rx);
            }
            Err(TryRecvError::Closed) => {
                error!("Failed to receive streaming usage from gateway: channel closed");
            }
        }
    }
}

async fn handle_stream_response<A>(
    stream: ChatResponseStream<AdapterFormat<A>>,
    usage_rx: oneshot::Receiver<Usage>,
    request_ctx: &mut RequestContext,
    span: Span,
) -> Result<Response, A::Error>
where
    A: FormatHandlerAdapter,
{
    use futures::stream::StreamExt;

    let stream_request_ctx = request_ctx.clone();
    let sse_stream = futures::stream::unfold(
        (
            stream,
            span,
            stream_request_ctx,
            false,
            false,
            Some(usage_rx),
            AdapterCollector::<A>::default(),
            false,
        ),
        |(
            mut stream,
            span,
            mut request_ctx,
            should_terminate,
            saw_item,
            mut usage_rx,
            mut output_collector,
            mut first_output_arrived,
        )| async move {
            if should_terminate {
                drop(span);
                return None;
            }

            match stream.next().await {
                Some(Ok(chunk)) => {
                    A::record_stream_item(&mut output_collector, &chunk);

                    let now_starts_output = !first_output_arrived && A::starts_output(&chunk);
                    if now_starts_output {
                        first_output_arrived = true;
                        hooks::observability::record_first_token_latency(&mut request_ctx).await;
                        span.add_event(
                            TraceEvent::new("first token arrived")
                                .with_property(|| ("kind", "first_token_arrived")),
                        );
                    }

                    A::apply_chunk_span_properties(&span, &chunk, !saw_item);

                    Some((
                        Ok::<SseEvent, Infallible>(A::serialize_stream_item(&chunk)),
                        (
                            stream,
                            span,
                            request_ctx,
                            false,
                            true,
                            usage_rx,
                            output_collector,
                            first_output_arrived,
                        ),
                    ))
                }
                Some(Err(err)) => {
                    error!("Gateway stream error: {}", err);
                    span.add_property(|| ("error.type", "stream_error"));
                    finalize_stream_usage(
                        &mut request_ctx,
                        &mut usage_rx,
                        &span,
                        A::output_message_span_properties(&output_collector),
                    )
                    .await;

                    if let Some(event) = A::stream_error_event(&err) {
                        Some((
                            Ok(event),
                            (
                                stream,
                                span,
                                request_ctx,
                                true,
                                saw_item,
                                usage_rx,
                                output_collector,
                                first_output_arrived,
                            ),
                        ))
                    } else {
                        drop(span);
                        None
                    }
                }
                None => {
                    finalize_stream_usage(
                        &mut request_ctx,
                        &mut usage_rx,
                        &span,
                        A::output_message_span_properties(&output_collector),
                    )
                    .await;

                    if let Some(event) = A::end_of_stream_event(saw_item) {
                        Some((
                            Ok(event),
                            (
                                stream,
                                span,
                                request_ctx,
                                true,
                                saw_item,
                                usage_rx,
                                output_collector,
                                first_output_arrived,
                            ),
                        ))
                    } else {
                        drop(span);
                        None
                    }
                }
            }
        },
    );

    let mut response = Sse::new(sse_stream).into_response();
    hooks::rate_limit::inject_response_headers(request_ctx, response.headers_mut()).await;
    Ok(response)
}
