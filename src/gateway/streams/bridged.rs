use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;
use pin_project::{pin_project, pinned_drop};
use tokio::sync::oneshot;

use super::HubChunkStream;
use crate::gateway::{
    error::Result,
    traits::{ChatFormat, ChatStreamState},
    types::common::{BridgeContext, Usage},
};

/// Buffered bridge stream from hub chunks into a concrete chat format.
///
/// `BridgedStream` sits on top of `HubChunkStream` and applies
/// `ChatFormat::from_hub_stream()` to each hub chunk. If one hub chunk expands
/// into multiple format-specific stream items, the first item is returned
/// immediately and the rest are queued in `buffer`.
///
/// When the hub stream ends, `BridgedStream` emits any
/// `ChatFormat::stream_end_events()` items and sends the latest accumulated hub
/// usage through `usage_tx`. The same usage snapshot is also sent on drop so
/// callers still receive partial usage if the client disconnects early.
#[pin_project(PinnedDrop)]
pub struct BridgedStream<F: ChatFormat> {
    #[pin]
    hub_stream: HubChunkStream,
    bridge_state: F::BridgeState,
    ctx: BridgeContext,
    buffer: VecDeque<F::StreamChunk>,
    ended: bool,
    usage_tx: Option<oneshot::Sender<Usage>>,
}

impl<F: ChatFormat> BridgedStream<F> {
    /// Creates a format bridge stream over hub chunks.
    pub fn new(
        hub_stream: HubChunkStream,
        ctx: BridgeContext,
        usage_tx: oneshot::Sender<Usage>,
    ) -> Self {
        Self {
            hub_stream,
            bridge_state: F::BridgeState::default(),
            ctx,
            buffer: VecDeque::new(),
            ended: false,
            usage_tx: Some(usage_tx),
        }
    }

    fn usage_from_hub_state(state: &ChatStreamState) -> Usage {
        Usage {
            input_tokens: state.input_tokens,
            output_tokens: state.output_tokens,
            cache_creation_input_tokens: state.cache_creation_input_tokens,
            cache_read_input_tokens: state.cache_read_input_tokens,
            ..Default::default()
        }
        .with_derived_total()
    }

    fn send_usage(usage_tx: &mut Option<oneshot::Sender<Usage>>, usage: Usage) {
        if let Some(tx) = usage_tx.take() {
            let _ = tx.send(usage);
        }
    }
}

impl<F: ChatFormat> Stream for BridgedStream<F> {
    type Item = Result<F::StreamChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if let Some(chunk) = this.buffer.pop_front() {
            return Poll::Ready(Some(Ok(chunk)));
        }

        if *this.ended {
            return Poll::Ready(None);
        }

        loop {
            match this.hub_stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(hub_chunk))) => {
                    match F::from_hub_stream(&hub_chunk, this.bridge_state, this.ctx) {
                        Ok(chunks) => {
                            if chunks.is_empty() {
                                continue;
                            }

                            let mut chunks = VecDeque::from(chunks);
                            let first = chunks.pop_front().unwrap();
                            this.buffer.extend(chunks);
                            return Poll::Ready(Some(Ok(first)));
                        }
                        Err(error) => return Poll::Ready(Some(Err(error))),
                    }
                }
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                Poll::Ready(None) => {
                    *this.ended = true;

                    let usage =
                        Self::usage_from_hub_state(&this.hub_stream.as_ref().get_ref().state);
                    Self::send_usage(this.usage_tx, usage);

                    let mut end_events =
                        VecDeque::from(F::stream_end_events(this.bridge_state, this.ctx));
                    if let Some(first) = end_events.pop_front() {
                        this.buffer.extend(end_events);
                        return Poll::Ready(Some(Ok(first)));
                    }

                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[pinned_drop]
impl<F: ChatFormat> PinnedDrop for BridgedStream<F> {
    fn drop(self: Pin<&mut Self>) {
        let this = self.project();
        let usage =
            BridgedStream::<F>::usage_from_hub_state(&this.hub_stream.as_ref().get_ref().state);
        BridgedStream::<F>::send_usage(this.usage_tx, usage);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use http::HeaderMap;
    use serde_json::Value;
    use tokio::sync::oneshot;

    use super::BridgedStream;
    use crate::gateway::{
        formats::OpenAIChatFormat,
        provider_instance::ProviderAuth,
        streams::HubChunkStream,
        traits::{ChatFormat, ChatTransform, ProviderCapabilities, ProviderMeta, StreamReaderKind},
        types::{
            common::BridgeContext,
            openai::{
                ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
                ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage,
            },
        },
    };

    struct DummyProvider;

    impl ProviderMeta for DummyProvider {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn default_base_url(&self) -> &'static str {
            "https://example.com"
        }

        fn stream_reader_kind(&self) -> StreamReaderKind {
            StreamReaderKind::Sse
        }

        fn build_auth_headers(
            &self,
            _auth: &ProviderAuth,
        ) -> crate::gateway::error::Result<HeaderMap> {
            Ok(HeaderMap::new())
        }
    }

    impl ChatTransform for DummyProvider {
        fn transform_stream_chunk(
            &self,
            raw: &str,
            _state: &mut crate::gateway::traits::ChatStreamState,
        ) -> crate::gateway::error::Result<Vec<ChatCompletionChunk>> {
            match raw {
                "data: buffered" => Ok(vec![
                    chunk_with_content("first", None),
                    chunk_with_content("second", None),
                ]),
                "data: usage" => Ok(vec![chunk_with_content("usage", Some((7, 11)))]),
                _ => Ok(vec![]),
            }
        }
    }

    impl ProviderCapabilities for DummyProvider {}

    struct BufferingFormat;

    impl ChatFormat for BufferingFormat {
        type Request = Value;
        type Response = Value;
        type StreamChunk = String;
        type BridgeState = usize;
        type NativeStreamState = ();

        fn name() -> &'static str {
            "buffering"
        }

        fn is_stream(_req: &Self::Request) -> bool {
            true
        }

        fn extract_model(_req: &Self::Request) -> &str {
            "dummy-model"
        }

        fn to_hub(
            _req: &Self::Request,
        ) -> crate::gateway::error::Result<(ChatCompletionRequest, BridgeContext)> {
            unreachable!("not used in this test")
        }

        fn from_hub(
            _resp: &ChatCompletionResponse,
            _ctx: &BridgeContext,
        ) -> crate::gateway::error::Result<Self::Response> {
            unreachable!("not used in this test")
        }

        fn from_hub_stream(
            chunk: &ChatCompletionChunk,
            state: &mut Self::BridgeState,
            _ctx: &BridgeContext,
        ) -> crate::gateway::error::Result<Vec<Self::StreamChunk>> {
            *state += 1;
            let content = chunk.choices[0].delta.content.clone().unwrap();
            Ok(vec![
                format!("{content}-{}a", *state),
                format!("{content}-{}b", *state),
            ])
        }

        fn stream_end_events(
            state: &mut Self::BridgeState,
            _ctx: &BridgeContext,
        ) -> Vec<Self::StreamChunk> {
            vec![format!("end-{state}")]
        }

        fn transform_native_stream_chunk(
            _provider: &dyn ProviderCapabilities,
            _raw: &str,
            _state: &mut Self::NativeStreamState,
        ) -> crate::gateway::error::Result<Vec<Self::StreamChunk>> {
            unreachable!("not used in this test")
        }

        fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
            chunk.clone()
        }
    }

    #[tokio::test]
    async fn bridged_stream_relays_openai_chunks_and_reports_usage() {
        let raw_stream = futures::stream::iter(vec![
            Ok("data: buffered".to_string()),
            Ok("data: usage".to_string()),
        ]);
        let hub_stream = HubChunkStream::new(raw_stream, Arc::new(DummyProvider));
        let (usage_tx, usage_rx) = oneshot::channel();
        let mut stream =
            BridgedStream::<OpenAIChatFormat>::new(hub_stream, BridgeContext::default(), usage_tx);

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        let usage_chunk = stream.next().await.unwrap().unwrap();

        assert_eq!(first.choices[0].delta.content.as_deref(), Some("first"));
        assert_eq!(second.choices[0].delta.content.as_deref(), Some("second"));
        assert_eq!(
            usage_chunk.choices[0].delta.content.as_deref(),
            Some("usage")
        );
        assert!(stream.next().await.is_none());

        let usage = usage_rx.await.unwrap();
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, Some(11));
        assert_eq!(usage.total_tokens, Some(18));
    }

    #[tokio::test]
    async fn bridged_stream_buffers_multi_chunk_bridge_output_and_end_events() {
        let raw_stream = futures::stream::iter(vec![Ok("data: buffered".to_string())]);
        let hub_stream = HubChunkStream::new(raw_stream, Arc::new(DummyProvider));
        let (usage_tx, usage_rx) = oneshot::channel();
        let mut stream =
            BridgedStream::<BufferingFormat>::new(hub_stream, BridgeContext::default(), usage_tx);

        assert_eq!(stream.next().await.unwrap().unwrap(), "first-1a");
        assert_eq!(stream.next().await.unwrap().unwrap(), "first-1b");
        assert_eq!(stream.next().await.unwrap().unwrap(), "second-2a");
        assert_eq!(stream.next().await.unwrap().unwrap(), "second-2b");
        assert_eq!(stream.next().await.unwrap().unwrap(), "end-2");
        assert!(stream.next().await.is_none());

        let usage = usage_rx.await.unwrap();
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());
        assert!(usage.total_tokens.is_none());
    }

    #[tokio::test]
    async fn bridged_stream_drop_reports_partial_usage() {
        let raw_stream = futures::stream::iter(vec![Ok("data: usage".to_string())]);
        let hub_stream = HubChunkStream::new(raw_stream, Arc::new(DummyProvider));
        let (usage_tx, usage_rx) = oneshot::channel();
        let mut stream =
            BridgedStream::<OpenAIChatFormat>::new(hub_stream, BridgeContext::default(), usage_tx);

        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.usage.as_ref().unwrap().prompt_tokens, 7);

        drop(stream);

        let usage = usage_rx.await.unwrap();
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, Some(11));
        assert_eq!(usage.total_tokens, Some(18));
    }

    fn chunk_with_content(content: &str, usage: Option<(u32, u32)>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk".into(),
            created: 1,
            model: "gpt-test".into(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: Some(content.into()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: usage.map(|(prompt_tokens, completion_tokens)| ChatCompletionUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            system_fingerprint: None,
        }
    }
}
