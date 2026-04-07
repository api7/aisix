use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::Stream;
use pin_project::pin_project;

use crate::gateway::{
    error::Result,
    traits::{ChatStreamState, ProviderCapabilities},
    types::openai::ChatCompletionChunk,
};

/// Buffered hub stream adapter for provider-produced raw stream lines.
///
/// `HubChunkStream` preserves output ordering when one raw input item expands
/// into multiple `ChatCompletionChunk` values. The first transformed chunk is
/// returned immediately and the remaining chunks are queued in `buffer` for
/// subsequent polls.
///
/// The stream mutates `state` as transformed chunks flow through it. In
/// particular, provider-specific stream metadata and the latest observed usage
/// totals are accumulated there so later pipeline stages can inspect them.
/// Provider-specific transformation behavior is delegated to `def`, held as an
/// `Arc<dyn ProviderCapabilities>`.
#[pin_project]
pub struct HubChunkStream {
    #[pin]
    inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
    def: Arc<dyn ProviderCapabilities>,
    pub(crate) state: ChatStreamState,
    buffer: VecDeque<ChatCompletionChunk>,
}

impl HubChunkStream {
    /// Creates a `HubChunkStream` from raw provider stream lines.
    ///
    /// The input stream must preserve line order. The returned stream stays
    /// `Send` as long as the input stream is `Send`, and every polled raw line
    /// is transformed through the supplied provider definition.
    pub fn new(
        inner: impl Stream<Item = Result<String>> + Send + 'static,
        def: Arc<dyn ProviderCapabilities>,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            def,
            state: ChatStreamState::default(),
            buffer: VecDeque::new(),
        }
    }
}

impl Stream for HubChunkStream {
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if let Some(chunk) = this.buffer.pop_front() {
            return Poll::Ready(Some(Ok(chunk)));
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(raw))) => {
                    match this.def.transform_stream_chunk(&raw, this.state) {
                        Ok(chunks) => {
                            if chunks.is_empty() {
                                continue;
                            }

                            this.state.chunk_index += chunks.len();
                            for chunk in &chunks {
                                if let Some(usage) = &chunk.usage {
                                    this.state.input_tokens = Some(usage.prompt_tokens);
                                    this.state.output_tokens = Some(usage.completion_tokens);
                                    if this.state.cache_creation_input_tokens.is_none()
                                        && this.state.cache_read_input_tokens.is_none()
                                    {
                                        this.state.cache_read_input_tokens = usage
                                            .prompt_tokens_details
                                            .as_ref()
                                            .and_then(|details| details.cached_tokens);
                                    }
                                }
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
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use http::HeaderMap;

    use super::HubChunkStream;
    use crate::gateway::{
        error::Result,
        provider_instance::ProviderAuth,
        traits::{ChatTransform, ProviderCapabilities, ProviderMeta, StreamReaderKind},
        types::openai::{
            ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
            ChatCompletionUsage,
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

        fn build_auth_headers(&self, _auth: &ProviderAuth) -> Result<HeaderMap> {
            Ok(HeaderMap::new())
        }
    }

    impl ChatTransform for DummyProvider {
        fn transform_stream_chunk(
            &self,
            raw: &str,
            _state: &mut crate::gateway::traits::ChatStreamState,
        ) -> Result<Vec<ChatCompletionChunk>> {
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

    #[tokio::test]
    async fn hub_chunk_stream_consumes_buffered_chunks_in_order() {
        let raw_stream = futures::stream::iter(vec![Ok("data: buffered".to_string())]);
        let mut stream = HubChunkStream::new(raw_stream, Arc::new(DummyProvider));

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();

        assert_eq!(first.choices[0].delta.content.as_deref(), Some("first"));
        assert_eq!(second.choices[0].delta.content.as_deref(), Some("second"));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn hub_chunk_stream_accumulates_usage_from_emitted_chunks() {
        let raw_stream = futures::stream::iter(vec![Ok("data: usage".to_string())]);
        let mut stream = HubChunkStream::new(raw_stream, Arc::new(DummyProvider));

        let chunk = stream.next().await.unwrap().unwrap();

        assert_eq!(chunk.usage.as_ref().unwrap().prompt_tokens, 7);
        assert_eq!(stream.state.input_tokens, Some(7));
        assert_eq!(stream.state.output_tokens, Some(11));
        assert_eq!(stream.state.chunk_index, 1);
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
