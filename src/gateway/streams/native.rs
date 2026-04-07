use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::Stream;
use pin_project::{pin_project, pinned_drop};
use tokio::sync::oneshot;

use crate::gateway::{
    error::Result,
    traits::{ChatFormat, ProviderCapabilities},
    types::common::Usage,
};

/// Buffered native-format stream adapter that bypasses the hub chunk layer.
///
/// `NativeStream` forwards raw provider stream lines directly into
/// `ChatFormat::transform_native_stream_chunk()`. Like the other stream
/// adapters, it preserves ordering when one raw line expands into multiple
/// output items by queueing the remainder in `buffer`.
///
/// Usage reporting is still wired through `usage_tx`, but the current generic
/// native stream layer does not yet expose a format-independent way to read
/// accumulated native usage from `NativeStreamState`, so completion and drop
/// currently report an empty `Usage` value.
#[pin_project(PinnedDrop)]
pub struct NativeStream<F: ChatFormat> {
    #[pin]
    inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
    def: Arc<dyn ProviderCapabilities>,
    native_state: F::NativeStreamState,
    buffer: VecDeque<F::StreamChunk>,
    ended: bool,
    usage_tx: Option<oneshot::Sender<Usage>>,
}

impl<F: ChatFormat> NativeStream<F> {
    /// Creates a native stream over raw provider output lines.
    pub fn new(
        inner: impl Stream<Item = Result<String>> + Send + 'static,
        def: Arc<dyn ProviderCapabilities>,
        usage_tx: oneshot::Sender<Usage>,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            def,
            native_state: F::NativeStreamState::default(),
            buffer: VecDeque::new(),
            ended: false,
            usage_tx: Some(usage_tx),
        }
    }

    fn send_usage(usage_tx: &mut Option<oneshot::Sender<Usage>>) {
        if let Some(tx) = usage_tx.take() {
            let _ = tx.send(Usage::default());
        }
    }
}

impl<F: ChatFormat> Stream for NativeStream<F> {
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
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(raw))) => {
                    match F::transform_native_stream_chunk(
                        this.def.as_ref(),
                        &raw,
                        this.native_state,
                    ) {
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
                    Self::send_usage(this.usage_tx);
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[pinned_drop]
impl<F: ChatFormat> PinnedDrop for NativeStream<F> {
    fn drop(self: Pin<&mut Self>) {
        let this = self.project();
        NativeStream::<F>::send_usage(this.usage_tx);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use http::HeaderMap;
    use serde_json::Value;
    use tokio::sync::oneshot;

    use super::NativeStream;
    use crate::gateway::{
        provider_instance::ProviderAuth,
        traits::{ChatFormat, ChatTransform, ProviderCapabilities, ProviderMeta, StreamReaderKind},
        types::{
            common::BridgeContext,
            openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
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

    impl ChatTransform for DummyProvider {}

    impl ProviderCapabilities for DummyProvider {}

    struct CountingNativeFormat;

    impl ChatFormat for CountingNativeFormat {
        type Request = Value;
        type Response = Value;
        type StreamChunk = String;
        type BridgeState = ();
        type NativeStreamState = usize;

        fn name() -> &'static str {
            "counting-native"
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
            _chunk: &ChatCompletionChunk,
            _state: &mut Self::BridgeState,
            _ctx: &BridgeContext,
        ) -> crate::gateway::error::Result<Vec<Self::StreamChunk>> {
            unreachable!("not used in this test")
        }

        fn transform_native_stream_chunk(
            provider: &dyn ProviderCapabilities,
            raw: &str,
            state: &mut Self::NativeStreamState,
        ) -> crate::gateway::error::Result<Vec<Self::StreamChunk>> {
            assert_eq!(provider.name(), "dummy");

            match raw {
                "data: buffered" => {
                    *state += 1;
                    Ok(vec![
                        format!("native-{}a", *state),
                        format!("native-{}b", *state),
                    ])
                }
                "data: skip" => Ok(vec![]),
                "data: single" => {
                    *state += 1;
                    Ok(vec![format!("native-{state}")])
                }
                _ => Ok(vec![]),
            }
        }

        fn serialize_chunk_payload(chunk: &Self::StreamChunk) -> String {
            chunk.clone()
        }
    }

    #[tokio::test]
    async fn native_stream_buffers_output_and_preserves_native_state() {
        let raw_stream = futures::stream::iter(vec![
            Ok("data: buffered".to_string()),
            Ok("data: skip".to_string()),
            Ok("data: single".to_string()),
        ]);
        let (usage_tx, usage_rx) = oneshot::channel();
        let mut stream = NativeStream::<CountingNativeFormat>::new(
            raw_stream,
            Arc::new(DummyProvider),
            usage_tx,
        );

        assert_eq!(stream.next().await.unwrap().unwrap(), "native-1a");
        assert_eq!(stream.next().await.unwrap().unwrap(), "native-1b");
        assert_eq!(stream.next().await.unwrap().unwrap(), "native-2");
        assert!(stream.next().await.is_none());

        let usage = usage_rx.await.unwrap();
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());
        assert!(usage.total_tokens.is_none());
    }

    #[tokio::test]
    async fn native_stream_drop_sends_empty_usage_signal() {
        let raw_stream = futures::stream::iter(vec![Ok("data: single".to_string())]);
        let (usage_tx, usage_rx) = oneshot::channel();
        let mut stream = NativeStream::<CountingNativeFormat>::new(
            raw_stream,
            Arc::new(DummyProvider),
            usage_tx,
        );

        assert_eq!(stream.next().await.unwrap().unwrap(), "native-1");

        drop(stream);

        let usage = usage_rx.await.unwrap();
        assert!(usage.input_tokens.is_none());
        assert!(usage.output_tokens.is_none());
        assert!(usage.total_tokens.is_none());
    }
}
