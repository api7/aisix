use std::pin::Pin;

use futures::Stream;
use tokio::sync::oneshot;

use crate::gateway::{error::Result, traits::ChatFormat, types::common::Usage};

/// Type-erased stream returned from typed chat responses.
pub type ChatResponseStream<F> =
    Pin<Box<dyn Stream<Item = Result<<F as ChatFormat>::StreamChunk>> + Send>>;

/// Format-parameterized chat response with usage attached to each mode.
pub enum ChatResponse<F: ChatFormat> {
    Complete {
        response: F::Response,
        usage: Usage,
    },
    Stream {
        stream: ChatResponseStream<F>,
        usage_rx: oneshot::Receiver<Usage>,
    },
}
