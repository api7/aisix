use std::pin::Pin;

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};

use crate::gateway::error::{GatewayError, Result};

struct ReaderState {
    buffer: BytesMut,
    terminated: bool,
}

/// `sse_reader` decodes a byte stream into newline-delimited SSE lines.
///
/// The returned stream yields UTF-8 `String` values split on newline
/// boundaries, filters out empty separator lines, and flushes any buffered
/// partial line when the upstream stream ends cleanly. Transport failures are
/// surfaced as `GatewayError::Http` and terminate the reader without emitting
/// buffered partial data.
pub fn sse_reader<S>(stream: S) -> Pin<Box<dyn Stream<Item = Result<String>> + Send>>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let stream = stream
        .chain(futures::stream::once(async {
            Ok(Bytes::from_static(b"\n"))
        }))
        .scan(
            ReaderState {
                buffer: BytesMut::new(),
                terminated: false,
            },
            |state, result| {
                if state.terminated {
                    return futures::future::ready(None);
                }

                match result {
                    Ok(chunk) => {
                        state.buffer.extend_from_slice(&chunk);

                        let mut lines = Vec::new();
                        if let Some(last_newline) =
                            state.buffer.iter().rposition(|&byte| byte == b'\n')
                        {
                            let complete_data = state.buffer.split_to(last_newline + 1);
                            let text = String::from_utf8_lossy(&complete_data);
                            for line in text.lines() {
                                if !line.is_empty() {
                                    lines.push(Ok(line.to_string()));
                                }
                            }
                        }

                        futures::future::ready(Some(futures::stream::iter(lines)))
                    }
                    Err(error) => {
                        state.buffer.clear();
                        state.terminated = true;
                        futures::future::ready(Some(futures::stream::iter(vec![Err(
                            GatewayError::Http(error),
                        )])))
                    }
                }
            },
        )
        .flatten();

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::StreamExt;

    use super::sse_reader;
    use crate::gateway::error::GatewayError;

    #[tokio::test]
    async fn sse_reader_splits_lines_across_chunks() {
        let byte_stream = futures::stream::iter(vec![
            Ok(Bytes::from("data: first\n")),
            Ok(Bytes::from("data: second")),
            Ok(Bytes::from("\n")),
        ]);

        let mut reader = sse_reader(byte_stream);

        assert_eq!(reader.next().await.unwrap().unwrap(), "data: first");
        assert_eq!(reader.next().await.unwrap().unwrap(), "data: second");
        assert!(reader.next().await.is_none());
    }

    #[tokio::test]
    async fn sse_reader_flushes_trailing_partial_line_on_eof() {
        let byte_stream = futures::stream::iter(vec![
            Ok(Bytes::from("data: first\n")),
            Ok(Bytes::from("data: second")),
        ]);

        let mut reader = sse_reader(byte_stream);

        assert_eq!(reader.next().await.unwrap().unwrap(), "data: first");
        assert_eq!(reader.next().await.unwrap().unwrap(), "data: second");
        assert!(reader.next().await.is_none());
    }

    #[tokio::test]
    async fn sse_reader_does_not_flush_partial_line_after_error() {
        let error = reqwest::Client::new()
            .get("http://[::1")
            .build()
            .unwrap_err();
        let byte_stream = futures::stream::iter(vec![Ok(Bytes::from("data: partial")), Err(error)]);

        let mut reader = sse_reader(byte_stream);

        assert!(matches!(
            reader.next().await.unwrap(),
            Err(GatewayError::Http(_))
        ));
        assert!(reader.next().await.is_none());
    }
}
