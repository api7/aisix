use std::pin::Pin;

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};

use crate::gateway::error::{GatewayError, Result};

pub fn sse_reader<S>(stream: S) -> Pin<Box<dyn Stream<Item = Result<String>> + Send>>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let stream = stream
        .chain(futures::stream::once(async {
            Ok(Bytes::from_static(b"\n"))
        }))
        .scan(BytesMut::new(), |buffer, result| match result {
            Ok(chunk) => {
                buffer.extend_from_slice(&chunk);

                let mut lines = Vec::new();
                if let Some(last_newline) = buffer.iter().rposition(|&byte| byte == b'\n') {
                    let complete_data = buffer.split_to(last_newline + 1);
                    let text = String::from_utf8_lossy(&complete_data);
                    for line in text.lines() {
                        if !line.is_empty() {
                            lines.push(Ok(line.to_string()));
                        }
                    }
                }

                futures::future::ready(Some(futures::stream::iter(lines)))
            }
            Err(error) => futures::future::ready(Some(futures::stream::iter(vec![Err(
                GatewayError::Http(error),
            )]))),
        })
        .flatten();

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::StreamExt;

    use super::sse_reader;

    #[tokio::test]
    async fn sse_reader_splits_lines_and_flushes_trailing_data() {
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
}
