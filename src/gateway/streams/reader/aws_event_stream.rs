use std::pin::Pin;

use aws_smithy_eventstream::{
    frame::{DecodedFrame, MessageFrameDecoder},
    smithy::parse_response_headers,
};
use aws_smithy_types::event_stream::Message;
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::gateway::error::{GatewayError, Result};

struct AwsEventStreamReaderState {
    buffer: BytesMut,
    decoder: MessageFrameDecoder,
    pending_frame: bool,
    terminated: bool,
}

enum AwsEventStreamInput {
    Chunk(std::result::Result<Bytes, reqwest::Error>),
    Eof,
}

/// `aws_event_stream_reader` decodes AWS EventStream frames into normalized JSON lines.
///
/// Each event frame is emitted as a JSON object with `type` and `payload` fields so
/// provider-specific transforms can distinguish Bedrock `ConverseStream` event kinds
/// without widening the shared streaming interface. Exception frames are surfaced as
/// `GatewayError::Stream`, and truncated frames at EOF fail closed instead of emitting
/// partial data.
pub fn aws_event_stream_reader<S>(stream: S) -> Pin<Box<dyn Stream<Item = Result<String>> + Send>>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let stream = stream
        .map(AwsEventStreamInput::Chunk)
        .chain(futures::stream::once(async { AwsEventStreamInput::Eof }))
        .scan(
            AwsEventStreamReaderState {
                buffer: BytesMut::new(),
                decoder: MessageFrameDecoder::new(),
                pending_frame: false,
                terminated: false,
            },
            |state, input| {
                if state.terminated {
                    return futures::future::ready(None);
                }

                let items = match input {
                    AwsEventStreamInput::Chunk(Ok(chunk)) => {
                        state.buffer.extend_from_slice(&chunk);
                        drain_aws_event_stream_messages(state)
                    }
                    AwsEventStreamInput::Chunk(Err(error)) => {
                        state.buffer.clear();
                        state.pending_frame = false;
                        state.terminated = true;
                        vec![Err(GatewayError::Http(error))]
                    }
                    AwsEventStreamInput::Eof => {
                        let mut items = drain_aws_event_stream_messages(state);
                        if !state.terminated && (state.pending_frame || !state.buffer.is_empty()) {
                            state.buffer.clear();
                            state.pending_frame = false;
                            state.terminated = true;
                            items.push(Err(GatewayError::Stream(
                                "aws event stream ended with an incomplete frame".into(),
                            )));
                        }
                        state.terminated = true;
                        items
                    }
                };

                futures::future::ready(Some(futures::stream::iter(items)))
            },
        )
        .flatten();

    Box::pin(stream)
}

fn drain_aws_event_stream_messages(state: &mut AwsEventStreamReaderState) -> Vec<Result<String>> {
    let mut items = Vec::new();

    loop {
        if state.buffer.is_empty() {
            break;
        }

        let buffered_len = state.buffer.len();
        match state.decoder.decode_frame(&mut state.buffer) {
            Ok(DecodedFrame::Incomplete) => {
                state.pending_frame = buffered_len > 0;
                break;
            }
            Ok(DecodedFrame::Complete(message)) => match normalize_aws_event_stream_message(&message)
            {
                Ok(line) => {
                    state.pending_frame = !state.buffer.is_empty();
                    items.push(Ok(line));
                }
                Err(error) => {
                    state.buffer.clear();
                    state.pending_frame = false;
                    state.terminated = true;
                    items.push(Err(error));
                    break;
                }
            },
            Err(error) => {
                state.buffer.clear();
                state.pending_frame = false;
                state.terminated = true;
                items.push(Err(GatewayError::Stream(format!(
                    "failed to decode aws event stream frame: {error}"
                ))));
                break;
            }
        }
    }

    items
}

fn normalize_aws_event_stream_message(message: &Message) -> Result<String> {
    let headers = parse_response_headers(message).map_err(|error| {
        GatewayError::Stream(format!("failed to parse aws event stream headers: {error}"))
    })?;

    match headers.message_type.as_str() {
        "event" => {
            let payload = if message.payload().is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(message.payload())
                    .map_err(|error| GatewayError::Transform(error.to_string()))?
            };

            serde_json::to_string(&json!({
                "type": headers.smithy_type.as_str(),
                "payload": payload,
            }))
            .map_err(|error| GatewayError::Transform(error.to_string()))
        }
        "exception" => Err(GatewayError::Stream(build_aws_event_stream_exception_message(
            headers.smithy_type.as_str(),
            message.payload(),
        ))),
        other => Err(GatewayError::Stream(format!(
            "unsupported aws event stream message type: {other}"
        ))),
    }
}

fn build_aws_event_stream_exception_message(exception_type: &str, payload: &[u8]) -> String {
    let detail = serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|value| {
            value
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| value.get("Message").and_then(Value::as_str))
                .map(str::to_owned)
                .or_else(|| (!value.is_null()).then(|| value.to_string()))
        })
        .or_else(|| {
            let text = String::from_utf8_lossy(payload).trim().to_string();
            (!text.is_empty()).then_some(text)
        });

    match detail {
        Some(detail) => format!("aws event stream exception {exception_type}: {detail}"),
        None => format!("aws event stream exception {exception_type}"),
    }
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use bytes::Bytes;
    use futures::StreamExt;
    use serde_json::json;

    use super::aws_event_stream_reader;
    use crate::gateway::error::GatewayError;

    #[tokio::test]
    async fn aws_event_stream_reader_decodes_split_event_frames() {
        let message_start = encode_event_message("messageStart", json!({
            "role": "assistant"
        }));
        let metadata = encode_event_message("metadata", json!({
            "usage": {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8}
        }));

        let split_at = message_start.len() / 2;
        let byte_stream = futures::stream::iter(vec![
            Ok(Bytes::copy_from_slice(&message_start[..split_at])),
            Ok(Bytes::copy_from_slice(&message_start[split_at..])),
            Ok(metadata),
        ]);

        let mut reader = aws_event_stream_reader(byte_stream);

        let first: serde_json::Value = serde_json::from_str(&reader.next().await.unwrap().unwrap())
            .unwrap();
        let second: serde_json::Value =
            serde_json::from_str(&reader.next().await.unwrap().unwrap()).unwrap();

        assert_eq!(first["type"], "messageStart");
        assert_eq!(first["payload"]["role"], "assistant");
        assert_eq!(second["type"], "metadata");
        assert_eq!(second["payload"]["usage"]["totalTokens"], 8);
        assert!(reader.next().await.is_none());
    }

    #[tokio::test]
    async fn aws_event_stream_reader_surfaces_exception_frames() {
        let byte_stream = futures::stream::iter(vec![Ok(encode_exception_message(
            "validationException",
            json!({"message": "bad request"}),
        ))]);

        let mut reader = aws_event_stream_reader(byte_stream);

        assert!(matches!(
            reader.next().await.unwrap(),
            Err(GatewayError::Stream(message))
                if message.contains("validationException") && message.contains("bad request")
        ));
        assert!(reader.next().await.is_none());
    }

    fn encode_event_message(event_type: &str, payload: serde_json::Value) -> Bytes {
        encode_message(
            vec![
                Header::new(":message-type", HeaderValue::String("event".into())),
                Header::new(
                    ":event-type",
                    HeaderValue::String(event_type.to_string().into()),
                ),
                Header::new(
                    ":content-type",
                    HeaderValue::String("application/json".into()),
                ),
            ],
            serde_json::to_vec(&payload).unwrap(),
        )
    }

    fn encode_exception_message(exception_type: &str, payload: serde_json::Value) -> Bytes {
        encode_message(
            vec![
                Header::new(":message-type", HeaderValue::String("exception".into())),
                Header::new(
                    ":exception-type",
                    HeaderValue::String(exception_type.to_string().into()),
                ),
                Header::new(
                    ":content-type",
                    HeaderValue::String("application/json".into()),
                ),
            ],
            serde_json::to_vec(&payload).unwrap(),
        )
    }

    fn encode_message(headers: Vec<Header>, payload: Vec<u8>) -> Bytes {
        let message = Message::new_from_parts(headers, payload);
        let mut buffer = Vec::new();
        write_message_to(&message, &mut buffer).unwrap();
        buffer.into()
    }
}