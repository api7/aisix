use std::error::Error;

use bytes::{Bytes, BytesMut};
use futures::{Stream, stream::BoxStream};

use crate::handlers::chat_completions::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
};

pub async fn chat_completion(
    client: reqwest::Client,
    url: &str,
    api_key: &str,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>> {
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, error_text).into());
    }

    let completion = response.json::<ChatCompletionResponse>().await?;
    Ok(completion)
}

pub async fn chat_completion_stream(
    client: reqwest::Client,
    url: &str,
    api_key: &str,
    request: ChatCompletionRequest,
) -> Result<
    BoxStream<'static, Result<ChatCompletionChunk, Box<dyn Error + Send + Sync>>>,
    Box<dyn Error + Send + Sync>,
> {
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, error_text).into());
    }

    Ok(Box::pin(parse_sse_stream(response.bytes_stream())))
}

fn parse_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
) -> impl Stream<Item = Result<ChatCompletionChunk, Box<dyn Error + Send + Sync>>> {
    use futures::stream::StreamExt;

    stream
        .chain(futures::stream::once(async {
            Ok(Bytes::from_static(b"\n"))
        }))
        .scan(BytesMut::new(), |buffer, result| {
            match result {
                Ok(chunk) => {
                    buffer.extend_from_slice(&chunk);

                    let mut lines = Vec::new();

                    // If there are incomplete lines, then this chunk will not terminate with a line break.
                    // We accumulate these remaining bytes and append them during the next extraction.
                    if let Some(last_newline) = buffer.iter().rposition(|&b| b == b'\n') {
                        let complete_data = buffer.split_to(last_newline + 1);

                        let text = String::from_utf8_lossy(&complete_data);
                        for line in text.lines() {
                            lines.push(Ok(line.to_string()));
                        }
                    }

                    futures::future::ready(Some(futures::stream::iter(lines)))
                }
                Err(e) => {
                    let err: Box<dyn Error + Send + Sync> = Box::new(e);
                    futures::future::ready(Some(futures::stream::iter(vec![Err(err)])))
                }
            }
        })
        .flatten()
        .filter_map(|line_result| async move {
            match line_result {
                Ok(line) => {
                    // Only process lines starting with "data: "
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        // Skip [DONE] events
                        // Stream termination signifies completion; we need not explicitly propagate them.
                        if json_str == "[DONE]" {
                            return None;
                        }

                        match serde_json::from_str::<ChatCompletionChunk>(json_str) {
                            Ok(chunk) => Some(Ok(chunk)),
                            Err(e) => {
                                // Propagate parse errors instead of skipping
                                let err: Box<dyn Error + Send + Sync> =
                                    format!("Failed to parse SSE chunk: {}", e).into();
                                Some(Err(err))
                            }
                        }
                    } else {
                        None
                    }
                }
                Err(e) => Some(Err(e)),
            }
        })
}
