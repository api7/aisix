use bytes::{Bytes, BytesMut};
use futures::{Stream, stream::BoxStream};
use serde::Serialize;

use crate::{
    providers::ProviderError,
    proxy::types::{
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
        EmbeddingResponse,
    },
};

pub async fn chat_completion(
    client: reqwest::Client,
    url: &str,
    api_key: &str,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ProviderError> {
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
        return Err(ProviderError::ServiceError(status, error_text));
    }

    let completion = response.json::<ChatCompletionResponse>().await?;
    Ok(completion)
}

pub async fn chat_completion_stream<T: Serialize>(
    client: reqwest::Client,
    url: &str,
    api_key: &str,
    request: T,
) -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> {
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
        return Err(ProviderError::ServiceError(status, error_text));
    }

    Ok(Box::pin(parse_sse_stream(response.bytes_stream())))
}

pub async fn embedding(
    client: reqwest::Client,
    url: &str,
    api_key: &str,
    request: EmbeddingRequest,
) -> Result<EmbeddingResponse, ProviderError> {
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
        return Err(ProviderError::ServiceError(status, error_text));
    }

    let embedding = response.json::<EmbeddingResponse>().await?;
    Ok(embedding)
}

fn parse_sse_stream(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
) -> impl Stream<Item = Result<ChatCompletionChunk, ProviderError>> {
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
                Err(err) => futures::future::ready(Some(futures::stream::iter(vec![Err(
                    ProviderError::RequestError(err),
                )]))),
            }
        })
        .flatten()
        .filter_map(|line| async move {
            match line {
                Ok(line) => {
                    // Only process lines starting with "data: "
                    if let Some(json_str) = line.strip_prefix("data: ") {
                        // Skip [DONE] events
                        // Stream termination signifies completion; we do not need to explicitly propagate them.
                        if json_str == "[DONE]" {
                            return None;
                        }

                        match serde_json::from_str::<ChatCompletionChunk>(json_str) {
                            Ok(chunk) => Some(Ok(chunk)),
                            Err(err) => Some(Err(ProviderError::CodecError(err))),
                        }
                    } else {
                        None
                    }
                }
                Err(e) => Some(Err(e)),
            }
        })
}
