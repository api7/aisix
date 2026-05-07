//! aisix-provider-deepseek — DeepSeek via its OpenAI-compatible endpoint.
//!
//! DeepSeek's chat API is a straight `/chat/completions` clone, so this
//! crate is a one-function factory around `aisix_provider_openai::OpenAiBridge`
//! with `name()` overridden to `"deepseek"` for metrics and log labels.
//!
//! The Model entity drives the actual base URL and API key — DeepSeek
//! users typically configure `api_base: "https://api.deepseek.com"` on
//! the Model.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use aisix_provider_openai::OpenAiBridge;

/// Default DeepSeek host. Only used if a Model is missing an `api_base`
/// override; operators should set one explicitly.
pub const DEEPSEEK_DEFAULT_BASE: &str = "https://api.deepseek.com";

/// Build a Bridge that speaks DeepSeek's OpenAI-compatible chat API.
///
/// Returns the underlying [`OpenAiBridge`] value so callers can still
/// reach `Bridge` methods — `Arc::new(deepseek_bridge()) as Arc<dyn Bridge>`
/// is how the Hub registers it at startup.
pub fn deepseek_bridge() -> OpenAiBridge {
    OpenAiBridge::new().with_name("deepseek")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{Bridge, BridgeContext, ChatFormat, ChatMessage};
    use std::sync::Arc;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn bridge_reports_deepseek_name() {
        let b = deepseek_bridge();
        assert_eq!(b.name(), "deepseek");
    }

    #[test]
    fn default_base_points_at_deepseek_host() {
        assert_eq!(DEEPSEEK_DEFAULT_BASE, "https://api.deepseek.com");
    }

    #[tokio::test]
    async fn forwards_chat_through_openai_transport() {
        // DeepSeek is OpenAI-compatible, so the wiremock expectation is
        // exactly an OpenAI /chat/completions call — proving this crate
        // is a thin relabel rather than a parallel transport.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer ds-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-ds",
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&server)
            .await;

        let model: aisix_core::Model = serde_json::from_str(
            r#"{
                "display_name": "my-deepseek",
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#,
        )
        .unwrap();
        let pk_cfg = format!(
            r#"{{"display_name":"ds-prod","secret":"ds-test","api_base":"{uri}"}}"#,
            uri = server.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_cfg).unwrap();
        let ctx = BridgeContext::new("req-1", Arc::new(model), Arc::new(pk));
        let req = ChatFormat::new("my-deepseek", vec![ChatMessage::user("ping")]);

        let resp = deepseek_bridge().chat(&req, &ctx).await.unwrap();
        assert_eq!(resp.message.content, "pong");
        assert_eq!(resp.usage.total_tokens, 2);
    }
}
