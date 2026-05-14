//! aisix-provider-google — Google Gemini via its OpenAI-compatible endpoint.
//!
//! Google exposes an OpenAI-shaped `/chat/completions` surface at
//! `generativelanguage.googleapis.com/v1beta/openai`. The wire format is
//! close enough to plain OpenAI that the upstream `OpenAiBridge` covers
//! every field we care about — this crate only relabels the bridge
//! (`name() == "google"`) so metrics and logs can distinguish traffic.
//!
//! Operators configure Gemini access by setting on the Model:
//!
//! ```yaml
//! provider_config:
//!   api_key: "AIza…"
//!   api_base: "https://generativelanguage.googleapis.com/v1beta/openai"
//! ```
//!
//! The Gemini native `:generateContent` format (different request/response
//! shape, split role model) is intentionally out of scope here; routing
//! to it would belong in its own crate with its own wire module.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use aisix_provider_openai::OpenAiBridge;

/// Default base for Google's OpenAI-compat endpoint. Only used when the
/// Model doesn't carry an explicit `api_base` — production configs should
/// set one.
pub const GOOGLE_DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/openai";

/// Build a Bridge that speaks Google's OpenAI-compatible chat API.
pub fn google_bridge() -> OpenAiBridge {
    OpenAiBridge::new().with_name("google")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{Bridge, BridgeContext, ChatFormat, ChatMessage};
    use std::sync::Arc;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn bridge_reports_gemini_name() {
        assert_eq!(google_bridge().name(), "google");
    }

    #[test]
    fn default_base_targets_v1beta_openai_shim() {
        assert!(GOOGLE_DEFAULT_BASE.contains("/v1beta/openai"));
    }

    #[tokio::test]
    async fn forwards_chat_through_openai_transport() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer AIzaTEST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-gem",
                "model": "gemini-2.5-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ciao"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            })))
            .mount(&server)
            .await;

        let model: aisix_core::Model = serde_json::from_str(
            r#"{
                "display_name": "my-gemini",
                "provider": "google",
                "model_name": "gemini-2.5-flash",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#,
        )
        .unwrap();
        let pk_cfg = format!(
            r#"{{"display_name":"gemini-prod","secret":"AIzaTEST","api_base":"{uri}"}}"#,
            uri = server.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_cfg).unwrap();
        let ctx = BridgeContext::new("req-1", Arc::new(model), Arc::new(pk));
        let req = ChatFormat::new("my-gemini", vec![ChatMessage::user("hola")]);

        let resp = google_bridge().chat(&req, &ctx).await.unwrap();
        assert_eq!(resp.message.content, "ciao");
        assert_eq!(resp.usage.total_tokens, 4);
    }
}
