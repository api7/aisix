pub mod anthropic;
pub mod azure;
pub mod bedrock;
pub mod deepseek;
pub mod gemini;
pub mod groq;
pub mod macros;
pub mod mistral;
pub mod openai;
pub mod openrouter;

pub use anthropic::AnthropicDef;
pub use azure::AzureDef;
pub use bedrock::BedrockDef;
pub use deepseek::DeepSeek;
pub use gemini::GoogleDef;
pub use groq::Groq;
pub use mistral::Mistral;
pub use openai::OpenAIDef;
pub use openrouter::OpenRouter;

pub mod identifiers {
    use super::{anthropic, azure, bedrock, deepseek, gemini, groq, mistral, openai, openrouter};

    pub const ANTHROPIC: &str = anthropic::IDENTIFIER;
    pub const AZURE: &str = azure::IDENTIFIER;
    pub const BEDROCK: &str = bedrock::IDENTIFIER;
    pub const DEEPSEEK: &str = deepseek::IDENTIFIER;
    pub const GEMINI: &str = gemini::IDENTIFIER;
    pub const GROQ: &str = groq::IDENTIFIER;
    pub const MISTRAL: &str = mistral::IDENTIFIER;
    pub const OPENAI: &str = openai::IDENTIFIER;
    pub const OPENROUTER: &str = openrouter::IDENTIFIER;
}

pub mod configs {
    pub use super::{
        anthropic::AnthropicProviderConfig, azure::AzureProviderConfig,
        bedrock::BedrockProviderConfig, deepseek::DeepSeekProviderConfig,
        gemini::GeminiProviderConfig, groq::GroqProviderConfig, mistral::MistralProviderConfig,
        openai::OpenAIProviderConfig, openrouter::OpenRouterProviderConfig,
    };
}

use crate::gateway::{error::Result, provider_instance::ProviderRegistry};

pub fn default_provider_registry() -> Result<ProviderRegistry> {
    let builder = ProviderRegistry::builder()
        .register(AnthropicDef)?
        .register(AzureDef)?
        .register(BedrockDef)?
        .register(DeepSeek)?
        .register(GoogleDef)?
        .register(Groq)?
        .register(Mistral)?
        .register(OpenAIDef)?
        .register(OpenRouter)?;
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::default_provider_registry;

    #[test]
    fn default_provider_registry_registers_builtin_providers() {
        let registry = default_provider_registry().unwrap();

        assert_eq!(registry.get("openai").unwrap().name(), "openai");
        assert_eq!(registry.get("azure").unwrap().name(), "azure");
        assert_eq!(registry.get("anthropic").unwrap().name(), "anthropic");
        assert_eq!(registry.get("bedrock").unwrap().name(), "bedrock");
        assert_eq!(registry.get("gemini").unwrap().name(), "gemini");
        assert_eq!(registry.get("groq").unwrap().name(), "groq");
        assert_eq!(registry.get("mistral").unwrap().name(), "mistral");
        assert_eq!(registry.get("deepseek").unwrap().name(), "deepseek");
        assert_eq!(registry.get("openrouter").unwrap().name(), "openrouter");
        assert!(registry.get("missing").is_none());
    }
}
