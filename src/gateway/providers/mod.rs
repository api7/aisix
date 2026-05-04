pub mod anthropic;
pub mod azure;
pub mod bedrock;
pub mod cohere;
pub mod deepseek;
pub mod fireworks;
pub mod gemini;
pub mod groq;
pub mod macros;
pub mod mistral;
pub mod modelscope;
pub mod moonshot;
pub mod openai;
pub mod openrouter;
pub mod siliconflow;
pub mod stepfun;
pub mod xai;
pub mod zhipuai;

pub use anthropic::AnthropicDef;
pub use azure::AzureDef;
pub use bedrock::BedrockDef;
pub use cohere::Cohere;
pub use deepseek::DeepSeek;
pub use fireworks::FireworksAi;
pub use gemini::GoogleDef;
pub use groq::Groq;
pub use mistral::Mistral;
pub use modelscope::{ModelScope, ModelScopeCn};
pub use moonshot::{MoonshotAi, MoonshotAiCn};
pub use openai::OpenAIDef;
pub use openrouter::OpenRouter;
pub use siliconflow::{SiliconFlow, SiliconFlowCn};
pub use stepfun::StepFun;
pub use xai::Xai;
pub use zhipuai::ZhipuAi;

pub mod identifiers {
    use super::{
        anthropic, azure, bedrock, cohere, deepseek, fireworks, gemini, groq, mistral, modelscope,
        moonshot, openai, openrouter, siliconflow, stepfun, xai, zhipuai,
    };

    pub const ANTHROPIC: &str = anthropic::IDENTIFIER;
    pub const AZURE: &str = azure::IDENTIFIER;
    pub const BEDROCK: &str = bedrock::IDENTIFIER;
    pub const COHERE: &str = cohere::IDENTIFIER;
    pub const DEEPSEEK: &str = deepseek::IDENTIFIER;
    pub const FIREWORKS_AI: &str = fireworks::IDENTIFIER;
    pub const GEMINI: &str = gemini::IDENTIFIER;
    pub const GROQ: &str = groq::IDENTIFIER;
    pub const MISTRAL: &str = mistral::IDENTIFIER;
    pub const MODELSCOPE: &str = modelscope::IDENTIFIER;
    pub const MODELSCOPE_CN: &str = modelscope::CN_IDENTIFIER;
    pub const SILICONFLOW: &str = siliconflow::IDENTIFIER;
    pub const SILICONFLOW_CN: &str = siliconflow::CN_IDENTIFIER;
    pub const STEPFUN: &str = stepfun::IDENTIFIER;
    pub const MOONSHOT_AI: &str = moonshot::IDENTIFIER;
    pub const MOONSHOT_AI_CN: &str = moonshot::CN_IDENTIFIER;
    pub const OPENAI: &str = openai::IDENTIFIER;
    pub const OPENROUTER: &str = openrouter::IDENTIFIER;
    pub const XAI: &str = xai::IDENTIFIER;
    pub const ZHIPUAI: &str = zhipuai::IDENTIFIER;
}

pub mod configs {
    pub use super::{
        anthropic::AnthropicProviderConfig,
        azure::AzureProviderConfig,
        bedrock::BedrockProviderConfig,
        cohere::CohereProviderConfig,
        deepseek::DeepSeekProviderConfig,
        fireworks::FireworksAiProviderConfig,
        gemini::GeminiProviderConfig,
        groq::GroqProviderConfig,
        mistral::MistralProviderConfig,
        modelscope::{ModelScopeCnProviderConfig, ModelScopeProviderConfig},
        moonshot::{MoonshotAiCnProviderConfig, MoonshotAiProviderConfig},
        openai::OpenAIProviderConfig,
        openrouter::OpenRouterProviderConfig,
        siliconflow::{SiliconFlowCnProviderConfig, SiliconFlowProviderConfig},
        stepfun::StepFunProviderConfig,
        xai::XaiProviderConfig,
        zhipuai::ZhipuAiProviderConfig,
    };
}

use crate::gateway::{error::Result, provider_instance::ProviderRegistry};

pub fn default_provider_registry() -> Result<ProviderRegistry> {
    let builder = ProviderRegistry::builder()
        .register(AnthropicDef)?
        .register(AzureDef)?
        .register(BedrockDef)?
        .register(Cohere)?
        .register(DeepSeek)?
        .register(FireworksAi)?
        .register(GoogleDef)?
        .register(Groq)?
        .register(Mistral)?
        .register(ModelScope)?
        .register(ModelScopeCn)?
        .register(SiliconFlow)?
        .register(SiliconFlowCn)?
        .register(StepFun)?
        .register(MoonshotAi)?
        .register(MoonshotAiCn)?
        .register(OpenAIDef)?
        .register(OpenRouter)?
        .register(Xai)?
        .register(ZhipuAi)?;
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::default_provider_registry;

    #[test]
    fn default_provider_registry_registers_builtin_providers() {
        let registry = default_provider_registry().unwrap();

        assert_eq!(registry.get("openai").unwrap().name(), "openai");
        assert_eq!(registry.get("azure").unwrap().name(), "azure");
        assert_eq!(registry.get("anthropic").unwrap().name(), "anthropic");
        assert_eq!(registry.get("bedrock").unwrap().name(), "bedrock");
        assert_eq!(registry.get("cohere").unwrap().name(), "cohere");
        assert_eq!(registry.get("fireworks-ai").unwrap().name(), "fireworks-ai");
        assert_eq!(registry.get("gemini").unwrap().name(), "gemini");
        assert_eq!(registry.get("groq").unwrap().name(), "groq");
        assert_eq!(registry.get("mistral").unwrap().name(), "mistral");
        assert_eq!(registry.get("modelscope").unwrap().name(), "modelscope");
        assert_eq!(
            registry.get("modelscope-cn").unwrap().name(),
            "modelscope-cn"
        );
        assert_eq!(registry.get("siliconflow").unwrap().name(), "siliconflow");
        assert_eq!(
            registry.get("siliconflow-cn").unwrap().name(),
            "siliconflow-cn"
        );
        assert_eq!(registry.get("stepfun").unwrap().name(), "stepfun");
        assert_eq!(registry.get("moonshotai").unwrap().name(), "moonshotai");
        assert_eq!(
            registry.get("moonshotai-cn").unwrap().name(),
            "moonshotai-cn"
        );
        assert_eq!(registry.get("deepseek").unwrap().name(), "deepseek");
        assert_eq!(registry.get("openrouter").unwrap().name(), "openrouter");
        assert_eq!(registry.get("xai").unwrap().name(), "xai");
        assert_eq!(registry.get("zhipuai").unwrap().name(), "zhipuai");
        assert!(registry.get("missing").is_none());
    }
}
