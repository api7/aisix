pub mod bedrock;

pub use bedrock::{BedrockGuardrailMeta, BedrockGuardrailRuntime};

pub mod identifiers {
    use super::bedrock;

    pub const BEDROCK: &str = bedrock::IDENTIFIER;
}

pub mod configs {
    pub use super::bedrock::BedrockGuardrailConfig;
}
