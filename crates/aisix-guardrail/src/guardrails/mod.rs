pub mod bedrock;
pub mod regex;

pub use self::{
    bedrock::{BedrockGuardrailMeta, BedrockGuardrailRuntime},
    regex::{RegexGuardrailMeta, RegexGuardrailRuntime},
};

pub mod identifiers {
    use super::{bedrock, regex};

    pub const BEDROCK: &str = bedrock::IDENTIFIER;
    pub const REGEX: &str = regex::IDENTIFIER;
}

pub mod configs {
    pub use super::{bedrock::BedrockGuardrailConfig, regex::RegexGuardrailConfig};
}
