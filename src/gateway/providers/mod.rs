pub mod deepseek;
pub mod macros;
pub mod openai;

pub use deepseek::DeepSeek;
pub use openai::OpenAIDef;

use crate::gateway::{error::Result, provider_instance::ProviderRegistry};

pub fn default_provider_registry() -> Result<ProviderRegistry> {
    let builder = ProviderRegistry::builder()
        .register(OpenAIDef)?
        .register(DeepSeek)?;
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::default_provider_registry;

    #[test]
    fn default_provider_registry_registers_builtin_providers() {
        let registry = default_provider_registry().unwrap();

        assert_eq!(registry.get("openai").unwrap().name(), "openai");
        assert_eq!(registry.get("deepseek").unwrap().name(), "deepseek");
        assert!(registry.get("missing").is_none());
    }
}
