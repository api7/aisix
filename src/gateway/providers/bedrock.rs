use serde::{Deserialize, Serialize};

pub const IDENTIFIER: &str = "bedrock";

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct BedrockProviderConfig {
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::BedrockProviderConfig;

    #[test]
    fn bedrock_provider_config_deserializes_static_credentials() {
        let config: BedrockProviderConfig = serde_json::from_value(json!({
            "region": "us-east-1",
            "access_key_id": "AKIA123",
            "secret_access_key": "secret",
            "session_token": "token",
            "endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com"
        }))
        .unwrap();

        assert_eq!(config.region, "us-east-1");
        assert_eq!(config.access_key_id, "AKIA123");
        assert_eq!(config.secret_access_key, "secret");
        assert_eq!(config.session_token.as_deref(), Some("token"));
        assert_eq!(
            config.endpoint.as_deref(),
            Some("https://bedrock-runtime.us-east-1.amazonaws.com")
        );
    }
}
