use reqwest::Url;
use serde_json::{Map, Value};

use crate::{
    gateway::{
        traits::ProviderCapabilities,
        types::{
            common::Usage,
            embed::{EmbeddingRequest, EmbeddingResponse, OneOrMany},
        },
    },
    proxy::utils::trace::span_attributes::append_usage_properties,
};

pub(super) fn request_span_properties(
    request: &EmbeddingRequest,
    provider: &dyn ProviderCapabilities,
    base_url: Option<&Url>,
) -> Vec<(String, String)> {
    let provider_semantics = provider.semantic_conventions();
    let input_texts = request_input_texts(request);
    let mut properties = vec![
        ("gen_ai.operation.name".into(), "embeddings".into()),
        ("openinference.span.kind".into(), "EMBEDDING".into()),
        (
            "gen_ai.provider.name".into(),
            provider_semantics.gen_ai_provider_name.to_string(),
        ),
        ("gen_ai.request.model".into(), request.model.clone()),
        ("embedding.model_name".into(), request.model.clone()),
        ("input.mime_type".into(), "application/json".into()),
    ];

    if let Some(value) = encoding_formats_json(request.encoding_format.as_deref()) {
        properties.push(("gen_ai.request.encoding_formats".into(), value));
    }

    if let Some(value) = embedding_invocation_parameters(request) {
        properties.push(("embedding.invocation_parameters".into(), value));
    }

    if let Some(value) = serde_json::to_string(&request.input).ok() {
        properties.push(("input.value".into(), value));
    }

    if let Some(user_id) = request.user.as_ref().filter(|user_id| !user_id.is_empty()) {
        properties.push(("user.id".into(), user_id.clone()));
    }

    for (index, text) in input_texts.iter().enumerate() {
        properties.push((
            format!("embedding.embeddings.{index}.embedding.text"),
            text.clone(),
        ));
    }

    if let Some(base_url) = base_url {
        if let Some(address) = base_url.host_str() {
            properties.push(("server.address".into(), address.to_string()));
        }
        if let Some(port) = base_url.port_or_known_default() {
            properties.push(("server.port".into(), port.to_string()));
        }
    }

    properties
}

pub(super) fn response_span_properties(
    response: &EmbeddingResponse,
    usage: &Usage,
) -> Vec<(String, String)> {
    let mut properties = vec![
        ("gen_ai.response.model".into(), response.model.clone()),
        ("output.mime_type".into(), "application/json".into()),
    ];

    if let Some(first_embedding) = response.data.first() {
        properties.push((
            "gen_ai.embeddings.dimension.count".into(),
            first_embedding.embedding.len().to_string(),
        ));
    }

    if let Some(value) = serde_json::to_string(response).ok() {
        properties.push(("output.value".into(), value));
    }

    for (index, data) in response.data.iter().enumerate() {
        if let Ok(value) = serde_json::to_string(&data.embedding) {
            properties.push((
                format!("embedding.embeddings.{index}.embedding.vector"),
                value,
            ));
        }
    }

    append_usage_properties(&mut properties, usage);
    properties
}

fn request_input_texts(request: &EmbeddingRequest) -> Vec<String> {
    match &request.input {
        OneOrMany::One(value) => vec![value.clone()],
        OneOrMany::Many(values) => values.clone(),
    }
}

fn encoding_formats_json(encoding_format: Option<&str>) -> Option<String> {
    let encoding_format = encoding_format?.trim();
    if encoding_format.is_empty() {
        return None;
    }

    serde_json::to_string(&vec![encoding_format]).ok()
}

fn embedding_invocation_parameters(request: &EmbeddingRequest) -> Option<String> {
    let mut params = Map::new();
    params.insert("model".into(), Value::String(request.model.clone()));

    if let Some(value) = request.dimensions {
        params.insert("dimensions".into(), Value::from(value));
    }

    if let Some(value) = request.encoding_format.as_ref() {
        params.insert("encoding_format".into(), Value::String(value.clone()));
    }

    serde_json::to_string(&Value::Object(params)).ok()
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use reqwest::Url;
    use serde_json::{Value, json};

    use super::{request_span_properties, response_span_properties};
    use crate::gateway::{
        providers::openai::OpenAIDef,
        types::{
            common::Usage,
            embed::{EmbeddingRequest, EmbeddingResponse},
        },
    };

    fn property_value<'a>(properties: &'a [(String, String)], key: &str) -> Option<&'a str> {
        properties
            .iter()
            .find(|(property_key, _)| property_key == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn request_span_properties_follow_embedding_semantic_conventions() {
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "text-embedding-3-large",
            "input": ["hello", "world"],
            "dimensions": 256,
            "encoding_format": "float",
            "user": "user-123"
        }))
        .unwrap();
        let provider = OpenAIDef;
        let base_url = Url::parse("https://api.openai.com/v1").unwrap();

        let properties = request_span_properties(&request, &provider, Some(&base_url));

        assert_eq!(
            property_value(&properties, "gen_ai.operation.name"),
            Some("embeddings")
        );
        assert_eq!(
            property_value(&properties, "openinference.span.kind"),
            Some("EMBEDDING")
        );
        assert_eq!(
            property_value(&properties, "gen_ai.provider.name"),
            Some("openai")
        );
        assert_eq!(
            property_value(&properties, "gen_ai.request.model"),
            Some("text-embedding-3-large")
        );
        assert_eq!(
            property_value(&properties, "embedding.model_name"),
            Some("text-embedding-3-large")
        );
        assert_eq!(
            property_value(&properties, "input.mime_type"),
            Some("application/json")
        );
        assert_eq!(property_value(&properties, "user.id"), Some("user-123"));
        assert_eq!(
            property_value(&properties, "embedding.embeddings.0.embedding.text"),
            Some("hello")
        );
        assert_eq!(
            property_value(&properties, "embedding.embeddings.1.embedding.text"),
            Some("world")
        );
        assert_eq!(property_value(&properties, "server.address"), Some("api.openai.com"));
        assert_eq!(property_value(&properties, "server.port"), Some("443"));
        assert_eq!(property_value(&properties, "llm.system"), None);
        assert_eq!(property_value(&properties, "llm.provider"), None);

        let input_value: Value =
            serde_json::from_str(property_value(&properties, "input.value").unwrap()).unwrap();
        assert_eq!(input_value, json!(["hello", "world"]));

        let encoding_formats: Value = serde_json::from_str(
            property_value(&properties, "gen_ai.request.encoding_formats").unwrap(),
        )
        .unwrap();
        assert_eq!(encoding_formats, json!(["float"]));

        let invocation_parameters: Value = serde_json::from_str(
            property_value(&properties, "embedding.invocation_parameters").unwrap(),
        )
        .unwrap();
        assert_eq!(
            invocation_parameters,
            json!({
                "model": "text-embedding-3-large",
                "dimensions": 256,
                "encoding_format": "float"
            })
        );
    }

    #[test]
    fn response_span_properties_include_vectors_and_usage() {
        let response: EmbeddingResponse = serde_json::from_value(json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": [0.1, 0.2],
                "index": 0
            }, {
                "object": "embedding",
                "embedding": [0.3, 0.4],
                "index": 1
            }],
            "model": "text-embedding-3-large",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }))
        .unwrap();
        let usage = Usage {
            input_tokens: Some(8),
            total_tokens: Some(8),
            ..Default::default()
        };

        let properties = response_span_properties(&response, &usage);

        assert_eq!(
            property_value(&properties, "gen_ai.response.model"),
            Some("text-embedding-3-large")
        );
        assert_eq!(
            property_value(&properties, "output.mime_type"),
            Some("application/json")
        );
        assert_eq!(
            property_value(&properties, "gen_ai.embeddings.dimension.count"),
            Some("2")
        );
        assert_eq!(property_value(&properties, "llm.token_count.prompt"), Some("8"));
        assert_eq!(property_value(&properties, "llm.token_count.total"), Some("8"));

        let output_value: Value =
            serde_json::from_str(property_value(&properties, "output.value").unwrap()).unwrap();
        assert_eq!(output_value["model"], "text-embedding-3-large");

        let vector0: Value = serde_json::from_str(
            property_value(&properties, "embedding.embeddings.0.embedding.vector").unwrap(),
        )
        .unwrap();
        let vector1: Value = serde_json::from_str(
            property_value(&properties, "embedding.embeddings.1.embedding.vector").unwrap(),
        )
        .unwrap();
        assert_eq!(vector0, json!([0.1, 0.2]));
        assert_eq!(vector1, json!([0.3, 0.4]));
    }
}