use std::borrow::Cow;

use aisix_core::Model;
use aisix_gateway::ChatFormat;
use serde_json::Value;

/// Apply the final direct target's mapping to an OpenAI Chat Completions
/// request. The caller-owned request stays untouched so retries against a
/// different target always start from the original effort.
pub(crate) fn chat_request<'a>(request: &'a ChatFormat, model: &Model) -> Cow<'a, ChatFormat> {
    let Some(requested) = request
        .extra
        .get("reasoning_effort")
        .and_then(Value::as_str)
    else {
        return Cow::Borrowed(request);
    };
    let Some(mapped) = model.mapped_effort(requested) else {
        return Cow::Borrowed(request);
    };
    if mapped == requested {
        return Cow::Borrowed(request);
    }

    let mut outbound = request.clone();
    outbound
        .extra
        .insert("reasoning_effort".to_string(), mapped.into());
    Cow::Owned(outbound)
}

/// Apply the final direct target's mapping to an Anthropic Messages request.
pub(crate) fn anthropic_request<'a>(body: &'a Value, model: &Model) -> Cow<'a, Value> {
    json_request(body, model, "/output_config/effort")
}

/// Apply the final direct target's mapping to an OpenAI Responses request.
pub(crate) fn responses_request<'a>(body: &'a Value, model: &Model) -> Cow<'a, Value> {
    json_request(body, model, "/reasoning/effort")
}

fn json_request<'a>(body: &'a Value, model: &Model, pointer: &str) -> Cow<'a, Value> {
    let Some(requested) = body.pointer(pointer).and_then(Value::as_str) else {
        return Cow::Borrowed(body);
    };
    let Some(mapped) = model.mapped_effort(requested) else {
        return Cow::Borrowed(body);
    };
    if mapped == requested {
        return Cow::Borrowed(body);
    }

    let mut outbound = body.clone();
    *outbound
        .pointer_mut(pointer)
        .expect("effort pointer resolved before cloning") = Value::String(mapped.to_string());
    Cow::Owned(outbound)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use aisix_gateway::{ChatFormat, ChatMessage};
    use serde_json::json;

    use super::*;

    fn model() -> Model {
        serde_json::from_value(json!({
            "display_name": "glm",
            "provider": "openai",
            "model_name": "glm-5.3",
            "provider_key_id": "pk-1",
            "effort_mapping": {
                "medium": "high",
                "high": "max"
            }
        }))
        .unwrap()
    }

    #[test]
    fn maps_each_supported_request_shape_once() {
        let model = model();

        let mut chat = ChatFormat::new("glm", vec![ChatMessage::user("hi")]);
        chat.extra
            .insert("reasoning_effort".to_string(), "medium".into());
        let mapped = chat_request(&chat, &model);
        assert_eq!(mapped.extra["reasoning_effort"], "high");
        assert_eq!(chat.extra["reasoning_effort"], "medium");

        let messages = json!({
            "output_config": {"effort": "medium", "format": {"type": "json_schema"}}
        });
        let mapped = anthropic_request(&messages, &model);
        assert_eq!(mapped["output_config"]["effort"], "high");
        assert_eq!(mapped["output_config"]["format"]["type"], "json_schema");

        let responses = json!({"reasoning": {"effort": "medium", "summary": "auto"}});
        let mapped = responses_request(&responses, &model);
        assert_eq!(mapped["reasoning"]["effort"], "high");
        assert_eq!(mapped["reasoning"]["summary"], "auto");
    }

    #[test]
    fn does_not_chain_or_clone_for_an_unmapped_value() {
        let model = model();
        let body = json!({"reasoning": {"effort": "low"}});
        assert!(matches!(responses_request(&body, &model), Cow::Borrowed(_)));

        let body = json!({"reasoning": {"effort": "medium"}});
        let mapped = responses_request(&body, &model);
        assert_eq!(mapped["reasoning"]["effort"], "high");
    }

    #[test]
    fn leaves_missing_and_non_string_effort_untouched() {
        let model = model();
        for body in [json!({}), json!({"output_config": {"effort": 3}})] {
            assert!(matches!(anthropic_request(&body, &model), Cow::Borrowed(_)));
        }
    }
}
