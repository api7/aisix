//! The output-token cap rename for OpenAI's reasoning models
//! (AISIX-Cloud#936).
//!
//! OpenAI's o-series and gpt-5 family reject `max_tokens` on Chat
//! Completions and take the cap as `max_completion_tokens` instead
//! (<https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens>),
//! so a client that sends the long-standing `max_tokens` would get a 400.
//! Both OpenAI-wire bridges apply this before the provider key's
//! operator-configured `param_renames`, so an operator's explicit rename
//! still has the last word.
//!
//! The control plane's Playground classifies models with the same rule;
//! the two must stay identical. Matching is literal and case-sensitive.

use serde_json::Value;

/// Which adapter's naming convention to classify the model name by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFamily {
    /// `adapter: openai`.
    Openai,
    /// `adapter: azure-openai`, where the name is usually an
    /// operator-chosen deployment name.
    AzureOpenai,
}

/// Whether `upstream_model` names a reasoning model under `family`'s
/// convention. Only the last `/`-separated segment is classified.
pub fn is_reasoning_model(family: ReasoningFamily, upstream_model: &str) -> bool {
    let name = upstream_model.rsplit('/').next().unwrap_or(upstream_model);
    let gpt_reasoning = name.contains("gpt-5") || name.contains("gpt-6");
    match family {
        ReasoningFamily::Openai => {
            let mut chars = name.chars();
            let o_series =
                chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit());
            o_series || (gpt_reasoning && !name.starts_with("gpt-5-chat"))
        }
        ReasoningFamily::AzureOpenai => {
            name.contains("o1") || name.contains("o3") || name.contains("o4") || gpt_reasoning
        }
    }
}

/// Move `max_tokens` to `max_completion_tokens`; when the caller sent
/// both, the explicit `max_completion_tokens` wins and `max_tokens` is
/// dropped. Every other field is left as it is.
pub fn apply_reasoning_token_cap(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let Some(max_tokens) = obj.remove("max_tokens") else {
        return;
    };
    obj.entry("max_completion_tokens").or_insert(max_tokens);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_classification() {
        use ReasoningFamily::Openai;
        for reasoning in [
            "o1",
            "o3-mini",
            "o4-mini-2025-04-16",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1-codex",
            "gpt-6",
            "openai/o3",
            "org/team/gpt-5-nano",
        ] {
            assert!(is_reasoning_model(Openai, reasoning), "{reasoning}");
        }
        for plain in [
            "gpt-4o",
            "gpt-4.1",
            "gpt-5-chat-latest",
            "openai/gpt-5-chat",
            "o",
            "omni-moderation-latest",
            "my-o3-deployment",
            "O3",
            "GPT-5",
        ] {
            assert!(!is_reasoning_model(Openai, plain), "{plain}");
        }
    }

    #[test]
    fn azure_openai_classification() {
        use ReasoningFamily::AzureOpenai;
        for reasoning in [
            "o1",
            "my-o3-deployment",
            "prod-o4-mini",
            "gpt-5",
            "gpt-5-chat",
            "team-gpt-6",
            "deployments/o1-preview",
        ] {
            assert!(is_reasoning_model(AzureOpenai, reasoning), "{reasoning}");
        }
        for plain in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-35-turbo",
            "O3-DEPLOY",
            "GPT-5",
        ] {
            assert!(!is_reasoning_model(AzureOpenai, plain), "{plain}");
        }
    }

    #[test]
    fn token_cap_renames_max_tokens() {
        let mut body = json!({"model": "o3", "max_tokens": 64, "temperature": 0.2});
        apply_reasoning_token_cap(&mut body);
        assert_eq!(
            body,
            json!({"model": "o3", "max_completion_tokens": 64, "temperature": 0.2})
        );
    }

    #[test]
    fn token_cap_keeps_an_explicit_max_completion_tokens() {
        let mut body = json!({"max_tokens": 64, "max_completion_tokens": 128});
        apply_reasoning_token_cap(&mut body);
        assert_eq!(body, json!({"max_completion_tokens": 128}));
    }

    #[test]
    fn token_cap_is_a_no_op_without_max_tokens() {
        let mut body = json!({"max_completion_tokens": 128, "temperature": 0.5});
        apply_reasoning_token_cap(&mut body);
        assert_eq!(
            body,
            json!({"max_completion_tokens": 128, "temperature": 0.5})
        );
    }
}
