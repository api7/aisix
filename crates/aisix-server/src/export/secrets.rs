//! Secret redaction for `aisix export`.
//!
//! Stored resource documents hold live upstream credentials in the
//! clear — a provider key's `api_key` and per-key request overrides, an
//! MCP / A2A server's `secret`, a guardrail's moderation-service
//! `api_key` / `access_key_secret` / `secret_access_key`, an OTLP
//! exporter's auth `headers`. The default export must never write any of
//! these verbatim: each is replaced with an `${ENV_VAR}` placeholder the
//! resources file interpolates at load time, and the operator populates
//! the variable out of band.
//!
//! The placeholder name is derived deterministically from the entry's
//! identity and the field so it is stable across exports and greppable.
//! It deliberately does **not** start with `AISIX_`: the gateway's
//! config loader claims that prefix (`Environment::with_prefix("AISIX")`)
//! and rejects unknown keys, so an `AISIX_…` secret variable set in the
//! data plane's environment would be misread as a bad config override
//! and fail boot. The same reason the e2e harness and the codebase's own
//! `SLS_CRED_…` / `OBJSTORE_CRED_…` conventions keep secret variables off
//! the `AISIX_` prefix.

use serde_json::Value;

/// Namespace every derived secret variable shares. Config-safe (does not
/// begin with the `AISIX_` config-override prefix) and greppable.
pub const SECRET_ENV_PREFIX: &str = "AISIXSECRET";

/// One placeholder emitted in place of a live credential. Collected so
/// the command can print the operator a "set these before loading" list
/// on stderr — never into the file itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretPlaceholder {
    /// Environment variable the resources file now references.
    pub env_var: String,
    /// Resource kind the credential belongs to (`provider_keys`, …).
    pub kind: &'static str,
    /// The entry's file identity (display_name / name).
    pub identity: String,
    /// Human label for the redacted field (e.g. `api_key`, `header x-…`).
    pub field: String,
}

/// Shared context every redaction helper carries: which kind / entry is
/// being redacted, whether to reveal instead of redact, and the sink to
/// record placeholders in. Bundled into one struct so the helpers stay
/// two- to four-argument and the same-typed `kind_token` / `kind` /
/// `identity` strings cannot be transposed at a call site.
pub struct RedactionCtx<'a> {
    /// Uppercased kind token in the env-var name (e.g. `PROVIDER_KEY`).
    pub kind_token: &'a str,
    /// Canonical kind name recorded on the placeholder (`provider_keys`).
    pub kind: &'static str,
    /// The entry's file identity.
    pub identity: &'a str,
    /// Emit the real values inline instead of placeholders.
    pub reveal: bool,
    /// Where recorded placeholders accumulate.
    pub out: &'a mut Vec<SecretPlaceholder>,
}

impl RedactionCtx<'_> {
    /// Record a placeholder and return its interpolation value.
    fn placeholder(&mut self, field_key: &str, field_label: String) -> Value {
        let env_var = secret_var(self.kind_token, self.identity, field_key);
        self.out.push(SecretPlaceholder {
            env_var: env_var.clone(),
            kind: self.kind,
            identity: self.identity.to_string(),
            field: field_label,
        });
        Value::String(format!("${{{env_var}}}"))
    }
}

/// Uppercase a label and replace every non-alphanumeric byte with `_`
/// so it is a legal, stable environment-variable fragment. Not collapsed
/// or trimmed — determinism matters more than prettiness, and the raw
/// shape stays recognizable (`openai-prod` → `OPENAI_PROD`).
pub fn sanitize(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// `AISIXSECRET_<KIND>_<IDENTITY>_<FIELD>` — the derived variable name.
pub fn secret_var(kind_token: &str, identity: &str, field: &str) -> String {
    format!(
        "{SECRET_ENV_PREFIX}_{kind_token}_{}_{}",
        sanitize(identity),
        sanitize(field),
    )
}

/// True when `value` is a non-empty string.
fn is_nonempty_str(value: &Value) -> bool {
    value.as_str().is_some_and(|s| !s.is_empty())
}

/// Replace one top-level string field with a placeholder. No-op when
/// `reveal` is set, when the field is absent, or when it is not a
/// non-empty string.
pub fn redact_top_level(doc: &mut Value, field: &str, ctx: &mut RedactionCtx<'_>) {
    if ctx.reveal {
        return;
    }
    let Some(slot) = doc.get_mut(field).filter(|v| is_nonempty_str(v)) else {
        return;
    };
    *slot = ctx.placeholder(field, field.to_string());
}

/// Recursively replace every non-empty string whose object key is in
/// `secret_keys`, at any depth. Used for guardrails, whose credential
/// fields (`api_key`, `access_key_secret`, and the Bedrock
/// `aws_credentials.secret_access_key` one level down) are always
/// secrets in that kind's closed schema. No-op when `reveal` is set.
pub fn redact_by_key(value: &mut Value, secret_keys: &[&str], ctx: &mut RedactionCtx<'_>) {
    if ctx.reveal {
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if secret_keys.contains(&key.as_str()) && is_nonempty_str(child) {
                    *child = ctx.placeholder(key, key.clone());
                } else {
                    redact_by_key(child, secret_keys, ctx);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_by_key(item, secret_keys, ctx);
            }
        }
        _ => {}
    }
}

/// Replace every value under a top-level `headers` object (an OTLP
/// exporter's request headers, which routinely carry a vendor auth
/// token) with a per-header placeholder. No-op when `reveal` is set or
/// there is no `headers` object.
pub fn redact_headers(doc: &mut Value, ctx: &mut RedactionCtx<'_>) {
    if ctx.reveal {
        return;
    }
    let Some(Value::Object(headers)) = doc.get_mut("headers") else {
        return;
    };
    for (name, value) in headers.iter_mut() {
        if is_nonempty_str(value) {
            let field_key = format!("HEADER_{}", sanitize(name));
            *value = ctx.placeholder(&field_key, format!("header {name}"));
        }
    }
}

/// Replace every non-empty **string** value under `doc[field]` (an object)
/// with a per-entry placeholder. Non-string values are left intact, so a
/// `default_body_fields: { safe_prompt: true }` flag survives while a
/// string credential in the same map is redacted. Used for a provider
/// key's `request.default_headers` (outbound auth headers to the upstream)
/// and `request.default_body_fields`, whose values are operator-supplied
/// and may carry a secondary credential the dedicated `api_key` field does
/// not cover. No-op when `reveal` is set or `field` is absent / not an
/// object.
pub fn redact_string_map(doc: &mut Value, field: &str, label: &str, ctx: &mut RedactionCtx<'_>) {
    if ctx.reveal {
        return;
    }
    let Some(Value::Object(map)) = doc.get_mut(field) else {
        return;
    };
    for (name, value) in map.iter_mut() {
        if is_nonempty_str(value) {
            let field_key = format!("{}_{}", sanitize(field), sanitize(name));
            *value = ctx.placeholder(&field_key, format!("{label} {name}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a redaction context over a fresh placeholder sink.
    fn ctx<'a>(
        identity: &'a str,
        reveal: bool,
        out: &'a mut Vec<SecretPlaceholder>,
    ) -> RedactionCtx<'a> {
        RedactionCtx {
            kind_token: "PROVIDER_KEY",
            kind: "provider_keys",
            identity,
            reveal,
            out,
        }
    }

    #[test]
    fn sanitize_uppercases_and_replaces_non_alphanumeric() {
        assert_eq!(sanitize("openai-prod"), "OPENAI_PROD");
        assert_eq!(sanitize("gpt-4o"), "GPT_4O");
        assert_eq!(sanitize("x.honeycomb/team"), "X_HONEYCOMB_TEAM");
    }

    #[test]
    fn secret_var_is_deterministic_and_prefixed() {
        let a = secret_var("PROVIDER_KEY", "openai-prod", "api_key");
        assert_eq!(a, "AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY");
        assert_eq!(a, secret_var("PROVIDER_KEY", "openai-prod", "api_key"));
        // Never the reserved config-override prefix.
        assert!(!a.starts_with("AISIX_"));
    }

    #[test]
    fn redact_top_level_replaces_value_and_records_placeholder() {
        let mut doc = json!({"display_name": "openai-prod", "api_key": "sk-live-SECRET"});
        let mut out = Vec::new();
        redact_top_level(
            &mut doc,
            "api_key",
            &mut ctx("openai-prod", false, &mut out),
        );
        assert_eq!(
            doc["api_key"],
            json!("${AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY}")
        );
        assert!(!doc.to_string().contains("sk-live-SECRET"));
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].env_var,
            "AISIXSECRET_PROVIDER_KEY_OPENAI_PROD_API_KEY"
        );
        assert_eq!(out[0].field, "api_key");
    }

    #[test]
    fn redact_top_level_reveal_leaves_value_untouched() {
        let mut doc = json!({"api_key": "sk-live-SECRET"});
        let mut out = Vec::new();
        redact_top_level(&mut doc, "api_key", &mut ctx("pk", true, &mut out));
        assert_eq!(doc["api_key"], json!("sk-live-SECRET"));
        assert!(out.is_empty());
    }

    #[test]
    fn redact_top_level_ignores_absent_or_empty_field() {
        let mut out = Vec::new();
        let mut absent = json!({"display_name": "pk"});
        redact_top_level(&mut absent, "api_key", &mut ctx("pk", false, &mut out));
        let mut empty = json!({"api_key": ""});
        redact_top_level(&mut empty, "api_key", &mut ctx("pk", false, &mut out));
        assert!(out.is_empty());
        assert_eq!(empty["api_key"], json!(""));
    }

    #[test]
    fn redact_by_key_reaches_nested_bedrock_secret() {
        // The Bedrock guardrail nests its secret one level down under
        // aws_credentials; the recursive walk must still catch it while
        // leaving the sibling access_key_id (an identifier, not a secret)
        // alone.
        let mut doc = json!({
            "name": "bedrock-guard",
            "kind": "bedrock",
            "api_key": "top-secret",
            "aws_credentials": {
                "kind": "static",
                "access_key_id": "AKIAEXAMPLE",
                "secret_access_key": "wJalr-SECRET"
            }
        });
        let mut out = Vec::new();
        let mut c = RedactionCtx {
            kind_token: "GUARDRAIL",
            kind: "guardrails",
            identity: "bedrock-guard",
            reveal: false,
            out: &mut out,
        };
        redact_by_key(
            &mut doc,
            &["api_key", "access_key_secret", "secret_access_key"],
            &mut c,
        );
        let text = doc.to_string();
        assert!(!text.contains("top-secret"), "{text}");
        assert!(!text.contains("wJalr-SECRET"), "{text}");
        // access_key_id is not in the secret set — preserved verbatim.
        assert_eq!(
            doc["aws_credentials"]["access_key_id"],
            json!("AKIAEXAMPLE")
        );
        assert_eq!(
            doc["aws_credentials"]["secret_access_key"],
            json!("${AISIXSECRET_GUARDRAIL_BEDROCK_GUARD_SECRET_ACCESS_KEY}")
        );
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn redact_string_map_masks_string_values_and_keeps_non_strings() {
        let mut request = json!({
            "default_headers": { "x-tenant-token": "tok-SECRET", "x-request-source": "" },
            "default_body_fields": { "api_key": "body-SECRET", "safe_prompt": true }
        });
        let mut out = Vec::new();
        let mut c = ctx("pk", false, &mut out);
        redact_string_map(&mut request, "default_headers", "default header", &mut c);
        redact_string_map(
            &mut request,
            "default_body_fields",
            "default body field",
            &mut c,
        );
        let text = request.to_string();
        assert!(!text.contains("tok-SECRET"), "{text}");
        assert!(!text.contains("body-SECRET"), "{text}");
        assert_eq!(
            request["default_headers"]["x-tenant-token"],
            json!("${AISIXSECRET_PROVIDER_KEY_PK_DEFAULT_HEADERS_X_TENANT_TOKEN}")
        );
        // Empty string skipped; non-string (bool) left intact.
        assert_eq!(request["default_headers"]["x-request-source"], json!(""));
        assert_eq!(request["default_body_fields"]["safe_prompt"], json!(true));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn redact_headers_masks_each_value() {
        let mut doc = json!({
            "name": "honeycomb",
            "kind": "otlp_http",
            "endpoint": "https://api.honeycomb.io/v1/traces",
            "headers": { "x-honeycomb-team": "hcaik_SECRET" }
        });
        let mut out = Vec::new();
        let mut c = RedactionCtx {
            kind_token: "OBSERVABILITY_EXPORTER",
            kind: "observability_exporters",
            identity: "honeycomb",
            reveal: false,
            out: &mut out,
        };
        redact_headers(&mut doc, &mut c);
        assert!(!doc.to_string().contains("hcaik_SECRET"));
        assert_eq!(
            doc["headers"]["x-honeycomb-team"],
            json!("${AISIXSECRET_OBSERVABILITY_EXPORTER_HONEYCOMB_HEADER_X_HONEYCOMB_TEAM}")
        );
        assert_eq!(out.len(), 1);
        // The endpoint (not a credential) is untouched.
        assert_eq!(doc["endpoint"], json!("https://api.honeycomb.io/v1/traces"));
    }
}
