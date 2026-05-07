//! Hand-written OpenAPI 3.1 document + Scalar mount.
//!
//! Covers every route mounted in `crate::build_router` so what you see
//! at `/admin/openapi-scalar` matches what the binary actually serves.
//! The proxy `/v1/chat/completions` surface is OpenAI-compatible and
//! operators refer to OpenAI's published spec for that — duplicating
//! it here adds drift risk without adding signal.
//!
//! The Scalar UI is a single static HTML page that loads the JSON spec
//! over HTTP — no JS bundling required.

use axum::http::header;
use axum::response::{Html, IntoResponse, Response};

/// Hand-written JSON spec. Small enough that maintaining it by hand is
/// less effort than wiring `utoipa` derive macros across every handler;
/// the surface is stable enough that drift is easy to spot in review.
/// Update this whenever a route is added/removed in `lib.rs`.
const OPENAPI_JSON: &str = r##"{
  "openapi": "3.1.0",
  "info": {
    "title": "aisix admin API",
    "version": "0.1.0",
    "description": "Admin surface for the standalone aisix gateway. All `/admin/v1/*` routes require Bearer admin-key auth (configured via `admin.admin_keys`). Errors use {\"error_msg\": \"...\"}.\n\nIn managed mode (aisix.cloud tenant) the admin listener is not bound — the dashboard owns CRUD via the AISIX-Cloud control plane."
  },
  "paths": {
    "/health": {
      "get": {
        "summary": "process health + snapshot counts",
        "security": [],
        "responses": {"200": {"description": "OK"}}
      }
    },
    "/metrics": {
      "get": {
        "summary": "Prometheus metrics (text/plain; version=0.0.4)",
        "security": [],
        "responses": {
          "200": {"description": "OK"},
          "503": {"description": "metrics recorder not configured"}
        }
      }
    },
    "/admin/openapi.json": {
      "get": {
        "summary": "this OpenAPI document",
        "security": [],
        "responses": {"200": {"description": "OK"}}
      }
    },
    "/admin/openapi-scalar": {
      "get": {
        "summary": "Scalar UI (HTML) for browsing this spec",
        "security": [],
        "responses": {"200": {"description": "OK"}}
      }
    },
    "/admin/v1/models": {
      "get": {
        "summary": "list models",
        "responses": {
          "200": {
            "description": "OK",
            "content": {"application/json": {"schema": {"type": "array", "items": {"$ref": "#/components/schemas/ModelEntry"}}}}
          }
        }
      },
      "post": {
        "summary": "create model",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Model"}}}},
        "responses": {
          "200": {"description": "OK", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ModelEntry"}}}},
          "400": {"description": "schema validation failed"},
          "409": {"description": "duplicate display_name"}
        }
      }
    },
    "/admin/v1/models/{id}": {
      "parameters": [{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}],
      "get":    { "summary": "get model",    "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} },
      "put":    {
        "summary": "update model",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Model"}}}},
        "responses": {"200": {"description": "OK"}, "400": {"description": "schema validation failed"}, "404": {"description": "not found"}, "409": {"description": "duplicate display_name"}}
      },
      "delete": { "summary": "delete model", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} }
    },
    "/admin/v1/apikeys": {
      "get":  { "summary": "list api keys",  "responses": {"200": {"description": "OK"}} },
      "post": {
        "summary": "create api key",
        "description": "Body carries `key_hash` (SHA-256 of plaintext). The plaintext is generated client-side; the gateway never sees it.",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ApiKey"}}}},
        "responses": {"200": {"description": "OK"}, "400": {"description": "schema validation failed"}, "409": {"description": "duplicate key_hash"}}
      }
    },
    "/admin/v1/apikeys/{id}": {
      "parameters": [{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}],
      "get":    { "summary": "get api key",    "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} },
      "put":    {
        "summary": "update api key",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ApiKey"}}}},
        "responses": {"200": {"description": "OK"}, "400": {"description": "schema validation failed"}, "404": {"description": "not found"}, "409": {"description": "duplicate key_hash"}}
      },
      "delete": { "summary": "delete api key", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} }
    },
    "/admin/v1/apikeys/{id}/rotate": {
      "parameters": [{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}],
      "post": {
        "summary": "rotate api key",
        "description": "Generates a new plaintext bearer, persists its SHA-256 hash, and returns the plaintext **once** under `plaintext`. Caller MUST capture — subsequent GETs only expose the hash.",
        "responses": {
          "200": {"description": "OK", "content": {"application/json": {"schema": {
            "type": "object",
            "required": ["entry", "plaintext"],
            "properties": {
              "entry":     {"$ref": "#/components/schemas/ApiKeyEntry"},
              "plaintext": {"type": "string", "example": "sk-abcd1234ef567890"}
            }
          }}}},
          "404": {"description": "not found"}
        }
      }
    },
    "/admin/v1/provider_keys": {
      "get":  { "summary": "list provider keys",  "responses": {"200": {"description": "OK"}} },
      "post": {
        "summary": "create provider key",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ProviderKey"}}}},
        "responses": {"200": {"description": "OK"}, "400": {"description": "schema validation failed"}, "409": {"description": "duplicate display_name"}}
      }
    },
    "/admin/v1/provider_keys/{id}": {
      "parameters": [{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}],
      "get":    { "summary": "get provider key",    "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} },
      "put":    {
        "summary": "update provider key",
        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ProviderKey"}}}},
        "responses": {"200": {"description": "OK"}, "400": {"description": "schema validation failed"}, "404": {"description": "not found"}, "409": {"description": "duplicate display_name"}}
      },
      "delete": { "summary": "delete provider key", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} }
    },
    "/admin/v1/health": {
      "get": {
        "summary": "per-Model upstream health",
        "description": "Returns each Model's health level: 0=Healthy, 1=Degraded (4-7 consecutive upstream failures), 2=Down (8+).",
        "responses": {"200": {"description": "OK"}}
      }
    },
    "/playground/chat/completions": {
      "post": {
        "summary": "in-process forward to the proxy router",
        "description": "Forwards a chat completion through the proxy in-process — no extra network hop, but the request is fully audited as if it had arrived on the proxy listener. Auth is a **proxy** ApiKey (`Authorization: Bearer sk-aisix-...`), not an admin key — the proxy middleware validates it inside the forwarded request.",
        "security": [{"ProxyBearer": []}],
        "responses": {"200": {"description": "OK (OpenAI-shape chat completion)"}, "401": {"description": "missing/invalid api key"}}
      }
    }
  },
  "components": {
    "securitySchemes": {
      "AdminBearer": {
        "type": "http",
        "scheme": "bearer",
        "description": "Admin key from `config.admin.admin_keys`. Required for every `/admin/v1/*` route."
      },
      "ProxyBearer": {
        "type": "http",
        "scheme": "bearer",
        "description": "Proxy ApiKey (`sk-aisix-...`) — only used by `/playground/chat/completions`."
      }
    },
    "schemas": {
      "Model": {
        "type": "object",
        "required": ["display_name"],
        "properties": {
          "display_name":    {"type": "string", "example": "my-gpt4"},
          "provider":        {"type": "string", "enum": ["openai","anthropic","gemini","deepseek"]},
          "model_name":      {"type": "string", "example": "gpt-4o"},
          "provider_key_id": {"type": "string", "example": "11111111-1111-1111-1111-111111111111"},
          "timeout":         {"type": "integer", "minimum": 0, "description": "Request timeout in milliseconds. Absent or 0 = no timeout."},
          "rate_limit":      {"$ref": "#/components/schemas/RateLimit"},
          "routing":         {"$ref": "#/components/schemas/Routing"},
          "cost":            {"$ref": "#/components/schemas/ModelCost"}
        },
        "description": "A direct model ships `provider` + `model_name` + `provider_key_id`; a routing model ships `routing` and omits the upstream triple."
      },
      "ModelEntry": {
        "type": "object",
        "required": ["id", "value", "revision"],
        "properties": {
          "id":       {"type": "string"},
          "value":    {"$ref": "#/components/schemas/Model"},
          "revision": {"type": "integer"}
        }
      },
      "ApiKey": {
        "type": "object",
        "required": ["key_hash", "allowed_models"],
        "properties": {
          "key_hash":       {"type": "string", "description": "SHA-256 hex of the plaintext bearer. Lowercase.", "example": "91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c"},
          "allowed_models": {"type": "array", "items": {"type": "string"}, "description": "Allowed Model display_names. `[\"*\"]` for all; `[]` denies everything."},
          "rate_limit":     {"$ref": "#/components/schemas/RateLimit"},
          "max_budget_usd": {"type": "number", "minimum": 0, "description": "Per-month USD spend cap. Absent = unlimited."}
        }
      },
      "ApiKeyEntry": {
        "type": "object",
        "required": ["id", "value", "revision"],
        "properties": {
          "id":       {"type": "string"},
          "value":    {"$ref": "#/components/schemas/ApiKey"},
          "revision": {"type": "integer"}
        }
      },
      "ProviderKey": {
        "type": "object",
        "required": ["display_name", "secret"],
        "properties": {
          "display_name": {"type": "string", "example": "openai-prod"},
          "secret":       {"type": "string", "description": "Upstream provider API key, plaintext.", "example": "sk-prod-xxxx"},
          "api_base":     {"type": "string", "description": "Override for the upstream base URL. Empty/absent uses the provider default."}
        }
      },
      "RateLimit": {
        "type": "object",
        "properties": {
          "tpm":         {"type": "integer", "minimum": 0, "description": "Tokens per minute"},
          "tpd":         {"type": "integer", "minimum": 0, "description": "Tokens per day"},
          "rpm":         {"type": "integer", "minimum": 0, "description": "Requests per minute"},
          "rpd":         {"type": "integer", "minimum": 0, "description": "Requests per day"},
          "concurrency": {"type": "integer", "minimum": 0, "description": "Max in-flight"}
        }
      },
      "Routing": {
        "type": "object",
        "required": ["targets"],
        "properties": {
          "strategy":     {"type": "string", "enum": ["round_robin", "weighted", "failover"]},
          "targets": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object",
              "required": ["model"],
              "properties": {
                "model":  {"type": "string", "description": "Target Model.display_name"},
                "weight": {"type": "integer", "minimum": 0}
              }
            }
          },
          "retry_budget": {"type": "integer", "minimum": 0}
        }
      },
      "ModelCost": {
        "type": "object",
        "required": ["input_per_1k", "output_per_1k"],
        "properties": {
          "input_per_1k":  {"type": "number", "minimum": 0, "description": "USD per 1,000 input (prompt) tokens"},
          "output_per_1k": {"type": "number", "minimum": 0, "description": "USD per 1,000 output (completion) tokens"}
        }
      },
      "AdminError": {
        "type": "object",
        "required": ["error_msg"],
        "properties": {"error_msg": {"type": "string"}}
      }
    }
  },
  "security": [{ "AdminBearer": [] }]
}"##;

const SCALAR_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>aisix admin OpenAPI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/admin/openapi.json"
      type="application/javascript"
    ></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"#;

pub async fn openapi_json() -> Response {
    ([(header::CONTENT_TYPE, "application/json")], OPENAPI_JSON).into_response()
}

pub async fn openapi_scalar() -> Html<&'static str> {
    Html(SCALAR_HTML)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn openapi_json_is_well_formed_and_documents_admin_paths() {
        let resp = openapi_json().await;
        assert_eq!(resp.status(), 200);
        // Validate by parsing — guards against typos in the literal block.
        let parsed: serde_json::Value =
            serde_json::from_str(OPENAPI_JSON).expect("OPENAPI_JSON must parse");
        // Every route mounted in build_router should be documented.
        for path in [
            "/health",
            "/metrics",
            "/admin/openapi.json",
            "/admin/openapi-scalar",
            "/admin/v1/models",
            "/admin/v1/models/{id}",
            "/admin/v1/apikeys",
            "/admin/v1/apikeys/{id}",
            "/admin/v1/apikeys/{id}/rotate",
            "/admin/v1/provider_keys",
            "/admin/v1/provider_keys/{id}",
            "/admin/v1/health",
            "/playground/chat/completions",
        ] {
            assert!(
                parsed["paths"][path].is_object(),
                "OPENAPI_JSON missing path {path}"
            );
        }
        // Reusable schemas referenced from the path bodies.
        for schema in [
            "Model",
            "ModelEntry",
            "ApiKey",
            "ApiKeyEntry",
            "ProviderKey",
            "RateLimit",
            "Routing",
            "ModelCost",
            "AdminError",
        ] {
            assert!(
                parsed["components"]["schemas"][schema].is_object(),
                "OPENAPI_JSON missing schema {schema}"
            );
        }
    }

    #[tokio::test]
    async fn openapi_unauthenticated_routes_carry_empty_security() {
        // /health, /metrics, and the openapi self-references are public
        // (mirrors the `unauthenticated like /metrics` design note in
        // build_router). The spec must mark them with security: [] so
        // Scalar's "Try it" doesn't prompt for an admin key on those
        // routes.
        let parsed: serde_json::Value =
            serde_json::from_str(OPENAPI_JSON).expect("OPENAPI_JSON must parse");
        for path in [
            "/health",
            "/metrics",
            "/admin/openapi.json",
            "/admin/openapi-scalar",
        ] {
            let security = &parsed["paths"][path]["get"]["security"];
            assert!(
                security.is_array() && security.as_array().unwrap().is_empty(),
                "{path} should declare security: [] but got {security:?}"
            );
        }
    }

    #[tokio::test]
    async fn scalar_html_loads_the_spec_url() {
        let html = openapi_scalar().await;
        let body = html.0;
        assert!(body.contains("/admin/openapi.json"));
        assert!(body.contains("scalar"));
    }
}
