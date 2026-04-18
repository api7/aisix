//! Minimal OpenAPI document + Scalar mount.
//!
//! Only the admin CRUD endpoints are described here today. The proxy
//! `/v1/chat/completions` surface is OpenAI-compatible and operators
//! refer to OpenAI's published spec for that — duplicating it here adds
//! drift risk without adding signal. Future PRs that introduce
//! aisix-specific request/response shapes can extend the spec inline.
//!
//! The Scalar UI is a single static HTML page that loads the JSON spec
//! over HTTP — no JS bundling required.

use axum::http::header;
use axum::response::{Html, IntoResponse, Response};

/// Hand-written JSON spec. Small enough that maintaining it by hand is
/// less effort than wiring `utoipa` derive macros across every handler;
/// the surface is stable enough that drift is easy to spot in review.
const OPENAPI_JSON: &str = r#"{
  "openapi": "3.1.0",
  "info": {
    "title": "aisix admin API",
    "version": "0.1.0",
    "description": "CRUD for Models and ApiKeys. All endpoints require Bearer admin-key auth. Errors use {error_msg}."
  },
  "paths": {
    "/admin/v1/models": {
      "get":  { "summary": "list models",  "responses": {"200": {"description": "OK"}} },
      "post": { "summary": "create model", "responses": {"200": {"description": "OK"}, "409": {"description": "duplicate name"}} }
    },
    "/admin/v1/models/{id}": {
      "get":    { "summary": "get model",    "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} },
      "put":    { "summary": "update model", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}, "409": {"description": "duplicate name"}} },
      "delete": { "summary": "delete model", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} }
    },
    "/admin/v1/apikeys": {
      "get":  { "summary": "list api keys",  "responses": {"200": {"description": "OK"}} },
      "post": { "summary": "create api key", "responses": {"200": {"description": "OK"}, "409": {"description": "duplicate key"}} }
    },
    "/admin/v1/apikeys/{id}": {
      "get":    { "summary": "get api key",    "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} },
      "put":    { "summary": "update api key", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}, "409": {"description": "duplicate key"}} },
      "delete": { "summary": "delete api key", "responses": {"200": {"description": "OK"}, "404": {"description": "not found"}} }
    }
  },
  "components": {
    "securitySchemes": {
      "AdminBearer": { "type": "http", "scheme": "bearer" }
    }
  },
  "security": [{ "AdminBearer": [] }]
}"#;

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
        assert!(parsed["paths"]["/admin/v1/models"].is_object());
        assert!(parsed["paths"]["/admin/v1/apikeys/{id}"].is_object());
    }

    #[tokio::test]
    async fn scalar_html_loads_the_spec_url() {
        let html = openapi_scalar().await;
        let body = html.0;
        assert!(body.contains("/admin/openapi.json"));
        assert!(body.contains("scalar"));
    }
}
