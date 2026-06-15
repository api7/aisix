//! Emit the canonical AISIX AI Gateway Admin API OpenAPI document.
//!
//! Invocation:
//!
//! ```bash
//! cargo run -p aisix-admin --bin dump-openapi
//! ```
//!
//! Writes the same merged OpenAPI JSON served by `GET /admin/openapi.json`
//! to stdout. Redirect it to `schemas/openapi/admin-api.json` after changing
//! Admin API routes, OpenAPI metadata, or resource schemas.

fn main() {
    println!("{}", aisix_admin::admin_openapi_json());
}
