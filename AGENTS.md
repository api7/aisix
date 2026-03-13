# AGENTS.md

AI Gateway codebase guide for agentic coding assistants.

## Project Overview

Rust-based AI gateway proxy supporting OpenAI, Anthropic, Gemini, and DeepSeek APIs. Built with Axum for HTTP, Tokio for async runtime, and etcd for configuration storage.

## Build, Lint, and Test Commands

### Build
```bash
cargo build                    # Debug build
cargo build --release          # Release build
```

### Run
```bash
RUST_LOG=info cargo run                    # Standard run
RUST_LOG=info cargo run --features trace   # With OTel tracing
```

### Lint
```bash
cargo clippy --all-targets --all-features --locked -- -D warnings
```
Clippy warnings are treated as errors. Fix all warnings before committing.

### Test
```bash
cargo test                                    # Run all tests
cargo test --test api                         # Run specific test file (tests/api.rs)
cargo test test_crud                          # Run specific test by name
cargo test --test admin::models_api           # Run tests in specific module
cargo test -- --nocapture                     # Show test output
```

### Format
```bash
cargo fmt                                     # Format all code
cargo fmt -- --check                          # Check formatting without changes
```

## Code Style Guidelines

### Imports

Imports are auto-organized by `rustfmt` with these rules (see `rustfmt.toml`):
- `reorder_imports = true` — Sort imports alphabetically
- `imports_granularity = "Crate"` — Merge imports from same crate
- `group_imports = "StdExternalCrate"` — Group: std → external crates → local

```rust
// Standard library first
use std::sync::Arc;

// External crates (alphabetical)
use anyhow::Result;
use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};
use tokio::select;

// Local modules last
use crate::config::entities::Model;
```

### Naming Conventions

- **Types/Structs/Enums**: `PascalCase` (e.g., `ProviderConfig`, `ChatCompletionError`)
- **Functions/Methods**: `snake_case` (e.g., `chat_completions`, `create_provider`)
- **Constants/Statics**: `SCREAMING_SNAKE_CASE` (e.g., `MODELS_PATTERN`, `SCHEMA_VALIDATOR`)
- **Modules**: `snake_case` (e.g., `chat_completions`, `rate_limit`)
- **Local variables**: `snake_case`

### Error Handling

Use `thiserror` for library/domain errors, `anyhow` for application errors:

```rust
// Domain errors with thiserror
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("Not implemented")]
    NotYetImplemented,
    #[error("API error {0}: {1}")]
    ServiceError(http::StatusCode, String),
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
}

// Application code uses anyhow::Result
pub async fn create_provider(config: &ProviderConfig) -> Result<Box<dyn Provider>> {
    // ...
}
```

Error types should implement `IntoResponse` for Axum handlers:

```rust
impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        match self {
            AuthError::MissingApiKey => (
                http::StatusCode::UNAUTHORIZED,
                Json(json!({ "error": { "message": "Missing API key" } })),
            ).into_response(),
        }
    }
}
```

### Async Patterns

- Use `tokio` as the async runtime
- Async functions: `async fn`
- Async tests: `#[tokio::test]`
- Traits with async methods: `#[async_trait]`

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat_completion(&self, request: ChatCompletionRequest) 
        -> Result<ChatCompletionResponse, ProviderError>;
}
```

### Tracing

Use `fastrace` for distributed tracing:

```rust
#[fastrace::trace]
pub async fn chat_completions(...) -> Result<Response, ChatCompletionError> {
    // Function is automatically traced
}

#[fastrace::trace(short_name = true)]
pub fn create_provider(config: &ProviderConfig) -> Box<dyn Provider> {
    // Short name in trace spans
}
```

### Documentation

Use `///` for doc comments on public items:

```rust
/// Creates a new provider instance based on the configuration.
pub fn create_provider(config: &ProviderConfig) -> Box<dyn Provider> {
    // ...
}
```

## Testing Patterns

### Test Organization

- Integration tests in `tests/` directory
- Unit tests in `#[cfg(test)] mod tests` within source files
- Test utilities in `tests/utils/`

### Test Attributes

```rust
#[test]                    // Synchronous unit test
fn test_valid_jsonschema() { }

#[tokio::test]             // Async test
async fn test_crud() { }

#[rstest]                  // Parameterized test
#[case::ok(json!({...}), true, None)]
#[case::error(json!({...}), false, Some("error message"))]
fn schemas(#[case] input: Value, #[case] ok: bool, #[case] err: Option<String>) { }
```

### Test Utilities

```rust
// tests/utils/http.rs
pub fn build_req(method: Method, uri: &str, body: Option<Value>, auth_key: &str) -> Request<Body>;
pub async fn oneshot_json(router: &Router, req: Request<Body>) -> (StatusCode, Value);
```

### Test Assertions

```rust
use pretty_assertions::assert_eq;  // Better diff output
use assert_matches::assert_matches;
```

## Project Structure

```
src/
├── main.rs              # Entry point, server setup
├── lib.rs               # Library exports
├── admin/               # Admin API (port 3001)
│   ├── mod.rs
│   ├── apikeys.rs
│   └── models.rs
├── config/              # Configuration loading
│   ├── mod.rs
│   ├── etcd.rs          # etcd provider
│   └── entities/        # Data models (ApiKey, Model)
├── providers/           # AI provider implementations
│   ├── mod.rs           # Provider trait
│   ├── openai.rs
│   ├── anthropic/
│   ├── gemini.rs
│   └── mock.rs
├── proxy/               # Proxy API (port 3000)
│   ├── mod.rs
│   ├── handlers/        # Request handlers
│   ├── middlewares/     # Auth, tracing, body parsing
│   └── hooks/           # Rate limiting, metrics
└── utils/               # Utilities (jsonschema, metrics)

tests/
├── api.rs               # Test entry point
├── admin/               # Admin API tests
├── proxy/               # Proxy tests
└── utils/               # Test utilities
```

## Key Dependencies

- `axum` — Web framework
- `tokio` — Async runtime
- `reqwest` — HTTP client for upstream providers
- `serde`/`serde_json` — Serialization
- `anyhow`/`thiserror` — Error handling
- `etcd-client` — etcd client
- `fastrace` — Distributed tracing
- `utoipa` — OpenAPI generation

## Configuration

Configuration via `config.yaml`:

```yaml
deployment:
  etcd:
    host: ["http://127.0.0.1:2379"]
    prefix: /aisix
    timeout: 30
  admin:
    admin_key:
      - key: "admin"
```

## CI/CD

GitHub Actions workflow (`.github/workflows/build.yaml`):
1. `cargo clippy` — Lint (warnings = error)
2. `cargo test` — Run tests
3. `cargo build` — Build binary

## VSCode Setup

Recommended extensions (`.vscode/extensions.json`):
- `rust-lang.rust-analyzer`
- `vadimcn.vscode-lldb`
- `tamasfe.even-better-toml`
- `fill-labs.dependi`

Format on save is enabled.
