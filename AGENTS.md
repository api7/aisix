# AGENTS.md

> **For AI coding assistants** (OpenCode, Cursor, Copilot, etc.): This file is the
> primary context source for AI assistants working on this codebase. Use it to
> understand project structure, coding conventions, and build commands.

> **For human contributors**: See [CONTRIBUTING.md](CONTRIBUTING.md) for the
> contribution guide.

## Project Overview

Rust-based AI gateway proxy supporting OpenAI, Anthropic, Gemini, and DeepSeek APIs. Built with Axum for HTTP, Tokio for async runtime, and etcd for configuration storage.

Includes a React-based admin UI (in `ui/`) for managing models, API keys, and a playground for testing chat completions.

## Build, Lint, and Test Commands

### Build
```bash
cargo build           # Debug build
cargo build --release # Release build
```

### Run
```bash
RUST_LOG=info cargo run
```

### UI Development
```bash
cd ui
pnpm install --frozen-lockfile    # Install dependencies
pnpm dev        # Start dev server
pnpm build      # Build for production
pnpm lint       # Run ESLint
pnpm format     # Format with Prettier
pnpm typecheck  # Type check without emit
pnpm preview    # Preview production build
```

### Lint
```bash
cargo clippy --all-targets --all-features --locked -- -D warnings
```
Clippy warnings are treated as errors. Fix all warnings before committing.

### Test
```bash
cargo test                             # Run all tests
cargo test --verbose                   # Run tests with verbose output
cargo test --test api                  # Run specific test file (tests/api.rs)
cargo test test_crud                   # Run specific test by name
cargo test --test admin::models_api    # Run tests in specific module
cargo test -- --nocapture              # Show test output
```

### E2E Test
```bash
pnpm -C tests/e2e install --frozen-lockfile  # Install e2e dependencies
pnpm -C tests/e2e test                       # Run all e2e tests
```

### Format
```bash
cargo fmt          # Format all code
cargo fmt -- --check  # Check formatting without changes
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
    async fn chat_completion_stream(&self, request: ChatCompletionRequest) 
        -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError>;
    async fn embedding(&self, request: EmbeddingRequest) 
        -> Result<EmbeddingResponse, ProviderError>;
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
│   ├── apikeys.rs       # API key CRUD
│   ├── models.rs        # Model CRUD
│   ├── playground.rs    # Playground chat completions
│   ├── types.rs         # Admin types
│   └── ui.rs            # Static UI file server
├── config/              # Configuration loading
│   ├── mod.rs
│   ├── etcd.rs          # etcd provider
│   ├── types.rs         # Config types
│   └── entities/        # Data models (ApiKey, Model)
│       ├── mod.rs
│       ├── apikeys.rs
│       ├── models.rs
│       └── types.rs
├── providers/           # AI provider implementations
│   ├── mod.rs           # Provider trait
│   ├── types.rs         # Provider types
│   ├── openai.rs
│   ├── openai_compatible.rs
│   ├── anthropic/
│   │   ├── mod.rs
│   │   ├── types.rs
│   │   └── README.md
│   ├── gemini.rs
│   ├── deepseek.rs
│   └── mock.rs
├── proxy/               # Proxy API (port 3000)
│   ├── mod.rs
│   ├── handlers/        # Request handlers
│   │   ├── mod.rs
│   │   ├── models.rs
│   │   ├── chat_completions/
│   │   │   ├── mod.rs
│   │   │   └── types.rs
│   │   └── embeddings/
│   │       ├── mod.rs
│   │       └── types.rs
│   ├── middlewares/     # Auth, tracing, body parsing
│   │   ├── mod.rs
│   │   ├── auth.rs
│   │   ├── parse_body.rs
│   │   └── trace.rs
│   └── hooks/           # Rate limiting, metrics, validation
│       ├── mod.rs
│       ├── metric.rs
│       ├── validate_model.rs
│       └── rate_limit/
│           ├── mod.rs
│           ├── concurrent/  # Concurrency limiting
│           └── ratelimit/    # Token/request rate limiting
└── utils/               # Utilities
    ├── mod.rs
    ├── future.rs
    ├── jsonschema.rs
    └── metrics.rs

ui/                      # React admin UI
├── src/
│   ├── assets/          # Static assets
│   ├── components/      # UI components (shadcn/ui based)
│   │   ├── apikeys/
│   │   ├── layout/
│   │   ├── models/
│   │   ├── playground/
│   │   ├── theme-provider.tsx
│   │   └── ui/
│   ├── hooks/           # Custom React hooks
│   ├── i18n/            # Internationalization
│   ├── lib/             # API client, queries, utilities
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   └── types.ts
│   │   ├── queries/
│   │   │   ├── apikeys.ts
│   │   │   ├── models.ts
│   │   │   └── index.ts
│   │   └── utils.ts
│   ├── routes/          # TanStack Router routes
│   ├── index.css
│   ├── main.tsx
│   └── routeTree.gen.ts
└── package.json

tests/
├── api.rs               # Test entry point
├── admin/               # Admin API tests
│   ├── mod.rs
│   ├── apikeys_api.rs
│   ├── auth.rs
│   ├── models_api.rs
│   └── ui.rs
├── proxy/               # Proxy tests
│   ├── mod.rs
│   └── timeout.rs
├── utils/               # Test utilities
│   ├── mod.rs
│   └── http.rs
└── e2e/                 # End-to-end tests (TypeScript/Vitest)
    ├── tests/
    │   ├── admin/       # Admin API e2e tests
    │   ├── proxy/       # Proxy e2e tests
    │   └── server.test.ts
    ├── utils/           # E2E test utilities
    └── vitest.config.ts
```

## Key Dependencies

### Rust
- `axum` — Web framework
- `tokio` — Async runtime
- `reqwest` — HTTP client for upstream providers
- `serde`/`serde_json` — Serialization
- `anyhow`/`thiserror` — Error handling
- `etcd-client` — etcd client
- `fastrace` + `fastrace-*` — Distributed tracing (axum, opentelemetry, reqwest integrations)
- `opentelemetry` + `opentelemetry-*` — OpenTelemetry SDK for tracing export
- `logforth` — Logging with fastrace integration
- `utoipa` + `utoipa-scalar` — OpenAPI generation and Scalar UI
- `rust-embed` — Embed static files (UI)
- `skp-ratelimit` — Rate limiting
- `jsonschema` — JSON Schema validation
- `async-trait` — Async trait support
- `config` — Configuration file parsing
- `dashmap` — Concurrent map for rate limiting
- `uuid` — UUID generation
- `tower` — Middleware utilities
- `clap` — CLI argument parsing
- `validator` — Input validation
- `axum-server` — Axum server runtime with TLS support
- `backon` — Retry with backoff
- `metrics` + `metrics-exporter-otel` — Prometheus metrics export
- `opentelemetry-semantic-conventions` — OpenTelemetry semantic conventions
- `fastrace-tracing` + `fastrace-reqwest` — Additional fastrace integrations
- `openssl` + `tokio-openssl` — TLS support for inbound connections (axum-server SNI)
- `reqwest` (`native-tls`) — TLS for outbound connections to upstream providers

### UI (React)
- `@tanstack/react-router` — File-based routing
- `@tanstack/react-query` — Data fetching
- `@tanstack/react-table` — Table components
- `@tanstack/react-form` — Form state management
- `shadcn` — UI component library (via `radix-ui`)
- `tailwindcss` — Styling
- `@monaco-editor/react` — Code editor (JSON config)
- `i18next` + `react-i18next` — Internationalization
- `lucide-react` — Icon library
- `next-themes` — Theme switching (dark mode)
- `openai` — OpenAI SDK for playground

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

## Git Commit Guidelines

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <subject>

<body>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

### Example

```
docs: rewrite README for open source users

- Add features list, architecture diagram, quick start guide
- Include API reference for Proxy and Admin APIs
- Document configuration options for models and API keys
```

### Rules

1. **Subject line**: Brief description of the change (imperative mood)
2. **Body**: Bullet points explaining what changed and why (optional for trivial changes)
3. **NO Co-authored-by**: Do not add `Co-authored-by:` trailers
4. **NO attribution links**: Do not add "Ultraworked with" or similar attribution
5. **Keep it minimal**: Only include information relevant to the change itself

### Anti-Patterns

```
# WRONG - contains unnecessary attribution
docs: rewrite README

Ultraworked with [SomeTool](https://...)
Co-authored-by: Bot <bot@example.com>

# CORRECT - clean and focused
docs: rewrite README

- Add features list and architecture diagram
- Include API reference documentation
```

## CI/CD

GitHub Actions workflow (`.github/workflows/build.yaml`):
1. Setup dependencies (protobuf-compiler, pnpm)
2. Setup Node.js (LTS)
3. Setup Rust toolchain (stable)
4. Setup environment (docker compose for etcd)
5. Build UI (`pnpm -C ui install --frozen-lockfile && pnpm -C ui build`)
6. `cargo clippy` — Lint (warnings = error)
7. `cargo test` — Run tests
8. E2E Test (`pnpm -C tests/e2e install --frozen-lockfile && pnpm -C tests/e2e test`)
9. `cargo build` — Build binary
10. Upload artifact (debug binary)

## VSCode Setup

Recommended extensions (`.vscode/extensions.json`):
- `rust-lang.rust-analyzer`
- `vadimcn.vscode-lldb`
- `tamasfe.even-better-toml`
- `fill-labs.dependi`

Format on save is enabled.
