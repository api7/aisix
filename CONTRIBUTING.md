# Contributing to AISIX

Thank you for your interest in contributing to AISIX! This guide covers everything you need to set up a development environment, follow project conventions, and submit contributions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Building](#building)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Guidelines](#commit-guidelines)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [CI/CD](#cicd)
- [Security](#security)
- [License](#license)

---

## Prerequisites

Install the following before starting:

| Tool | Version | Notes |
|------|---------|-------|
| Rust | latest stable | Install via [rustup](https://rustup.rs/) |
| Node.js | LTS | Used for the admin UI |
| pnpm | (bundled) | `corepack enable pnpm` after Node.js install |
| Docker & Docker Compose | any recent | Required for etcd and test services |
| protobuf-compiler | any | Required to build Rust gRPC dependencies |

**Install system dependencies (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install -y protobuf-compiler docker.io docker-compose-v2
corepack enable pnpm
```

---

## Setting Up the Development Environment

### 1. Clone the repository

```bash
git clone https://github.com/api7/aisix.git
cd aisix
```

### 2. Start etcd

```bash
docker compose -f ci/docker-compose.yaml up -d etcd
```

> To run E2E tests, use `docker compose -f ci/docker-compose.yaml up -d` to bring up all required services.

### 3. Build the admin UI

```bash
pnpm -C ui install --frozen-lockfile
pnpm -C ui build
```

The UI build output is embedded into the Rust binary at compile time via `rust-embed`.

> **Tip:** For UI-only development, run `pnpm -C ui dev` to start a hot-reload dev server that proxies API calls to the running gateway.

### 4. Build and run the gateway

```bash
RUST_LOG=info cargo run
```

The gateway starts two servers:
- **Proxy API** on `0.0.0.0:3000` — OpenAI-compatible endpoint for LLM requests
- **Admin API** on `127.0.0.1:3001` — manages models, API keys, and the playground

---

## Configuration

The gateway reads `config.yaml` at startup (pass `--config <path>` to use a different file):

```yaml
deployment:
  etcd:
    host:
      - "http://127.0.0.1:2379"
    prefix: /aisix
    timeout: 30
  admin:
    admin_key:
      - key: admin          # Admin API key — change for production

server:
  proxy:
    listen: 0.0.0.0:3000
    tls:
      enabled: false
      cert_file: cert.pem
      key_file: key.pem
  admin:
    listen: 127.0.0.1:3001
```

---

## Project Structure

```
src/
├── main.rs              # Entry point, server setup
├── lib.rs               # Library exports
├── admin/               # Admin API (port 3001)
├── config/              # Configuration loading and etcd provider
├── providers/           # AI provider implementations (OpenAI, Anthropic, Gemini, DeepSeek)
├── proxy/               # Proxy API (port 3000) — handlers, middlewares, hooks
└── utils/               # Shared utilities

ui/                      # React admin UI (Vite + TanStack Router/Query)
tests/
├── api.rs               # Rust integration test entry point
├── admin/               # Admin API tests
├── proxy/               # Proxy tests
├── utils/               # Test helpers
└── e2e/                 # TypeScript E2E tests (Vitest)
```

---

## Building

```bash
# Debug build (faster compile, slower runtime)
cargo build

# Release build
cargo build --release
```

---

## Running Tests

### Rust unit and integration tests

Make sure etcd is running before starting:

```bash
# Run all tests
cargo test

# Run a specific test file
cargo test --test api

# Run a specific test by name
cargo test test_crud

# Run tests in a specific module
cargo test --test admin::models_api

# Show println!/dbg! output
cargo test -- --nocapture
```

### E2E tests (TypeScript / Vitest)

The E2E test suite runs the gateway binary directly — it requires:
1. A built binary at `target/debug/aisix`
2. etcd and other required services running (`ci/docker-compose.yaml`)

```bash
# Build the binary first
cargo build

# Install E2E test dependencies
pnpm -C tests install --frozen-lockfile

# Run E2E tests
pnpm -C tests test
```

### Admin UI checks

```bash
pnpm -C ui lint         # ESLint
pnpm -C ui typecheck    # TypeScript type check (no emit)
```

---

## Code Style

### Rust

**Format:** Run `cargo fmt` before every commit. CI rejects unformatted code.

```bash
cargo fmt              # Format all Rust code
cargo fmt -- --check   # Verify formatting without changes
```

**Lint:** Fix all Clippy warnings — they are treated as errors in CI:

```bash
cargo clippy --all-targets --all-features --locked -- -D warnings
```

**Import order** (enforced by `rustfmt.toml`):

```rust
// 1. Standard library
use std::sync::Arc;

// 2. External crates (alphabetical)
use anyhow::Result;
use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

// 3. Local modules
use crate::config::entities::Model;
```

**Naming conventions:**

| Item | Convention | Example |
|------|-----------|---------|
| Types / Structs / Enums | `PascalCase` | `ProviderConfig` |
| Functions / Methods | `snake_case` | `chat_completions` |
| Constants / Statics | `SCREAMING_SNAKE_CASE` | `MODELS_PATTERN` |
| Modules | `snake_case` | `chat_completions` |

**Error handling:**
- Use `thiserror` for domain/library errors
- Use `anyhow` for application-level errors
- Axum handler errors must implement `IntoResponse`

**Async:**
- Runtime: `tokio`
- Async trait methods require `#[async_trait]`
- Async tests use `#[tokio::test]`

**Tracing:** Annotate handler and provider functions with `#[fastrace::trace]`.

### TypeScript (UI and E2E tests)

```bash
pnpm -C ui format      # Prettier (writes)
pnpm -C ui lint        # ESLint
```

---

## Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <subject>

<body>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

**Rules:**
- Subject line: imperative mood, concise description
- Body: bullet points explaining what changed and why (optional for trivial changes)
- Do **not** add `Co-authored-by:` trailers
- Do **not** add attribution links

**Examples:**

```
feat: add rate limiting per API key

- Implement concurrent and token-per-minute limits
- Store rate limit state in DashMap for lock-free access
```

```
fix: handle empty etcd prefix in config loader
```

---

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes following the code style guidelines above.
3. Run the full verification suite locally before pushing:
   ```bash
   cargo fmt -- --check
   cargo clippy --all-targets --all-features --locked -- -D warnings
   cargo test
  cargo build && pnpm -C tests install --frozen-lockfile && pnpm -C tests test
   ```
4. Commit using Conventional Commits format.
5. Open a pull request against `main`. Describe **why** the change is needed, not just what it does.
6. CI runs automatically on every PR — all checks must pass before merging.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/build.yaml`) runs on every push and PR to `main`:

| Step | Command |
|------|---------|
| Build UI | `pnpm -C ui install --frozen-lockfile && pnpm -C ui build` |
| Lint | `cargo clippy --all-targets --all-features --locked -- -D warnings` |
| Test | `cargo test --verbose` |
| E2E Test | `pnpm -C tests install --frozen-lockfile && pnpm -C tests test` |
| Build | `cargo build --verbose` |

The CI environment uses `ci/docker-compose.yaml` to start etcd and other required services.

---

## Security

If you discover a security vulnerability, please **do not** open a public GitHub issue.
Follow the process described in [SECURITY.md](./SECURITY.md) to report it privately.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
