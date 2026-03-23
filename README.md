[![Build Status](https://github.com/api7/ai-gateway-stash/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/api7/ai-gateway-stash/actions/workflows/build.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/api7/ai-gateway-stash/blob/main/LICENSE)

<p align="center">
  <img src="./docs/images/aisix_logo.svg" alt="AISIX AI Gateway" width="400" />
</p>

<p align="center">
  <strong>A high-performance, Rust-based AI Gateway for unified LLM access.</strong><br/>
  <em>OpenAI-compatible API across OpenAI, Anthropic, Gemini, DeepSeek, and more.</em><br/><br/>
  🦀 <strong>Rust</strong> • 🔌 <strong>OpenAI Compatible</strong> • 🗄️ <strong>etcd</strong>
</p>


## Why AISIX

- 🦀 **Rust + Tokio** — Extreme performance with low resource footprint; ships as a single binary
- 🔌 **OpenAI-Compatible** — One API to call all LLMs; drop-in replacement with zero code changes
- ⚡ **Dynamic Config** — Hot-reload via etcd; update models and keys without restarting
- 🛡️ **Enterprise-Grade Control** — API key auth, rate limiting (RPM / TPM / concurrent), and per-model access control
- 📊 **Observability** — OpenTelemetry distributed tracing and Prometheus metrics out of the box
- 🎨 **Admin UI** — Built-in management dashboard for models, API keys, and a chat playground

---

## Features

### 🌐 Multi-Provider Support

| Provider | Chat Completions | Streaming | Embeddings |
|---|:---:|:---:|:---:|
| 🟢 OpenAI | ✅ | ✅ | ✅ |
| 🟠 Anthropic | ✅ | ✅ | — |
| 🔵 Gemini | ✅ | ✅ | ✅ |
| 🐋 DeepSeek | ✅ | ✅ | — |
| 🔌 OpenAI-Compatible | ✅ | ✅ | ✅ |

### 🚦 Traffic Management

- Rate limiting — RPM, TPM, and concurrent request limits
- Per-model and per-key access control
- Request validation with JSON Schema

### 🛡️ Security & Auth

- API key authentication on all proxy requests
- Per-key model allowlist
- Admin API key protection

### 📊 Observability

- OpenTelemetry distributed tracing (Jaeger / Zipkin)
- Prometheus metrics export
- Structured logging via [`logforth`](https://crates.io/crates/logforth)

### 🎨 Admin & Management

- RESTful Admin API with OpenAPI spec + Scalar UI
- React-based Admin Dashboard
- Model CRUD / API Key CRUD / Chat Playground
- etcd-backed dynamic configuration (no restarts needed)

---

## Architecture

<a href="#architecture"><img src="docs/images/architecture_basic.svg" alt="AISIX Architecture" width="100%" /></a>

---

## Quick Start

*Will link to new documentation about development setup and contribution guidelines*

---

## Development

*Will link to new documentation about development setup and contribution guidelines*

### Prerequisites

- Rust (latest stable/nightly version)

### Build & Run

1. Build UI

    ```bash
    cd ui
    pnpm install --frozen-lockfile
    pnpm build

    ## Or if you don't want to, then create a stub folder.
    ## Run this command in the root directory of the project.
    mkdir -p ui/dist
    ```

2. Build gateway

    ```bash
    cargo run
    ```

## Roadmap
- [ ] Load Balancing / Fallback across providers
- [ ] Prompt caching
- [ ] Cost tracking & usage analytics
- [ ] More providers (Azure, Bedrock, Ollama...)
- [ ] Kubernetes Helm chart
- [ ] New protocol support
    - [ ] OpenAI Responses API
    - [ ] Anthropic Messages API
    - [ ] Google Gemini GenerateContent API
- [ ] Multimodal APIs: Image, audio, video
- [ ] MCP proxy

## Community
<!-- Inspired by Apache APISIX and Kong community entry points -->

- Use [GitHub Discussions](https://github.com/api7/ai-gateway-stash/discussions) for questions, ideas, and architecture discussions
- Use [GitHub Issues](https://github.com/api7/ai-gateway-stash/issues) for bug reports, feature requests, and actionable tasks
- Follow repository activity for ongoing documentation and product updates

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
