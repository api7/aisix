<div align="center">

# AISIX AI Gateway

### The open-source, Rust-native AI gateway for LLMs and AI agents

**One OpenAI-compatible API in front of every model.** Route, govern, secure, cache, and
observe all your LLM and AI-agent traffic from a single control point — shipped as one
static binary with low per-request overhead. Self-host for free, forever.

*Built by the original creators of [Apache APISIX](https://apisix.apache.org/).*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Built with Rust](https://img.shields.io/badge/Built%20with-Rust-orange.svg)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/badge/docs-read-3aa757.svg)](docs/index.md)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2.svg)](https://discord.gg/dUmRZ7Rvf)
[![Website](https://img.shields.io/badge/website-api7.ai-1a73e8.svg)](https://api7.ai/ai-gateway)

[**Start free**](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=ai-gateway) ·
[**Documentation**](docs/index.md) ·
[**Quickstart**](docs/quickstart/self-hosted.md) ·
[**AISIX Cloud**](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud) ·
[**Roadmap**](docs/roadmap.md)

<br>

<img src="docs/assets/aisix-architecture.svg" alt="AISIX AI Gateway architecture — one OpenAI- or Anthropic-compatible API in front of OpenAI, Anthropic, Gemini/Vertex, Bedrock, Azure OpenAI, and DeepSeek, with API key auth, rate limits and budgets, guardrails, caching, routing and failover, and observability in between" width="100%">

</div>

---

**AISIX AI Gateway** is a Rust-native gateway that puts a single, OpenAI-compatible API in
front of every LLM provider — OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI,
DeepSeek, and any OpenAI-compatible endpoint. It gives platform teams one place to route,
govern, secure, and observe LLM traffic, with first-class SSE streaming and low gateway
overhead.

It runs as a **single static binary** — low cold-start, lock-free config reads, dynamic
configuration over etcd with no restarts. Run it **self-hosted and free**, or connect it to
**[AISIX Cloud](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud)**
for a managed control plane with team governance, budgets, audit, and a dashboard.

> **AISIX AI Gateway (this repo)** is the open-source core — the gateway/data plane.
> **[AISIX Cloud](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud)**
> is the managed SaaS that adds the multi-tenant control plane on top. The proxy API is
> identical in both. **New to AISIX Cloud? [Start free →](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud)**

## ⚡ Quickstart

AISIX is etcd-backed, so the fastest local run is Docker Compose (gateway + etcd). Grab the
ready-to-run `docker-compose.yml` and example `config.yaml` from the
[self-hosted quickstart](docs/quickstart/self-hosted.md), then:

```bash
docker compose up          # proxy → :3000, admin API → :3001
```

Configure a model and an API key through the admin API on `:3001`
([first model, first key, first request](docs/quickstart/first-model-first-key-first-request.md)),
then call the gateway exactly like OpenAI:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $AISIX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"}]}'
```

## ✨ Why AISIX

- **One API, every model.** Speak the OpenAI *or* Anthropic wire format in; the gateway
  translates to whichever provider each model points at. Point an OpenAI or Claude SDK at
  one `base_url` and switch models without changing code.
- **A real gateway, in Rust.** Single static binary, low cold-start, lock-free config reads
  on the hot path, native streaming.
- **Open-source core, free forever.** Apache-2.0, self-hostable end to end. Reach for
  AISIX Cloud only when you want the managed control plane.
- **Production controls built in.** Routing & failover, rate limits, budgets, guardrails,
  caching, and observability ship in the box.
- **Proven lineage.** Built by [API7](https://api7.ai) — the original creators of
  [Apache APISIX](https://apisix.apache.org/), a top-level project at the Apache Software Foundation.

## 🧩 Features — available today

Anchored to the [feature matrix](docs/overview/feature-matrix.md); covered by 90+ E2E tests.

- **OpenAI-compatible proxy** (`:3000`) — `chat/completions`, `responses`, `embeddings`,
  `rerank`, `images/generations`, `audio/{speech,transcriptions,translations}`,
  `GET /v1/models`, and a `passthrough/:provider/*` escape hatch. Native SSE streaming,
  tool/function calling, JSON mode, vision/multimodal input, and reasoning-content support.
- **Anthropic Messages API** — `POST /v1/messages` as a first-class route, working against
  **any** configured upstream: requests and responses (including streaming) are translated
  both ways when a model points at a non-Anthropic provider.
- **Routing & failover** — virtual/routing models, weighted load balancing, automatic
  failover, retry budgets, cooldowns, and per-attempt timeouts.
- **Rate limiting & concurrency** — RPM/RPD + TPM/TPD + concurrency caps, AND-combined
  across `ApiKey`, `Model`, and policy scopes (`api_key` / `model` / `team` / `member`).
- **Guardrails** — content-policy enforcement on input and output: keyword/regex
  (in-process), AWS Bedrock Guardrails, Azure AI Content Safety (Prompt Shield + text
  moderation), and Aliyun content moderation. A block returns `422 content_filter`.
- **Caching** — exact-match response cache with per-policy TTL and model/key scope matchers;
  memory and Redis backends; cost-saved telemetry on every hit.
- **Observability** — Prometheus `/metrics`, structured per-request access logs, usage
  events, OTLP/GenAI span export (Langfuse, Honeycomb, Grafana Cloud, or any OTLP receiver),
  plus dedicated Datadog and Aliyun SLS log exporters and object-storage (S3/GCS/Azure Blob)
  telemetry.
- **Admin API** (`:3001`) — JSON-Schema-validated CRUD for every resource, OpenAPI 3 with a
  Scalar UI at `/admin/openapi-scalar`, per-model upstream health, and a built-in playground.

## 🔌 Supported providers

AISIX dispatches through **five native adapter families** — distinct wire-protocol bridges,
not one generic relabel. Whatever the upstream protocol, the client-facing API stays
OpenAI-shaped.

| Adapter family | Reaches | Wire shape · auth |
|---|---|---|
| `openai` | OpenAI **+ any OpenAI-compatible vendor** — DeepSeek, Groq, Mistral, Together, Fireworks, Perplexity, vLLM, Ollama, self-hosted | OpenAI chat completions · Bearer |
| `anthropic` | Anthropic Claude | Anthropic Messages · `x-api-key` |
| `bedrock` | AWS Bedrock — Anthropic, Meta Llama, Mistral, Cohere, Amazon Titan/Nova, AI21 | Bedrock Converse + `/invoke` · SigV4 |
| `vertex` | Google Vertex AI (Gemini) | Vertex `:generateContent` · OAuth2 |
| `azure-openai` | Azure OpenAI | Azure deployments · api-key / Entra ID |

Plus specialized handling for vendor quirks (e.g. DeepSeek reasoning content) and dedicated
**rerank / embeddings** vendors (Cohere, Jina). Details in
[adapter protocol families](docs/reference/adapters.md). More providers on the
[roadmap](docs/roadmap.md).

## ☁️ Self-hosted vs AISIX Cloud

Same gateway binary, same proxy API. **AISIX Cloud** adds the managed control plane on top.

<p align="center">
  <img src="docs/assets/aisix-cloud-console-tour.gif" alt="AISIX Cloud dashboard tour — overview metrics, multi-provider models, guardrails, budgets with hard-stop spend caps, and observability exporters" width="100%">
  <br>
  <em>The AISIX Cloud dashboard — overview metrics, multi-provider models, guardrails, budgets (with hard-stop spend caps), and observability exporters, across all your gateways.</em>
</p>

| | Self-hosted (this repo) | [AISIX Cloud](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud) (managed) |
|---|---|---|
| Price | Free · Apache-2.0 · forever | Managed SaaS — [see pricing](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pricing) |
| Configuration | Admin API on `:3001` + etcd | Dashboard + API, multi-environment |
| Tenancy | Single instance / namespace | Org → Team → Member → Environment |
| Provider keys | Stored in etcd (mTLS channel) | Envelope-encrypted at rest |
| API keys | Hashed, shown once, rotation | Hashed + masked reveal, rotation, PATs |
| Budgets | Per-key rate limits; budgets are Cloud-only | Per key / provider / env / org, hard-stop & alerts |
| RBAC | Admin key = full access | Org roles (owner / admin / member), invites |
| Audit log | — | Full org-scoped audit with diff viewer |
| Billing & metering | — | Plans, usage metering, Stripe portal |
| Surface | OpenAPI + playground | Full dashboard + per-environment playground |

→ **Want the managed control plane, governance, budgets, and dashboard?**
**[Start free](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud)** or
**[book a demo](https://api7.ai/contact?utm_source=github&utm_medium=readme&utm_campaign=demo)**.

## 🏗️ Architecture

A single Cargo workspace; one binary (`aisix-server`) wires the crates together.

```
crates/
├── aisix-core           Config, snapshot, resource model, errors
├── aisix-etcd           Config provider + watch supervisor
├── aisix-gateway        Hub & bridge, SSE parser, provider trait
├── aisix-proxy          /v1/* handlers, routing, middleware
├── aisix-admin          CRUD + playground + OpenAPI
├── aisix-provider-*     openai · anthropic · azure-openai · bedrock · vertex
├── aisix-ratelimit      fixed-window + token accounting + concurrency
├── aisix-cache          memory + redis backends
├── aisix-guardrails     pre/post content-policy hooks
├── aisix-obs            tracing, metrics, access log, exporters
└── aisix-server         single binary — bootstrap + CLI
```

Deep dives: [protocol translation](docs/architecture/protocol-translation.md) ·
[snapshot & watch](docs/architecture/snapshot-and-watch.md) ·
[two-phase rate limiting](docs/architecture/two-phase-rate-limit.md).

## 🗺️ Roadmap

Highlights on the [roadmap](docs/roadmap.md); tracked live in
[issues](https://github.com/api7/ai-gateway/issues):

- 100+ additional provider integrations (Together, Fireworks, Replicate, …)
- Semantic (embedding-similarity) caching + pgvector backend
- More guardrails — Lakera, Presidio, OpenAI Moderation, Llama-Guard
- More observability sinks — Langsmith, Helicone, Slack alerts
- JWT / OIDC auth for proxy clients (Entra ID, Okta, Google Workspace)
- Distributed (Redis-backed) rate limiting
- MCP gateway — registration, transports, auth, cost tracking

## 🛠️ Development

Prerequisites: the Rust toolchain pinned in `rust-toolchain.toml`, plus Docker (for etcd).

```bash
cargo check --workspace
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Coverage (matches the CI gate)
cargo llvm-cov --workspace --lcov --output-path lcov.info

# Run locally (needs a reachable etcd + a config.yaml — see the quickstart)
cargo run -p aisix-server --bin aisix -- --config config.yaml
```

## 💬 Community

- **Discord** — [discord.gg/dUmRZ7Rvf](https://discord.gg/dUmRZ7Rvf)
- **Issues & discussions** — [github.com/api7/ai-gateway/issues](https://github.com/api7/ai-gateway/issues)
- **Website** — [api7.ai/ai-gateway](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme)

If AISIX is useful to you, a ⭐ helps other engineers find it.

## 📄 License

[Apache 2.0](LICENSE).
