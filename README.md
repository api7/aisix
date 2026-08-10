<div align="center">

# AISIX AI Gateway

### The open-source, Rust-native AI gateway for LLMs and AI agents

**One OpenAI-compatible API in front of every model.** Route, govern, secure, cache, and
observe all your LLM and AI-agent traffic from a single control point — shipped as one
static binary with low per-request overhead. Run it in your infrastructure for free,
forever.

*Built by the original creators of [Apache APISIX](https://apisix.apache.org/).*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Built with Rust](https://img.shields.io/badge/Built%20with-Rust-orange.svg)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/badge/docs-read-3aa757.svg)](https://docs.api7.ai/ai-gateway/)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2.svg)](https://discord.gg/dUmRZ7Rvf)
[![Website](https://img.shields.io/badge/website-api7.ai-1a73e8.svg)](https://api7.ai/ai-gateway)

[**Start free**](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=ai-gateway) ·
[**Documentation**](https://docs.api7.ai/ai-gateway/) ·
[**Quickstart**](https://docs.api7.ai/ai-gateway/getting-started/gateway-quickstart) ·
[**AISIX Cloud**](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud) ·
[**Roadmap**](ROADMAP.md)

<br>

<img src="assets/aisix-architecture.svg" alt="AISIX AI Gateway architecture — one OpenAI- or Anthropic-compatible API in front of OpenAI, Anthropic, Gemini/Vertex, Bedrock, Azure OpenAI, and DeepSeek, with API key auth, rate and token limits, guardrails, caching, routing and failover, and observability in between" width="100%">

</div>

---

**AISIX AI Gateway** is a Rust-native gateway that puts a single, OpenAI-compatible API in
front of every LLM provider — OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure OpenAI,
DeepSeek, and any OpenAI-compatible endpoint. It gives platform teams one place to route,
govern, secure, and observe LLM traffic, with first-class SSE streaming and low gateway
overhead.

It runs as a **single static binary** — low cold-start, lock-free config reads, and hot
configuration reloads with no restarts: declare resources in one `resources.yaml` and
reload on `SIGHUP`, or point the gateway at etcd for a multi-replica cluster. Run the
open-source gateway in your infrastructure, or connect it to
**[AISIX Cloud](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud)**
for centralized management with team governance, budgets, audit, and a dashboard.

> **AISIX AI Gateway (this repo)** is the open-source product. It runs without a control
> plane using declarative configuration or etcd. When connected to
> **[AISIX Cloud](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud)**,
> the same gateway serves as the data plane. AISIX Cloud adds a commercial control plane,
> either hosted by API7 (**Hybrid Cloud**) or hosted by you in your infrastructure
> (**On-Premises**). In both options, the gateway runs in your environment and calls
> providers directly; live AI traffic does not pass through the control plane or API7.
> The proxy API is identical throughout.
> **[Talk to us about AISIX Cloud →](https://api7.ai/contact?utm_source=github&utm_medium=readme&utm_campaign=cloud)**

## ⚡ Quickstart

One container. No control plane, no database, no configuration store — the gateway reads
every dynamic resource from one declarative `resources.yaml`.

```yaml
# config.yaml
resources_file: /etc/aisix/resources.yaml
proxy:
  addr: "0.0.0.0:3000"
admin:
  enabled: false          # a declarative gateway needs no admin listener
observability:
  metrics:
    prometheus:
      enabled: true
      addr: "0.0.0.0:9090"
```

```yaml
# resources.yaml
_format_version: "1"

provider_keys:
  - display_name: openai-main
    provider: openai
    api_key: ${OPENAI_API_KEY}        # interpolated from the environment

models:
  - display_name: my-model
    provider: openai
    model_name: gpt-4o-mini
    provider_key: openai-main

api_keys:
  - display_name: local-dev
    key_env: CALLER_API_KEY           # hashed at load; the plaintext is never stored
    allowed_models: ["my-model"]
```

```bash
export OPENAI_API_KEY="YOUR_PROVIDER_KEY"
export CALLER_API_KEY="YOUR_CALLER_KEY"

docker run -d --name aisix \
  --platform linux/amd64 \
  -v "$(pwd)/config.yaml:/etc/aisix/config.yaml:ro" \
  -v "$(pwd)/resources.yaml:/etc/aisix/resources.yaml:ro" \
  -e OPENAI_API_KEY -e CALLER_API_KEY \
  -p 3000:3000 -p 127.0.0.1:9090:9090 \
  ghcr.io/api7/aisix:latest        # proxy → :3000, metrics + status → :9090
#                                  ^ the metrics/status listener is unauthenticated;
#                                    keep it on loopback or a private network
```

Then call the gateway exactly like OpenAI:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"}]}'
```

Edit `resources.yaml` and send `SIGHUP` (`docker kill -s HUP aisix`) to apply changes with
no restart — an invalid file is rejected whole and the last good configuration keeps
serving. Check a file before booting with `aisix validate --resources resources.yaml`.

Full walkthrough: the
[Gateway Quickstart](https://docs.api7.ai/ai-gateway/getting-started/gateway-quickstart) ·
every field: the [resources file reference](https://docs.api7.ai/ai-gateway/reference/resources-file).
For a multi-replica cluster, point the gateway at etcd instead — `resources_file` and
`etcd` are mutually exclusive.

## ✨ Why AISIX

- **One API, every model.** Speak the OpenAI *or* Anthropic wire format in; the gateway
  translates to whichever provider each model points at. Point an OpenAI or Claude SDK at
  one `base_url` and switch models without changing code.
- **A real gateway, in Rust.** Single static binary, low cold-start, lock-free config reads
  on the hot path, native streaming.
- **Open source, free forever.** Apache-2.0 licensed and built to run in your
  infrastructure. Choose AISIX Cloud when you want centralized management through a
  control plane and dashboard.
- **Production controls built in.** Routing & failover, rate limits, guardrails, caching,
  and observability ship in the box. (Budgets and spend caps are an AISIX Cloud feature —
  the gateway enforces the control plane's decisions.)

## 🧩 Features — available today

Covered by 185 end-to-end test files (503 cases) that run against real gateway processes.

- **OpenAI-compatible proxy** (`:3000`) — `chat/completions`, `completions`, `responses`,
  `embeddings`, `rerank`, `images/generations`, `audio/{speech,transcriptions,translations}`,
  `videos` (submit → poll → fetch), `files`, `batches`, `fine_tuning/jobs`, `realtime`,
  `GET /v1/models`, plus a root-level `/passthrough/:provider/*` escape hatch. Native SSE streaming,
  tool/function calling, JSON mode, vision/multimodal input, and reasoning-content support.
- **Anthropic Messages API** — `POST /v1/messages` as a first-class route, working against
  **any** configured upstream: requests and responses (including streaming) are translated
  both ways when a model points at a non-Anthropic provider.
- **Routing & failover** — virtual/routing models with six strategies: `round_robin`,
  `weighted` (with sticky/canary hashing), `failover`, plus metric-based `least_cost`,
  `least_latency`, and `least_busy`. Retry budgets, cooldowns, tag-conditional targets,
  and per-attempt timeouts.
- **Ensemble models** — fan one request out to a panel of models concurrently, then have a
  judge model synthesize a single answer, with a minimum-successful-responses threshold.
- **Semantic routing** — one virtual model that dispatches by the *meaning* of each
  request: it embeds the prompt, scores it against per-route example utterances, and routes
  to the best match (or a default). See the
  [semantic routing docs](https://docs.api7.ai/ai-gateway/routing/semantic-routing).
- **Rate limiting & concurrency** — RPS/RPM/RPH/RPD + TPM/TPD + concurrency caps,
  AND-combined across caller keys, models, and policy scopes (`api_key` / `model` / `team` /
  `member` / `team_member`). Counters are per-process by default, or shared across replicas
  with the Redis backend.
- **Guardrails** — content-policy enforcement on input and output, in-process or through a
  provider: keyword/regex, built-in PII detection and redaction, Presidio, Lakera, OpenAI
  Moderation, AWS Bedrock Guardrails, Azure AI Content Safety (Prompt Shield + text
  moderation), and two Alibaba Cloud services. A block returns `422 content_filter`;
  monitor mode records what would have happened without blocking.
- **Caching** — exact-match response cache with per-policy TTL and model/key scope matchers;
  memory and Redis backends; cost-saved telemetry on every hit. Separately, **automatic
  prompt caching** can be enabled per direct Anthropic model to inject cache breakpoints, so
  callers get provider-side prompt discounts without changing their requests.
- **MCP gateway** — register upstream MCP servers as first-class resources and front them
  all behind one endpoint (`/mcp`), with the gateway holding each server's upstream
  credential, namespacing tools per server, and enforcing per-caller tool access. Also
  exposes a REST API as MCP tools from its OpenAPI description.
- **A2A agent gateway** — front A2A (Agent-to-Agent) agents at `/a2a/:agent`, serving each
  agent's card with URLs rewritten to the gateway, over JSON-RPC 2.0.
- **Inbound authentication** — caller API keys (SHA-256 hashed, model allowlists, expiry,
  rotation), or OIDC/JWT bearer tokens validated against registered providers (Entra ID,
  Okta, Google Workspace, or any OIDC issuer) with JWKS caching.
- **Observability** — Prometheus `/metrics`, structured per-request access logs, usage
  events, OTLP/GenAI span export (Langfuse, Honeycomb, Grafana Cloud, or any OTLP receiver),
  plus dedicated Datadog and Aliyun SLS log exporters and object-storage (S3/GCS/Azure Blob)
  telemetry.
- **Declarative configuration** — one `resources.yaml` carries all ten resource collections
  (provider keys, models, caller keys, guardrails, MCP servers, A2A agents, cache policies,
  observability exporters, rate-limit policies, OIDC providers), validated against the same
  JSON Schemas the gateway uses at runtime. `aisix validate` checks a file offline; `SIGHUP`
  reloads it atomically.
- **Operational endpoints** — `/livez` and `/readyz` on the proxy listener; `/status/config`,
  `/status/ready`, `/status/models`, and Prometheus `/metrics` on a dedicated metrics
  listener (`:9090`). The admin listener (`:3001`) additionally serves a **read-only**
  resource surface, OpenAPI 3 with a Scalar UI, and a playground. Resources are managed
  declaratively — through the `resources_file` (reloaded on SIGHUP) or direct etcd
  writes — not through the admin listener; its former write endpoints were removed.

## 🔌 Supported providers

AISIX dispatches through **five native adapter families** — distinct wire-protocol bridges,
not one generic relabel. Whatever the upstream protocol, the client-facing API stays
OpenAI-shaped.

| Adapter family | Reaches | Wire shape · auth |
|---|---|---|
| `openai` | OpenAI **+ any OpenAI-compatible vendor** — DeepSeek, Groq, Mistral, Together, Fireworks, Perplexity, vLLM, Ollama, or self-hosted OpenAI-compatible endpoints | OpenAI chat completions · Bearer |
| `anthropic` | Anthropic Claude | Anthropic Messages · `x-api-key` |
| `bedrock` | AWS Bedrock — Anthropic, Meta Llama, Mistral, Cohere, Amazon Titan/Nova, AI21 | Bedrock Converse + `/invoke` · SigV4 |
| `vertex` | Google Vertex AI (Gemini) | Vertex `:generateContent` · OAuth2 |
| `azure-openai` | Azure OpenAI | Azure deployments · api-key / Entra ID |

Plus specialized handling for vendor quirks (e.g. DeepSeek reasoning content) and dedicated
**rerank / embeddings** vendors (Cohere, Jina). Details in
[adapter protocol families](https://docs.api7.ai/ai-gateway/providers/adapters).

## ☁️ Open source vs AISIX Cloud

Same gateway binary, same proxy API — in every form the gateway runs in your environment.
**AISIX Cloud** adds a commercial control plane, either hosted by API7
(**Hybrid Cloud**) or hosted in your infrastructure (**On-Premises**).

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="assets/console-overview.png" alt="AISIX Cloud overview — requests, latency p50/p99, error rate and cost today, with a 7-day request-and-cost trend and data-plane health" width="100%"><br>
      <sub><b>Overview</b> — traffic, latency, error rate &amp; spend at a glance</sub>
      <br><br>
      <img src="assets/console-models.png" alt="AISIX Cloud models — alias an upstream LLM per provider (OpenAI, Anthropic, AWS Bedrock, DeepSeek) with model IDs and per-model rate limits" width="100%"><br>
      <sub><b>Models</b> — one alias per upstream: OpenAI, Anthropic, Bedrock, DeepSeek…</sub>
      <br><br>
      <img src="assets/console-guardrails.png" alt="AISIX Cloud guardrails — pre-input and post-output content policies (keyword blocklist, Azure Content Safety, AWS Bedrock) that block on violation" width="100%"><br>
      <sub><b>Guardrails</b> — pre-input &amp; post-output policies, block on violation</sub>
    </td>
    <td width="50%" valign="top">
      <img src="assets/console-playground.png" alt="AISIX Cloud playground — pick a model, set system and user prompts, run, and read the response with live token and cost metering" width="100%"><br>
      <sub><b>Playground</b> — test any model with live token &amp; cost metering</sub>
      <br><br>
      <img src="assets/console-observability.png" alt="AISIX Cloud observability exporters — fan out chat-completion telemetry to OTLP, Datadog and object storage, with per-target delivery health" width="100%"><br>
      <sub><b>Observability</b> — fan out traces &amp; logs to OTLP, Datadog, object storage</sub>
      <br><br>
      <img src="assets/console-budgets.png" alt="AISIX Cloud budgets — organization and per-environment spend caps with progress bars, hard-stop versus warn-only, including an over-budget policy" width="100%"><br>
      <sub><b>Budgets</b> — hard-stop spend caps with warn-only tiers</sub>
    </td>
  </tr>
</table>

<p align="center">
  <em>The AISIX Cloud dashboard — overview metrics, multi-provider models, guardrails, budgets (with hard-stop spend caps), and observability exporters, across all your gateways.</em>
  <br><br>
  <a href="https://aisix-demo.api7.ai/"><b>▶ Try the live dashboard demo — aisix-demo.api7.ai</b></a>
</p>

| | Open-source gateway (this repo) | [AISIX Cloud](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=cloud) (Hybrid Cloud or On-Premises) |
|---|---|---|
| Price | Free · Apache-2.0 · forever | Commercial — [talk to us](https://api7.ai/contact?utm_source=github&utm_medium=readme&utm_campaign=pricing) |
| Configuration | Declarative `resources.yaml`, or etcd for a cluster | Dashboard + Cloud Admin API, multi-environment |
| Tenancy | Single instance / namespace | Org → Team → Member → Environment |
| Provider keys | In the resources file as `${VAR}` env references, or in etcd | Envelope-encrypted at rest, write-only, in-place rotation |
| Inbound auth | Caller keys (SHA-256 hashed, model allowlists, expiry), or OIDC/JWT bearers | Same, plus masked reveal, key ownership, and PATs |
| Budgets | — (rate and token limits only) | Per key / provider / env / org / team, hard-stop & alerts |
| RBAC | Admin key = read-only resource surface | Org roles (owner / admin / member), invites |
| Audit log | — | Full org-scoped audit with diff viewer |
| Usage & cost | Export logs, metrics, and usage events yourself | Managed usage views, model pricing catalog, spend reporting |
| Surface | Status endpoints, OpenAPI read surface, playground | Full dashboard + per-environment playground |

→ **Want the AISIX Cloud control plane, governance, budgets, and dashboard?**
**[Talk to API7](https://api7.ai/contact?utm_source=github&utm_medium=readme&utm_campaign=cloud)** about
Hybrid Cloud or On-Premises, or **[book a demo](https://api7.ai/contact?utm_source=github&utm_medium=readme&utm_campaign=demo)**.

## 🏗️ Architecture

A single Cargo workspace; the `aisix-server` crate builds one binary named `aisix` that
wires the crates together.

```text
crates/
├── aisix-core           Config, snapshot, resource model, resources.yaml source, errors
├── aisix-etcd           Config provider + watch supervisor
├── aisix-gateway        Hub & bridge, SSE parser, provider trait
├── aisix-proxy          /v1/*, /mcp, /a2a handlers, routing, middleware
├── aisix-admin          Read-only resource surface + playground + OpenAPI
├── aisix-provider-*     openai · anthropic · azure-openai · bedrock · vertex
├── aisix-mcp            MCP gateway — server registry, tool ACL, transports
├── aisix-a2a            A2A agent gateway — agent cards, JSON-RPC bridge
├── aisix-ratelimit      fixed-window + token accounting + concurrency (local | redis)
├── aisix-cache          memory + redis backends
├── aisix-redis          shared Redis connection for cache + rate limits
├── aisix-guardrails     pre/post content-policy hooks
├── aisix-obs            tracing, metrics, access log, exporters
└── aisix-server         the `aisix` binary — bootstrap + CLI
```

## 🗺️ Roadmap

Highlights on the [roadmap](ROADMAP.md); tracked live in
[issues](https://github.com/api7/aisix/issues):

- Semantic (embedding-similarity) response caching
- More observability sinks — Langsmith, Helicone, Slack alerts
- Prompt templates managed as gateway resources
- Llama-Guard as a guardrail provider

Shipped since this list was last written: the MCP gateway, the A2A agent gateway,
OIDC/JWT inbound auth, Redis-backed distributed rate limiting, and the Lakera, Presidio,
PII, and OpenAI Moderation guardrails — see **Features** above.

## 🛠️ Development

Prerequisites: the Rust toolchain pinned in `rust-toolchain.toml`. Docker is only needed
for the tests that exercise etcd, Redis, or provider emulators.

```bash
cargo check --workspace
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Coverage (matches the CI gate)
cargo llvm-cov --workspace --lcov --output-path lcov.info

# Run locally against a resources.yaml (no etcd needed). Copy the Quickstart's two files
# and change resources_file to the local path, e.g. resources_file: ./resources.yaml
cargo run -p aisix-server --bin aisix -- --config config.local.yaml

# Check a resources file without starting a listener
cargo run -p aisix-server --bin aisix -- validate --resources resources.yaml
```

## 💬 Community

- **Discord** — [discord.gg/dUmRZ7Rvf](https://discord.gg/dUmRZ7Rvf)
- **Issues & discussions** — [github.com/api7/aisix/issues](https://github.com/api7/aisix/issues)
- **Contributing** — [CONTRIBUTING.md](CONTRIBUTING.md)
- **Website** — [api7.ai/ai-gateway](https://api7.ai/ai-gateway?utm_source=github&utm_medium=readme)

If AISIX is useful to you, a ⭐ helps other engineers find it.

## 📄 License

[Apache 2.0](LICENSE).
