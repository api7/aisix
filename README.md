# aisix — AI Gateway

> A single-binary, Rust-native AI gateway. OpenAI-compatible proxy + Admin API.
> Config lives in etcd. Lock-free reads. First-class streaming. >90% E2E coverage gate.

`aisix` is a Rust-native AI inference gateway: low cold-start, native streaming, single static binary.

It runs in two modes from the same binary:

- **Standalone** — operator drives configuration through the Admin API on `:3001`. Self-hosted; no control plane.
- **Managed** — runs as a Data Plane (DP) tenant of the [AISIX-Cloud](https://github.com/api7/AISIX-Cloud) control plane (CP). Admin API is unbound; configuration arrives over an mTLS-authenticated etcd watch from cp-api.

## What's shipped today

Surfaces and capabilities currently in main:

- **Proxy API (`:3000`)** — OpenAI-compatible
  - Chat completions: `POST /v1/chat/completions` with native SSE streaming
  - Anthropic-shape: `POST /v1/messages` (Claude SDK works against `base_url`)
  - OpenAI Responses: `POST /v1/responses`
  - Embeddings: `POST /v1/embeddings`
  - Rerank: `POST /v1/rerank`
  - Audio: `POST /v1/audio/transcriptions`, `/translations`, `/speech`
  - Images: `POST /v1/images/generations`
  - Listing: `GET /v1/models`
  - Provider passthrough escape hatch: `ANY /passthrough/{provider}/*rest`

- **Providers** — OpenAI, Anthropic, Gemini, DeepSeek (one bridge crate per vendor; OpenAI-shape `model: "<provider>/<id>"` selects the bridge)

- **Admin API (`:3001`)** — CRUD on the routing-related entities, JSON-Schema validated, OpenAPI 3 + Scalar UI at `/admin/openapi-scalar`
  - `/admin/v1/models`
  - `/admin/v1/apikeys` (+ `POST .../rotate`)
  - `/admin/v1/provider_keys`
  - `/admin/v1/health` — per-model upstream health (Healthy / Degraded / Down)
  - `/playground/chat/completions` — in-process forward to the proxy router

  Guardrails, cache policies, and observability exporters exist as
  resource types in `aisix-core` and are honoured at runtime, but
  standalone CRUD over those still goes through direct etcd writes
  today — Admin handlers will follow.

- **Config plane** — etcd with watch-driven `ArcSwap` snapshot. Lock-free reads on the hot path. Schema-rejected rows logged + skipped, never fatal.

- **Caching** — moka in-process exact-match cache + cost-saved telemetry (cache_hit_saved_input/output_tokens). Per-policy TTL. `applies_to` matcher on model + api_key scopes (Stage 3). Cache key includes `tools`, `response_format`, `seed`, `stop`. Backends: `memory`, `redis` (via feature flag).

- **Guardrails** — input + output hooks, fail-open opt-in. Two kinds shipped: `keyword` (literal + regex blocklist), `bedrock` (AWS Bedrock Guardrails dispatch via `aws-sdk-bedrockruntime`). Live config from etcd; chain bypass recorded on telemetry for compliance audit.

- **Rate limiting** — fixed-window RPM/RPD + post-deduct TPM/TPD + concurrency semaphore. Two-phase commit so token cost is known before the counter advances. Per-ApiKey scope today.

- **Observability** — Prometheus `/metrics`, OTLP traces/metrics/logs export, per-request structured access log, Langfuse sender, per-env OTLP/HTTP fan-out exporter (managed-mode supports multi-vendor sinks driven by etcd config).

- **Telemetry events** — DP-side `UsageEvent` per request with cache_status, reasoning, provider-id detail, guardrail bypass reason. Posted to cp-api in managed mode; consumed by `/admin/v1/spend` in standalone.

- **Managed-mode bootstrap** — cert-bundle path (no `/dp/register` round-trip): cp-api signs the mTLS leaf at mint time and ships PEMs as env vars at `docker run`. Snapshot persisted to `config_cache.json` so the proxy survives CP outages and restarts (offline resilience per PRD-09 §9.7.2).

- **Per-key budgets** — `ApiKey.max_budget_usd` inline cap; managed mode delegates evaluation to cp-api `/dp/budget_check` with a 5 s LRU on the DP side.

## Workspace

```
crates/
├── aisix-core                 Config, Snapshot, ResourceEntry, errors
├── aisix-etcd                 ConfigProvider, watch supervisor
├── aisix-gateway              Hub & Bridge, SSE parser, provider trait
├── aisix-provider-openai
├── aisix-provider-anthropic
├── aisix-provider-gemini
├── aisix-provider-deepseek
├── aisix-proxy                /v1/* handlers + middleware
├── aisix-admin                CRUD + playground + OpenAPI
├── aisix-obs                  tracing, metrics, access log
├── aisix-ratelimit            fixed-window + semaphore
├── aisix-cache                in-mem + redis + qdrant
├── aisix-guardrails           pre/during/post hooks
└── aisix-server               single binary — bootstrap + CLI
```

## Standalone vs Managed (DP) — what's where

The same binary runs both modes; the table is about **which surface owns the feature**, not about whether it works at all in one mode.

> A few resources (Budget, Team, Member/Role, Audit Log, Billing) belong to the SaaS control plane and are intentionally **absent** from the standalone DP — standalone uses inline per-key alternatives (`ApiKey.max_budget_usd`, `ApiKey.rate_limit`).

| Capability | Standalone (DP only) | Managed (DP + AISIX-Cloud CP) |
|---|---|---|
| Configuration entry point | `/admin/v1/*` on `:3001`, static `admin_keys` bearer | Dashboard / cp-api `/api/*`, Better Auth session or PAT |
| Multi-tenant model | None — single instance, single namespace | Org → Team → Member → Environment hierarchy |
| ProviderKey storage | Plaintext `secret` in etcd (mTLS-only channel) | Master-key envelope-encrypted at rest, decrypted on projection |
| API key handling | Hash on create, plaintext shown once | Hash + masked / one-time reveal in dashboard, rotation flow |
| Budget enforcement | Per-ApiKey inline cap (`max_budget_usd`) | Per-ApiKey + per-ProviderKey + per-Environment + per-Org budgets, hard-stop / warn-only modes, alerts, audit |
| Audit log | None | Full org-scoped audit with diff viewer, RBAC-gated views |
| RBAC / Roles | None — admin key is binary access | Org-scoped roles (owner / admin / developer / viewer), invitations |
| Auth for proxy clients | Inbound `ApiKey` only | Inbound `ApiKey` only (proxy contract is identical) |
| Pricing / cost | Per-Model `cost.input_per_1k` / `cost.output_per_1k` | Pricing rows synced from models.dev + per-model overrides |
| Personal Access Tokens | None | `aisix_pat_*` for CLI / CI |
| Billing | None | Stripe portal handoff, plan management, metering |
| Guardrail / cache / exporter CRUD | `/admin/v1/*` direct write | Dashboard CRUD → cp-api validates → projects to env's etcd via outbox |
| DP cert provisioning | N/A — single-process, no DP/CP split | cp-api `/api/.../gateway_certificates` issues the mTLS bundle; DP boots with PEMs in env vars |
| etcd-side authz | N/A | env-prefix enforcement (etcdauth interceptor): each DP cert is scoped to `/aisix/<env>/`, can't read another tenant's keyspace |
| Playground | `POST /playground/chat/completions` (admin API) | Per-env playground in dashboard, audited |

The DP container image (`ghcr.io/api7/ai-gateway:main`) is the same in both modes. The `managed.enabled` flag in config selects the bootstrap path.

## Roadmap

Tracked as parent issues on this repo. P0 = pre-1.0 must-have, P1 = first follow-up wave, P2 = long-tail.

**P0 (in flight, blocks 1.0)**

- [#43] Wire guardrail config loading from etcd / YAML across all kinds
- [#44] AWS Bedrock provider (chat + embeddings + image)
- [#45] Azure OpenAI provider
- [#46] 429 from upstream triggers routing fallback
- [#48] Persist budget tracker (currently resets on restart)
- [#49] Real OpenTelemetry tracing (currently scaffold-only)
- [#50] Multimodal content blocks (vision, image input) on chat completions
- [#51] AWS Bedrock Guardrails as a first-class guardrail kind

**P1 (post-1.0)**

- [#47] Distributed (Redis-backed) rate limiting
- [#52] Lakera + Presidio + OpenAI Moderation guardrail trio
- [#53] JWT / OIDC auth for proxy clients (Entra ID / Google Workspace / Okta)
- [#54] Latency-based + cost-based + tag-based routing strategies
- [#55] Semantic cache (embedding-similarity matching)
- [#56] Per-team / per-user rate limits + budgets — depends on the SaaS team model landing in cp-api
- [#57] Helicone / Langsmith / Datadog logs / Slack alerts as exporter kinds
- [#58] MCP gateway (registration, transports, auth, access control, cost tracking)
- [#59] Vertex AI / Google AI Studio native generateContent (today is OpenAI-shape relabel)
- [#89] Manual cache-purge endpoint per policy
- [#90] pgvector semantic-cache backend (Stage 4b)

**P2 (long-tail)**

- [#60] ~95 long-tail provider integrations (Together, Fireworks, Replicate, …)
- [#61] ~25 long-tail guardrail integrations
- [#62] ~30 long-tail observability sinks
- [#63] Enterprise-tier features (SCIM, customer portal, prompt management, batch / files / fine-tune passthrough)

The full live tracker: [github.com/api7/ai-gateway/issues](https://github.com/api7/ai-gateway/issues).

## Development

Prerequisites: Rust toolchain (pinned in `rust-toolchain.toml`), Docker (for etcd).

```bash
# Rust workspace
cargo check --workspace
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Coverage (matches CI gate)
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --lcov --output-path lcov.info

# Run (scaffold — full startup arrives in PR #5)
cargo run -p aisix-server -- --config config.example.yaml
```

## License

MIT — see [LICENSE](LICENSE).
