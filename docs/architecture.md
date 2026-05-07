# Architecture

This document describes how the **aisix AI Gateway** is wired together at the
component, data, and process level. Audience: platform engineers and
architects evaluating, operating, or extending aisix. For the user-facing
API surface see [`api-proxy.md`](./api-proxy.md) and
[`api-admin.md`](./api-admin.md).

## 1. Goals & non-goals

**Goals**

- Single-binary deployment. No external runtime dependencies beyond etcd.
- OpenAI-compatible client surface — drop-in for any SDK that targets OpenAI.
- Lock-free hot path. Reads from the configuration snapshot must never block
  on a writer.
- First-class streaming. SSE responses pass through with controlled buffering
  and accurate end-of-stream telemetry.
- Operator-driven configuration. Models, ApiKeys, ProviderKeys, Guardrails,
  CachePolicies, and ObservabilityExporters all live in etcd and are
  mutated via the Admin API at runtime.
- >90% combined unit + E2E line coverage as a CI gate.

**Non-goals (today)**

- Per-tenant data residency.
- Native model hosting. aisix dispatches to upstream providers; it does not
  serve weights itself.
- A full BYOC workflow. Bring your own API keys to upstream providers and
  configure them through the ProviderKey entity.

## 2. Component model

```
                ┌──────────────────────────────────────────────┐
                │                  aisix binary                │
                │                                              │
   client ─────►│  :3000  Proxy router (OpenAI-compatible)     │
   (OpenAI SDK) │           │                                  │
                │           ▼                                  │
                │       Hub  ──►  Bridge (per provider)  ──►   │── upstream
                │           │     (OpenAI, Anthropic, Gemini,  │   provider
                │           │      DeepSeek, Cohere…)          │
                │           │                                  │
                │       ┌───┴────────────────────┐             │
                │       │     ProxyState         │             │
                │       │  (Arc-cloned per-req)  │             │
                │       │  • SnapshotHandle      │             │
                │       │  • Limiter             │             │
                │       │  • BudgetTracker       │             │
                │       │  • HealthTracker       │             │
                │       │  • Cache               │             │
                │       │  • GuardrailChain      │             │
                │       │  • LangfuseSender?     │             │
                │       └────────────────────────┘             │
                │           ▲                                  │
                │           │                                  │
   operator ───►│  :3001  Admin router (CRUD + Playground +    │
   (curl / SDK) │           │  OpenAPI Scalar)                 │
                │           ▼                                  │
                │       AdminState                             │
                │       • SnapshotHandle (shared with proxy)   │
                │       • ConfigStore (etcd writes)            │
                │       • Metrics (shared)                     │
                │                                              │
                │           │                                  │
                │           ▼                                  │
                │    Watch Supervisor  ◄── etcd watch          │
                │       │                  (one stream)        │
                │       ▼                                      │
                │    AisixSnapshot  (ArcSwap, lock-free read)  │
                └──────────────────────────────────────────────┘
                                      ▲
                                      │ watch + put / range / delete
                                      │
                                ┌─────┴─────┐
                                │   etcd    │
                                │  cluster  │
                                └───────────┘
```

Two HTTP listeners run inside the same process:

- **Proxy (`:3000`)** — public surface, accepts caller traffic.
- **Admin (`:3001`)** — operator surface, accepts admin-key-authenticated
  traffic.

Each listener is an `axum::Router` with its own state struct
(`ProxyState` / `AdminState`). They share the same `SnapshotHandle` and
the same `Metrics` handle, but otherwise their middleware stacks and
auth models are independent. See `crates/aisix-server/src/main.rs` for
the orchestration.

## 3. Configuration data plane

### 3.1 Snapshot model

The single source of truth for all runtime configuration is etcd. The
process boots an `EtcdConfigProvider`, reads every key under the
configured prefix, validates each entry against its JSON Schema, and
populates an `AisixSnapshot`:

```
AisixSnapshot
├─ models                   : ResourceTable<Model>
├─ apikeys                  : ResourceTable<ApiKey>
├─ provider_keys            : ResourceTable<ProviderKey>
├─ guardrails               : ResourceTable<Guardrail>
├─ cache_policies           : ResourceTable<CachePolicy>
└─ observability_exporters  : ResourceTable<ObservabilityExporter>
```

Each `ResourceTable<T>` is a primary `DashMap<id, ResourceEntry<T>>`
plus a secondary `DashMap<name, id>` for O(1) name → id lookup
(spec §3). Snapshots are atomically swapped via `ArcSwap` — readers
hold the snapshot they cloned at request start, writers publish a new
one without blocking any reader. This is the lock-free read path that
drives every per-request lookup.

### 3.2 Watch supervisor

A long-running tokio task (`Supervisor`) opens a single `etcd.watch()`
on the configuration prefix. Events are demultiplexed by sub-prefix
(`/models/`, `/apikeys/`, …) and dispatched into the right table.

Failure modes:

- **Connection drop** — exponential backoff (1 → 60 s) and reconnect.
- **Compaction** (`ErrGRPCCompacted`) — broadcast a `Resync` event,
  reload the entire prefix, atomically `ArcSwap::store` a fresh
  `AisixSnapshot`.
- **Schema-invalid entry on Put** — log a warning, skip the entry,
  leave previous good entries untouched.

End-to-end propagation latency from Admin write to readable snapshot
is bounded at 500 ms in practice; the E2E harness verifies this for
every CRUD test.

### 3.3 Why etcd

aisix lifts a pattern from Envoy / Higress / xDS: configuration is a
distributed state machine, not a file. etcd was chosen over PostgreSQL
(LiteLLM's choice) because it gives us:

- Native long-poll watch streams.
- A linearizable, transactional KV with leases.
- Multi-replica out of the box.
- Operational footprint many platform teams already run.

If a deployment cannot run etcd, the `ConfigProvider` trait is
deliberately small and a future filesystem-backed provider would slot
in cleanly behind it.

## 4. Hub & Bridge dispatch

The provider-facing layer is a two-tier "hub-and-bridge" design:

- **Hub** (`aisix-gateway::Hub`) holds an `Arc<dyn Bridge>` per provider
  prefix (`openai`, `anthropic`, `gemini`, `deepseek`).
- **Bridge** (per provider crate) implements three async methods:
  `chat`, `chat_stream`, and `embeddings`. It owns the format
  translation between the gateway's normalised `ChatFormat` and the
  upstream wire shape.

Two paths through the Hub:

1. **Native pass-through** — for upstream-native endpoints
   (e.g. `/v1/messages` against an Anthropic Model), the Bridge skips
   format translation and just forwards.
2. **Hub-translated** — for cross-provider client requests
   (e.g. an OpenAI client hitting an Anthropic Model), the Hub
   converts via the normalised `ChatFormat`, dispatches, and converts
   the response back.

This factoring means adding a new upstream is a self-contained crate:
implement `Bridge`, register on `Hub::new()`, ship. See
`crates/aisix-provider-anthropic` for a non-trivial example
(SSE event-state machine, content-block ↔ message translation,
cache_control passthrough).

### 4.1 Streaming (SSE)

For streaming requests:

1. The Bridge's `chat_stream` returns a
   `Stream<Item = Result<ChatChunk, BridgeError>>`.
2. The proxy wraps it in `axum::response::Sse::new(stream)` with
   `KeepAlive` enabled.
3. Each chunk is rendered via the `OpenAiChunkRenderer` so the wire
   format is byte-identical to OpenAI's reference client.
4. End-of-stream telemetry (token totals, first-token latency) is
   computed from accumulated deltas and emitted to metrics + access
   log + (optionally) Langfuse.

If the client disconnects mid-stream, post-processing still runs
inside `tokio::spawn` so usage is not lost.

## 5. Request middleware stack

Outer → inner, on the proxy router:

```
trace      → DefaultBodyLimit(10 MB) → Auth (ApiKey)
           → handler:
                model authz
              → guardrails.check_input
              → budget pre-check
              → rate-limit reservation (commit RPM)
              → routing pick (if virtual model)
              → bridge dispatch
              → guardrails.check_output (post)
              → rate-limit commit (post-deduct TPM, budget USD)
              → render headers
                  • x-ratelimit-{limit,remaining,reset}-{requests,tokens,concurrent}
                  • Retry-After (rate-limit only)
                  • x-aisix-call-id (request UUID)
                  • x-aisix-cache: hit|miss
              → emit access log + metrics + langfuse event
```

The two-phase rate limit is intentional. RPM is committed *before*
dispatch (we reserve the slot), TPM is post-deducted from the upstream
`usage` field. This is the only correct way to enforce TPM without
pre-tokenising every prompt locally.

## 6. Cache layer

`aisix-cache::Cache` is a small async trait — `get(key) -> Option<ChatResponse>`
and `put(key, value)`. The proxy looks up the cache before dispatch and
falls through on miss.

Two backends ship today:

- **MemoryCache** (default) — in-process via `moka`, LRU with a TTL.
- **RedisCache** (`feature = "redis"`) — single-node via
  `redis::aio::ConnectionManager`. JSON-encoded values under a
  configurable namespace prefix.

Streaming responses are not cached — there is no terminal value to
store. A future semantic cache PR will land Qdrant behind the same
trait.

The `CacheKey` is a stable hash of the request fingerprint
(model, messages, temperature, top_p, max_tokens). Anything else
(request id, deadlines, the caller's ApiKey, custom headers) is
excluded so two callers asking the same question hit the same entry.

## 7. Rate limit, budget, and health

Three separate trackers in `ProxyState`, all process-local for V1:

- **Limiter** — `aisix-ratelimit::Limiter`. Fixed-window RPM/TPM
  counters and a `Semaphore` for concurrency. Two-phase:
  `reserve()` commits RPM, `add()` commits TPM after the upstream
  responds. `peek()` is a read-only check used to render
  `x-ratelimit-*` headers without a side effect.
- **BudgetTracker** — per-ApiKey monthly USD spend. Driven off
  `Model.cost.{input,output}_per_1k` × the upstream `usage` field.
  `would_exceed` is a pre-check; `add` is the post-commit.
- **HealthTracker** — per-Model rolling window of consecutive
  upstream failures. Promotes to Degraded after 4 fails, Down after
  8. Read by the routing layer (cooldown) and exposed via
  `GET /admin/v1/health`.

A future PR can swap any of these behind a trait for Redis-backed
durability across replicas.

## 8. Observability

`aisix-obs` provides four sinks, layered cleanly:

- **`tracing`** — structured logs, default writer is `stderr`. Filter
  via `RUST_LOG`/`AISIX_LOG`. Spans get the W3C `traceparent` parent
  context if the inbound request carried one.
- **`metrics` + Prometheus** — counters and histograms exported at
  `/metrics` on the admin listener. The same `Metrics` handle is
  shared between proxy and admin so a single scrape sees every event.
- **OTLP** — opt-in tracer/meter/log providers exported to an OTLP
  collector. Pin set in `Cargo.toml` to avoid the well-known
  ecosystem version-skew traps.
- **Langfuse** — opt-in custom HTTP exporter that batches generation
  events (50 / s flush window) and POSTs them to
  `{host}/api/public/ingestion`. Designed to never block the request
  hot path; full queues drop the event with a debug log.

Five canonical metrics:

- `aisix_request_count` — labelled by provider, model, status, outcome.
- `aisix_request_latency` — histogram, end-to-end including body drain.
- `aisix_llm_latency` — histogram, upstream-only (excludes our own work).
- `aisix_token_count` — by direction (input / output) and model.
- `aisix_first_token_latency` — histogram, populated only for streaming
  requests (recorded at `idx == 0`).

## 9. Bootstrap & shutdown

```
1. Parse CLI (clap).
2. Load + validate config (YAML/TOML/JSON, AISIX__* env overrides).
3. Init tracing.
4. Connect to etcd (5 s × 5 retry).
5. Load initial AisixSnapshot.
6. Spawn watch Supervisor.
7. Build Hub + Limiter + Metrics + Cache + (optional Langfuse).
8. Build proxy router; build admin router (sharing snapshot + metrics).
9. Bind both TCP listeners; tokio::join! the two services with a
   common graceful-shutdown signal.
10. On SIGTERM/SIGINT: cancel supervisor, stop accepting, drain
   in-flight requests, flush metrics + traces, exit.
```

The whole sequence lives in `crates/aisix-server/src/main.rs`. The
contract is: any failure in steps 1-6 panics with a clear cause; once
the listeners bind in step 9 the process is committed to running until
a shutdown signal arrives.

## 10. Testing pyramid

- **Rust unit + integration tests** — per-crate, run by `cargo test`.
  Coverage measured with `cargo-llvm-cov`; the rust-unit job uploads
  an LCOV artifact.
- **TypeScript E2E harness** — `tests/e2e/` (Vitest). Each test spawns
  the binary against a unique etcd prefix (`/aisix-e2e-<uuid>`),
  writes via the Admin API, waits 500 ms for snapshot propagation,
  asserts via the Proxy API. The CI job spins etcd + Redis service
  containers.
- **Coverage gate** — `coverage-gate` job merges the rust-unit and
  e2e LCOV files and fails the build below the configured threshold
  (90% in steady state; soft during scaffold milestones).

See [`testing.md`](./testing.md) for the full test layout.

## 11. Crate boundaries (workspace)

| Crate | Responsibility |
|---|---|
| `aisix-core` | Config, snapshot, resource entries, errors. No I/O. |
| `aisix-etcd` | `EtcdConfigProvider`, watch supervisor, schema-validated put/range. |
| `aisix-gateway` | Hub, Bridge trait, SSE parser, normalised `ChatFormat`. |
| `aisix-provider-{openai,anthropic,gemini,deepseek}` | Per-provider Bridge impls. |
| `aisix-proxy` | `/v1/*` handlers, middleware, ProxyState, request rendering. |
| `aisix-admin` | Admin CRUD, playground, OpenAPI Scalar. |
| `aisix-obs` | `tracing` init, metrics, access log, OTLP, Langfuse. |
| `aisix-ratelimit` | Fixed-window limiter + concurrency semaphore. |
| `aisix-cache` | `Cache` trait, MemoryCache, RedisCache. |
| `aisix-guardrails` | Pre/during/post-call content policy hooks. |
| `aisix-server` | The single binary — bootstrap + signal handling. |

The dependency graph is strictly acyclic: `aisix-server` imports
everything; `aisix-proxy` and `aisix-admin` are siblings that share
`aisix-core` + `aisix-gateway` + `aisix-obs`; provider crates depend
only on `aisix-gateway` (and `aisix-core` for Provider enums).

## 12. Where to look next

- API surfaces: [`api-proxy.md`](./api-proxy.md), [`api-admin.md`](./api-admin.md).
- Test architecture: [`testing.md`](./testing.md).
- Spec: the canonical product brief lives in `ai-gateway.md` at the
  repository root.
