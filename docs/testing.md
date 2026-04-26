# Testing

This document describes the testing strategy, harness layout, and CI
gating model for the aisix AI Gateway. Spec §0.3 mandates >90%
combined line coverage with end-to-end tests driven by GitHub
Actions.

## 1. Test pyramid

```
                 ┌─────────────────────────┐
                 │   Playwright (UI E2E)   │   small — UI flows only
                 ├─────────────────────────┤
                 │   Vitest E2E (Rust bin) │   medium — full request paths
                 ├─────────────────────────┤
                 │   Cargo integration     │   medium — per-crate, real backends
                 ├─────────────────────────┤
                 │   Cargo unit tests      │   large — fast, hermetic
                 └─────────────────────────┘
```

Coverage targets, all measured with line coverage:

| Layer | Target | Tooling |
|---|---|---|
| Rust unit | ≥85% per crate | `cargo-llvm-cov` |
| Combined (rust unit + E2E LCOV merge) | ≥90% | `lcov-result-merger` + the `coverage-gate` job |
| Frontend unit | informational | `@vitest/coverage-v8` |

The gate is enforced by the `coverage-gate` GitHub Actions job. It
merges the LCOV artifacts from `rust-unit` and `e2e`, computes the
combined line coverage, and fails the build when below the configured
threshold (90% in steady state, soft during scaffold milestones).

## 2. Rust unit tests

Located inside each crate as `#[cfg(test)] mod tests` blocks. Run
locally with:

```bash
cargo test --workspace                          # everything
cargo test --workspace --all-features           # incl. redis-feature paths
cargo test -p aisix-cache                       # one crate
cargo test -p aisix-obs langfuse::tests         # one module
```

Coverage:

```bash
cargo llvm-cov                                  # text summary
cargo llvm-cov --html                           # browseable HTML report
cargo llvm-cov --workspace --all-features --lcov --output-path lcov.info
```

Conventions:

- One `#[cfg(test)] mod tests` per file with the unit under test.
- Test names describe the scenario in plain English:
  `rejects_unknown_top_level_fields`, not `test_validate_3`.
- AAA structure (Arrange / Act / Assert) — not enforced, but
  prevailing.
- Use `wiremock` for HTTP-level mocking of upstream providers
  (already a workspace dep).
- Use `rstest` for parameterised tests where it improves clarity.

## 3. Cargo integration tests

Lives in each crate's `tests/` directory; each file is its own binary.
Used when a test needs a real external service.

Current integration suites:

- `crates/aisix-cache/tests/redis_integration.rs` — runs against a
  real Redis. Requires `CACHE_TEST_REDIS_URL`; tests no-op silently if the
  env var is unset, so local `cargo test` stays hermetic.

CI exposes the env vars these tests need:

| Test | Service | Env var |
|---|---|---|
| `redis_integration` | `redis:7-alpine` | `CACHE_TEST_REDIS_URL=redis://127.0.0.1:6379` |

Future suites that need etcd, an OTLP collector, or a Langfuse mock
will follow the same pattern: a service container in CI, a no-op
guard for local dev.

## 4. Vitest E2E harness

Lives in `tests/e2e/`. Each test:

1. Pings etcd; skips if unreachable.
2. Picks two free TCP ports (proxy + admin).
3. Generates a fresh admin key (UUID) and a fresh etcd prefix
   (`/aisix-e2e-<uuid>`) so concurrent runs cannot collide.
4. Writes a YAML config to a temp dir.
5. Spawns `target/debug/aisix --config <path>`.
6. Waits up to 10 s for `/health` and `/admin/v1/health` to respond.
7. Drives the request scenario via the Admin and Proxy clients.
8. Tears down: SIGTERM + 3 s grace + SIGKILL fallback, plus etcd
   prefix cleanup.

### 4.1 Layout

```
tests/e2e/
├── package.json
├── vitest.config.ts
├── tsconfig.json
└── src/
    ├── harness/
    │   ├── app.ts             # spawn + wait + cleanup
    │   ├── admin.ts           # typed Admin client
    │   ├── proxy.ts           # typed Proxy client
    │   ├── etcd.ts            # JSON gateway client (probe + prefix delete)
    │   ├── http.ts            # undici wrapper that ignores HTTP_PROXY
    │   ├── ports.ts           # free-port picker
    │   ├── upstream-openai.ts # node http mock with stream support
    │   └── index.ts           # barrel re-exports
    └── cases/
        └── *.test.ts          # one file per scenario
```

### 4.2 Running locally

```bash
cd tests/e2e
npm install
# Bring up etcd (one-time, leave running):
docker run --rm -d -p 2379:2379 quay.io/coreos/etcd:v3.6.1 \
  etcd --listen-client-urls http://0.0.0.0:2379 \
       --advertise-client-urls http://0.0.0.0:2379
# Then:
npm test
```

Tests skip silently if etcd is not reachable, so the suite can also
run unconditionally without manual setup — it just doesn't assert
anything in that case.

### 4.3 Writing a new case

```ts
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient, ProxyClient, EtcdClient,
  spawnApp, startOpenAiUpstream, waitConfigPropagation,
  type OpenAiUpstream, type SpawnedApp,
} from "../harness/index.js";

describe("my feature", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({ nonStreamBody: { /*…*/ } });
    app = await spawnApp();
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("does the thing", async (ctx) => {
    if (!etcdReachable || !app || !upstream) { ctx.skip(); return; }
    // … your scenario here
  });
});
```

The contract you must respect:

- **Per-test isolation** — never reuse `app` or `upstream` between
  describe blocks; each `beforeAll`/`afterAll` pair is independent.
- **Wait for propagation** — call `await waitConfigPropagation()`
  (500 ms) between an Admin write and a Proxy read.
- **Clean teardown** — always `await app?.exit()` in `afterAll`,
  even if `beforeAll` failed (use the `?.` operator so undefined
  doesn't throw).

## 5. Mock upstreams

`startOpenAiUpstream(opts)` boots a standalone Node HTTP server that
returns the same canned response for every method/path combination.
Use it whenever a test needs to assert the proxy actually reached
the upstream and what it sent.

Options:

| Option | Effect |
|---|---|
| `nonStreamBody` | JSON body returned for non-streaming requests (default: a minimal `chat.completion` shape) |
| `streamEvents` | Array of pre-stringified SSE event payloads; presence implies streaming response |
| `responseDelayMs` | Delay before first byte |
| `eventDelayMs` | Delay between SSE events |
| `status` | Override response status (e.g. `500` to test error mapping) |
| `errorBody` | Body returned when `status >= 400` |
| `disconnectAfterEvents` | Drop the connection mid-stream after N events |

The mock records every received request as `{method, path, headers,
body}` in `upstream.receivedRequests`, so a single assertion can
verify both the request shape (`headers.authorization === 'Bearer
sk-real-…'`) and the path (`/v1/chat/completions`).

A future PR will add an Anthropic-shaped mock with the same surface
plus content-block / tool_use builders.

## 6. CI workflow

`.github/workflows/ci.yml` defines six jobs that fan out from an
initial commit:

```
lint ─┬─ rust-unit (with redis service)
      │
      ├─ build-ui ─┬─ build-bin ─ e2e (with etcd + redis services)
      │
      └─ … ─────────────────────────────── coverage-gate
```

Job descriptions:

| Job | Purpose |
|---|---|
| `lint` | `cargo fmt --check`, `cargo clippy -D warnings`, `pnpm typecheck` |
| `rust-unit` | `cargo llvm-cov --workspace --all-features` → uploads `lcov-unit.info`. Spins a `redis:7-alpine` service so `CACHE_TEST_REDIS_URL` integration tests can run |
| `build-ui` | `pnpm build` in `ui/` → uploads `ui-dist` artifact for the next job |
| `build-bin` | `cargo build` of `aisix-server` with `RUSTFLAGS=-C instrument-coverage` → uploads the binary artifact for the e2e job |
| `e2e` | Spins etcd + redis services, downloads the binary artifact, runs Vitest. Currently `continue-on-error: true` while the harness stabilises across CI runners |
| `coverage-gate` | Merges `lcov-unit.info` + `lcov-e2e.info` → fails if below threshold |

## 7. Frontend tests

`ui/` ships a small Vitest unit suite for hooks and pure utilities.
Heavier UI flows belong in Playwright (planned, not yet wired). The
build-ui job runs `pnpm test` plus `pnpm build` so a broken build
fails CI before the binary ever picks up the static assets.

## 8. Coverage policy

The `coverage-gate` job is the source of truth. It reads
`COVERAGE_THRESHOLD` (default `90`) and fails the build below that
percentage on the merged LCOV.

We deliberately do **not** enforce per-crate thresholds — some crates
(`aisix-server`, glue code) are exercised primarily by E2E tests and
their unit-only coverage will always lag. The combined number is
what matters for product quality.

When a PR drops combined coverage:

1. Identify the uncovered region in the failing diff
   (`cargo llvm-cov --html` locally).
2. Either add a unit test that pins the missing branch, or add an
   E2E case that walks the new code path.
3. If the new code is genuinely defensive (`unreachable!()`,
   `expect("invariant")`), document why coverage is acceptable and
   adjust the gate threshold conservatively in a follow-up.

## 9. See also

- [`architecture.md`](./architecture.md) — the system being tested.
- [`api-proxy.md`](./api-proxy.md), [`api-admin.md`](./api-admin.md)
  — the contracts the tests pin.
- `.github/workflows/ci.yml` — the canonical CI definition.
