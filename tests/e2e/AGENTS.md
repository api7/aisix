# AGENTS.md — end-to-end specs

Applies to specs under `src/cases/`. The repo root `AGENTS.md` carries the
general testing policy; this file covers what is specific to this harness.

## A readiness gate must imply everything the spec then asserts

Seeded resources reach the gateway as etcd watch events applied in revision
order, so waiting on resource *N* proves only that everything written up to *N*
has landed. Gating on a resource seeded in the middle of `beforeAll` and then
asserting on ones written after it is a race — and test ordering hides it,
because a later test in the same file runs against an already-warm snapshot.

Seed every caller API key after every other resource, then gate on that key
authenticating (`GET /v1/models` returning `200`). That single condition implies
the whole seed set is in the snapshot.

Two things the gate must not be:

- **A request that exercises the behavior under test.** The gate would then
  fail by exhausting its 30s timeout instead of by an assertion, hiding what
  actually broke.
- **Wrapped in a catch-all that swallows every error.** A `401` before the key
  propagates is the normal transient state, but an upstream, transport, or
  invalid-response failure must surface as itself rather than as a timeout.
  Prefer a gate that cannot throw (`ProxyClient.listModels`) over a `try/catch`
  around an SDK call.

## etcd is per-fork, and on CI it must be there

CI runs one etcd cluster per vitest fork and hands each fork its own
(`AISIX_E2E_ETCD_ENDPOINTS`, mapped by `VITEST_POOL_ID`). Two
consequences a spec author cannot see from the 200-odd call sites they
would otherwise copy:

- **`etcdEndpoint()` is the only permitted way to name an etcd.** Reading
  `AISIX_E2E_ETCD` / `AISIX_E2E_ETCD_ENDPOINTS`, or writing a
  `127.0.0.1:2379` literal, pins ONE fork's cluster while the gateway
  the spec just spawned talks to its own — surfacing as a
  `waitConfigPropagation` timeout that reads like a product bug.
  `harness/etcd-endpoint.test.ts` fails the build on it.

- **`ping()` throws on CI**, so the familiar
  `if (!(await etcd.ping())) ctx.skip()` prologue is a local-only escape
  hatch there; do not add a second `if (process.env.CI) throw` beside
  it. The skip exists for a developer with no etcd running. On CI a
  skipped file contributes no assertions while the leg still reports
  green, and with a cluster per fork one dead endpoint would take out
  only the quarter of the suite bound to it — quietly.
