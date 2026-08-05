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
