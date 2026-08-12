# aisix-proxy

## Response-body streams and spawned tasks must re-attach the request span

`request_id::ensure_request_id` opens the `request{request_id=…}` span that puts a
`request_id` on every log line a request emits — that field is what joins a deep
diagnostic (e.g. the Aliyun guardrail's `aliyun_request_id`) back to the
`x-aisix-request-id` the caller was handed.

Two places fall outside it, and neither errors when missed — the logs are just
silently uncorrelated, which reads exactly like working code:

- **Streamed response bodies.** Hyper polls the generator after the middleware has
  returned. Wrap it in `request_id::in_request_span(…)` **from the handler's
  stack** (it captures `Span::current()`, so calling it elsewhere attaches a no-op
  span). Every `async_stream::stream!` returned as a body needs this.
- **Detached tasks.** Anything reached via `tokio::spawn` or axum's
  `WebSocketUpgrade::on_upgrade` inherits nothing; attach the span to the future
  with `.instrument()` (see `realtime::realtime`).

Do not hold a span guard across an await to work around this — it leaks the span
onto whatever the executor runs next on that thread.

A `text/event-stream` body needs a second wrapper for the same reason — nothing
errors when it is missed. Pass it through `sse_keepalive::with_heartbeat(…,
sse_keepalive::interval())` (or, on an axum `Sse`, `keep_alive` with that
interval) so a model that is slow to its first token doesn't look like an
abandoned connection to a proxy in front. Only for SSE: the same wrapper on an
opaque binary passthrough (audio, images) corrupts it.

## Every terminal path emits the access log — including the ones that give up early

The access log and `request_metrics::record` are emitted **by the handler**, at
the end of dispatch, because that is the only place that knows the provider, model
and token counts. A path that returns before reaching that tail therefore logs
nothing, and nothing errors: the caller gets a correct status while the gateway
keeps no record of the request, which is indistinguishable from the request never
arriving.

Two shapes give up early, and both must answer through
`reject::reject_before_dispatch` (it renders the envelope *and* emits the
telemetry, so the two can't drift apart):

- **Middleware short-circuits** — anything that returns instead of calling
  `next.run(request)` (see `enforce_request_body_limit`). These run ahead of
  authentication, so they pass `api_key_id: None`.
- **Extractor rejections a handler unwraps at its top** — the
  `Result<Json<T>, JsonRejection>` / `Result<Bytes, BytesRejection>` parameters.
  Auth already ran here, so pass the key id.

A handler that instead wraps its whole dispatch and logs the wrapper's status
(`/mcp`, `/a2a`, `/passthrough`, `/v1/videos`, `/v1/files`) is already covered —
don't add a second emit to those, or the request logs twice.

Emit the request metrics through `request_metrics::record` and nothing else. It
writes the legacy `aisix_requests_total` **and** the detailed `aisix_proxy_*` /
`aisix_llm_*` families from one call, so calling `Metrics::record_request`
directly silently produces a request that exists in one family and not the
others — the bug AISIX-Cloud#1234 fixed across ten endpoints.

## A new proxy route has to be declared in three places

Adding a `.route(…)` in `build_router` is not enough, and nothing fails loudly
if you stop there:

1. `normalize_endpoint_label` — an unlisted path collapses to `"other"`, so the
   route is invisible per-endpoint in every request series (how `/v1/videos`
   shipped).
2. `request_metrics::LLM_ENDPOINTS` — decides whether the route counts as model
   inference. Unlisted means proxy-only, which is the safe default but a silent
   one.
3. The `ROUTES` table in `request_metrics`' tests — the only thing that makes
   (1) and (2) fail loudly. It is a hand-maintained list of every route; a route
   missing from it is a route the tests cannot check.

## A per-model gate must say whether it binds the requested entry or each target

`resolve_attempt_models` expands a routing model into targets, so `model_entry` /
`virtual_entry` is the **group**, which carries none of a member's config. A gate
written against it silently never runs for group traffic, and nothing errors —
requests keep succeeding on a target that should have been excluded.

**The default is that a per-model gate binds each target.** Anything an operator
configures ON a model — rate limits, `allowed_cidrs`, cooldown, health, timeouts —
is a statement about that model, and reaching it through a group must not strip it.
The only deliberately entry-scoped gate is the group's own copy of any of the
above. Anything else that only checks `model_entry` / `virtual_entry` is a bug.

Guardrail attachment is the **known open exception, not a settled design**: the
chain resolves from `RequestContext.model_id` before dispatch, so a guardrail
scoped to a member never runs for group traffic (measured: direct 422, via group
200). It is unfixed because the semantics are undecided, not because entry scope
is correct — input guardrails run before a target is picked, and under failover
there is no single "winning member" to resolve against. Tracked in
AISIX-Cloud#1090; do not cite it as precedent for scoping a new gate to the entry.
The 2026-08 model-kind audit re-confirmed the same gap for **ensemble panel/judge
sub-calls and semantic-router targets**, and the ruling (project decision) is that
all three kinds stay under #1090's one unified design pass: the operator can
attach the guardrail to the parent entry, so member scope is a mitigable gap, not
an unavoidable bypass. Do not piecemeal-fix one kind ahead of that decision, and
do not re-audit it as a new finding.

Two shapes, both already implemented — copy the nearest one:

- **Filter the candidate set** (static per-caller predicates like `allowed_cidrs`):
  drop ineligible targets in `routing::resolve_attempt_models` *before* the strategy
  picks, so `max_fallbacks` budgets attempts across reachable targets and a
  metric-based strategy ranks only those. Empty result → the gate's own error.
  Do NOT fold these into `filter_attempt_models`: its
  `when_all_unavailable: try_anyway` policy hands back the unfiltered list, which
  would defeat an allowlist. See `routing::targets_allowed_for_ip`.
- **Check per attempt** (dynamic/stateful gates like a rate-limit reservation):
  resolve from the attempt model *inside* the dispatch loop, in all four
  group-capable endpoints (chat, messages, count_tokens, responses) and in both the
  streaming and non-streaming branches; skip the target and continue rather than
  failing the whole request. See `quota::reserve_routing_target`, which also shows
  the non-double-charge rule: it returns `None` for non-routing dispatch, whose
  model layers the pre-dispatch `quota::enforce*` already reserved.

Whichever shape, the group's own gate stays enforced pre-dispatch — the two tiers
are additive, not either/or — and a caller-visible rejection must keep the
direct-model envelope (`ModelIpRestricted` names no model and no CIDR), so a group
never becomes a probe for which members exist.

## `request_id` is caller-controlled input, not a gateway-minted UUID

Since AISIX-Cloud#1288 a caller can supply the request id via a configured inbound
header and `request_id::ensure_request_id` adopts it verbatim, so every
`ClientContext.request_id` / `RequestId` value downstream may be a string the
caller chose. It is only guaranteed to be 1..=256 bytes of visible ASCII
(`request_id::is_acceptable`) — **not** a UUID, and not unique: nothing stops two
requests carrying the same id, so an id is a grouping key, never an identity.

New code that consumes it must therefore treat it as untrusted: escape it for the
sink rather than interpolating it raw (a URL, a file path, a shell argument, a log
format that isn't structured). Never make it a Prometheus label — unbounded
cardinality straight from the caller. The existing sinks are already safe and show
the shape: an OTLP span attribute, an HTTP header value, a `tracing` field, a
parameterised SQL bind.

`is_acceptable` is half of a cross-repo contract with cp-api's `validRequestID`
(AISIX-Cloud `internal/dpmgr/api/telemetry.go`): tightening this side alone
silently strands ids, and tightening THAT side alone silently drops the request
from billing and /logs while the caller still gets a 200 carrying the id. Change
both or neither.
