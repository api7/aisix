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
