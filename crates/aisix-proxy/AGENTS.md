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

Decide, and encode the decision at the call site:

- **Binds each target** (anything protecting the upstream behind it — rate limits,
  cooldown, health, timeouts): resolve it from the attempt model *inside* the
  dispatch loop, in all four group-capable endpoints (chat, messages,
  count_tokens, responses) and in both the streaming and non-streaming branches.
  A limit-shaped gate should skip the target and let dispatch continue rather than
  failing the whole request — see `quota::reserve_routing_target`.
- **Binds the requested entry** (anything scoped to the alias the caller named —
  `allowed_cidrs`, guardrail attachment): keep it pre-dispatch, and say so in the
  user-facing docs, because the group/member split is otherwise invisible.

A reservation-shaped gate additionally must not double-charge: `reserve_routing_target`
returns `None` for non-routing dispatch, whose model layers the pre-dispatch
`quota::enforce*` already reserved.
