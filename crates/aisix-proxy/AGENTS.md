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

## Turning a buffered relay into a streamed one moves four things, not one

Swapping `resp.bytes()` for `resp.bytes_stream()` is the visible part of the
change and the least of it. The handler now returns while the response is still
being delivered, so three request-scoped things that used to be finished by then
are not — and a fourth rule decides whether the path may stream at all. None of
the four errors when missed:

- **The reqwest request-level timeout has to go.** `RequestBuilder::timeout`
  bounds the body read too, so it cuts a long stream off mid-response. Bound the
  connect phase with `stream_timeout::send_with_deadline` and each chunk with
  `stream_timeout::with_read_timeout_bytes`, both from the model's `stream`
  budget (`routing::effective_timeouts`).
- **The rate-limit reservation has to become a `into_stream_hold()`.** Dropping it
  at handler return releases the concurrency slot while the stream is still
  running, which is how a key capped at N serves more than N at once (#450). The
  terminal token cost then rides `Limiter::add_tokens_post_stream` instead of
  `commit_tokens`.
- **The UsageEvent moves into the stream's Drop guard.** Counts that arrive on a
  terminal SSE event are unknown at handler return, so the guard owns the emit —
  fired at end-of-stream AND on client disconnect, since SDKs routinely close on
  the terminal frame. Whatever flag says "the guard owns it" must gate the
  handler's own emit, or the request is counted twice.
- **A block- or mask-capable output guardrail keeps the buffered path.** It has to
  see the whole response before any of it reaches the caller, or the streaming
  request is a way around the check the same request gets without it. That is
  `runs_on_output(chain) && chain.stream_output_policy().holds_back()`. A
  monitor-only chain resolves to `EndOfStreamCheck` and must NOT hold the stream
  back (AISIX-Cloud#1010) — it takes the live path and scans once at
  end-of-stream via `guardrail_stream::EosOutputScan`.

Retry semantics change too, and that is intended: the retryable unit ends at the
status check, because once the first frame is on the wire there is no failing
over left to do.

An e2e that asserts the response CONTENT cannot see any of this — a buffered relay
and a streamed one return identical bytes. Assert the delivery shape instead: a
mock upstream that trickles its response over ~1s, and a client read loop that
counts chunks and measures first-byte against last-byte
(`audio-stream-relay-e2e.test.ts`).

## A guardrail-raised refusal is gated on the direction, not on `is_empty()`

`GuardrailIndex::resolve` matches on SCOPE only — env / model / mcp_server /
api_key / team. It does not filter by `hook_point`, so a chain resolved for a
request is non-empty whenever any attachment is in scope, including one attached
on the output hook alone. Each guardrail then no-ops on the hook it is not
configured for, which makes `!chain.is_empty()` a safe gate for *running* the
checks and a wrong gate for *refusing*: a refusal the proxy raises itself —
`unscannable_body`, and anything else the code decides rather than a verdict —
is only justified when something would have read that side of the exchange.

Gate the refusal on `Guardrail::refuses_unevaluable_input(&chain)` for a
request-side one and `refuses_unevaluable_output(&chain)` for a response-side
one. Those fold TWO conditions per member — the member reads that side, and its
failure policy is fail-closed — and a chain is `any` over its members. Both
halves must hold on the SAME member, which is why there is a combined predicate
rather than two you could `&&` together: a chain of [output-only fail-closed,
input-only fail-open] reads the request and contains a fail-closed row, yet
nothing in it justifies refusing a request.

The failure policy is per hook: the row's `fail_open` governs the input hook,
and each remote kind's own `<Kind>Config::output_fail_open` governs the output
hook. `keyword` and `pii` have no `output_fail_open` — they never call out — so
their row-level `fail_open` governs both of their hooks.

`/mcp` is the sharp case for the direction half: it resolves ONE chain and uses
it in both directions, so an input-only row reaches the output arm and vice
versa. Note also that `hook_point` defaults to `both` and `fail_open` to
`false`, so this only bites a deployment that set one of them explicitly.

Not gated, deliberately: the refusals that need a HELD-BACK stream
(`output_buffer_exceeded`, `mask_writeback_failed`, and `unscannable_body` on a
buffered SSE body). Honouring `fail_open` there does not mean skipping a
refusal, it means releasing already-buffered bytes that were never scanned —
a different decision, and one nobody has taken.

## The bypass reason rides the audit log, and every emitter must set it

`usage_events.guardrail_bypassed_reason` is the record that a guardrail on
this request did not evaluate and its failure policy let the request past
it. Its failure mode is silent in both directions: the caller gets a normal
200, and an emitter that leaves the field at its `String::default()`
reports "nothing was bypassed" for a request that was — indistinguishable
from a screened one.

**It is not mutually exclusive with `guardrail_blocked`, and must not be
suppressed when that is set.** A chain can fail open on one member and be
refused by another, and an input hook can fail open on a prompt the
provider already answered before the output hook refused it. Both are
requests where something really did go unscreened, and the second is the
more compliance-relevant of the two — `chat.rs` has always carried the
input bypass onto the billed-then-blocked event through `UpstreamCharge`.
"Reached a provider unscreened" is the two fields read together, never this
one alone.

So it is not threaded per handler. The chain folds record the first `Bypass`
they see onto the request's `GuardrailAuditLog`, which every handler already
clones for `guardrail_enforced_hits`, and every emitter reads it back with
`usage_attr::bypass_reason(audit)`. Not terminal-only, unlike the enforced
hits: every attempt of a request that failed open went upstream unscreened.
`chat.rs` still threads its own copy because its per-attempt and ensemble
sub-call emitters are handed the value captured earlier in the request,
which a request-scoped snapshot cannot express; the two agree by
construction.

Two rules follow, and both are enforced mechanically rather than by review:

- A new `UsageEvent` literal in this crate sets the field, or carries a
  `NO-GUARDRAIL-CHAIN: <why>` comment saying no chain was ever resolved
  (`usage_attr`'s `every_usage_event_this_crate_builds_answers_the_bypass_question`
  parses the crate's own source for both).
- A pass-through the PROXY performs on a body it could not scan is recorded
  with `GuardrailChain::record_unevaluable_{input,output}_bypass`, never
  `record_bypass` directly. That pass has two causes and only one is a
  bypass: a chain whose readers of that side are all fail-open let an
  unscreened request through, while a chain where NOTHING reads that side
  never offered to screen it. Tagging the second would fire the field on
  requests that were never going to be screened, which breaks the negative
  answer just as thoroughly as dropping the first.

Deliberately NOT recorded, so the next person does not read the gated
unscannable sites as the full set: the places that scan a mangled copy
unconditionally, with no failure policy involved. `jobs::scan_output_blob`
scans `String::from_utf8_lossy` of a batch or fine-tuning response and, where
the verdict lets it through, relays the original bytes whatever the row's
failure policy says; `passthrough_route` scans the lossy body; `audio.rs`'s
`transcription_output_text` falls back to the lossy body for the plain-text
transcript formats (`text` / `srt` / `vtt`). None of these consults
`refuses_unevaluable_*`, so a fail-CLOSED row does not refuse there either —
not something telemetry can paper over, and making them refuse is a behaviour
change rather than an observability one. Tagging them instead
would fire the field on every job response, every passthrough body and every
plain-text transcript, which destroys the negative answer just as thoroughly.
Note that `record_unevaluable_*` is the wrong helper at such a site: its
predicate assumes the fail-closed case was already refused, so at a site that
never refuses it would silently drop exactly the case worth reporting.

What is no longer in that set is the multipart `prompt` on `audio.rs` and
`images_edits.rs`: since #1016 both call
`dispatch::require_utf8_prompt_fields` after model resolution and before the
chain runs, so an undecodable `prompt` part is answered 400 and the
`filter_map` in their scan builders can no longer be reached by one. Do not
describe those surfaces as silently dropping non-UTF-8 prompt parts — audio's
transcript OUTPUT scan, listed above, is a separate site and still lossy.

Values come from the guardrail kind's own `bypass_tag()` — the same bounded
vocabulary `GuardrailVerdict::block_unavailable` carries, so one outage reads
the same whichever way the row is configured — plus `unscannable_body` for
the proxy-raised pass. The audit log clamps whatever it is given through
`bounded_failure_tag`, because the value lands on an unsanitized metric
label and on a wire field the control plane stores as `varchar(64)`.

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

## A passthrough route that detects an envelope must observe what the typed endpoint does

`passthrough_route.rs` relays bytes verbatim, but it *detects* the request
envelope and extracts from it. Whatever it extracts is the whole observation —
there is no bridge behind it to fill anything in. So a new observation added to
a typed endpoint has a second home: a new `UsageEvent` token dimension needs
reading in `usage_of`, a new guardrail scan input needs adding to
`message_scan_text` / `request_guardrail_text`, a new attribution field needs
setting on `RouteTelemetry`. Miss it and the route keeps answering 200 while
metering and enforcement silently weaken — the shape of #988, where the cache
and reasoning counters existed everywhere except here.

Two rules bound the extraction:

- An **opaque** (`Raw`) body is never mined for meaning. Buffered opaque
  responses are not probed for usage at all, and an opaque stream's flat token
  fields count only on a frame the server itself labelled one
  (`event: token_usage`) — a caller-shaped body must not be able to mint tokens.
- Usage accumulates **field-wise max** across frames, never last-wins: one
  stream reports it in pieces (Anthropic's `message_start` carries the input
  and cache counters, its `message_delta` only the output ones), so an
  assignment truncates whatever arrived earlier.

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
do not re-audit it as a new finding. The same project ruling holds for the OTHER
member gates on **ensemble** sub-calls (member `allowed_cidrs`, cooldown/health
consumption, Prometheus usage, caching): ensemble is an experimental surface and
its parity gaps are deliberate TODOs, not fresh findings — semantic-router
targets got these gates first because they share the single-winner dispatch
shape; graduate ensemble deliberately, in one pass.

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
