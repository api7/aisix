//! Prometheus metrics registry shared across the proxy middleware and
//! the admin `/metrics` endpoint.
//!
//! Existing compatibility series cover spec §7:
//! - `aisix_requests_total{provider,model,status,outcome}` — counter
//!   incremented once per proxy request that produced a response. A request
//!   the caller abandoned before the response head never reaches it; see
//!   [`M_PROXY_CLIENT_CANCELLED`].
//! - `aisix_request_duration_seconds{provider,model,status}` — histogram
//!   of proxy latency, recorded when the response is handed to the client.
//!   That is the full request for a non-streamed response and only time to
//!   response START for a streamed one; `aisix_request_e2e_latency_seconds`
//!   is the series that is end-to-end on both. See `request_metrics`.
//! - `aisix_ratelimit_rejections_total{scope,layer,policy_id}` — counter for
//!   429 flows.
//! - `aisix_tokens_consumed_total{provider,model}` — counter of
//!   `usage.total_tokens` summed across completed calls. Streamed calls are
//!   included: they contribute from the end-of-stream emit, not at response
//!   open like the two series above.
//!
//! Newer AISIX-native series use `aisix_proxy_*` and `aisix_llm_*`
//! names with bounded, DP-stable labels. They intentionally do not
//! copy label names from other LLM gateways that the data plane does
//! not have.
//!
//! A single [`Metrics`] instance is held `Arc`'d inside `ObsState` and
//! cloned into axum state. The exposition format is emitted via
//! `metrics-exporter-prometheus`'s text renderer; no global recorder is
//! installed, so tests can spin up isolated instances per case.

use metrics_exporter_prometheus::{
    Matcher, PrometheusBuilder, PrometheusHandle, PrometheusRecorder,
};
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

/// Metric names (public so the admin `/metrics` handler and tests can
/// refer to them without typo risk).
pub const M_REQUESTS_TOTAL: &str = "aisix_requests_total";
pub const M_REQUEST_DURATION: &str = "aisix_request_duration_seconds";
pub const M_RATELIMIT_REJECTIONS: &str = "aisix_ratelimit_rejections_total";
pub const M_TOKENS_CONSUMED: &str = "aisix_tokens_consumed_total";
pub const M_LLM_SPEND_MICRO_USD_TOTAL: &str = "aisix_llm_spend_micro_usd_total";
pub const M_LLM_INPUT_TOKENS_TOTAL: &str = "aisix_llm_input_tokens_total";
pub const M_LLM_OUTPUT_TOKENS_TOTAL: &str = "aisix_llm_output_tokens_total";
pub const M_LLM_TOTAL_TOKENS_TOTAL: &str = "aisix_llm_total_tokens_total";
// ── Upstream prompt-cache token detail (AISIX-Cloud#1404) ───────────────
//
// The upstream provider's OWN prompt cache, not the gateway's response /
// semantic cache — that one is `aisix_cache_requests_total{outcome}`, and
// mixing the two is the whole reason these carry `input` in the name.
//
// Three counters rather than two because the providers report cache reads
// under two INCOMPATIBLE accounting rules, and folding them together makes
// every cross-protocol ratio wrong:
//
// - OpenAI-shape (`prompt_tokens_details.cached_tokens`, DeepSeek's
//   `prompt_cache_hit_tokens`, Gemini's `cachedContentTokenCount`) is a
//   SUBSET of the reported prompt tokens — already inside
//   `aisix_llm_input_tokens_total`.
// - Anthropic-shape (`cache_read_input_tokens` / `cache_creation_input_tokens`,
//   Bedrock's `cacheReadInputTokens` / `cacheWriteInputTokens`) is reported
//   ALONGSIDE `input_tokens` — outside `aisix_llm_input_tokens_total`, and
//   folded into `aisix_llm_total_tokens_total` by `total_tokens_with_cache`.
//
// So the cache-inclusive input a request really consumed is
// `input + cache_read + cache_creation`, and the part of it served from
// cache is `cached_input + cache_read`. Both expressions hold for either
// provider shape precisely because the two reads stay separate here.
//
// All three are sparse: a series appears only once its value is non-zero,
// so providers that report no cache detail mint nothing.
/// Prompt-cache hits the upstream reported as part of its prompt token
/// count — already included in [`M_LLM_INPUT_TOKENS_TOTAL`].
pub const M_LLM_CACHED_INPUT_TOKENS_TOTAL: &str = "aisix_llm_cached_input_tokens_total";
/// Prompt-cache hits the upstream reported as a counter separate from its
/// input tokens — NOT included in [`M_LLM_INPUT_TOKENS_TOTAL`], and
/// included in [`M_LLM_TOTAL_TOKENS_TOTAL`].
pub const M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL: &str = "aisix_llm_cache_read_input_tokens_total";
/// Input tokens written INTO the upstream's prompt cache. Reported as a
/// counter separate from input tokens, and billed at a premium by every
/// provider that reports it.
pub const M_LLM_CACHE_CREATION_INPUT_TOKENS_TOTAL: &str =
    "aisix_llm_cache_creation_input_tokens_total";
pub const M_LLM_REQUESTS_TOTAL: &str = "aisix_llm_requests_total";
pub const M_LLM_REQUEST_DURATION: &str = "aisix_llm_request_duration_seconds";
pub const M_LLM_TTFT: &str = "aisix_llm_time_to_first_token_seconds";
/// Issue #890 req-4: token volume sliced by inbound client type — a
/// DEDICATED low-cardinality series so the client dimension never multiplies
/// the per-key `aisix_llm_*_tokens_total` families. `client_type` is
/// normalised to a bounded allowlist by [`client_type_from_user_agent`]; the
/// raw user-agent + client version stay in logs / `UsageEvent`, never here.
/// AISIX-Cloud#1044 adds a `model` label (the requested logical model, same
/// value as the `aisix_llm_*` families' `model`) so the series answers
/// "which models is each client spending tokens on". The label set stays
/// client_type × model × token_type — per-key/team/user dimensions belong to
/// the `aisix_llm_*_tokens_total` families (or UsageEvent/logs), never here.
pub const M_LLM_TOKENS_BY_CLIENT_TOTAL: &str = "aisix_llm_tokens_by_client_total";
/// Requests currently inside a handler, by `endpoint` and
/// `inbound_protocol`.
///
/// Deliberately carries NO `upstream_protocol` (AISIX-Cloud#1403). This
/// gauge is a matched pair of edges — `InFlightGuard::new` increments on
/// the way in, its `Drop` decrements on the way out — and the increment
/// runs in the proxy middleware, BEFORE routing has selected a target, so
/// at that moment the label has no value but `unknown`. Resolving it only
/// on the decrement would be worse than useless: the two edges would
/// address different series, leaving the `unknown` one pinned at a
/// non-zero count forever on an endpoint that has long gone idle.
///
/// Callers wanting concurrency by upstream protocol take it from the
/// request families, which label the completed request with the target it
/// actually reached.
pub const M_PROXY_IN_FLIGHT: &str = "aisix_proxy_in_flight_requests";
pub const M_PROXY_REQUESTS_TOTAL: &str = "aisix_proxy_requests_total";
pub const M_PROXY_FAILED_REQUESTS_TOTAL: &str = "aisix_proxy_failed_requests_total";
pub const M_PROXY_REQUEST_DURATION: &str = "aisix_proxy_request_duration_seconds";
/// Requests the client abandoned before the response head was written.
/// Every other proxy series needs a handler to produce a response first,
/// which a cancelled request never does — without this one those requests
/// are absent from the metrics entirely, not counted as failures.
///
/// Labelled by `endpoint` plus whatever the request had resolved before
/// the caller hung up — the model it addressed and the ProviderKey of the
/// target it was waiting on, `unknown` when it was cancelled before
/// reaching them (AISIX-Cloud#1317). Without those, "which model do
/// callers give up on" was unanswerable, which is the question this
/// series exists to answer. Every value arrives already bounded from the
/// proxy layer — the route template, the configured model set, and a
/// ProviderKey name read off the row its id names (#451).
///
/// Deliberately NOT the full `RequestLabels` set: a cancelled request has
/// no status, no outcome and often no team / user, so the caller-identity
/// dimensions would be `unknown` on every sample and cost series for
/// nothing.
pub const M_PROXY_CLIENT_CANCELLED_TOTAL: &str = "aisix_proxy_client_cancelled_requests_total";
/// Requests refused by the `request_body_limit_bytes` cap before any
/// handler ran, split by how the gateway's drain of the refused body
/// ended. `outcome != "completed"` means the gateway stopped reading
/// while the caller was still sending, so that caller most likely saw a
/// connection reset instead of the 413 it was owed.
///
/// This is the amplification-safe channel for the same event the
/// `aisix::body_limit` log carries: a flood of oversize requests can
/// suppress the log's rate-limited warnings, never this counter. Both
/// label sets are bounded by construction — `endpoint` is a route
/// template (#451) and `outcome` a fixed vocabulary.
///
/// Deliberately carries NO `upstream_protocol` (AISIX-Cloud#1403): the
/// cap rejects the request while draining its body, before any handler
/// runs and therefore before a target is selected, so the label would be
/// the constant `unknown` on every sample this family can ever produce.
pub const M_PROXY_BODY_LIMIT_REJECTIONS_TOTAL: &str =
    "aisix_proxy_request_body_limit_rejections_total";
/// Per-DEPLOYMENT (one concrete Model row = one upstream target) call
/// counters, incremented once per upstream ATTEMPT rather than once per
/// client request — the granularity the `aisix_proxy_*` / `aisix_llm_*`
/// families deliberately do NOT have. A request that fails over across
/// three targets is one sample there and three here.
///
/// That difference is the whole point of the family, and the reason a
/// gateway-wide 5xx count read off `aisix_proxy_requests_total` can sit
/// orders of magnitude below the number of failed attempts an operator
/// sees in the usage log (AISIX-Cloud#1299): most failed attempts belong
/// to requests a fallback went on to serve, and those requests are a
/// `status="200"` sample in the request families.
///
/// Scope: emitted from the Model-Group dispatch loops
/// (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) — the
/// endpoints that keep per-attempt telemetry at all. The single-target
/// handlers dispatch once per request and are already covered by the
/// request families.
///
/// Only attempts that REACHED the upstream are counted: an attempt
/// refused by its target's own rate-limit layers before dispatch
/// produced no upstream response, so counting it would put a gateway-side
/// refusal into a family operators read as upstream health. It is still a
/// real attempt in the usage log and in the fallback classification below.
pub const M_DEPLOYMENT_REQUESTS_TOTAL: &str = "aisix_deployment_requests_total";
pub const M_DEPLOYMENT_SUCCESS_TOTAL: &str = "aisix_deployment_success_responses_total";
pub const M_DEPLOYMENT_FAILURE_TOTAL: &str = "aisix_deployment_failure_responses_total";
pub const M_DEPLOYMENT_STATE: &str = "aisix_deployment_state";
pub const M_DEPLOYMENT_COOLED_DOWN_TOTAL: &str = "aisix_deployment_cooled_down_total";
/// Fallback outcomes, counted once per fallback ATTEMPT — the attempt
/// that moved to a different target than the previous one. The attempt
/// that served the request bumps the successful family, one that failed
/// in turn bumps the failed family, so a request rescued by its second
/// fallback contributes one of each.
///
/// `model` is what the caller asked for (the Model-Group name);
/// `fallback_model` is the target the gateway moved to. Both are
/// configured names, so the label set is bounded by the resource set.
pub const M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL: &str = "aisix_routing_successful_fallbacks_total";
pub const M_ROUTING_FAILED_FALLBACKS_TOTAL: &str = "aisix_routing_failed_fallbacks_total";
pub const M_RATELIMIT_REMAINING_REQUESTS: &str = "aisix_ratelimit_remaining_requests";
pub const M_RATELIMIT_REMAINING_TOKENS: &str = "aisix_ratelimit_remaining_tokens";
pub const M_BUDGET_LIMIT_USD: &str = "aisix_budget_limit_usd";
pub const M_BUDGET_SPENT_USD: &str = "aisix_budget_spent_usd";
pub const M_BUDGET_REMAINING_USD: &str = "aisix_budget_remaining_usd";
pub const M_BUDGET_RESET_SECONDS: &str = "aisix_budget_reset_seconds";
pub const M_BUDGET_DETAILS_PRESENT: &str = "aisix_budget_details_present";
pub const M_REDIS_FAILURES_TOTAL: &str = "aisix_redis_failures_total";
pub const M_USAGE_EVENT_DROPS_TOTAL: &str = "aisix_usage_event_drops_total";
/// Guardrail outcomes (#379 observability). `aisix_guardrail_blocks_total`
/// counts guardrail executions that rejected a request (input or output hook;
/// policy or fail-closed combined). `aisix_guardrail_bypasses_total` counts
/// fail-open executions — a remote-API guardrail's upstream was unreachable
/// but `fail_open` let the request through — sliced by the bounded DP-internal
/// `reason` (e.g. `bedrock_5xx` / `bedrock_timeout` /
/// `bedrock_throttled`). Both counters share the all-endpoint execution
/// chokepoint with `aisix_guardrail_latency_seconds`.
pub const M_GUARDRAIL_BLOCKS_TOTAL: &str = "aisix_guardrail_blocks_total";
pub const M_GUARDRAIL_BYPASSES_TOTAL: &str = "aisix_guardrail_bypasses_total";

/// Per-request inbound authentication decisions
/// (AISIX-Cloud#1080/#1081). Before this series, a rejected credential
/// was invisible: the auth extractor short-circuits ahead of every
/// handler, so no request counter and no access-log line ever fired
/// for a 401. Labels, all bounded:
/// - `method`: `api_key` / `jwt` / `none` (no credential presented).
/// - `result`: `allowed` / `denied`.
/// - `reason`: the DP-internal denial reason class (e.g. `unknown_key`,
///   `key_expired`, `jwt_bad_signature`, `jwt_untrusted_issuer`,
///   `jwt_identity_unmapped`); `none` when allowed.
pub const M_AUTH_DECISIONS_TOTAL: &str = "aisix_auth_decisions_total";
/// Per-execution guardrail latency histogram (AISIX-Cloud#1076), recorded
/// by the chain fold for every member consulted on any handler — chat,
/// messages, responses, embeddings, streaming end-of-stream/window scans,
/// cache-hit output checks, and the segment (Bedrock-style) pass alike.
/// Labels:
/// - `env_id`: constant per DP process (`unknown` standalone).
/// - `guardrail`: the configured (row) name.
/// - `kind`: the guardrail kind discriminator (`keyword`/`pii` run
///   in-process; every other kind calls a remote service, so this label
///   splits local vs remote latency).
/// - `phase`: `input` / `output`.
/// - `result`: `allowed` / `blocked` / `masked` / `bypassed` (remote
///   failure + fail-open) / `would_block` / `would_mask` (monitor mode).
/// - `error_type`: bounded failure tag (e.g. `lakera_timeout`,
///   `custom_unknown_action`) whenever the guardrail could not EVALUATE
///   the content, else `none`. It is populated on `result="bypassed"`
///   (fail-open) and equally on the `result="blocked"` a fail-CLOSED row
///   produces for the same cause (AISIX-Cloud#1365) — `result` stays
///   `blocked` there so a shipped alert on it keeps counting, and
///   `error_type != "none"` is what separates "this content violated a
///   policy" from "this guardrail is broken or its provider is down".
///
/// The `_count` series doubles as a per-guardrail execution counter, so
/// there is no separate `aisix_guardrail_requests_total` (LiteLLM's
/// `litellm_guardrail_requests_total` equivalent = `sum by (...)` of it).
pub const M_GUARDRAIL_LATENCY_SECONDS: &str = "aisix_guardrail_latency_seconds";
/// Issue #408: counter for UsageEvent emission attempts, incremented before
/// the `UsageSink` accepts or rejects the event. Operators slice this by:
/// - `handler`: the bounded handler family that emitted the event (for
///   example `chat`, `messages`, `count_tokens`, `responses`, `mcp`, or
///   `passthrough_route`). Fixed enumeration, low cardinality.
/// - `status_code`: bucketed as `2xx` / `4xx` / `5xx`.
/// - `status`: the raw code (`429`, `502`, ...), so a query can name one
///   failure mode instead of a whole family (AISIX-Cloud#1389). Named to
///   match `aisix_proxy_requests_total`'s own raw-code label, so one
///   spelling works across both families — and deliberately NOT
///   `http_status_code`, which contains `status_code` as a substring and
///   would make every `status_code="…"` matcher ambiguous. It adds no
///   series over `status_code` alone (the raw code determines the family),
///   and `status_code` is kept so dashboards and alerts written against
///   `status_code="4xx"` keep working unchanged.
/// - `user_id` / `user_name`: the org member behind the request and
///   their display name, from [`UsageEventLabels`]. Both come off one
///   ApiKey row, so the name is determined by the id and the pair is one
///   series, not one per name.
/// - `inbound_protocol`: `openai` / `anthropic` / `mcp` / `other`, with
///   non-HTTP and tunnel protocols folded into the bounded `other` bucket.
/// - `upstream_protocol`: the wire protocol of the ProviderKey the event
///   was attributed to, so this family joins with the request and usage
///   families on that dimension (AISIX-Cloud#1403). Read off the SAME
///   resolved row as `provider_key_id` / `provider_key_name`, and
///   `unknown` for an event that never selected an upstream (a
///   pre-dispatch rejection, or a non-inference tunnel like `/mcp`).
///
/// Paired with `aisix_usage_event_drops_total{reason}` for the
/// `try_send` failure paths (sink full / closed).
pub const M_USAGE_EVENT_EMITS_TOTAL: &str = "aisix_usage_events_emitted_total";
/// Cache gate outcomes, one increment per request that reached an
/// enabled cache policy with an available backend. Labels:
/// - `policy`: the policy's operator-facing name (bounded by the
///   configured policy count).
/// - `outcome`: `hit_exact` / `hit_semantic` / `miss` / `bypass`
///   (`bypass` = the caller's `Cache-Control: no-cache` skipped the
///   read path).
///
/// Requests with no matching policy or an unavailable backend
/// (`cache_status=disabled`) are not counted — the gate never opened.
pub const M_CACHE_REQUESTS_TOTAL: &str = "aisix_cache_requests_total";
/// Latency of the cache semantic layer's embedding calls, by `policy`.
/// Summary series (no fixed buckets), like the other legacy
/// `histogram!` series here.
pub const M_CACHE_SEMANTIC_EMBED_SECONDS: &str = "aisix_cache_semantic_embedding_seconds";
/// Embedding failures on the cache semantic layer, by `policy` and
/// `cause`. `resolve` (embedding model missing or not an embedding
/// model) is counted once per eligible request — including requests
/// that then hit the exact layer; `embed` (provider call failed or
/// timed out) is counted per attempted call. Each failure leaves that
/// request exact-only, so a nonzero rate here with a flat
/// `hit_semantic` outcome is the "semantic layer silently down"
/// signal.
pub const M_CACHE_SEMANTIC_EMBED_FAILURES_TOTAL: &str =
    "aisix_cache_semantic_embedding_failures_total";
/// Semantic-store operation failures, by `policy` and `op`
/// (`lookup` / `store`). Failures degrade to an ordinary miss, which
/// makes a broken store indistinguishable from a healthy low hit rate
/// in the outcome counter alone — this series is the disambiguator.
/// The in-process store cannot fail today; shared (redis) stores can.
pub const M_CACHE_SEMANTIC_STORE_FAILURES_TOTAL: &str = "aisix_cache_semantic_store_failures_total";
pub const M_OTLP_FANOUT_DROPS_TOTAL: &str = "aisix_otlp_fanout_drops_total";
pub const M_OTLP_FANOUT_FAILURES_TOTAL: &str = "aisix_otlp_fanout_failures_total";
/// AISIX-Cloud#1011: SLO-grade latency distributions as REAL bucketed
/// histograms (`_bucket{le=…}`), aggregatable across DP instances with
/// `histogram_quantile()`. Every other `histogram!` series in this file
/// renders as a summary (no buckets configured) whose quantiles cannot
/// be re-aggregated — these two get explicit buckets in [`Metrics::new`]
/// and a DEDICATED low-cardinality label set ([`LatencyLabels`]) so the
/// per-key/per-user dimensions never multiply the bucket count.
///
/// `aisix_request_e2e_latency_seconds` observes the client-perceived
/// end-to-end latency once per request: at handler return for
/// non-streaming requests and failures, at stream completion for
/// committed streams (full stream duration, matching the usage event's
/// `latency_ms` — NOT the time-to-first-byte the summary series record).
/// A stream the client cancels mid-flight still observes once, with the
/// committed status (2xx) and the duration up to the abort — the same
/// client-perceived semantics as the usage event.
pub const M_REQUEST_E2E_LATENCY_SECONDS: &str = "aisix_request_e2e_latency_seconds";
/// Time-to-first-token for streaming requests, same label set as
/// [`M_REQUEST_E2E_LATENCY_SECONDS`] (with `streaming="true"` always).
pub const M_REQUEST_TTFT_SECONDS: &str = "aisix_request_ttft_seconds";

// ── A2A gateway series (AISIX-Cloud#1215) ──────────────────────────────────
//
// The `aisix_proxy_*` families already count `/a2a` traffic and time it, but
// only by route: which agent was reached and which operation was invoked are
// not labels there, so "is the invoice agent's `message/stream` failing?" —
// the question an agent-platform operator actually asks — cannot be answered
// from them. These four carry that dimension and nothing else, so they stay
// bounded: `agent` is a registered resource name, `operation` the canonical
// bounded set, `state` the specification's task states. Task, context and
// JSON-RPC request ids are deliberately absent — they belong in logs and
// traces, never in a label.
//
// Request DURATION is not repeated here: `aisix_proxy_request_duration_seconds`
// already times `/a2a` end to end, and a second histogram of the same quantity
// would only be a second thing to keep in sync.
/// A2A calls by agent, canonical operation and status class.
///
/// It does NOT agree with `aisix_proxy_requests_total{endpoint="/a2a"}`, and
/// two differences are deliberate. A call refused before its agent is resolved
/// — a bad key, a denied agent, an unknown one — has no agent or operation to
/// file under, so it is counted there and not here. And a stream the caller
/// abandoned is `4xx` here but `2xx` there, because the response head really
/// did go out as a 200. Read this family for agent health and that one for
/// route traffic; do not expect the totals to match.
pub const M_A2A_REQUESTS_TOTAL: &str = "aisix_a2a_requests_total";
/// Time from an A2A streaming call starting to the FIRST event the upstream
/// agent pushed — the agent's own "time to first byte".
///
/// Named for the event rather than a token because an agent stream carries
/// task updates, not tokens. It defaults to [`M_REQUEST_TTFT_SECONDS`]'s
/// bucket edges — the same shape of wait — but takes its own `a2a_ttfb`
/// operator override, so tuning one histogram never silently retunes the
/// other.
pub const M_A2A_TTFB_SECONDS: &str = "aisix_a2a_ttfb_seconds";
/// Events relayed downstream on A2A streaming calls. Divided by
/// [`M_A2A_REQUESTS_TOTAL`] over the streaming operations it gives events per
/// call — how chatty an agent is, and whether that changed.
pub const M_A2A_STREAM_EVENTS_TOTAL: &str = "aisix_a2a_stream_events_total";
/// A2A calls by the task state they ended on, normalized to the
/// specification's set plus `unknown`. The series an operator watches for
/// tasks piling up in `input-required` or `failed`.
pub const M_A2A_TASK_STATE_TOTAL: &str = "aisix_a2a_task_state_total";

// ── Config load-observability series (load-observability contract) ─────────
// Reflected from [`aisix_core::ConfigMetricsView`] at scrape time via
// [`Metrics::sync_config_status`]. Standard Prometheus config-reload naming so
// the series read the same as the control plane exposes.
pub const M_CONFIG_LAST_RELOAD_SUCCESSFUL: &str = "aisix_config_last_reload_successful";
pub const M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP: &str =
    "aisix_config_last_reload_success_timestamp_seconds";
pub const M_CONFIG_RELOADS_TOTAL: &str = "aisix_config_reloads_total";
pub const M_CONFIG_RELOAD_FAILURES_TOTAL: &str = "aisix_config_reload_failures_total";
pub const M_CONFIG_REJECTED_RESOURCES: &str = "aisix_config_rejected_resources";
/// Served resources per kind carrying fields this gateway version does not
/// know (loaded with those fields ignored — partially compatible, #871).
/// Non-zero means this data plane and the control plane writing to it are
/// not the same generation; which one is ahead is not knowable from here
/// (AISIX-Cloud#1435), since a field this build retired is as unknown to
/// it as one it has never seen.
pub const M_CONFIG_PARTIALLY_COMPATIBLE_RESOURCES: &str =
    "aisix_config_partially_compatible_resources";
/// Served resources per kind whose latest source bytes are rejected and
/// whose last known good value serves instead (#871). Non-zero means the
/// gateway is running stale config for those rows; the per-row detail
/// (which resource, stale since when) lives in `/status/config`
/// `rejected[]`.
pub const M_CONFIG_STALE_SERVED_RESOURCES: &str = "aisix_config_stale_served_resources";
pub const M_CONFIG_OBSERVED_REVISION: &str = "aisix_config_observed_revision";
pub const M_CONFIG_APPLIED_REVISION: &str = "aisix_config_applied_revision";
pub const M_CONFIG_HASH_INFO: &str = "aisix_config_hash_info";
pub const M_CONFIG_SOURCE_CONNECTED: &str = "aisix_config_source_connected";

/// Default bucket edges for [`M_REQUEST_E2E_LATENCY_SECONDS`], spanning the
/// full client-perceived range: a millisecond-scale rejection or cache hit
/// through a multi-minute generation. The low edges earn their keep here —
/// requests refused before dispatch and cache hits really do return in
/// single-digit milliseconds — and the 420/600 s edges keep
/// `histogram_quantile()` interpolating instead of pinning P99 at the top
/// edge, since `upstream.timeout_ms` allows far longer requests.
/// 17 edges → 18 `_bucket` series per label combination; keep
/// [`LatencyLabels`] lean before adding edges.
pub const DEFAULT_E2E_LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 420.0,
    600.0,
];

/// Default bucket edges for [`M_REQUEST_TTFT_SECONDS`] (AISIX-Cloud#1226).
/// Deliberately NOT the same set as [`DEFAULT_E2E_LATENCY_BUCKETS`]: this
/// histogram observes the *upstream* time-to-first-token, so a request the
/// gateway answers itself (cache hit, pre-dispatch rejection) never lands
/// here at all and the millisecond edges stay empty forever against a
/// hosted provider — one dead series per label combination each. The floor
/// stays at 50 ms rather than higher so a co-located self-hosted upstream
/// (vLLM / Ollama) remains distinguishable; deployments that only ever call
/// hosted providers can raise it via `observability.metrics.buckets`.
/// The top edge stays at 300 s where the e2e set continues to 600 s:
/// time-to-*first*-token beyond five minutes is not a range worth two more
/// permanently-empty series.
pub const DEFAULT_TTFT_BUCKETS: &[f64] = &[
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Default bucket edges for [`M_GUARDRAIL_LATENCY_SECONDS`] — shifted an
/// order of magnitude below the request-latency sets: local (keyword/pii)
/// checks run in microseconds, the added-latency budget under scrutiny is
/// ~50 ms (AISIX-Cloud#1076), and remote guardrail timeouts default to 5 s.
/// The 30 s top edge outlives any configurable guardrail timeout.
pub const DEFAULT_GUARDRAIL_LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Resolved bucket edges for the three real (non-summary) histograms,
/// each either the default above or an operator override from
/// `observability.metrics.buckets` (AISIX-Cloud#1226).
///
/// Every edge here costs one `_bucket` time series **per label
/// combination** on a metric labelled by endpoint × model × provider ×
/// status_class × streaming, so an override is a cardinality decision as
/// much as a resolution one.
#[derive(Debug, Clone)]
pub struct HistogramBuckets {
    pub request_e2e_latency: Vec<f64>,
    pub request_ttft: Vec<f64>,
    pub guardrail_latency: Vec<f64>,
    pub a2a_ttfb: Vec<f64>,
}

impl Default for HistogramBuckets {
    fn default() -> Self {
        Self {
            request_e2e_latency: DEFAULT_E2E_LATENCY_BUCKETS.to_vec(),
            request_ttft: DEFAULT_TTFT_BUCKETS.to_vec(),
            guardrail_latency: DEFAULT_GUARDRAIL_LATENCY_BUCKETS.to_vec(),
            a2a_ttfb: DEFAULT_TTFT_BUCKETS.to_vec(),
        }
    }
}

impl HistogramBuckets {
    /// Upper bound on operator-supplied edges. Prometheus wants a bucket
    /// count in the tens, not the hundreds; the built-in sets use 12–17.
    pub const MAX_EDGES: usize = 64;

    /// Resolve `observability.metrics.buckets`, falling back to the
    /// defaults per metric. Errors are boot-fatal by design: a silently
    /// ignored override would leave the deployment reading quantiles off
    /// bucket edges it did not choose.
    pub fn from_config(cfg: &aisix_core::HistogramBucketsConfig) -> Result<Self, String> {
        Ok(Self {
            request_e2e_latency: resolve_edges(
                "request_e2e_latency",
                cfg.request_e2e_latency.as_deref(),
                DEFAULT_E2E_LATENCY_BUCKETS,
            )?,
            request_ttft: resolve_edges(
                "request_ttft",
                cfg.request_ttft.as_deref(),
                DEFAULT_TTFT_BUCKETS,
            )?,
            guardrail_latency: resolve_edges(
                "guardrail_latency",
                cfg.guardrail_latency.as_deref(),
                DEFAULT_GUARDRAIL_LATENCY_BUCKETS,
            )?,
            a2a_ttfb: resolve_edges("a2a_ttfb", cfg.a2a_ttfb.as_deref(), DEFAULT_TTFT_BUCKETS)?,
        })
    }
}

/// Validate one override list, or hand back the default when unset.
/// `+Inf` is rejected rather than tolerated: the exporter appends it
/// itself, so accepting one here would emit a duplicate `le="+Inf"`.
fn resolve_edges(field: &str, edges: Option<&[f64]>, default: &[f64]) -> Result<Vec<f64>, String> {
    let Some(edges) = edges else {
        return Ok(default.to_vec());
    };
    let ctx = format!("observability.metrics.buckets.{field}");
    if edges.is_empty() {
        return Err(format!("{ctx}: must list at least one bucket edge"));
    }
    if edges.len() > HistogramBuckets::MAX_EDGES {
        return Err(format!(
            "{ctx}: {} edges exceed the limit of {}",
            edges.len(),
            HistogramBuckets::MAX_EDGES
        ));
    }
    for (i, &edge) in edges.iter().enumerate() {
        if !edge.is_finite() || edge <= 0.0 {
            return Err(format!(
                "{ctx}[{i}]: {edge} must be a finite positive number of seconds \
                 (the +Inf bucket is added automatically)"
            ));
        }
        if i > 0 && edge <= edges[i - 1] {
            return Err(format!(
                "{ctx}[{i}]: {edge} must be greater than the preceding edge {} \
                 — bucket edges are cumulative and must strictly ascend",
                edges[i - 1]
            ));
        }
    }
    Ok(edges.to_vec())
}

/// Holds an isolated `PrometheusRecorder` plus its render handle.
/// `metrics::*` macros talk to whatever recorder is in scope; we use
/// `metrics::with_local_recorder` so each write lands on the instance
/// this struct owns — no global state, tests can run in parallel.
#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    recorder: PrometheusRecorder,
    handle: PrometheusHandle,
    /// Per-(endpoint, protocol) in-flight counts. A linear scan over a
    /// bounded slot list (route templates × protocols): the steady-state
    /// hit is two pointer-length string compares with no allocation and
    /// no hashing, where the map this replaced allocated two `String`
    /// keys and ran SipHash on every request edge. Slots for drained
    /// pairs stay at zero rather than being removed — the set is bounded,
    /// and reuse beats churn.
    proxy_in_flight: Mutex<Vec<(String, String, i64)>>,
    /// Process-unique id prefixed onto every worker-cache key, so
    /// thread-local entries minted for one instance's recorder can never
    /// serve another instance (parallel tests build many `Metrics`).
    /// Pre-formatted once — per-emit `fmt::write` was a visible cost.
    worker_key_prefix: String,
    /// Constant `env_id` label for the SLO latency histograms — one DP
    /// process serves exactly one environment. `"unknown"` when the DP
    /// runs standalone (no control plane).
    env_id: String,
    /// Last labels emitted for the config load-observability gauges, so
    /// [`Metrics::sync_config_status`] can zero out stale label series (the
    /// applied hash changed, or a kind's rejections cleared) instead of
    /// leaving a second, contradictory sample in the exposition.
    config_labels: Mutex<ConfigLabelState>,
    /// Every label set seen on a gauge family whose identity includes a
    /// resource that can be deleted, rebound or renamed, so
    /// [`Metrics::retire_stale_gauges`] can retire the ones that stopped
    /// describing anything.
    ///
    /// These families are written ONLY from the request path, and the
    /// recorder registers no idle timeout, so a series whose key is gone
    /// is never touched again and keeps asserting its last value as
    /// though it were current — `aisix_budget_details_present` stays at
    /// `1` for a deleted key, latching the documented "over 90%" alert
    /// with no way to clear short of restarting the process.
    ///
    /// Populated on the registration path only (a worker-cache miss), so
    /// the steady-state emit never touches this lock.
    retirable: Mutex<HashMap<Box<str>, RetirableSeries>>,
}

/// A gauge family whose label set can outlive what it describes.
///
/// Counters and histograms are deliberately absent: a counter that stops
/// being written reads as a flat rate and its total stays historically
/// correct, so nothing has to be retired. Only a gauge asserts a CURRENT
/// value, which is what goes wrong once the thing it names is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaugeFamily {
    /// `aisix_budget_*` — keyed by an api key plus the team / member it
    /// was bound to at the time.
    Budget,
    /// `aisix_ratelimit_remaining_{requests,tokens}` — keyed by an api
    /// key and a model.
    RatelimitRemaining,
}

// `aisix_deployment_state` is NOT here, though its label set can strand
// the same way. It is edge-triggered: `health.rs`'s `sync_deployment_state`
// returns early unless the state CHANGED, so a live model is written only
// when its health flips. Retiring it on a label change — repointing a
// model at another provider key, or renaming the model — would blank the
// series with nothing to rewrite it until the next health transition, and
// "is this deployment up?" going dark is worse than the stale label it
// would replace. Making it sweepable means having the health tracker
// re-publish on a snapshot change, which is a different change.

fn family_key(family: GaugeFamily) -> &'static str {
    match family {
        GaugeFamily::Budget => "budget",
        GaugeFamily::RatelimitRemaining => "ratelimit_remaining",
    }
}

struct RetirableSeries {
    family: GaugeFamily,
    /// Label VALUES in the family's fixed order — the same order
    /// [`LiveGaugeSeries`] destructures them in.
    labels: Vec<Box<str>>,
    /// The metrics actually registered under this label set, and the only
    /// ones retirement writes.
    ///
    /// A family's members are not all present for every label set: a key
    /// with no budget reaches `clear_budget_gauges`, which writes the
    /// presence flag alone. Retiring the family's full list would then
    /// CREATE the four amount series — `metrics::gauge!` registers on
    /// first use — minting exactly the per-key cardinality that
    /// `clear_budget_gauges` is careful not to.
    metrics: Vec<&'static str>,
}

/// One live label set, handed to the liveness predicate of
/// [`Metrics::retire_stale_gauges`]. Variants carry named fields rather
/// than a slice so a caller cannot silently compare the wrong label
/// against the wrong resource.
#[derive(Debug, Clone, Copy)]
pub enum LiveGaugeSeries<'a> {
    Budget {
        api_key_id: &'a str,
        team_id: &'a str,
        user_id: &'a str,
        user_name: &'a str,
    },
    RatelimitRemaining {
        api_key_id: &'a str,
        /// Reported for context only. It is the model string the CALLER
        /// sent, not a resolved name, so a liveness check must not try to
        /// look it up — see the emit site's note.
        model: &'a str,
    },
}

#[derive(Default)]
struct ConfigLabelState {
    last_hash: Option<String>,
    last_rejected_kinds: std::collections::HashSet<String>,
    last_partial_kinds: std::collections::HashSet<String>,
    last_stale_kinds: std::collections::HashSet<String>,
}

/// Per-thread cap on each worker-cache map. Label sets are bounded by
/// construction (operator-configured names, fixed vocabularies), so
/// production never approaches this; it is a safety valve against an
/// unforeseen unbounded dimension pinning memory in every worker.
const WORKER_CACHE_CAPACITY: usize = 1024;

/// Cap on the retirable-series registry — the process-wide count of
/// distinct label sets across the gauge families that can outlive what
/// they name.
///
/// The bound that makes one number safe for all of them is that every
/// dimension is CONFIGURED: api keys, and models collapsed to the
/// configured set by `usage_attr::metric_model_label`. A caller-shaped
/// dimension would break it — the registry is shared, so one family able
/// to mint entries per request would fill the cap and silently stop every
/// other family from being tracked, which is exactly the failure this
/// whole mechanism exists to prevent.
const RETIRABLE_CAPACITY: usize = 16_384;

/// Separator joining label values into a worker-cache key — a control
/// byte that no bounded label vocabulary contains. A value that DOES
/// contain it (nothing today) falls back to the uncached emit path
/// rather than risk two label sets aliasing one key.
const WORKER_KEY_SEP: char = '\u{1f}';

/// The per-request series handles every request-shaped emit resolves
/// through. Each field registers lazily on first use so the proxy-only
/// paths never mint `aisix_llm_*` series (and vice versa) — the same
/// series-sparsity the plain macro path had.
#[derive(Default)]
struct RequestSeriesHandles {
    proxy_requests: OnceLock<metrics::Counter>,
    proxy_failed_requests: OnceLock<metrics::Counter>,
    proxy_duration: OnceLock<metrics::Histogram>,
    llm_requests: OnceLock<metrics::Counter>,
    llm_duration: OnceLock<metrics::Histogram>,
}

/// Usage dimensions are registered independently so a zero-valued token or
/// spend field does not create an otherwise absent Prometheus series.
#[derive(Default)]
struct UsageSeriesHandles {
    input_tokens: OnceLock<metrics::Counter>,
    output_tokens: OnceLock<metrics::Counter>,
    total_tokens: OnceLock<metrics::Counter>,
    cached_input_tokens: OnceLock<metrics::Counter>,
    cache_read_input_tokens: OnceLock<metrics::Counter>,
    cache_creation_input_tokens: OnceLock<metrics::Counter>,
    spend_micro_usd: OnceLock<metrics::Counter>,
}

// ── Per-worker handle cache ────────────────────────────────────────────────
//
// `metrics::counter!`-family macros rebuild a `Key` (one owned `String` per
// label), hash it, and probe the recorder's sharded registry on EVERY emit.
// `metrics::Counter`/`Gauge`/`Histogram` are `Arc`-backed handles wired
// straight to the series' storage, so registering once per label set and
// reusing the handle removes all of that from the steady-state path.
//
// The cache is `thread_local!`, not shared: this process runs one
// current-thread runtime per pinned core (thread-per-core), and a shared
// map — like the two `RwLock` caches this replaced — puts one contended
// cache line (the lock word) in front of every emit on every worker. The
// spike for AISIX-Cloud#1259 item 3b measured that shared-lock variant
// recovering almost nothing (+0.5% throughput) while the thread-local
// variant recovered +4.4%; per-worker duplication of a bounded handle set
// is the whole trick.
//
// Correctness properties:
// - Keys are `instance id \x1f site \x1f label values…`, so two `Metrics`
//   instances on one thread (parallel tests) can never serve each other's
//   recorder, and one site's entries can never answer another site.
// - Values never alias: label values are joined with a control byte no
//   bounded label vocabulary contains, and a value that does contain it
//   falls back to the uncached emit instead of being cached.
// - Eviction only drops OUR reference. The series and its value live in
//   the recorder's registry; re-registering the same labels returns a
//   handle to the SAME storage, so counts continue exactly where they
//   left off.

/// FNV-1a, hand-rolled: three instructions per byte, no dependency, and
/// no DoS surface — every byte hashed here is operator-bounded config
/// vocabulary, never attacker-chosen cardinality (#451 keeps raw client
/// strings out of label values by contract).
struct FnvHasher(u64);

impl Default for FnvHasher {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

type FnvMap<T> = HashMap<Box<str>, T, std::hash::BuildHasherDefault<FnvHasher>>;

/// One worker's handle maps, one per handle shape. Field-per-shape rather
/// than a value enum so every site gets its concrete handle type back
/// without a runtime discriminant.
#[derive(Default)]
struct WorkerCache {
    counters: FnvMap<metrics::Counter>,
    gauges: FnvMap<metrics::Gauge>,
    histograms: FnvMap<metrics::Histogram>,
    request_series: FnvMap<RequestSeriesHandles>,
    usage_series: FnvMap<UsageSeriesHandles>,
    /// Reused key buffer: the steady-state emit builds its lookup key in
    /// place and allocates nothing; only a first-sight miss allocates the
    /// stored `Box<str>`.
    key_buf: String,
}

thread_local! {
    static WORKER_CACHE: std::cell::RefCell<WorkerCache> =
        std::cell::RefCell::new(WorkerCache::default());
}

/// Selects which [`WorkerCache`] map a handle shape lives in.
/// Hands back a handle-shape's map TOGETHER with the built key. The pair
/// comes from disjoint `WorkerCache` fields, which the per-impl field
/// split proves to the borrow checker — a `&mut WorkerCache -> &mut map`
/// signature would pin the whole cache and forbid reading the key.
trait WorkerCached: Sized {
    fn slot_with_key(cache: &mut WorkerCache) -> (&mut FnvMap<Self>, &str);
}

impl WorkerCached for metrics::Counter {
    fn slot_with_key(cache: &mut WorkerCache) -> (&mut FnvMap<Self>, &str) {
        (&mut cache.counters, cache.key_buf.as_str())
    }
}

impl WorkerCached for metrics::Gauge {
    fn slot_with_key(cache: &mut WorkerCache) -> (&mut FnvMap<Self>, &str) {
        (&mut cache.gauges, cache.key_buf.as_str())
    }
}

impl WorkerCached for metrics::Histogram {
    fn slot_with_key(cache: &mut WorkerCache) -> (&mut FnvMap<Self>, &str) {
        (&mut cache.histograms, cache.key_buf.as_str())
    }
}

impl WorkerCached for RequestSeriesHandles {
    fn slot_with_key(cache: &mut WorkerCache) -> (&mut FnvMap<Self>, &str) {
        (&mut cache.request_series, cache.key_buf.as_str())
    }
}

impl WorkerCached for UsageSeriesHandles {
    fn slot_with_key(cache: &mut WorkerCache) -> (&mut FnvMap<Self>, &str) {
        (&mut cache.usage_series, cache.key_buf.as_str())
    }
}

/// Writes one emit's label values into the worker-cache key buffer.
/// Numeric/bool writers exist so `u16` statuses and flags key without a
/// heap allocation; `dirty` flips when a value contains the separator,
/// which sends that emit down the uncached path.
struct WorkerKey<'a> {
    buf: &'a mut String,
    dirty: bool,
}

impl WorkerKey<'_> {
    fn label(&mut self, value: &str) {
        if value.contains(WORKER_KEY_SEP) {
            self.dirty = true;
        }
        self.buf.push(WORKER_KEY_SEP);
        self.buf.push_str(value);
    }

    fn label_u16(&mut self, value: u16) {
        self.buf.push(WORKER_KEY_SEP);
        // Hand-rolled digits: `fmt::write` machinery was a visible
        // per-emit cost for what is a five-byte-max ASCII render.
        let mut digits = [0u8; 5];
        let mut i = digits.len();
        let mut v = value;
        loop {
            i -= 1;
            digits[i] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        self.buf
            .push_str(std::str::from_utf8(&digits[i..]).expect("ascii digits"));
    }

    fn label_bool(&mut self, value: bool) {
        self.buf.push(WORKER_KEY_SEP);
        self.buf.push(if value { 't' } else { 'f' });
    }
}

impl std::fmt::Debug for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metrics").finish_non_exhaustive()
    }
}

impl Metrics {
    /// Build an isolated recorder. `install_global` is kept for future
    /// use but currently has no effect — every Metrics instance runs
    /// with a local recorder so parallel tests don't collide.
    pub fn new(_install_global: bool) -> Self {
        Self::new_with_env_id("unknown")
    }

    /// Like [`Metrics::new`], stamping `env_id` onto the SLO latency
    /// histograms. Empty ids (standalone DP) collapse to `"unknown"`,
    /// matching the missing-dimension convention used elsewhere.
    pub fn new_with_env_id(env_id: &str) -> Self {
        Self::new_with_buckets(env_id, &HistogramBuckets::default())
    }

    /// Like [`Metrics::new_with_env_id`], with operator-supplied histogram
    /// bucket edges (`observability.metrics.buckets`, AISIX-Cloud#1226).
    pub fn new_with_buckets(env_id: &str, buckets: &HistogramBuckets) -> Self {
        // Buckets ONLY for the SLO histograms and the guardrail latency
        // histogram: with `metrics-exporter-prometheus`, a distribution
        // without configured buckets renders as a summary — which is what
        // every legacy `histogram!` series here intentionally stays as.
        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Full(M_REQUEST_E2E_LATENCY_SECONDS.to_string()),
                &buckets.request_e2e_latency,
            )
            .expect("bucket lists are validated non-empty")
            .set_buckets_for_metric(
                Matcher::Full(M_REQUEST_TTFT_SECONDS.to_string()),
                &buckets.request_ttft,
            )
            .expect("bucket lists are validated non-empty")
            .set_buckets_for_metric(
                Matcher::Full(M_GUARDRAIL_LATENCY_SECONDS.to_string()),
                &buckets.guardrail_latency,
            )
            .expect("bucket lists are validated non-empty")
            .set_buckets_for_metric(
                Matcher::Full(M_A2A_TTFB_SECONDS.to_string()),
                &buckets.a2a_ttfb,
            )
            .expect("bucket lists are validated non-empty")
            .build_recorder();
        let handle = recorder.handle();
        static NEXT_INSTANCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            inner: Arc::new(MetricsInner {
                recorder,
                handle,
                proxy_in_flight: Mutex::new(Vec::new()),
                worker_key_prefix: format!(
                    "{:x}",
                    NEXT_INSTANCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                ),
                env_id: if env_id.is_empty() {
                    "unknown".to_string()
                } else {
                    env_id.to_string()
                },
                config_labels: Mutex::new(ConfigLabelState::default()),
                retirable: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Remember one label set of a retirable gauge family.
    ///
    /// Called from the REGISTRATION path only — a worker-cache miss — so
    /// the steady-state emit never takes this lock. Re-registration (a
    /// second worker, or a cache eviction) is an idempotent no-op.
    fn track_retirable(&self, family: GaugeFamily, metric: &'static str, labels: &[&str]) {
        let mut key = String::with_capacity(64);
        key.push_str(family_key(family));
        for value in labels {
            key.push(WORKER_KEY_SEP);
            key.push_str(value);
        }
        let mut map = self.inner.retirable.lock().expect("retirable registry");
        if let Some(entry) = map.get_mut(key.as_str()) {
            // A label set is registered once per metric per worker, and the
            // members appear at different times — a key gets its presence
            // flag on the request that finds no budget and its amounts only
            // once one exists. So an existing entry still has to learn about
            // a metric it has not seen.
            if !entry.metrics.contains(&metric) {
                entry.metrics.push(metric);
            }
            return;
        }
        // Safety valve, same reasoning as `WORKER_CACHE_CAPACITY`: these
        // label sets are bounded by the configured resources, but an
        // unforeseen unbounded dimension must not pin memory here. Losing
        // a registration only means that series is not retirable — never
        // a wrong value.
        if map.len() >= RETIRABLE_CAPACITY {
            return;
        }
        map.insert(
            Box::from(key.as_str()),
            RetirableSeries {
                family,
                labels: labels.iter().map(|v| Box::from(*v)).collect(),
                metrics: vec![metric],
            },
        );
    }

    /// Retire the gauge series whose label set no longer describes
    /// anything, returning how many were retired this pass.
    ///
    /// `is_live` answers, for one label set, "does the configuration
    /// still contain this exact thing?" — the caller holds the snapshot,
    /// so this crate needs to know nothing about resource lifecycles.
    ///
    /// Retirement writes a value, because the exporter has no API to drop
    /// a series. The value is deliberately NOT zero for anything that
    /// carries a quantity: `aisix_ratelimit_remaining_requests 0` reads as
    /// "quota exhausted" and `aisix_budget_remaining_usd 0` as "spent it
    /// all", so zeroing would replace a stale claim with a false one.
    /// `NaN` is the honest marker — it renders in the text format and
    /// every comparison against it is false, so an alert on a retired
    /// series stops firing instead of inverting. It does mean an
    /// aggregation over a family with a retired member reads `NaN`; the
    /// metric reference says so.
    /// `aisix_budget_details_present` is the exception: it is a boolean
    /// presence flag whose documented cleared value is `0`, and that is
    /// what the guard `details_present == 1` is written against.
    ///
    /// A retired label set is DROPPED from the registry, so it is marked
    /// exactly once rather than re-marked every pass. It is not re-tracked
    /// by a later write either — `with_worker_handle` returns straight out
    /// of the thread-local cache on a hit and never reaches the
    /// registration path. That costs nothing in practice: these ids are
    /// uuids, so an identical label set coming back would need a deleted id
    /// to be reissued, and the shapes that DO recur (a rebind, a rename)
    /// produce a different label set, which registers on its own.
    pub fn retire_stale_gauges(&self, is_live: impl Fn(LiveGaugeSeries<'_>) -> bool) -> usize {
        // Take the entries out under the lock and release it before doing
        // any work: `is_live` reads the caller's snapshot and the retire
        // path registers handles on the recorder, and holding this mutex
        // across either puts the sweeper in the request path's way — the
        // registration side takes the same lock on a worker-cache miss.
        let taken: Vec<RetirableSeries> = {
            let mut map = self.inner.retirable.lock().expect("retirable registry");
            map.drain().map(|(_, v)| v).collect()
        };
        let mut retired = 0usize;
        let mut keep: Vec<RetirableSeries> = Vec::with_capacity(taken.len());
        metrics::with_local_recorder(&self.inner.recorder, || {
            for series in taken {
                let l: Vec<&str> = series.labels.iter().map(|v| &**v).collect();
                let view = match (series.family, l.as_slice()) {
                    (GaugeFamily::Budget, [api_key_id, team_id, user_id, user_name]) => {
                        LiveGaugeSeries::Budget {
                            api_key_id,
                            team_id,
                            user_id,
                            user_name,
                        }
                    }
                    (GaugeFamily::RatelimitRemaining, [api_key_id, model]) => {
                        LiveGaugeSeries::RatelimitRemaining { api_key_id, model }
                    }
                    // Arity is fixed per family at the registration sites;
                    // a mismatch means a family gained a label without its
                    // view. Keep rather than retire on a guess.
                    _ => {
                        keep.push(series);
                        continue;
                    }
                };
                if is_live(view) {
                    keep.push(series);
                    continue;
                }
                retired += 1;
                // Only what this label set actually registered. Writing a
                // family's full list would CREATE the members it never had:
                // `metrics::gauge!` registers on first use, so retiring a
                // budgetless key would mint the four amount series that
                // `clear_budget_gauges` deliberately does not.
                for metric in &series.metrics {
                    // The presence flag is boolean and its documented
                    // cleared value is 0; everything else carries a
                    // quantity and retires to NaN.
                    let value = if *metric == M_BUDGET_DETAILS_PRESENT {
                        0.0
                    } else {
                        f64::NAN
                    };
                    match view {
                        LiveGaugeSeries::Budget {
                            api_key_id,
                            team_id,
                            user_id,
                            user_name,
                        } => metrics::gauge!(
                            *metric,
                            "api_key_id" => api_key_id.to_string(),
                            "team_id" => team_id.to_string(),
                            "user_id" => user_id.to_string(),
                            "user_name" => user_name.to_string(),
                        )
                        .set(value),
                        LiveGaugeSeries::RatelimitRemaining { api_key_id, model } => {
                            metrics::gauge!(
                                *metric,
                                "api_key_id" => api_key_id.to_string(),
                                "model" => model.to_string(),
                            )
                            .set(value)
                        }
                    }
                }
            }
        });
        // A retired label set is dropped: nothing writes it any more, so
        // re-marking it every tick is pure waste, and keeping it would let
        // dead entries fill `RETIRABLE_CAPACITY` and silently stop NEW
        // series from ever being tracked — which is the bug this exists to
        // fix. See `retire_stale_gauges` for why not re-tracking a revived
        // label set costs nothing.
        let mut map = self.inner.retirable.lock().expect("retirable registry");
        for series in keep {
            let mut key = String::with_capacity(64);
            key.push_str(family_key(series.family));
            for value in &series.labels {
                key.push(WORKER_KEY_SEP);
                key.push_str(value);
            }
            map.entry(Box::from(key.as_str())).or_insert(series);
        }
        retired
    }

    /// Render the current metric values in Prometheus text exposition format.
    pub fn render(&self) -> String {
        self.inner.handle.render()
    }

    /// Drain pending histogram samples into their distributions.
    ///
    /// `PrometheusBuilder::build_recorder` does not start the exporter's
    /// background upkeep task, so the server must call this periodically
    /// even when no Prometheus server is scraping the metrics endpoint.
    pub fn run_upkeep(&self) {
        self.inner.handle.run_upkeep();
    }

    /// Reflect the config load-observability state into the recorder. Called
    /// at scrape time by the metrics/status listener so `aisix_config_*`
    /// series always mirror the live [`aisix_core::ConfigStatus`]. Idempotent
    /// and cheap; safe to call on every scrape.
    ///
    /// Etcd-only series (`observed_revision`, `applied_revision`,
    /// `source_connected`) are emitted only in etcd mode. Label churn on the
    /// info/rejected gauges (`hash_info`, `rejected_resources`) zeroes the
    /// prior label set so the exposition never carries two live samples.
    ///
    /// Deliberately NOT routed through the per-worker handle cache: this
    /// runs once per scrape (not per request), and the zeroing discipline
    /// above works on churning label sets — exactly the shape a
    /// first-seen handle cache handles worst. The macro path's per-call
    /// registration cost is irrelevant at scrape frequency.
    pub fn sync_config_status(&self, view: &aisix_core::ConfigMetricsView) {
        use aisix_core::SourceKind;
        let etcd = matches!(view.source_kind, SourceKind::Etcd);
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::gauge!(M_CONFIG_LAST_RELOAD_SUCCESSFUL).set(if view.last_reload_successful {
                1.0
            } else {
                0.0
            });
            if let Some(ts) = view.last_reload_success_ts {
                metrics::gauge!(M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP).set(ts as f64);
            }
            // Counters are tracked authoritatively in ConfigStatus; mirror the
            // absolute value so the counter stays monotonic across scrapes.
            metrics::counter!(M_CONFIG_RELOADS_TOTAL).absolute(view.reloads_total);
            for (reason, count) in &view.reload_failures {
                metrics::counter!(M_CONFIG_RELOAD_FAILURES_TOTAL, "reason" => *reason)
                    .absolute(*count);
            }

            if etcd {
                metrics::gauge!(M_CONFIG_SOURCE_CONNECTED).set(if view.connected == Some(true) {
                    1.0
                } else {
                    0.0
                });
                if let Some(rev) = view.observed_revision {
                    metrics::gauge!(M_CONFIG_OBSERVED_REVISION).set(rev as f64);
                }
                if let Some(rev) = view.applied_revision {
                    metrics::gauge!(M_CONFIG_APPLIED_REVISION).set(rev as f64);
                }
            }

            let mut labels = self.inner.config_labels.lock().expect("config label state");

            // Info-style hash gauge: exactly one live `hash_info{hash=…} 1`;
            // the previously-current hash is zeroed on change so a scraper can
            // filter `== 1`. The `hash` label churns as the applied config
            // changes, and a zeroed series is retained by the recorder — but
            // the churn is bounded by the number of DISTINCT config states
            // (operator/CP edits, a low-frequency event — never per-request),
            // not by traffic. The series name is part of the frozen cross-plane
            // metric contract the control plane also exposes, so it stays.
            if labels.last_hash.as_deref() != view.config_hash.as_deref() {
                if let Some(prev) = labels.last_hash.take() {
                    metrics::gauge!(M_CONFIG_HASH_INFO, "hash" => prev).set(0.0);
                }
            }
            if let Some(hash) = &view.config_hash {
                metrics::gauge!(M_CONFIG_HASH_INFO, "hash" => hash.clone()).set(1.0);
                labels.last_hash = Some(hash.clone());
            }

            // Rejected-resource gauge per kind: set current, zero cleared kinds.
            for kind in &labels.last_rejected_kinds {
                if !view.rejected_by_kind.contains_key(kind) {
                    metrics::gauge!(M_CONFIG_REJECTED_RESOURCES, "kind" => kind.clone()).set(0.0);
                }
            }
            for (kind, count) in &view.rejected_by_kind {
                metrics::gauge!(M_CONFIG_REJECTED_RESOURCES, "kind" => kind.clone())
                    .set(*count as f64);
            }
            labels.last_rejected_kinds = view.rejected_by_kind.keys().cloned().collect();

            // Partially-compatible gauge per kind, same zeroing discipline.
            // Per-field detail deliberately stays off the labels (field paths
            // are not a bounded set) — it lives in `/status/config`.
            for kind in &labels.last_partial_kinds {
                if !view.partially_compatible_by_kind.contains_key(kind) {
                    metrics::gauge!(M_CONFIG_PARTIALLY_COMPATIBLE_RESOURCES, "kind" => kind.clone())
                        .set(0.0);
                }
            }
            for (kind, count) in &view.partially_compatible_by_kind {
                metrics::gauge!(M_CONFIG_PARTIALLY_COMPATIBLE_RESOURCES, "kind" => kind.clone())
                    .set(*count as f64);
            }
            labels.last_partial_kinds = view.partially_compatible_by_kind.keys().cloned().collect();

            // Stale-served gauge per kind (#871), same zeroing discipline.
            for kind in &labels.last_stale_kinds {
                if !view.stale_served_by_kind.contains_key(kind) {
                    metrics::gauge!(M_CONFIG_STALE_SERVED_RESOURCES, "kind" => kind.clone())
                        .set(0.0);
                }
            }
            for (kind, count) in &view.stale_served_by_kind {
                metrics::gauge!(M_CONFIG_STALE_SERVED_RESOURCES, "kind" => kind.clone())
                    .set(*count as f64);
            }
            labels.last_stale_kinds = view.stale_served_by_kind.keys().cloned().collect();
        });
    }

    /// Record the outcome of one proxy request.
    pub fn record_request(
        &self,
        provider: &str,
        model: &str,
        status: u16,
        outcome: RequestOutcome,
        duration: Duration,
    ) {
        self.cached_counter(
            M_REQUESTS_TOTAL,
            1,
            |k| {
                k.label(provider);
                k.label(model);
                k.label_u16(status);
                k.label(outcome.as_str());
            },
            || {
                metrics::counter!(
                    M_REQUESTS_TOTAL,
                    "provider" => provider.to_string(),
                    "model" => model.to_string(),
                    "status" => status.to_string(),
                    "outcome" => outcome.as_str().to_string(),
                )
            },
        );
        self.cached_histogram(
            M_REQUEST_DURATION,
            duration.as_secs_f64(),
            |k| {
                k.label(provider);
                k.label(model);
                k.label_u16(status);
            },
            || {
                metrics::histogram!(
                    M_REQUEST_DURATION,
                    "provider" => provider.to_string(),
                    "model" => model.to_string(),
                    "status" => status.to_string(),
                )
            },
        );
    }

    /// Record one inbound authentication decision on
    /// [`M_AUTH_DECISIONS_TOTAL`] (AISIX-Cloud#1081). Called from the
    /// proxy auth choke point for every credential judgment — allowed
    /// or denied, API-key and JWT paths alike.
    pub fn record_auth_decision(&self, method: &str, allowed: bool, reason: &str) {
        let result = if allowed { "allowed" } else { "denied" };
        let reason = if reason.is_empty() { "none" } else { reason };
        self.cached_counter(
            M_AUTH_DECISIONS_TOTAL,
            1,
            |k| {
                k.label(method);
                k.label(result);
                k.label(reason);
            },
            || {
                metrics::counter!(
                    M_AUTH_DECISIONS_TOTAL,
                    "method" => method.to_string(),
                    "result" => result.to_string(),
                    "reason" => reason.to_string(),
                )
            },
        );
    }

    /// Record the aggregate counters that accompany one timed guardrail
    /// execution. Keeping this beside the per-guardrail histogram makes the
    /// three families share the same all-endpoint coverage.
    fn record_guardrail_outcome(&self, result: &str, error_type: Option<&str>) {
        if result == "blocked" {
            self.cached_counter(
                M_GUARDRAIL_BLOCKS_TOTAL,
                1,
                |_| {},
                || metrics::counter!(M_GUARDRAIL_BLOCKS_TOTAL),
            );
        }
        if let ("bypassed", Some(reason)) = (result, error_type) {
            self.cached_counter(
                M_GUARDRAIL_BYPASSES_TOTAL,
                1,
                |k| k.label(reason),
                || {
                    metrics::counter!(
                        M_GUARDRAIL_BYPASSES_TOTAL,
                        "reason" => reason.to_string(),
                    )
                },
            );
        }
    }

    /// Record one guardrail member execution on
    /// [`M_GUARDRAIL_LATENCY_SECONDS`] (AISIX-Cloud#1076). Called by the
    /// chain fold through the `GuardrailMetricsSink` impl below — once per
    /// member per hook pass, on every handler.
    pub fn record_guardrail_execution(&self, exec: &aisix_core::GuardrailExecution<'_>) {
        // `env_id` is constant per instance and the cache key is already
        // instance-scoped, so it stays out of the key.
        self.cached_histogram(
            M_GUARDRAIL_LATENCY_SECONDS,
            exec.elapsed.as_secs_f64(),
            |k| {
                k.label(exec.guardrail_name);
                k.label(exec.kind);
                k.label(exec.phase);
                k.label(exec.result);
                k.label(exec.error_type.unwrap_or("none"));
            },
            || {
                metrics::histogram!(
                    M_GUARDRAIL_LATENCY_SECONDS,
                    "env_id" => self.inner.env_id.clone(),
                    "guardrail" => exec.guardrail_name.to_string(),
                    "kind" => exec.kind.to_string(),
                    "phase" => exec.phase.to_string(),
                    "result" => exec.result.to_string(),
                    "error_type" => exec.error_type.unwrap_or("none").to_string(),
                )
            },
        );
        self.record_guardrail_outcome(exec.result, exec.error_type);
    }

    /// Count one rate-limit rejection. `scope` is the exceeded
    /// dimension (`requests`/`tokens`/`concurrency`); `layer` names the
    /// limit source (`api_key`/`model`/`mcp`/`policy`); `policy_id`
    /// identifies the offending policy on the `policy` layer (bounded
    /// by the configured policy count) and is empty elsewhere.
    /// Recorded at the quota gate, the one point every endpoint funnels
    /// through (AISIX-Cloud#892).
    pub fn record_ratelimit_rejection(&self, scope: &str, layer: &str, policy_id: Option<&str>) {
        let policy_id = policy_id.unwrap_or_default();
        self.cached_counter(
            M_RATELIMIT_REJECTIONS,
            1,
            |k| {
                k.label(scope);
                k.label(layer);
                k.label(policy_id);
            },
            || {
                metrics::counter!(
                    M_RATELIMIT_REJECTIONS,
                    "scope" => scope.to_string(),
                    "layer" => layer.to_string(),
                    "policy_id" => policy_id.to_string(),
                )
            },
        );
    }

    /// Count one cache-gate outcome. `policy` is the matched policy's
    /// name, `outcome` one of the fixed [`M_CACHE_REQUESTS_TOTAL`]
    /// values.
    pub fn record_cache_event(&self, policy: &str, outcome: &str) {
        self.cached_counter(
            M_CACHE_REQUESTS_TOTAL,
            1,
            |k| {
                k.label(policy);
                k.label(outcome);
            },
            || {
                metrics::counter!(
                    M_CACHE_REQUESTS_TOTAL,
                    "policy" => policy.to_string(),
                    "outcome" => outcome.to_string(),
                )
            },
        );
    }

    /// Record one successful cache semantic-layer embedding call.
    pub fn record_cache_semantic_embed(&self, policy: &str, elapsed: Duration) {
        self.cached_histogram(
            M_CACHE_SEMANTIC_EMBED_SECONDS,
            elapsed.as_secs_f64(),
            |k| k.label(policy),
            || {
                metrics::histogram!(
                    M_CACHE_SEMANTIC_EMBED_SECONDS,
                    "policy" => policy.to_string(),
                )
            },
        );
    }

    /// Count one failed cache semantic-layer embedding attempt. `cause`
    /// is one of the fixed [`M_CACHE_SEMANTIC_EMBED_FAILURES_TOTAL`]
    /// values.
    pub fn record_cache_semantic_embed_failure(&self, policy: &str, cause: &str) {
        self.cached_counter(
            M_CACHE_SEMANTIC_EMBED_FAILURES_TOTAL,
            1,
            |k| {
                k.label(policy);
                k.label(cause);
            },
            || {
                metrics::counter!(
                    M_CACHE_SEMANTIC_EMBED_FAILURES_TOTAL,
                    "policy" => policy.to_string(),
                    "cause" => cause.to_string(),
                )
            },
        );
    }

    /// Count one failed semantic-store operation. `op` is one of the
    /// fixed [`M_CACHE_SEMANTIC_STORE_FAILURES_TOTAL`] values.
    pub fn record_cache_semantic_store_failure(&self, policy: &str, op: &str) {
        self.cached_counter(
            M_CACHE_SEMANTIC_STORE_FAILURES_TOTAL,
            1,
            |k| {
                k.label(policy);
                k.label(op);
            },
            || {
                metrics::counter!(
                    M_CACHE_SEMANTIC_STORE_FAILURES_TOTAL,
                    "policy" => policy.to_string(),
                    "op" => op.to_string(),
                )
            },
        );
    }

    /// Record one finished A2A call across the whole `aisix_a2a_*` family.
    ///
    /// A single entry point rather than four, so a new call site cannot land
    /// with half the series wired — the same anti-drift move
    /// [`crate::metrics`]' request families make. Every label here is bounded
    /// by construction: `agent` is a registered resource, `operation` and
    /// `task_state` come from fixed sets.
    ///
    /// `ttfb` and `stream_events` are `None` / `0` for a unary call, which
    /// observes neither series.
    pub fn record_a2a_call(&self, labels: A2aLabels<'_>, call: A2aCallOutcome) {
        let status = status_bucket(labels.status);
        self.cached_counter(
            M_A2A_REQUESTS_TOTAL,
            1,
            |k| {
                k.label(labels.agent);
                k.label(labels.operation);
                k.label(status);
            },
            || {
                metrics::counter!(
                    M_A2A_REQUESTS_TOTAL,
                    "agent" => labels.agent.to_string(),
                    "operation" => labels.operation.to_string(),
                    "status" => status.to_string(),
                )
            },
        );
        if let Some(ttfb) = call.ttfb {
            self.cached_histogram(
                M_A2A_TTFB_SECONDS,
                ttfb.as_secs_f64(),
                |k| {
                    k.label(labels.agent);
                    k.label(labels.operation);
                },
                || {
                    metrics::histogram!(
                        M_A2A_TTFB_SECONDS,
                        "agent" => labels.agent.to_string(),
                        "operation" => labels.operation.to_string(),
                    )
                },
            );
        }
        if call.stream_events > 0 {
            self.cached_counter(
                M_A2A_STREAM_EVENTS_TOTAL,
                u64::from(call.stream_events),
                |k| {
                    k.label(labels.agent);
                    k.label(labels.operation);
                },
                || {
                    metrics::counter!(
                        M_A2A_STREAM_EVENTS_TOTAL,
                        "agent" => labels.agent.to_string(),
                        "operation" => labels.operation.to_string(),
                    )
                },
            );
        }
        // Only a call that reached a task has a state to report; counting an
        // empty one would invent a bucket for "the upstream never answered".
        if !call.task_state.is_empty() {
            self.cached_counter(
                M_A2A_TASK_STATE_TOTAL,
                1,
                |k| {
                    k.label(labels.agent);
                    k.label(call.task_state);
                },
                || {
                    metrics::counter!(
                        M_A2A_TASK_STATE_TOTAL,
                        "agent" => labels.agent.to_string(),
                        "state" => call.task_state.to_string(),
                    )
                },
            );
        }
    }

    /// Count a request the client abandoned before it produced a response
    /// head. Every label must already be bounded by the caller — see
    /// [`M_PROXY_CLIENT_CANCELLED_TOTAL`].
    pub fn record_client_cancelled(&self, labels: CancelledLabels<'_>) {
        self.cached_counter(
            M_PROXY_CLIENT_CANCELLED_TOTAL,
            1,
            |k| {
                k.label(labels.endpoint);
                k.label(labels.model);
                k.label(labels.provider_key_id);
                k.label(labels.provider_key_name);
            },
            || {
                metrics::counter!(
                    M_PROXY_CLIENT_CANCELLED_TOTAL,
                    "endpoint" => labels.endpoint.to_string(),
                    "model" => labels.model.to_string(),
                    "provider_key_id" => labels.provider_key_id.to_string(),
                    "provider_key_name" => labels.provider_key_name.to_string(),
                )
            },
        );
    }

    /// Count a request refused by the request-body cap. `endpoint` must
    /// already be a bounded route template and `outcome` one of the fixed
    /// drain outcomes — see [`M_PROXY_BODY_LIMIT_REJECTIONS_TOTAL`].
    pub fn record_body_limit_rejection(
        &self,
        endpoint: &str,
        inbound_protocol: &str,
        outcome: &str,
    ) {
        self.cached_counter(
            M_PROXY_BODY_LIMIT_REJECTIONS_TOTAL,
            1,
            |k| {
                k.label(endpoint);
                k.label(inbound_protocol);
                k.label(outcome);
            },
            || {
                metrics::counter!(
                    M_PROXY_BODY_LIMIT_REJECTIONS_TOTAL,
                    "endpoint" => endpoint.to_string(),
                    "inbound_protocol" => inbound_protocol.to_string(),
                    "outcome" => outcome.to_string(),
                )
            },
        );
    }

    pub fn record_tokens(&self, provider: &str, model: &str, total_tokens: u64) {
        if total_tokens == 0 {
            return;
        }
        self.cached_counter(
            M_TOKENS_CONSUMED,
            total_tokens,
            |k| {
                k.label(provider);
                k.label(model);
            },
            || {
                metrics::counter!(
                    M_TOKENS_CONSUMED,
                    "provider" => provider.to_string(),
                    "model" => model.to_string(),
                )
            },
        );
    }

    pub fn increment_proxy_in_flight(&self, endpoint: &str, inbound_protocol: &str) {
        self.apply_in_flight_delta(endpoint, inbound_protocol, 1);
    }

    pub fn decrement_proxy_in_flight(&self, endpoint: &str, inbound_protocol: &str) {
        self.apply_in_flight_delta(endpoint, inbound_protocol, -1);
    }

    /// Apply one in-flight edge and publish the resulting count on the
    /// gauge WHILE the slot lock is held, so concurrent edges on one pair
    /// cannot publish out of order and strand a stale value on an
    /// endpoint that then goes idle. The count clamps at zero (a
    /// decrement without its increment must not wedge the gauge
    /// negative). No path takes this lock from inside the worker cache,
    /// so emitting under it cannot deadlock.
    ///
    /// Precondition: `endpoint` must already be a bounded route template
    /// (`normalize_endpoint_label` output, #451) and `inbound_protocol` a
    /// fixed vocabulary. The slot vector's boundedness — and the linear
    /// scan's cost — depend on it; a raw request path here would grow the
    /// vector without bound.
    fn apply_in_flight_delta(&self, endpoint: &str, inbound_protocol: &str, delta: i64) {
        let mut slots = self.inner.proxy_in_flight.lock().expect("lock in-flight");
        let value = if let Some(slot) = slots
            .iter_mut()
            .find(|(e, p, _)| e == endpoint && p == inbound_protocol)
        {
            slot.2 = (slot.2 + delta).max(0);
            slot.2
        } else {
            let value = delta.max(0);
            slots.push((endpoint.to_string(), inbound_protocol.to_string(), value));
            value
        };
        self.set_proxy_in_flight_gauge(endpoint, inbound_protocol, value);
    }

    /// Shared gauge emit for the increment/decrement pair — one cache
    /// entry, since both sides address the same series.
    fn set_proxy_in_flight_gauge(&self, endpoint: &str, inbound_protocol: &str, value: i64) {
        self.cached_gauge(
            M_PROXY_IN_FLIGHT,
            value as f64,
            |k| {
                k.label(endpoint);
                k.label(inbound_protocol);
            },
            || {
                metrics::gauge!(
                    M_PROXY_IN_FLIGHT,
                    "endpoint" => endpoint.to_string(),
                    "inbound_protocol" => inbound_protocol.to_string(),
                )
            },
        );
    }

    /// Emit through this worker's cached handle for `(site, label values)`,
    /// registering via `register` on the first sight of that label set on
    /// this thread.
    ///
    /// `build_key` writes the label VALUES in a fixed per-site order;
    /// `use_handle` performs the actual increment/set/record. On the
    /// steady-state path this allocates nothing and touches no shared
    /// state beyond the series' own value atomics.
    ///
    /// Invariant: `register` and `use_handle` must not re-enter any
    /// `Metrics` emit (they would hit the `RefCell` re-borrow) — they only
    /// register or write handles, and registration cannot emit.
    fn with_worker_handle<H: WorkerCached>(
        &self,
        site: &'static str,
        build_key: impl FnOnce(&mut WorkerKey<'_>),
        register: impl FnOnce() -> H,
        use_handle: impl FnOnce(&H),
    ) {
        WORKER_CACHE.with(|cell| {
            let cache = &mut *cell.borrow_mut();
            cache.key_buf.clear();
            let dirty = {
                let mut key = WorkerKey {
                    buf: &mut cache.key_buf,
                    dirty: false,
                };
                key.buf.push_str(&self.inner.worker_key_prefix);
                key.buf.push(WORKER_KEY_SEP);
                key.buf.push_str(site);
                build_key(&mut key);
                key.dirty
            };
            if dirty {
                // A label value contained the key separator: emit through
                // a freshly registered handle instead of risking two label
                // sets aliasing one cache key. Same series, slow path.
                use_handle(&register());
                return;
            }
            let (map, key) = H::slot_with_key(cache);
            if let Some(handle) = map.get(key) {
                use_handle(handle);
                return;
            }
            let handle = register();
            use_handle(&handle);
            if map.len() >= WORKER_CACHE_CAPACITY {
                if let Some(evicted) = map.keys().next().cloned() {
                    map.remove(&evicted);
                }
            }
            map.insert(Box::from(key), handle);
        });
    }

    /// Test-only view of this thread's cached request-series entry count.
    /// Each `#[test]` runs on its own thread, so the count starts at zero
    /// and covers exactly that test's emits.
    #[cfg(test)]
    fn worker_request_series_len() -> usize {
        WORKER_CACHE.with(|cell| cell.borrow().request_series.len())
    }

    #[cfg(test)]
    fn worker_usage_series_len() -> usize {
        WORKER_CACHE.with(|cell| cell.borrow().usage_series.len())
    }

    fn with_request_series(
        &self,
        labels: RequestLabels<'_>,
        record: impl FnOnce(&RequestSeriesHandles),
    ) {
        self.with_worker_handle(
            "request_series",
            |k| {
                k.label(labels.endpoint);
                k.label(labels.inbound_protocol);
                k.label(labels.upstream_protocol);
                k.label(labels.provider);
                k.label(labels.model);
                k.label(labels.upstream_model);
                k.label(labels.provider_key_id);
                k.label(labels.provider_key_name);
                k.label(labels.api_key_id);
                k.label(labels.team_id);
                k.label(labels.user_id);
                k.label(labels.user_name);
                k.label_bool(labels.stream);
                k.label_bool(labels.is_fallback);
                k.label_u16(labels.status);
                k.label(labels.outcome.as_str());
            },
            RequestSeriesHandles::default,
            record,
        );
    }

    fn with_usage_series(&self, labels: UsageLabels<'_>, record: impl FnOnce(&UsageSeriesHandles)) {
        self.with_worker_handle(
            "usage_series",
            |k| {
                k.label(labels.endpoint);
                k.label(labels.inbound_protocol);
                k.label(labels.upstream_protocol);
                k.label(labels.provider);
                k.label(labels.model);
                k.label(labels.upstream_model);
                k.label(labels.provider_key_id);
                k.label(labels.provider_key_name);
                k.label(labels.api_key_id);
                k.label(labels.team_id);
                k.label(labels.user_id);
                k.label(labels.user_name);
            },
            UsageSeriesHandles::default,
            record,
        );
    }

    /// Resolve one lazily-registered handle in a series bundle. The
    /// recorder guard is paid only on the one-time init path; a cache-hit
    /// emit never touches it.
    fn init_handle<'h, T>(&self, slot: &'h OnceLock<T>, init: impl FnOnce() -> T) -> &'h T {
        slot.get_or_init(|| metrics::with_local_recorder(&self.inner.recorder, init))
    }

    /// Increment a worker-cached counter. `register` runs under the
    /// recorder guard on the one-time miss path only.
    fn cached_counter(
        &self,
        site: &'static str,
        by: u64,
        build_key: impl FnOnce(&mut WorkerKey<'_>),
        register: impl FnOnce() -> metrics::Counter,
    ) {
        self.with_worker_handle(
            site,
            build_key,
            || metrics::with_local_recorder(&self.inner.recorder, register),
            |counter| counter.increment(by),
        );
    }

    /// Set a worker-cached gauge, same contract as [`Self::cached_counter`].
    fn cached_gauge(
        &self,
        site: &'static str,
        value: f64,
        build_key: impl FnOnce(&mut WorkerKey<'_>),
        register: impl FnOnce() -> metrics::Gauge,
    ) {
        self.with_worker_handle(
            site,
            build_key,
            || metrics::with_local_recorder(&self.inner.recorder, register),
            |gauge| gauge.set(value),
        );
    }

    /// Record into a worker-cached histogram, same contract as
    /// [`Self::cached_counter`].
    fn cached_histogram(
        &self,
        site: &'static str,
        value: f64,
        build_key: impl FnOnce(&mut WorkerKey<'_>),
        register: impl FnOnce() -> metrics::Histogram,
    ) {
        self.with_worker_handle(
            site,
            build_key,
            || metrics::with_local_recorder(&self.inner.recorder, register),
            |histogram| histogram.record(value),
        );
    }

    pub fn record_proxy_request(&self, labels: RequestLabels<'_>, duration: Duration) {
        self.with_request_series(labels, |h| {
            self.init_handle(&h.proxy_requests, || {
                labels.request_counter(M_PROXY_REQUESTS_TOTAL)
            })
            .increment(1);
            self.init_handle(&h.proxy_duration, || {
                labels.request_duration_histogram(M_PROXY_REQUEST_DURATION)
            })
            .record(duration.as_secs_f64());
            if labels.outcome != RequestOutcome::Success {
                self.init_handle(&h.proxy_failed_requests, || {
                    labels.request_counter(M_PROXY_FAILED_REQUESTS_TOTAL)
                })
                .increment(1);
            }
        });
    }

    pub fn record_llm_request(&self, labels: RequestLabels<'_>, duration: Duration) {
        self.with_request_series(labels, |h| {
            self.init_handle(&h.llm_requests, || {
                labels.request_counter(M_LLM_REQUESTS_TOTAL)
            })
            .increment(1);
            self.init_handle(&h.llm_duration, || {
                labels.request_duration_histogram(M_LLM_REQUEST_DURATION)
            })
            .record(duration.as_secs_f64());
        });
    }

    /// Record the paired proxy and LLM request series with one cache lookup.
    /// All request handlers emit these together with the same end-to-end
    /// duration.
    pub fn record_proxy_and_llm_request(&self, labels: RequestLabels<'_>, duration: Duration) {
        let secs = duration.as_secs_f64();
        self.with_request_series(labels, |h| {
            self.init_handle(&h.proxy_requests, || {
                labels.request_counter(M_PROXY_REQUESTS_TOTAL)
            })
            .increment(1);
            self.init_handle(&h.proxy_duration, || {
                labels.request_duration_histogram(M_PROXY_REQUEST_DURATION)
            })
            .record(secs);
            self.init_handle(&h.llm_requests, || {
                labels.request_counter(M_LLM_REQUESTS_TOTAL)
            })
            .increment(1);
            self.init_handle(&h.llm_duration, || {
                labels.request_duration_histogram(M_LLM_REQUEST_DURATION)
            })
            .record(secs);
            if labels.outcome != RequestOutcome::Success {
                self.init_handle(&h.proxy_failed_requests, || {
                    labels.request_counter(M_PROXY_FAILED_REQUESTS_TOTAL)
                })
                .increment(1);
            }
        });
    }

    /// The terminal token + spend emit for one request.
    ///
    /// The three prompt-cache counters (AISIX-Cloud#1404) are recorded
    /// ALONGSIDE `input_tokens` / `total_tokens`, never folded into
    /// them: this function must not change what the existing three
    /// counters report, or every cost figure built on them shifts. The
    /// caller supplies each field exactly as the upstream reported it —
    /// see [`LlmUsage`] for which accounting shape each one carries.
    pub fn record_llm_usage(&self, labels: UsageLabels<'_>, usage: LlmUsage) {
        let spend_micro_usd = usage.spend_micro_usd();
        if usage.input_tokens == 0
            && usage.output_tokens == 0
            && usage.total_tokens == 0
            && usage.cached_input_tokens == 0
            && usage.cache_read_input_tokens == 0
            && usage.cache_creation_input_tokens == 0
            && spend_micro_usd.is_none()
        {
            return;
        }
        self.with_usage_series(labels, |handles| {
            if usage.input_tokens > 0 {
                self.init_handle(&handles.input_tokens, || {
                    labels.counter(M_LLM_INPUT_TOKENS_TOTAL)
                })
                .increment(u64::from(usage.input_tokens));
            }
            if usage.output_tokens > 0 {
                self.init_handle(&handles.output_tokens, || {
                    labels.counter(M_LLM_OUTPUT_TOKENS_TOTAL)
                })
                .increment(u64::from(usage.output_tokens));
            }
            if usage.total_tokens > 0 {
                self.init_handle(&handles.total_tokens, || {
                    labels.counter(M_LLM_TOTAL_TOKENS_TOTAL)
                })
                .increment(u64::from(usage.total_tokens));
            }
            if usage.cached_input_tokens > 0 {
                self.init_handle(&handles.cached_input_tokens, || {
                    labels.counter(M_LLM_CACHED_INPUT_TOKENS_TOTAL)
                })
                .increment(u64::from(usage.cached_input_tokens));
            }
            if usage.cache_read_input_tokens > 0 {
                self.init_handle(&handles.cache_read_input_tokens, || {
                    labels.counter(M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL)
                })
                .increment(u64::from(usage.cache_read_input_tokens));
            }
            if usage.cache_creation_input_tokens > 0 {
                self.init_handle(&handles.cache_creation_input_tokens, || {
                    labels.counter(M_LLM_CACHE_CREATION_INPUT_TOKENS_TOTAL)
                })
                .increment(u64::from(usage.cache_creation_input_tokens));
            }
            if let Some(value) = spend_micro_usd {
                self.init_handle(&handles.spend_micro_usd, || {
                    labels.counter(M_LLM_SPEND_MICRO_USD_TOTAL)
                })
                .increment(value);
            }
        });
    }

    pub fn record_time_to_first_token(&self, labels: UsageLabels<'_>, ttft: Duration) {
        if ttft.is_zero() {
            return;
        }
        self.cached_histogram(
            M_LLM_TTFT,
            ttft.as_secs_f64(),
            |k| {
                k.label(labels.endpoint);
                k.label(labels.inbound_protocol);
                k.label(labels.upstream_protocol);
                k.label(labels.provider);
                k.label(labels.model);
                k.label(labels.upstream_model);
                k.label(labels.provider_key_id);
                k.label(labels.provider_key_name);
                k.label(labels.api_key_id);
                k.label(labels.team_id);
                k.label(labels.user_id);
                k.label(labels.user_name);
            },
            || {
                metrics::histogram!(
                    M_LLM_TTFT,
                    "endpoint" => labels.endpoint.to_string(),
                    "inbound_protocol" => labels.inbound_protocol.to_string(),
                    "upstream_protocol" => labels.upstream_protocol.to_string(),
                    "provider" => labels.provider.to_string(),
                    "model" => labels.model.to_string(),
                    "upstream_model" => labels.upstream_model.to_string(),
                    "provider_key_id" => labels.provider_key_id.to_string(),
                    "provider_key_name" => labels.provider_key_name.to_string(),
                    "api_key_id" => labels.api_key_id.to_string(),
                    "team_id" => labels.team_id.to_string(),
                    "user_id" => labels.user_id.to_string(),
                    "user_name" => labels.user_name.to_string(),
                )
            },
        );
    }

    /// #890 req-4: record token volume for the inbound `client_type` on the
    /// dedicated [`M_LLM_TOKENS_BY_CLIENT_TOTAL`] series. `client_type` MUST
    /// come from [`ClientTypeClassifier::classify`] (or the built-in
    /// [`client_type_from_user_agent`]) — never raw client input — so the
    /// value set stays bounded by built-ins ∪ boot-validated operator rules
    /// (AISIX-Cloud#1045); zero dims are skipped to keep the series sparse.
    ///
    /// `model` (AISIX-Cloud#1044) is the requested logical model — callers
    /// MUST pass the same value they put in [`UsageLabels::model`] (or its
    /// endpoint's equivalent), never the raw client string of an unresolved
    /// request nor the routed `upstream_model`, so the label stays bounded by
    /// the configured model set and joins cleanly with the `aisix_llm_*`
    /// families.
    ///
    /// `total_tokens` is the caller's canonical cache-inclusive total
    /// (`input + output + Anthropic cache_creation/cache_read`), emitted under
    /// `token_type="total"` (AISIX-Cloud#1002). It is passed in — not derived
    /// from `input + output` — because Anthropic reports cache tokens as
    /// counters SEPARATE from `input_tokens`, so a prompt+completion sum
    /// undercounts cached traffic (same reason as [`total_tokens_with_cache`]
    /// and the `aisix_llm_total_tokens_total` fix in #679).
    pub fn record_llm_tokens_by_client(
        &self,
        client_type: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
    ) {
        if input_tokens == 0 && output_tokens == 0 && total_tokens == 0 {
            return;
        }
        for (token_type, count) in [
            ("input", input_tokens),
            ("output", output_tokens),
            ("total", total_tokens),
        ] {
            if count == 0 {
                continue;
            }
            self.cached_counter(
                M_LLM_TOKENS_BY_CLIENT_TOTAL,
                count,
                |k| {
                    k.label(client_type);
                    k.label(model);
                    k.label(token_type);
                },
                || {
                    metrics::counter!(
                        M_LLM_TOKENS_BY_CLIENT_TOTAL,
                        "client_type" => client_type.to_string(),
                        "model" => model.to_string(),
                        "token_type" => token_type,
                    )
                },
            );
        }
    }

    /// Shared cached emit for the deployment counter family.
    fn cached_deployment_counter(&self, metric: &'static str, labels: DeploymentLabels<'_>) {
        self.cached_counter(
            metric,
            1,
            |k| {
                k.label(labels.provider);
                k.label(labels.model);
                k.label(labels.upstream_model);
                k.label(labels.provider_key_id);
            },
            || {
                metrics::counter!(
                    metric,
                    "provider" => labels.provider.to_string(),
                    "model" => labels.model.to_string(),
                    "upstream_model" => labels.upstream_model.to_string(),
                    "provider_key_id" => labels.provider_key_id.to_string(),
                )
            },
        );
    }

    pub fn record_deployment_request(&self, labels: DeploymentLabels<'_>, outcome: RequestOutcome) {
        self.cached_deployment_counter(M_DEPLOYMENT_REQUESTS_TOTAL, labels);
        match outcome {
            RequestOutcome::Success => {
                self.cached_deployment_counter(M_DEPLOYMENT_SUCCESS_TOTAL, labels)
            }
            _ => self.cached_deployment_counter(M_DEPLOYMENT_FAILURE_TOTAL, labels),
        }
    }

    pub fn set_deployment_state(&self, labels: DeploymentLabels<'_>, state: DeploymentState) {
        self.cached_gauge(
            M_DEPLOYMENT_STATE,
            state.as_f64(),
            |k| {
                k.label(labels.provider);
                k.label(labels.model);
                k.label(labels.upstream_model);
                k.label(labels.provider_key_id);
            },
            || {
                metrics::gauge!(
                    M_DEPLOYMENT_STATE,
                    "provider" => labels.provider.to_string(),
                    "model" => labels.model.to_string(),
                    "upstream_model" => labels.upstream_model.to_string(),
                    "provider_key_id" => labels.provider_key_id.to_string(),
                )
            },
        );
    }

    pub fn record_deployment_cooldown(&self, labels: DeploymentLabels<'_>) {
        self.cached_deployment_counter(M_DEPLOYMENT_COOLED_DOWN_TOTAL, labels);
    }

    pub fn record_routing_fallback(&self, success: bool, model: &str, fallback_model: &str) {
        let metric = if success {
            M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL
        } else {
            M_ROUTING_FAILED_FALLBACKS_TOTAL
        };
        self.cached_counter(
            metric,
            1,
            |k| {
                k.label(model);
                k.label(fallback_model);
            },
            || {
                metrics::counter!(
                    metric,
                    "model" => model.to_string(),
                    "fallback_model" => fallback_model.to_string(),
                )
            },
        );
    }

    pub fn set_rate_limit_remaining(
        &self,
        api_key_id: &str,
        model: &str,
        requests: Option<u64>,
        tokens: Option<u64>,
    ) {
        if let Some(value) = requests {
            self.cached_gauge(
                M_RATELIMIT_REMAINING_REQUESTS,
                value as f64,
                |k| {
                    k.label(api_key_id);
                    k.label(model);
                },
                || {
                    self.track_retirable(
                        GaugeFamily::RatelimitRemaining,
                        M_RATELIMIT_REMAINING_REQUESTS,
                        &[api_key_id, model],
                    );
                    metrics::gauge!(
                        M_RATELIMIT_REMAINING_REQUESTS,
                        "api_key_id" => api_key_id.to_string(),
                        "model" => model.to_string(),
                    )
                },
            );
        }
        if let Some(value) = tokens {
            self.cached_gauge(
                M_RATELIMIT_REMAINING_TOKENS,
                value as f64,
                |k| {
                    k.label(api_key_id);
                    k.label(model);
                },
                || {
                    self.track_retirable(
                        GaugeFamily::RatelimitRemaining,
                        M_RATELIMIT_REMAINING_TOKENS,
                        &[api_key_id, model],
                    );
                    metrics::gauge!(
                        M_RATELIMIT_REMAINING_TOKENS,
                        "api_key_id" => api_key_id.to_string(),
                        "model" => model.to_string(),
                    )
                },
            );
        }
    }

    /// Shared cached emit for the budget gauge family.
    fn cached_budget_gauge(&self, metric: &'static str, labels: BudgetLabels<'_>, value: f64) {
        self.cached_gauge(
            metric,
            value,
            |k| {
                k.label(labels.api_key_id);
                k.label(labels.team_id);
                k.label(labels.user_id);
                k.label(labels.user_name);
            },
            || {
                self.track_retirable(
                    GaugeFamily::Budget,
                    metric,
                    &[
                        labels.api_key_id,
                        labels.team_id,
                        labels.user_id,
                        labels.user_name,
                    ],
                );
                metrics::gauge!(
                    metric,
                    "api_key_id" => labels.api_key_id.to_string(),
                    "team_id" => labels.team_id.to_string(),
                    "user_id" => labels.user_id.to_string(),
                    "user_name" => labels.user_name.to_string(),
                )
            },
        );
    }

    pub fn set_budget_gauges(&self, labels: BudgetLabels<'_>, budget: BudgetGauges) {
        self.cached_budget_gauge(M_BUDGET_DETAILS_PRESENT, labels, 1.0);
        if let Some(value) = budget.limit_usd {
            self.cached_budget_gauge(M_BUDGET_LIMIT_USD, labels, value);
        }
        if let Some(value) = budget.spent_usd {
            self.cached_budget_gauge(M_BUDGET_SPENT_USD, labels, value);
        }
        if let Some(value) = budget.remaining_usd {
            self.cached_budget_gauge(M_BUDGET_REMAINING_USD, labels, value);
        }
        if let Some(value) = budget.reset_seconds {
            self.cached_budget_gauge(M_BUDGET_RESET_SECONDS, labels, value as f64);
        }
    }

    /// The key carries no budget: clear the presence flag and nothing
    /// else.
    ///
    /// Deliberately does NOT retract the four amounts, even though a key
    /// whose budget was removed leaves them frozen at their last budgeted
    /// values. This runs on EVERY request whose decision carries no
    /// budget, which includes every request on a deployment with budgets
    /// switched off (`budget::Mode::Disabled` answers `allow_all()`, and
    /// that is the default) — writing the amounts here would mint four
    /// `NaN` series per api key for a feature nobody enabled, five-ing
    /// the cardinality of this family to fix a case that only arises
    /// after a budget is deleted. The documented guard
    /// `aisix_budget_details_present == 1` is what covers that case, and
    /// [`Metrics::retire_stale_gauges`] covers the key going away.
    pub fn clear_budget_gauges(&self, labels: BudgetLabels<'_>) {
        self.cached_budget_gauge(M_BUDGET_DETAILS_PRESENT, labels, 0.0);
    }

    pub fn record_redis_failure(&self, operation: &str) {
        self.cached_counter(
            M_REDIS_FAILURES_TOTAL,
            1,
            |k| k.label(operation),
            || metrics::counter!(M_REDIS_FAILURES_TOTAL, "operation" => operation.to_string()),
        );
    }

    /// Count a usage event that never made it into the queue. Carries the
    /// same attribution as the emit counter (AISIX-Cloud#1317): a dropped
    /// event is a hole in the usage record, and the operator's question is
    /// whose data is missing, which `reason` alone cannot answer.
    ///
    /// "The same attribution" means every dimension [`UsageEventLabels`]
    /// supplies, `upstream_protocol` included (AISIX-Cloud#1403) — the
    /// pair is read as `emitted == delivered + dropped`, and a slice of
    /// that invariant exists only on dimensions BOTH sides carry. The
    /// protocol is a function of the ProviderKey row `provider_key_id`
    /// already names, so like `provider_key_name` it adds no series here.
    /// (`handler` / `status_code` / `status` / `inbound_protocol`
    /// are the emit counter's own arguments, not attribution, and stay
    /// one-sided — `emitted == delivered + dropped` is an invariant over
    /// the attribution dimensions, which is why the member pair `user_id`
    /// / `user_name` DOES appear on both.)
    pub fn record_usage_event_drop(&self, reason: &str, labels: UsageEventLabels<'_>) {
        self.cached_counter(
            M_USAGE_EVENT_DROPS_TOTAL,
            1,
            |k| {
                k.label(reason);
                k.label(labels.model);
                k.label(labels.provider_key_id);
                k.label(labels.provider_key_name);
                k.label(labels.user_id);
                k.label(labels.user_name);
                k.label(labels.upstream_protocol);
            },
            || {
                metrics::counter!(
                    M_USAGE_EVENT_DROPS_TOTAL,
                    "reason" => reason.to_string(),
                    "model" => labels.model.to_string(),
                    "provider_key_id" => labels.provider_key_id.to_string(),
                    "provider_key_name" => labels.provider_key_name.to_string(),
                    "user_id" => labels.user_id.to_string(),
                    "user_name" => labels.user_name.to_string(),
                    "upstream_protocol" => labels.upstream_protocol.to_string(),
                )
            },
        );
    }

    /// Issue #408: bump on every `UsageSink::try_emit` call (the
    /// handler's emission intent — paired with the drops counter so
    /// the invariant `emitted == delivered + dropped` holds strictly).
    ///
    /// The surface labels are `&'static str` so prometheus cardinality is
    /// type-system-bounded:
    /// - `handler`: bounded handler family (`chat`, `embeddings`,
    ///   `messages`, `mcp`, etc.)
    /// - `status_code`: bucketed by `status_bucket()` (one of `2xx` /
    ///   `3xx` / `4xx` / `5xx` / `other`) — never a raw u16
    /// - `inbound_protocol`: normalised by the caller to one of
    ///   `"openai"` / `"anthropic"` / `"mcp"` / `"other"` (audit MEDIUM-3 —
    ///   `&'static str` here prevents user-controlled cardinality)
    ///
    /// [`UsageEventLabels`] adds the model, the ProviderKey the event was
    /// attributed to (AISIX-Cloud#1317), that key's `upstream_protocol`
    /// (AISIX-Cloud#1403) and the member pair `user_id` / `user_name`. Those cannot be `&'static str`, so they are
    /// bounded at the proxy's emit chokepoint instead — see that type. The
    /// dimensions they add are ones `aisix_llm_requests_total` already
    /// carries, so this counter stays well inside the request families'
    /// series count.
    pub fn record_usage_event_emit(
        &self,
        handler: &'static str,
        status_code: u16,
        inbound_protocol: &'static str,
        labels: UsageEventLabels<'_>,
    ) {
        let status_class = status_bucket(status_code);
        self.cached_counter(
            M_USAGE_EVENT_EMITS_TOTAL,
            1,
            |k| {
                k.label(handler);
                k.label(status_class);
                k.label_u16(status_code);
                k.label(inbound_protocol);
                k.label(labels.model);
                k.label(labels.provider_key_id);
                k.label(labels.provider_key_name);
                k.label(labels.user_id);
                k.label(labels.user_name);
                k.label(labels.upstream_protocol);
            },
            || {
                metrics::counter!(
                    M_USAGE_EVENT_EMITS_TOTAL,
                    "handler" => handler,
                    "status_code" => status_class,
                    "status" => status_code.to_string(),
                    "inbound_protocol" => inbound_protocol,
                    "upstream_protocol" => labels.upstream_protocol.to_string(),
                    "model" => labels.model.to_string(),
                    "provider_key_id" => labels.provider_key_id.to_string(),
                    "provider_key_name" => labels.provider_key_name.to_string(),
                    "user_id" => labels.user_id.to_string(),
                    "user_name" => labels.user_name.to_string(),
                )
            },
        );
    }

    pub fn record_otlp_fanout_drop(&self, exporter: &str, reason: &str) {
        self.cached_counter(
            M_OTLP_FANOUT_DROPS_TOTAL,
            1,
            |k| {
                k.label(exporter);
                k.label(reason);
            },
            || {
                metrics::counter!(
                    M_OTLP_FANOUT_DROPS_TOTAL,
                    "exporter" => exporter.to_string(),
                    "reason" => reason.to_string(),
                )
            },
        );
    }

    pub fn record_otlp_fanout_failure(&self, exporter: &str) {
        self.cached_counter(
            M_OTLP_FANOUT_FAILURES_TOTAL,
            1,
            |k| k.label(exporter),
            || {
                metrics::counter!(
                    M_OTLP_FANOUT_FAILURES_TOTAL,
                    "exporter" => exporter.to_string(),
                )
            },
        );
    }

    /// Observe one request's client-perceived end-to-end latency on
    /// [`M_REQUEST_E2E_LATENCY_SECONDS`]. Call exactly once per request:
    /// at handler return for non-streaming requests and failures, at
    /// stream completion for committed streams.
    pub fn record_request_e2e_latency(&self, labels: LatencyLabels<'_>, elapsed: Duration) {
        self.cached_latency_histogram(M_REQUEST_E2E_LATENCY_SECONDS, labels, elapsed);
    }

    /// Shared cached emit for the two SLO latency histograms — identical
    /// label shape, `env_id` constant per instance (the cache key is
    /// already instance-scoped, so it stays out of the key).
    fn cached_latency_histogram(
        &self,
        metric: &'static str,
        labels: LatencyLabels<'_>,
        elapsed: Duration,
    ) {
        let model = if labels.model.is_empty() {
            "unknown"
        } else {
            labels.model
        };
        let provider = if labels.provider.is_empty() {
            "unknown"
        } else {
            labels.provider
        };
        self.cached_histogram(
            metric,
            elapsed.as_secs_f64(),
            |k| {
                k.label(labels.endpoint);
                k.label(model);
                k.label(provider);
                k.label(status_bucket(labels.status));
                k.label_bool(labels.streaming);
            },
            || {
                metrics::histogram!(
                    metric,
                    "env_id" => self.inner.env_id.clone(),
                    "endpoint" => labels.endpoint.to_string(),
                    "model" => model.to_string(),
                    "provider" => provider.to_string(),
                    "status_class" => status_bucket(labels.status),
                    "streaming" => bool_str(labels.streaming),
                )
            },
        );
    }

    /// Observe a streaming request's time-to-first-token on
    /// [`M_REQUEST_TTFT_SECONDS`]. Zero durations are skipped (TTFT was
    /// never measured — e.g. the stream died before the first token).
    pub fn record_request_ttft(&self, labels: LatencyLabels<'_>, ttft: Duration) {
        if ttft.is_zero() {
            return;
        }
        self.cached_latency_histogram(M_REQUEST_TTFT_SECONDS, labels, ttft);
    }
}

/// The injection point the guardrail chain records through
/// (AISIX-Cloud#1076): `aisix-guardrails` sees only this core trait, so it
/// stays free of a metrics dependency.
impl aisix_core::GuardrailMetricsSink for Metrics {
    fn record_guardrail_execution(&self, exec: &aisix_core::GuardrailExecution<'_>) {
        Metrics::record_guardrail_execution(self, exec);
    }
}

/// Who was called and how, for the `aisix_a2a_*` family. Every field is
/// bounded: a registered agent name and the canonical operation set.
#[derive(Clone, Copy)]
pub struct A2aLabels<'a> {
    /// Registered agent name. Borrowed: it comes from the snapshot, and an
    /// unregistered agent is refused before any call is metered.
    pub agent: &'a str,
    /// Canonical operation. `&'static str` so the fixed set is enforced by the
    /// compiler rather than by a comment — the same reason
    /// `record_usage_event_emit` takes its handler that way.
    pub operation: &'static str,
    /// Raw HTTP status; bucketed to `2xx` / `4xx` / … at record time.
    pub status: u16,
}

/// What one A2A call did, for the series that only some calls observe.
#[derive(Clone, Copy, Default)]
pub struct A2aCallOutcome {
    /// Time to the upstream agent's first streamed event. `None` for a unary
    /// call and for a stream that never produced one.
    pub ttfb: Option<Duration>,
    /// Events relayed downstream. 0 for a unary call.
    pub stream_events: u32,
    /// Normalized task state the call ended on; empty when no response
    /// carried one. `&'static str` for the same reason as
    /// [`A2aLabels::operation`].
    pub task_state: &'static str,
}

#[derive(Clone, Copy)]
pub struct LatencyLabels<'a> {
    /// Route template, e.g. `/v1/chat/completions`. Bounded set.
    pub endpoint: &'a str,
    /// Gateway-level model name (the dashboard alias the caller requested).
    pub model: &'a str,
    /// Provider kind (`openai`, `anthropic`, …); `unknown` pre-resolution.
    pub provider: &'a str,
    /// Raw HTTP status; bucketed to `2xx`/`4xx`/… at record time.
    pub status: u16,
    pub streaming: bool,
}

/// Missing dimensions default to `"unknown"`, never an empty label value.
/// Bucket an HTTP status code into one of `2xx` / `3xx` / `4xx` /
/// `5xx` / `other` (the last covers 1xx and out-of-range). Used by
/// the UsageEvent emission counter (#408) to keep prometheus label
/// cardinality bounded — raw `u16` would explode to ~1000 series per
/// handler×protocol combination.
fn status_bucket(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "other",
    }
}

/// Render a boolean metric dimension as a stable `"true"`/`"false"` label
/// value (#890 reqs 1 & 2: `stream`, `is_fallback`).
fn bool_str(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

/// Normalise a raw inbound `User-Agent` into a BOUNDED `client_type` label
/// for [`M_LLM_TOKENS_BY_CLIENT_TOTAL`] (#890 req-4).
///
/// The result is always one of a fixed allowlist (plus `"other"` /
/// `"unknown"`), returned as `&'static str`, so a client-controlled header
/// can never grow prometheus cardinality. Matching is case-insensitive and
/// substring-based, most-specific first (SDK/tool names win over the generic
/// HTTP-library buckets, whose UA a higher-level SDK often embeds). The full
/// user-agent and its version are preserved on the `UsageEvent`/logs — only
/// this coarse, bounded type ever becomes a metric label.
pub fn client_type_from_user_agent(user_agent: &str) -> &'static str {
    let ua = user_agent.trim().to_ascii_lowercase();
    if ua.is_empty() {
        return "unknown";
    }
    // (substring, label) — ordered: products/SDKs before generic libs.
    const TABLE: &[(&str, &str)] = &[
        ("claude-cli", "claude-code"),
        ("claude-code", "claude-code"),
        ("codex", "codex"),
        ("cline", "cline"),
        // Cline-family forks (AISIX-Cloud#1045). Each sends `<Product>/<ver>`
        // on its OpenAI-compatible provider path (Roo since PR #5492,
        // Kilo ≤5.16.2, Zoo ≥3.54.0); the second spelling covers the
        // `roo-code/<ver> (<os>; <arch>)`-style native-path variants.
        ("roo-code", "roo-code"),
        ("roocode", "roo-code"),
        ("kilo-code", "kilocode"),
        ("kilocode", "kilocode"),
        ("zoo-code", "zoo-code"),
        ("zoocode", "zoo-code"),
        // VS Code Copilot Chat BYOK sends `GitHubCopilotChat/<ver>` from
        // the user's machine; the broader needle also catches other
        // `GitHubCopilot*` variants should they surface in a UA.
        ("githubcopilot", "github-copilot"),
        // Cursor routes BYO-endpoint traffic through its own backend,
        // which presents `Cursor/1.0` (version segment is fixed).
        ("cursor", "cursor"),
        // Terminal agents / editors (AISIX-Cloud#1045). opencode PREFIXES
        // the Vercel AI SDK UA (`opencode/<ver> ai-sdk/…`), so it must
        // stay ahead of the `ai-sdk/provider-utils` bucket below. Qwen
        // Code sends `QwenCode/<ver> (<os>; <arch>)` on OpenAI paths but
        // masquerades as `claude-cli/…` toward non-Anthropic hosts on its
        // Anthropic path — that traffic lands in `claude-code`, which a
        // substring table cannot untangle. Gemini CLI embeds the surface
        // (`GeminiCLI-tui/<ver>/<model> (…)`). `zed/` keeps the slash so
        // the needle requires the `Zed/<ver>` token form.
        ("opencode", "opencode"),
        ("qwencode", "qwen-code"),
        ("geminicli", "gemini-cli"),
        ("charm-crush", "crush"),
        ("zed/", "zed"),
        ("aider", "aider"),
        ("openai-python", "openai-python"),
        ("openai/python", "openai-python"),
        ("openai-node", "openai-node"),
        ("openai/js", "openai-node"),
        ("anthropic-sdk-python", "anthropic-python"),
        ("anthropic/python", "anthropic-python"),
        ("anthropic-sdk-typescript", "anthropic-typescript"),
        ("anthropic/js", "anthropic-typescript"),
        ("langchain", "langchain"),
        ("llama-index", "llamaindex"),
        ("llama_index", "llamaindex"),
        ("llamaindex", "llamaindex"),
        ("litellm", "litellm"),
        // Vercel AI SDK default UA (`ai/<v> ai-sdk/provider-utils/<v>
        // runtime/<rt>`) — the whole-SDK bucket for tools that don't
        // override it (Cline 4.x, Kilo Code 7.x, …).
        ("ai-sdk/provider-utils", "vercel-ai-sdk"),
        ("curl", "curl"),
        ("python-requests", "python-requests"),
        ("python-httpx", "httpx"),
        ("httpx", "httpx"),
        ("aiohttp", "aiohttp"),
        ("okhttp", "okhttp"),
        ("go-http-client", "go-http-client"),
        ("node-fetch", "node"),
        ("undici", "node"),
        ("axios", "node"),
        ("postmanruntime", "postman"),
        ("mozilla", "browser"),
    ];
    for (needle, label) in TABLE {
        if ua.contains(needle) {
            return label;
        }
    }
    "other"
}

/// Boot-compiled `client_type` classifier: operator rules from
/// `observability.metrics.client_type_rules` (AISIX-Cloud#1045) tried in
/// config order first, then the built-in
/// [`client_type_from_user_agent`] allowlist. Custom rules deliberately
/// outrank built-ins so a deployment can re-bucket anything — e.g. an
/// in-house tool whose UA embeds `axios` and would otherwise land in
/// `node`. Cardinality stays bounded: a match emits the rule's fixed
/// `client` value (validated at compile), never request-derived text.
#[derive(Debug, Default)]
pub struct ClientTypeClassifier {
    rules: Vec<(regex::Regex, String)>,
}

impl ClientTypeClassifier {
    pub const MAX_RULES: usize = 64;
    pub const MAX_PATTERN_LEN: usize = 512;
    pub const MAX_CLIENT_LEN: usize = 64;

    /// Built-ins only — the behaviour of every deployment without
    /// `client_type_rules` configured.
    pub fn builtin() -> Self {
        Self::default()
    }

    /// Compile + validate operator rules. Errors are boot-fatal by design
    /// (a silently dropped rule would misattribute traffic until someone
    /// notices the label is missing).
    pub fn compile(rules: &[aisix_core::ClientTypeRule]) -> Result<Self, String> {
        if rules.len() > Self::MAX_RULES {
            return Err(format!(
                "observability.metrics.client_type_rules: {} rules exceed the limit of {}",
                rules.len(),
                Self::MAX_RULES
            ));
        }
        let mut compiled = Vec::with_capacity(rules.len());
        for (i, rule) in rules.iter().enumerate() {
            let ctx = format!("observability.metrics.client_type_rules[{i}]");
            if rule.pattern.is_empty() || rule.pattern.len() > Self::MAX_PATTERN_LEN {
                return Err(format!(
                    "{ctx}: pattern must be 1..={} bytes",
                    Self::MAX_PATTERN_LEN
                ));
            }
            if !valid_client_label(&rule.client) {
                return Err(format!(
                    "{ctx}: client {:?} must match [a-z0-9][a-z0-9._-]* and be at most {} chars",
                    rule.client,
                    Self::MAX_CLIENT_LEN
                ));
            }
            let re = regex::RegexBuilder::new(&rule.pattern)
                .case_insensitive(true)
                .build()
                .map_err(|e| format!("{ctx}: invalid pattern: {e}"))?;
            compiled.push((re, rule.client.clone()));
        }
        Ok(Self { rules: compiled })
    }

    /// Classify a raw inbound `User-Agent`. Empty/whitespace UA is always
    /// `unknown` (custom rules never see it — `unknown` keeps meaning "the
    /// client sent nothing"); then custom rules in config order (first
    /// match wins); then the built-in table; then `other`.
    pub fn classify<'a>(&'a self, user_agent: &str) -> &'a str {
        let ua = user_agent.trim();
        if ua.is_empty() {
            return "unknown";
        }
        for (re, client) in &self.rules {
            if re.is_match(ua) {
                return client;
            }
        }
        client_type_from_user_agent(ua)
    }
}

/// Prometheus-safe label value: lowercase alnum start, then alnum/`.`/`_`/`-`.
fn valid_client_label(label: &str) -> bool {
    if label.is_empty() || label.len() > ClientTypeClassifier::MAX_CLIENT_LEN {
        return false;
    }
    let mut chars = label.chars();
    let first = chars.next().expect("non-empty checked above");
    (first.is_ascii_lowercase() || first.is_ascii_digit())
        && chars
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
}

#[derive(Debug, Clone, Copy)]
pub struct RequestLabels<'a> {
    pub endpoint: &'a str,
    pub inbound_protocol: &'a str,
    /// The upstream API protocol the selected target actually speaks —
    /// the ProviderKey adapter's wire value, as resolved by the SAME
    /// two-tier rule dispatch uses (AISIX-Cloud#1403).
    ///
    /// A bounded set: `openai` / `anthropic` / `bedrock` / `vertex` /
    /// `azure-openai`, plus `unknown` for a request that never selected
    /// an upstream (a pre-dispatch rejection, or a non-inference route
    /// like `/mcp`). Distinct from `inbound_protocol`, which is the
    /// protocol the CALLER spoke: a request arriving on `/v1/messages`
    /// and served by an OpenAI-shape upstream carries
    /// `inbound_protocol="anthropic", upstream_protocol="openai"`.
    ///
    /// `provider` cannot stand in for it: vendor identity is an open
    /// string, and the same vendor can be reached through different
    /// adapters (a `byo` key with `adapter: anthropic`, say).
    pub upstream_protocol: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
    /// Readable provider-key name (#890 req-3). 1:1 with `provider_key_id`
    /// so it adds no new series; `"unknown"` when unresolved.
    pub provider_key_name: &'a str,
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub user_id: &'a str,
    /// Readable user display name (#890 req-3). 1:1 with `user_id`;
    /// `"unknown"` until cp-api syncs it onto the api-key config.
    pub user_name: &'a str,
    /// Whether the client requested a streaming (SSE) response (#890 req-1).
    /// Emitted on the request counter AND duration histogram so a
    /// TTFT-vs-E2E comparison can restrict the E2E latency to the same
    /// streaming-only sample TTFT is measured on.
    pub stream: bool,
    /// Whether serving this request involved a fallback to a different
    /// routing target (#890 req-2). Emitted on the request COUNTERS only
    /// (a success-rate dimension — kept off the bucketed histograms to
    /// avoid ×2 per latency bucket) so a success rate can exclude fallback
    /// requests from the denominator.
    pub is_fallback: bool,
    pub status: u16,
    pub outcome: RequestOutcome,
}

impl Default for RequestLabels<'_> {
    fn default() -> Self {
        Self {
            endpoint: "unknown",
            inbound_protocol: "openai",
            upstream_protocol: "unknown",
            provider: "unknown",
            model: "unknown",
            upstream_model: "unknown",
            provider_key_id: "unknown",
            provider_key_name: "unknown",
            api_key_id: "unknown",
            team_id: "unknown",
            user_id: "unknown",
            user_name: "unknown",
            stream: false,
            is_fallback: false,
            status: 0,
            outcome: RequestOutcome::UpstreamError,
        }
    }
}

impl RequestLabels<'_> {
    fn request_counter(&self, metric: &'static str) -> metrics::Counter {
        metrics::counter!(
            metric,
            "endpoint" => self.endpoint.to_string(),
            "inbound_protocol" => self.inbound_protocol.to_string(),
            "upstream_protocol" => self.upstream_protocol.to_string(),
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
            "provider_key_name" => self.provider_key_name.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
            "user_name" => self.user_name.to_string(),
            "stream" => bool_str(self.stream),
            "is_fallback" => bool_str(self.is_fallback),
            "status" => self.status.to_string(),
            "outcome" => self.outcome.as_str().to_string(),
        )
    }

    fn request_duration_histogram(&self, metric: &'static str) -> metrics::Histogram {
        metrics::histogram!(
            metric,
            "endpoint" => self.endpoint.to_string(),
            "inbound_protocol" => self.inbound_protocol.to_string(),
            "upstream_protocol" => self.upstream_protocol.to_string(),
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
            "provider_key_name" => self.provider_key_name.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
            "user_name" => self.user_name.to_string(),
            "stream" => bool_str(self.stream),
            "status" => self.status.to_string(),
            "outcome" => self.outcome.as_str().to_string(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UsageLabels<'a> {
    pub endpoint: &'a str,
    pub inbound_protocol: &'a str,
    /// The upstream API protocol the selected target actually speaks —
    /// the ProviderKey adapter's wire value, as resolved by the SAME
    /// two-tier rule dispatch uses (AISIX-Cloud#1403).
    ///
    /// A bounded set: `openai` / `anthropic` / `bedrock` / `vertex` /
    /// `azure-openai`, plus `unknown` for a request that never selected
    /// an upstream (a pre-dispatch rejection, or a non-inference route
    /// like `/mcp`). Distinct from `inbound_protocol`, which is the
    /// protocol the CALLER spoke: a request arriving on `/v1/messages`
    /// and served by an OpenAI-shape upstream carries
    /// `inbound_protocol="anthropic", upstream_protocol="openai"`.
    ///
    /// `provider` cannot stand in for it: vendor identity is an open
    /// string, and the same vendor can be reached through different
    /// adapters (a `byo` key with `adapter: anthropic`, say).
    pub upstream_protocol: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
    /// Readable provider-key name (#890 req-3). 1:1 with `provider_key_id`.
    pub provider_key_name: &'a str,
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub user_id: &'a str,
    /// Readable user display name (#890 req-3). 1:1 with `user_id`.
    pub user_name: &'a str,
}

impl Default for UsageLabels<'_> {
    fn default() -> Self {
        Self {
            endpoint: "unknown",
            inbound_protocol: "openai",
            upstream_protocol: "unknown",
            provider: "unknown",
            model: "unknown",
            upstream_model: "unknown",
            provider_key_id: "unknown",
            provider_key_name: "unknown",
            api_key_id: "unknown",
            team_id: "unknown",
            user_id: "unknown",
            user_name: "unknown",
        }
    }
}

impl UsageLabels<'_> {
    fn counter(&self, metric: &'static str) -> metrics::Counter {
        metrics::counter!(
            metric,
            "endpoint" => self.endpoint.to_string(),
            "inbound_protocol" => self.inbound_protocol.to_string(),
            "upstream_protocol" => self.upstream_protocol.to_string(),
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
            "provider_key_name" => self.provider_key_name.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
            "user_name" => self.user_name.to_string(),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LlmUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    /// Upstream prompt-cache hits already counted inside `input_tokens`
    /// — the OpenAI accounting shape (AISIX-Cloud#1404). See
    /// [`M_LLM_CACHED_INPUT_TOKENS_TOTAL`].
    pub cached_input_tokens: u32,
    /// Upstream prompt-cache hits reported ALONGSIDE `input_tokens` —
    /// the Anthropic accounting shape. See
    /// [`M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL`].
    pub cache_read_input_tokens: u32,
    /// Input tokens written into the upstream's prompt cache. See
    /// [`M_LLM_CACHE_CREATION_INPUT_TOKENS_TOTAL`].
    pub cache_creation_input_tokens: u32,
    pub spend_usd: f64,
}

impl LlmUsage {
    fn spend_micro_usd(self) -> Option<u64> {
        if !self.spend_usd.is_finite() || self.spend_usd <= 0.0 {
            return None;
        }
        let micro_usd = (self.spend_usd * 1_000_000.0).round();
        (micro_usd > 0.0).then_some(micro_usd as u64)
    }
}

/// The attribution dimensions `aisix_usage_events_emitted_total` and
/// `aisix_usage_event_drops_total` share (AISIX-Cloud#1317).
///
/// The two counters are read as one — `emitted == delivered + dropped` —
/// so they have to carry the SAME labels or the invariant only holds
/// after summing away every dimension, and "which model's usage records
/// were lost" stays unanswerable. Both take this one struct for that
/// reason.
///
/// Bounded by the caller, which is the proxy's emit chokepoint: `model`
/// is `usage_attr::metric_model_label` output (the configured set, never
/// the caller's raw `model` field — #451), and the ProviderKey pair is
/// read off one snapshot row so the name can never be paired with an id
/// it did not come from.
#[derive(Debug, Clone, Copy)]
pub struct UsageEventLabels<'a> {
    pub model: &'a str,
    pub provider_key_id: &'a str,
    pub provider_key_name: &'a str,
    /// Org member the authenticating key belongs to (AISIX-Cloud#1389) —
    /// the same `user_id` `aisix_proxy_requests_total` carries, so a
    /// member's request rate and their usage-event rate slice alike.
    /// `unknown` when the key is bound to no member, or when auth never
    /// resolved a key.
    ///
    /// This is attribution, so it sits on BOTH counters: "which member's
    /// usage records were lost" is exactly the question the shared label
    /// set exists to keep answerable. `UsageSink::try_emit` fills it from
    /// the event's own `user_id`, so the counter and the row cp-api
    /// persists can never name different people.
    pub user_id: &'a str,
    /// Readable display name of the member `user_id` names
    /// (AISIX-Cloud#1455). One series per (id, name) pair rather than per
    /// name: the two are read off one ApiKey row, so at any instant the
    /// name is determined by the id and the label costs nothing. The name
    /// is a SNAPSHOT of the row, not a live read: cp-api stamps it when it
    /// projects the ApiKey, and nothing reprojects a key just because its
    /// member was renamed — so a rename shows up here only once that key is
    /// next written for some other reason, and splits the counter into a
    /// before and an after series when it does.
    ///
    /// Filled from the event beside `user_id` in `UsageSink::try_emit`
    /// rather than by each handler's label builder, so the two cannot
    /// name different people. `unknown` whenever `user_id` is.
    pub user_name: &'a str,
    /// The upstream wire protocol of the key named by `provider_key_id`
    /// (AISIX-Cloud#1403) — the same value, resolved the same way, that
    /// the request and usage families carry, so an operator can align
    /// this counter with them under `sum by (upstream_protocol)`.
    ///
    /// Must be read off the SAME ProviderKey row as the id and name
    /// beside it (`usage_attr::ResolvedPk`), never looked up separately:
    /// three dimensions describing three different keys is exactly the
    /// drift `PkLabels` exists to prevent. `unknown` when nothing was
    /// resolved — the pre-dispatch failures, and the tunnels that have
    /// no LLM upstream at all (`/mcp`, `/a2a`).
    pub upstream_protocol: &'a str,
}

impl Default for UsageEventLabels<'_> {
    fn default() -> Self {
        Self {
            model: "unknown",
            provider_key_id: "unknown",
            provider_key_name: "unknown",
            user_id: "unknown",
            user_name: "unknown",
            upstream_protocol: "unknown",
        }
    }
}

/// Label set for [`M_PROXY_CLIENT_CANCELLED_TOTAL`]. Every field is a
/// value the proxy has already bounded — see that constant.
#[derive(Debug, Clone, Copy)]
pub struct CancelledLabels<'a> {
    pub endpoint: &'a str,
    /// What the caller addressed, collapsed to the configured model set;
    /// `unknown` when the request was cancelled before it resolved one.
    pub model: &'a str,
    /// The ProviderKey of the target the request was waiting on, and the
    /// readable name read off that same row. Both `unknown` when the
    /// request never selected a target.
    pub provider_key_id: &'a str,
    pub provider_key_name: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct DeploymentLabels<'a> {
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
}

impl Default for DeploymentLabels<'_> {
    fn default() -> Self {
        Self {
            provider: "unknown",
            model: "unknown",
            upstream_model: "unknown",
            provider_key_id: "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentState {
    Healthy,
    PartialFailure,
    Down,
}

impl DeploymentState {
    fn as_f64(self) -> f64 {
        match self {
            Self::Healthy => 0.0,
            Self::PartialFailure => 1.0,
            Self::Down => 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetLabels<'a> {
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub user_id: &'a str,
    /// Readable display name of the member `user_id` names
    /// (AISIX-Cloud#1455), read off the same ApiKey row so the name is
    /// determined by the id and the pair costs no series over `user_id`
    /// alone. `unknown` whenever `user_id` is.
    ///
    /// These are GAUGES, and this recorder registers no idle timeout, so
    /// a label set that stops being written stays at its last value: if a
    /// rename ever does reach the ApiKey row (see [`UsageEventLabels`] —
    /// it takes a later write of that key), the pre-rename sample stays
    /// behind, and a `sum` that does not group by the member counts both.
    /// That is true of every label on this family that can change under a
    /// fixed `api_key_id`: rebinding a key's team or owner strands its
    /// samples the same way. `clear_budget_gauges` does not help — it
    /// zeroes `details_present` for the label set it is handed, which is
    /// the new one.
    pub user_name: &'a str,
}

impl Default for BudgetLabels<'_> {
    fn default() -> Self {
        Self {
            api_key_id: "unknown",
            team_id: "unknown",
            user_id: "unknown",
            user_name: "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BudgetGauges {
    pub limit_usd: Option<f64>,
    pub spent_usd: Option<f64>,
    pub remaining_usd: Option<f64>,
    pub reset_seconds: Option<u64>,
}

/// Canonical outcome label for [`Metrics::record_request`]. Keeps the
/// `outcome` dimension bounded so Prometheus cardinality stays sane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestOutcome {
    Success,
    ClientError,
    UpstreamError,
    RateLimited,
}

impl RequestOutcome {
    pub fn from_status(status: u16) -> Self {
        match status {
            429 => Self::RateLimited,
            200..=399 => Self::Success,
            400..=499 => Self::ClientError,
            _ => Self::UpstreamError,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::ClientError => "client_error",
            Self::UpstreamError => "upstream_error",
            Self::RateLimited => "rate_limited",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// This file's own source with the test module cut off. Every guard
    /// below reads the production half only — a mention from inside the
    /// tests is not a use.
    fn production_source() -> &'static str {
        include_str!("metrics.rs")
            .split("\n#[cfg(test)]\nmod tests {")
            .next()
            .expect("production half")
    }

    /// Every metric-name constant this module declares: `M_FOO` ->
    /// "aisix_foo", so failures name the metric an operator would grep for
    /// rather than a Rust identifier. Whitespace-tolerant: the longer
    /// constants wrap their value onto the next line.
    ///
    /// Read out of the source so the set is never a list someone has to
    /// remember to extend.
    fn declared_metric_names() -> BTreeMap<&'static str, &'static str> {
        let production = production_source();
        let names: BTreeMap<&str, &str> = production
            .match_indices("pub const M_")
            .filter_map(|(at, marker)| {
                let decl = &production[at + marker.len()..];
                let (tail, rest) = decl.split_once(": &str =")?;
                let value = rest.trim_start().strip_prefix('"')?;
                let start = at + marker.len() - "M_".len();
                let ident = &production[start..start + "M_".len() + tail.len()];
                Some((ident, value.split('"').next()?))
            })
            .collect();
        assert!(
            names.len() > 40,
            "the constant scan found only {} metric names — the `pub const` \
             spelling it keys off has changed and the guards built on it are \
             no longer looking at anything",
            names.len()
        );
        names
    }

    /// One `metrics::{counter,histogram,gauge}!` invocation in this file's
    /// production half: the family it emits (resolved to the metric NAME
    /// where the first argument is one of this module's `M_*` constants,
    /// otherwise the argument text itself) and the label keys it writes.
    ///
    /// Read out of the source deliberately. Only `aisix-obs` depends on
    /// the `metrics` crate and every invocation lives in this file, so
    /// the scan sees the WHOLE emit surface — including families no test
    /// exercises, which is precisely where the omission this guards
    /// against hides.
    fn emit_sites() -> Vec<(String, BTreeSet<String>)> {
        let production = production_source();
        let names = declared_metric_names();

        let mut sites = Vec::new();
        for macro_name in ["counter", "histogram", "gauge"] {
            let opener = format!("metrics::{macro_name}!(");
            let mut cursor = 0usize;
            while let Some(offset) = production[cursor..].find(&opener) {
                let open = cursor + offset + opener.len() - 1;
                let mut depth = 0usize;
                let mut end = open;
                for (i, c) in production[open..].char_indices() {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                end = open + i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                assert!(end > open, "unbalanced `{opener}` at byte {open}");
                let body = &production[open + 1..end];
                let line = production[..open].lines().count();
                let family = body
                    .split(',')
                    .next()
                    .expect("a family argument")
                    .trim()
                    .to_string();
                let display = names
                    .get(family.as_str())
                    .map(|name| (*name).to_string())
                    // A shared helper takes the family as a parameter, so
                    // the site has no name of its own — identify it by the
                    // helper it lives in, which is what a reader has to go
                    // edit.
                    .unwrap_or_else(|| {
                        let enclosing = production[..open]
                            .rsplit_once("    fn ")
                            .and_then(|(_, tail)| tail.split('(').next())
                            .unwrap_or(family.as_str());
                        format!("{enclosing}() (metrics.rs:{line})")
                    });
                let labels = body
                    .split('"')
                    .skip(1)
                    .step_by(2)
                    .zip(body.split('"').skip(2).step_by(2))
                    .filter(|(_, after)| after.trim_start().starts_with("=>"))
                    .map(|(key, _)| key.to_string())
                    .collect();
                sites.push((display, labels));
                cursor = end;
            }
        }
        assert!(
            sites.len() > 20,
            "the emit-site scan found only {} sites — the macro spelling it \
             keys off has changed and this guard is no longer looking at \
             anything",
            sites.len()
        );
        sites
    }

    /// Every `M_*` constant this module declares must be referred to by
    /// the production code below it — a name nothing else mentions cannot
    /// reach an emit site, whatever the plumbing in between.
    ///
    /// `aisix_llm_api_latency_seconds` sat here declared and emitted by
    /// nothing at all, found by accident rather than by a check. Both
    /// sides of this assertion are read out of the source, so a constant
    /// added tomorrow is covered without anyone remembering to extend a
    /// list — a list-shaped test would have been typed from the families
    /// that DO emit and would have passed.
    ///
    /// Deliberately checks the mention and not the emit itself. A family
    /// argument reaches `metrics::{counter,histogram,gauge}!` through
    /// five different helper shapes and, in `record_routing_fallback`,
    /// through a local binding chosen by an `if` — resolving that
    /// statically needs a dataflow model of the module, which would be
    /// far more likely to break on an innocent refactor than to catch a
    /// second dead constant. The weaker signal is the one that is both
    /// robust and sufficient for the defect it guards.
    #[test]
    fn every_metric_name_constant_is_used_by_production_code() {
        // Doc comments cross-reference constants by the dozen, and a
        // rustdoc link is not a use.
        let code: String = production_source()
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");

        let is_ident_char = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let mentions = |ident: &str| -> usize {
            let mut count = 0usize;
            let mut cursor = 0usize;
            while let Some(offset) = code[cursor..].find(ident) {
                let at = cursor + offset;
                let end = at + ident.len();
                let bounded = !code[..at].chars().next_back().is_some_and(&is_ident_char)
                    && !code[end..].chars().next().is_some_and(&is_ident_char);
                if bounded {
                    count += 1;
                }
                cursor = end;
            }
            count
        };

        // The declaration itself is one mention; anything reachable has a
        // second one somewhere.
        let unused: Vec<&str> = declared_metric_names()
            .into_iter()
            .filter(|(ident, _)| mentions(ident) < 2)
            .map(|(_, name)| name)
            .collect();
        assert!(
            unused.is_empty(),
            "declared but never used by any production code — nothing can \
             emit these, so the series do not exist to scrape: {unused:?}"
        );
    }

    /// The families that carry `inbound_protocol` and deliberately do NOT
    /// carry `upstream_protocol`. Both are counted at a point where no
    /// upstream has been selected — the reason is spelled out at each
    /// metric's own definition, which is what a reader hits first.
    ///
    /// Membership here is a decision, not a backlog: adding a family is
    /// how you state "this one is excluded on purpose", and the assertions
    /// below reject an entry that is stale (the family gained the label
    /// anyway) or dangling (the family no longer carries
    /// `inbound_protocol`).
    fn upstream_protocol_exempt_families() -> BTreeSet<&'static str> {
        BTreeSet::from([
            // Incremented in the proxy middleware, before routing picks a
            // target; the matching decrement must address the same series.
            M_PROXY_IN_FLIGHT,
            // The body cap rejects before any handler runs.
            M_PROXY_BODY_LIMIT_REJECTIONS_TOTAL,
        ])
    }

    /// Every rendered sample that carries `inbound_protocol` must carry
    /// `upstream_protocol` too, unless its family is a stated exclusion.
    ///
    /// The exposition half of
    /// `upstream_protocol_covers_every_inbound_protocol_family`: that one
    /// reads the emit sites and cannot see whether the label survives to
    /// the wire; this one reads what a Prometheus scrape would actually
    /// get. Both derive the family set instead of listing it — the caller
    /// hands over a rendering, not a set of names to check.
    fn assert_upstream_protocol_rides_along(rendered: &str) {
        let exempt = upstream_protocol_exempt_families();
        let mut checked = 0usize;
        for line in rendered.lines() {
            if line.starts_with('#') || !line.contains("inbound_protocol=") {
                continue;
            }
            let family = line.split('{').next().expect("a sample name");
            // Histogram suffixes are the same family.
            let base = family
                .strip_suffix("_bucket")
                .or_else(|| family.strip_suffix("_sum"))
                .or_else(|| family.strip_suffix("_count"))
                .unwrap_or(family);
            if exempt.contains(base) {
                continue;
            }
            // Every sample of a family must carry the label, or
            // `sum by (upstream_protocol)` silently drops the ones that
            // do not — an operator's dashboard then joins across a
            // subset and reads as merely incomplete data.
            assert!(
                line.contains("upstream_protocol="),
                "{base} sample carries inbound_protocol without \
                 upstream_protocol: {line}"
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "no sample carrying inbound_protocol was rendered — this \
             assertion checked nothing"
        );
    }

    /// AISIX-Cloud#1403: `upstream_protocol` must sit on every family that
    /// carries `inbound_protocol`, minus the ones deliberately excluded.
    ///
    /// Derived from the emit sites rather than restated as a list of
    /// family names. A hand-written list only ever freezes the set that
    /// existed when someone typed it: the label originally reached
    /// exactly the families that share a label STRUCT — `RequestLabels`,
    /// `UsageLabels` — and the three that build their label list inline
    /// here were never considered. Two of those are the deliberate
    /// exclusions below; the third,
    /// `aisix_usage_events_emitted_total`, had the value in hand beside
    /// `provider_key_id` and simply did not emit it, so
    /// `sum by (upstream_protocol)` dropped it out of every join with the
    /// rest. A list-shaped test would have been written to match the
    /// families that did get it, and passed.
    #[test]
    fn upstream_protocol_covers_every_inbound_protocol_family() {
        let sites = emit_sites();
        let with = |label: &str| -> BTreeSet<String> {
            sites
                .iter()
                .filter(|(_, labels)| labels.contains(label))
                .map(|(family, _)| family.clone())
                .collect()
        };
        let inbound = with("inbound_protocol");
        let upstream = with("upstream_protocol");
        let exempt: BTreeSet<String> = upstream_protocol_exempt_families()
            .into_iter()
            .map(str::to_string)
            .collect();

        let missing: Vec<&String> = inbound
            .iter()
            .filter(|f| !upstream.contains(*f) && !exempt.contains(*f))
            .collect();
        assert!(
            missing.is_empty(),
            "these families carry `inbound_protocol` without \
             `upstream_protocol`: {missing:?}. Either resolve the value off \
             the request's `usage_attr::ResolvedPk` the way the request \
             families do — never a second ProviderKey lookup — or add the \
             family to `upstream_protocol_exempt_families` and say at its \
             definition why the value cannot exist at emit time.",
        );

        let stale: Vec<&String> = exempt.iter().filter(|f| upstream.contains(*f)).collect();
        assert!(
            stale.is_empty(),
            "these families are listed as deliberately without \
             `upstream_protocol` but now emit it: {stale:?}. Drop them from \
             `upstream_protocol_exempt_families` and from the exclusion note \
             at their definition.",
        );

        let dangling: Vec<&String> = exempt.iter().filter(|f| !inbound.contains(*f)).collect();
        assert!(
            dangling.is_empty(),
            "these families are listed as deliberately without \
             `upstream_protocol` but no longer carry `inbound_protocol` \
             either: {dangling:?}. The exemption is about a comparison that \
             no longer exists — remove it.",
        );
    }

    #[test]
    fn outcome_from_status_maps_correctly() {
        assert_eq!(RequestOutcome::from_status(200), RequestOutcome::Success);
        assert_eq!(RequestOutcome::from_status(301), RequestOutcome::Success);
        assert_eq!(
            RequestOutcome::from_status(404),
            RequestOutcome::ClientError
        );
        assert_eq!(
            RequestOutcome::from_status(429),
            RequestOutcome::RateLimited
        );
        assert_eq!(
            RequestOutcome::from_status(502),
            RequestOutcome::UpstreamError
        );
    }

    #[test]
    fn recording_a_request_renders_in_exposition_format() {
        let m = Metrics::new(false);
        m.record_request(
            "openai",
            "my-gpt4",
            200,
            RequestOutcome::Success,
            Duration::from_millis(120),
        );
        let rendered = m.render();
        assert!(rendered.contains(M_REQUESTS_TOTAL));
        assert!(rendered.contains("provider=\"openai\""));
        assert!(rendered.contains("outcome=\"success\""));
        assert!(rendered.contains(M_REQUEST_DURATION));
    }

    /// AISIX-Cloud#1076: the per-execution guardrail histogram renders with
    /// real `_bucket{le=…}` series (quantile-aggregatable, not a summary)
    /// and the full bounded label set; `error_type` defaults to `none`.
    #[test]
    fn guardrail_execution_renders_bucketed_histogram_with_labels() {
        let m = Metrics::new_with_env_id("env-7");
        m.record_guardrail_execution(&aisix_core::GuardrailExecution {
            guardrail_name: "block-secrets",
            kind: "keyword",
            phase: "input",
            result: "blocked",
            error_type: None,
            elapsed: Duration::from_micros(300),
        });
        m.record_guardrail_execution(&aisix_core::GuardrailExecution {
            guardrail_name: "lakera-prod",
            kind: "lakera",
            phase: "output",
            result: "bypassed",
            error_type: Some("lakera_timeout"),
            elapsed: Duration::from_secs(5),
        });
        let out = m.render();
        assert!(
            out.contains("aisix_guardrail_latency_seconds_bucket"),
            "{out}"
        );
        // 0.3 ms lands in the first (1 ms) bucket — the sub-SLO edges exist.
        assert!(out.contains("le=\"0.001\""), "{out}");
        assert!(out.contains("env_id=\"env-7\""));
        assert!(out.contains("guardrail=\"block-secrets\""));
        assert!(out.contains("kind=\"keyword\""));
        assert!(out.contains("phase=\"input\""));
        assert!(out.contains("result=\"blocked\""));
        assert!(out.contains("error_type=\"none\""));
        assert!(out.contains("guardrail=\"lakera-prod\""));
        assert!(out.contains("result=\"bypassed\""));
        assert!(out.contains("error_type=\"lakera_timeout\""));
        // No per-key/per-user dimension may ride a bucketed histogram.
        assert!(!out.contains("api_key_id="));
    }

    #[test]
    fn ratelimit_rejection_counter_increments() {
        let m = Metrics::new(false);
        m.record_ratelimit_rejection("requests", "api_key", None);
        m.record_ratelimit_rejection("requests", "api_key", None);
        m.record_ratelimit_rejection("requests", "policy", Some("pol-1"));
        let rendered = m.render();
        assert!(rendered.contains(M_RATELIMIT_REJECTIONS));
        assert!(rendered.contains("scope=\"requests\""));
        assert!(rendered.contains("layer=\"api_key\""));
        // Policy-layer rejections carry the offending policy id; other
        // layers leave the label empty.
        assert!(rendered.contains("policy_id=\"pol-1\""));
        assert!(rendered.contains("policy_id=\"\""));
    }

    #[test]
    fn a2a_family_slices_by_agent_and_operation() {
        let m = Metrics::new(false);
        m.record_a2a_call(
            A2aLabels {
                agent: "invoices",
                operation: "message/stream",
                status: 200,
            },
            A2aCallOutcome {
                ttfb: Some(Duration::from_millis(80)),
                stream_events: 17,
                task_state: "completed",
            },
        );
        let rendered = m.render();
        // The dimension the proxy families cannot answer: which agent, which
        // operation.
        assert!(rendered.contains(&format!(
            "{M_A2A_REQUESTS_TOTAL}{{agent=\"invoices\",operation=\"message/stream\",status=\"2xx\"}} 1"
        )));
        assert!(rendered.contains(&format!(
            "{M_A2A_STREAM_EVENTS_TOTAL}{{agent=\"invoices\",operation=\"message/stream\"}} 17"
        )));
        assert!(rendered.contains(&format!(
            "{M_A2A_TASK_STATE_TOTAL}{{agent=\"invoices\",state=\"completed\"}} 1"
        )));
        // A real histogram, not a summary — the buckets are registered.
        assert!(rendered.contains(&format!("{M_A2A_TTFB_SECONDS}_bucket")));
        // Task and context ids are the whole reason this family is safe to
        // label; neither may ever appear.
        assert!(!rendered.contains("task_id="));
        assert!(!rendered.contains("context_id="));
    }

    #[test]
    fn a2a_unary_call_observes_only_the_series_it_has_data_for() {
        let m = Metrics::new(false);
        m.record_a2a_call(
            A2aLabels {
                agent: "invoices",
                operation: "tasks/get",
                status: 502,
            },
            // No stream, and the upstream never answered, so no task state.
            A2aCallOutcome::default(),
        );
        let rendered = m.render();
        assert!(rendered.contains(&format!(
            "{M_A2A_REQUESTS_TOTAL}{{agent=\"invoices\",operation=\"tasks/get\",status=\"5xx\"}} 1"
        )));
        // An absent figure must not be recorded as zero: a call with no stream
        // is not a call whose stream carried nothing, and a call with no
        // answer has no task state to bucket.
        assert!(!rendered.contains(M_A2A_STREAM_EVENTS_TOTAL));
        assert!(!rendered.contains(M_A2A_TASK_STATE_TOTAL));
        assert!(!rendered.contains(M_A2A_TTFB_SECONDS));
    }

    #[test]
    fn guardrail_outcome_counters_increment() {
        let m = Metrics::new(false);
        for (result, error_type) in [
            ("blocked", None),
            ("bypassed", Some("bedrock_5xx")),
            ("allowed", None),
        ] {
            m.record_guardrail_execution(&aisix_core::GuardrailExecution {
                guardrail_name: "policy",
                kind: "bedrock",
                phase: "input",
                result,
                error_type,
                elapsed: Duration::from_millis(1),
            });
        }
        let rendered = m.render();
        // Exactly one block (the clean call must not increment it).
        assert!(
            rendered.contains(&format!("{M_GUARDRAIL_BLOCKS_TOTAL} 1")),
            "want one block, got:\n{rendered}"
        );
        // Exactly one bypass, sliced by the bounded reason — pinning the count
        // proves the blocked + clean calls didn't touch the bypass counter.
        assert!(
            rendered.contains(&format!(
                "{M_GUARDRAIL_BYPASSES_TOTAL}{{reason=\"bedrock_5xx\"}} 1"
            )),
            "want exactly one bedrock_5xx bypass, got:\n{rendered}"
        );
    }

    #[test]
    fn zero_tokens_do_not_emit_a_sample() {
        let m = Metrics::new(false);
        m.record_tokens("openai", "my-gpt4", 0);
        let rendered = m.render();
        // Counter family is never touched so it doesn't appear.
        assert!(!rendered.contains(M_TOKENS_CONSUMED));
    }

    #[test]
    fn token_counts_accumulate_across_calls() {
        let m = Metrics::new(false);
        m.record_tokens("openai", "my-gpt4", 10);
        m.record_tokens("openai", "my-gpt4", 32);
        let rendered = m.render();
        // The rendered counter should be 42. Keep the assertion robust to
        // whitespace variations by searching for the literal value.
        assert!(
            rendered.contains("42"),
            "expected total 42 in exposition, got:\n{rendered}"
        );
    }

    #[test]
    fn aisix_native_request_usage_and_latency_metrics_render() {
        let m = Metrics::new(false);
        let labels = RequestLabels {
            endpoint: "/v1/chat/completions",
            inbound_protocol: "openai",
            upstream_protocol: "anthropic",
            provider: "openai",
            model: "gpt",
            upstream_model: "gpt-4o",
            provider_key_id: "pk-1",
            provider_key_name: "my-openai-key",
            api_key_id: "ak-1",
            team_id: "team-1",
            user_id: "user-1",
            user_name: "alice",
            stream: true,
            is_fallback: true,
            status: 200,
            outcome: RequestOutcome::Success,
        };
        let usage_labels = UsageLabels {
            endpoint: "/v1/chat/completions",
            inbound_protocol: "openai",
            upstream_protocol: "anthropic",
            provider: "openai",
            model: "gpt",
            upstream_model: "gpt-4o",
            provider_key_id: "pk-1",
            provider_key_name: "my-openai-key",
            api_key_id: "ak-1",
            team_id: "team-1",
            user_id: "user-1",
            user_name: "alice",
        };
        assert_eq!(Metrics::worker_request_series_len(), 0);
        m.record_proxy_and_llm_request(labels, Duration::from_millis(25));
        m.record_proxy_and_llm_request(labels, Duration::from_millis(20));
        assert_eq!(
            Metrics::worker_request_series_len(),
            1,
            "repeated labels must reuse one registered handle set"
        );
        m.record_llm_usage(
            usage_labels,
            LlmUsage {
                input_tokens: 5,
                output_tokens: 7,
                total_tokens: 12,
                cached_input_tokens: 3,
                cache_read_input_tokens: 800,
                cache_creation_input_tokens: 200,
                spend_usd: 0.001,
            },
        );
        m.record_time_to_first_token(usage_labels, Duration::from_millis(42));

        let rendered = m.render();
        assert!(rendered.contains(M_PROXY_REQUESTS_TOTAL));
        assert!(rendered.contains(M_LLM_REQUESTS_TOTAL));
        assert!(rendered
            .lines()
            .filter(|line| line.starts_with(M_PROXY_REQUESTS_TOTAL))
            .all(|line| line.ends_with(" 2")));
        assert!(rendered
            .lines()
            .filter(|line| line.starts_with(M_LLM_REQUESTS_TOTAL))
            .all(|line| line.ends_with(" 2")));
        assert!(rendered.contains(M_LLM_INPUT_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_OUTPUT_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_TOTAL_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_SPEND_MICRO_USD_TOTAL));
        assert!(rendered.contains(M_LLM_REQUEST_DURATION));
        assert!(rendered.contains(M_LLM_TTFT));
        assert!(rendered.contains("endpoint=\"/v1/chat/completions\""));
        assert!(rendered.contains("team_id=\"team-1\""));
        assert!(rendered.contains("user_id=\"user-1\""));
        // #890 req-3: readable names ride alongside the ids (1:1).
        assert!(rendered.contains("provider_key_name=\"my-openai-key\""));
        assert!(rendered.contains("user_name=\"alice\""));
        // #890 req-1/req-2: stream on counter + duration; is_fallback on
        // the counter only (verified absent from the duration below).
        assert!(rendered.contains("stream=\"true\""));
        assert!(rendered.contains("is_fallback=\"true\""));
        // is_fallback must NOT appear on the duration histogram series.
        for line in rendered.lines() {
            if line.starts_with(M_LLM_REQUEST_DURATION)
                || line.starts_with(M_PROXY_REQUEST_DURATION)
            {
                assert!(
                    !line.contains("is_fallback="),
                    "is_fallback must stay off the duration histogram: {line}"
                );
            }
        }
        // AISIX-Cloud#1403: the cross-protocol pair. The labels above spell
        // an OpenAI caller served by an Anthropic upstream, and both halves
        // have to survive onto the same sample or the conversion is
        // invisible.
        assert!(rendered.contains("inbound_protocol=\"openai\""));
        assert!(rendered.contains("upstream_protocol=\"anthropic\""));
        assert_upstream_protocol_rides_along(&rendered);
        // AISIX-Cloud#1404: each cache counter carries its own value and
        // none of them was folded into input or total.
        assert!(rendered.contains(M_LLM_CACHED_INPUT_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_CACHE_CREATION_INPUT_TOKENS_TOTAL));
        let value_of = |family: &str| -> f64 {
            rendered
                .lines()
                .find(|line| line.starts_with(family) && line.contains('{'))
                .and_then(|line| line.rsplit(' ').next())
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{family} rendered no sample"))
        };
        assert_eq!(value_of(M_LLM_CACHED_INPUT_TOKENS_TOTAL), 3.0);
        assert_eq!(value_of(M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL), 800.0);
        assert_eq!(value_of(M_LLM_CACHE_CREATION_INPUT_TOKENS_TOTAL), 200.0);
        assert_eq!(value_of(M_LLM_INPUT_TOKENS_TOTAL), 5.0);
        assert_eq!(value_of(M_LLM_TOTAL_TOKENS_TOTAL), 12.0);
    }

    /// A request that reported only cache tokens still records them. The
    /// all-zero short-circuit in `record_llm_usage` guards series
    /// sparsity, and before #1404 it only knew about input / output /
    /// total / spend — so a usage report whose only non-zero field is a
    /// cache counter must not be swallowed by it.
    #[test]
    fn cache_only_usage_is_not_swallowed_by_the_zero_short_circuit() {
        let m = Metrics::new(false);
        m.record_llm_usage(
            UsageLabels::default(),
            LlmUsage {
                cache_read_input_tokens: 64,
                ..Default::default()
            },
        );
        let rendered = m.render();
        assert!(rendered.contains(M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL));
        // …and the counters that reported nothing stay absent, so the
        // scrape does not gain a zero-valued series per request.
        assert!(!rendered.contains(M_LLM_CACHED_INPUT_TOKENS_TOTAL));
        assert!(!rendered.contains(M_LLM_INPUT_TOKENS_TOTAL));
    }

    #[test]
    fn request_series_handle_cache_is_bounded() {
        let metrics = Metrics::new(false);
        // Two passes over capacity+1 distinct label sets: pass one fills
        // the worker cache and forces at least one eviction; pass two
        // re-emits everything, so every evicted entry re-registers.
        for _ in 0..2 {
            for index in 0..=WORKER_CACHE_CAPACITY {
                let model = format!("model-{index}");
                metrics.record_proxy_and_llm_request(
                    RequestLabels {
                        model: &model,
                        ..RequestLabels::default()
                    },
                    Duration::from_millis(1),
                );
            }
        }

        assert_eq!(
            Metrics::worker_request_series_len(),
            WORKER_CACHE_CAPACITY,
            "request series handle cache must stay at its fixed capacity"
        );

        // Every label set — evicted or cached — must have continued its
        // own Prometheus series: eviction drops our handle, never the
        // series or its value.
        let rendered = metrics.render();
        for index in 0..=WORKER_CACHE_CAPACITY {
            let model_label = format!("model=\"model-{index}\"");
            assert!(
                rendered.lines().any(|line| {
                    line.starts_with(M_PROXY_REQUESTS_TOTAL)
                        && line.contains(&model_label)
                        && line.ends_with(" 2")
                }),
                "series for model-{index} must show both emits (eviction must not reset it)"
            );
        }
    }

    #[test]
    fn paired_request_metrics_increment_the_failure_counter_once() {
        let metrics = Metrics::new(false);
        metrics.record_proxy_and_llm_request(
            RequestLabels {
                status: 502,
                outcome: RequestOutcome::UpstreamError,
                ..RequestLabels::default()
            },
            Duration::from_millis(10),
        );

        let rendered = metrics.render();
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with(M_PROXY_FAILED_REQUESTS_TOTAL)
                    && line.ends_with(" 1")),
            "failed request counter was not incremented:\n{rendered}"
        );
    }

    #[test]
    fn separator_bearing_label_values_never_alias_cached_series() {
        let metrics = Metrics::new(false);
        // These two label sets join to the SAME byte sequence under the
        // cache-key separator; the dirty-key fallback must keep them
        // distinct series (uncached, correct) rather than folding them
        // into one cache entry.
        let labels_a = RequestLabels {
            model: "x\u{1f}y",
            upstream_model: "z",
            ..RequestLabels::default()
        };
        let labels_b = RequestLabels {
            model: "x",
            upstream_model: "y\u{1f}z",
            ..RequestLabels::default()
        };
        metrics.record_proxy_and_llm_request(labels_a, Duration::from_millis(1));
        metrics.record_proxy_and_llm_request(labels_b, Duration::from_millis(1));
        metrics.record_proxy_and_llm_request(labels_b, Duration::from_millis(1));

        assert_eq!(
            Metrics::worker_request_series_len(),
            0,
            "a separator-bearing label value must never be cached"
        );
        let rendered = metrics.render();
        let series: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with(M_PROXY_REQUESTS_TOTAL))
            .collect();
        assert_eq!(
            series.len(),
            2,
            "aliasing would fold the two label sets into one series: {series:?}"
        );
        assert!(series.iter().any(|line| line.ends_with(" 1")));
        assert!(series.iter().any(|line| line.ends_with(" 2")));
    }

    #[test]
    fn two_instances_on_one_thread_never_share_cached_handles() {
        let a = Metrics::new(false);
        let b = Metrics::new(false);
        a.record_tokens("openai", "shared-model", 1);
        b.record_tokens("openai", "shared-model", 2);

        // If the worker cache ignored the instance id, `b`'s emit would
        // land on `a`'s handle: `a` would render 3 and `b` nothing.
        assert!(a
            .render()
            .lines()
            .any(|line| line.starts_with(M_TOKENS_CONSUMED) && line.ends_with(" 1")));
        assert!(b
            .render()
            .lines()
            .any(|line| line.starts_with(M_TOKENS_CONSUMED) && line.ends_with(" 2")));
    }

    /// Pins the deployment counter family against a cache-key/label
    /// drift: the key builder and the register closure in
    /// `cached_deployment_counter` list the labels independently, and
    /// dropping one from the KEY would silently alias two deployments
    /// onto one series while every single-label-set test stayed green.
    #[test]
    fn deployment_counters_stay_distinct_per_label_set() {
        let metrics = Metrics::new(false);
        let dep_a = DeploymentLabels {
            provider: "openai",
            model: "gpt",
            upstream_model: "gpt-4o",
            provider_key_id: "pk-a",
        };
        // Differs ONLY in provider_key_id — the label a key-builder
        // regression is most likely to drop.
        let dep_b = DeploymentLabels {
            provider_key_id: "pk-b",
            ..dep_a
        };
        metrics.record_deployment_request(dep_a, RequestOutcome::Success);
        metrics.record_deployment_request(dep_a, RequestOutcome::Success);
        metrics.record_deployment_request(dep_b, RequestOutcome::UpstreamError);

        let rendered = metrics.render();
        let series = |metric: &str, key: &str| {
            rendered
                .lines()
                .find(|l| {
                    l.starts_with(metric) && l.contains(&format!("provider_key_id=\"{key}\""))
                })
                .map(str::to_owned)
        };
        let requests_a = series(M_DEPLOYMENT_REQUESTS_TOTAL, "pk-a")
            .expect("deployment A must have its own requests series");
        assert!(requests_a.ends_with(" 2"), "got: {requests_a}");
        for label in [
            "provider=\"openai\"",
            "model=\"gpt\"",
            "upstream_model=\"gpt-4o\"",
        ] {
            assert!(requests_a.contains(label), "missing {label}: {requests_a}");
        }
        let requests_b = series(M_DEPLOYMENT_REQUESTS_TOTAL, "pk-b")
            .expect("deployment B must have its own requests series");
        assert!(requests_b.ends_with(" 1"), "got: {requests_b}");
        assert!(series(M_DEPLOYMENT_SUCCESS_TOTAL, "pk-a").is_some_and(|l| l.ends_with(" 2")));
        assert!(series(M_DEPLOYMENT_FAILURE_TOTAL, "pk-b").is_some_and(|l| l.ends_with(" 1")));
        // The outcome split must not cross-pollinate.
        assert!(series(M_DEPLOYMENT_SUCCESS_TOTAL, "pk-b").is_none());
        assert!(series(M_DEPLOYMENT_FAILURE_TOTAL, "pk-a").is_none());
    }

    /// Same cache-key/label drift guard for the fallback family, whose
    /// second label (`fallback_model`) is the one a group with several
    /// targets differs on: dropping it from the KEY would fold "fell back
    /// to B" and "fell back to C" into one series and make the counter
    /// useless for the question it exists to answer.
    #[test]
    fn fallback_counters_stay_distinct_per_target() {
        let metrics = Metrics::new(false);
        metrics.record_routing_fallback(true, "group", "target-b");
        metrics.record_routing_fallback(true, "group", "target-b");
        metrics.record_routing_fallback(true, "group", "target-c");
        metrics.record_routing_fallback(false, "group", "target-c");

        let rendered = metrics.render();
        let series = |metric: &str, target: &str| {
            rendered
                .lines()
                .find(|l| {
                    l.starts_with(metric) && l.contains(&format!("fallback_model=\"{target}\""))
                })
                .map(str::to_owned)
        };
        let to_b = series(M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL, "target-b")
            .expect("fallbacks to target-b must have their own series");
        assert!(to_b.ends_with(" 2"), "got: {to_b}");
        assert!(to_b.contains("model=\"group\""), "got: {to_b}");
        assert!(series(M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL, "target-c")
            .is_some_and(|l| l.ends_with(" 1")));
        assert!(
            series(M_ROUTING_FAILED_FALLBACKS_TOTAL, "target-c").is_some_and(|l| l.ends_with(" 1"))
        );
        // The success/failure split must not cross-pollinate.
        assert!(series(M_ROUTING_FAILED_FALLBACKS_TOTAL, "target-b").is_none());
    }

    /// Pins the budget gauge family's values, field gating and clear
    /// semantics. (Key-drift protection is NOT this test's job — its two
    /// label sets differ in every dimension, so dropping one key label
    /// still yields distinct keys; `budget_gauge_key_covers_every_label`
    /// below is the drift guard.)
    #[test]
    fn budget_gauges_stay_distinct_per_label_set_and_clear() {
        let metrics = Metrics::new(false);
        let key_a = BudgetLabels {
            api_key_id: "ak-a",
            team_id: "team-a",
            user_id: "user-a",
            user_name: "alice",
        };
        let key_b = BudgetLabels {
            api_key_id: "ak-b",
            team_id: "team-b",
            user_id: "user-b",
            user_name: "bob",
        };
        metrics.set_budget_gauges(
            key_a,
            BudgetGauges {
                limit_usd: Some(100.0),
                spent_usd: Some(40.0),
                remaining_usd: Some(60.0),
                reset_seconds: Some(3600),
            },
        );
        metrics.set_budget_gauges(
            key_b,
            BudgetGauges {
                spent_usd: Some(7.0),
                ..BudgetGauges::default()
            },
        );

        let rendered = metrics.render();
        let series = |metric: &str, team: &str| {
            rendered
                .lines()
                .find(|l| l.starts_with(metric) && l.contains(&format!("team_id=\"{team}\"")))
                .map(str::to_owned)
        };
        for (metric, value) in [
            (M_BUDGET_DETAILS_PRESENT, "1"),
            (M_BUDGET_LIMIT_USD, "100"),
            (M_BUDGET_SPENT_USD, "40"),
            (M_BUDGET_REMAINING_USD, "60"),
            (M_BUDGET_RESET_SECONDS, "3600"),
        ] {
            let line = series(metric, "team-a")
                .unwrap_or_else(|| panic!("{metric} missing for team-a:\n{rendered}"));
            assert!(line.ends_with(&format!(" {value}")), "got: {line}");
            assert!(
                line.contains("api_key_id=\"ak-a\"")
                    && line.contains("user_id=\"user-a\"")
                    // AISIX-Cloud#1455: the member's readable name rides
                    // beside their id here for the same reason it does on
                    // the request families — a budget alert that can only
                    // name a UUID is one an operator has to go look up.
                    && line.contains("user_name=\"alice\"")
            );
        }
        // Key B must have its OWN spent series, not overwrite A's.
        assert!(series(M_BUDGET_SPENT_USD, "team-b").is_some_and(|l| l.ends_with(" 7")));
        // Unset dimensions register nothing for B.
        assert!(series(M_BUDGET_LIMIT_USD, "team-b").is_none());

        metrics.clear_budget_gauges(key_a);
        let rendered = metrics.render();
        let details = |team: &str| {
            rendered
                .lines()
                .find(|l| {
                    l.starts_with(M_BUDGET_DETAILS_PRESENT)
                        && l.contains(&format!("team_id=\"{team}\""))
                })
                .map(str::to_owned)
        };
        assert!(details("team-a").is_some_and(|l| l.ends_with(" 0")));
        assert!(details("team-b").is_some_and(|l| l.ends_with(" 1")));
    }

    /// Pins the legacy spec-7 pair's full label sets, accumulation, and
    /// the outcome-stays-off-durations invariant. (Its two label sets
    /// differ in several dimensions at once, so this test cannot catch a
    /// single dropped key label; `legacy_request_key_covers_every_label`
    /// below is the drift guard.)
    #[test]
    fn legacy_request_series_stay_fully_labelled_and_distinct() {
        let metrics = Metrics::new(false);
        metrics.record_request(
            "openai",
            "my-gpt4",
            200,
            RequestOutcome::Success,
            Duration::from_millis(120),
        );
        metrics.record_request(
            "openai",
            "my-gpt4",
            200,
            RequestOutcome::Success,
            Duration::from_millis(80),
        );
        metrics.record_request(
            "openai",
            "other-model",
            404,
            RequestOutcome::ClientError,
            Duration::from_millis(5),
        );

        let rendered = metrics.render();
        let counter_a = rendered
            .lines()
            .find(|l| l.starts_with(M_REQUESTS_TOTAL) && l.contains("status=\"200\""))
            .expect("200 series must render");
        assert!(
            counter_a.ends_with(" 2"),
            "cached handle must accumulate: {counter_a}"
        );
        for label in [
            "provider=\"openai\"",
            "model=\"my-gpt4\"",
            "outcome=\"success\"",
        ] {
            assert!(counter_a.contains(label), "missing {label}: {counter_a}");
        }
        let counter_b = rendered
            .lines()
            .find(|l| l.starts_with(M_REQUESTS_TOTAL) && l.contains("status=\"404\""))
            .expect("404 series must render distinctly");
        assert!(counter_b.ends_with(" 1"), "got: {counter_b}");
        assert!(counter_b.contains("model=\"other-model\""));

        // Duration summary: carries provider/model/status, never outcome,
        // and counts per (model, status) series independently.
        let dur_count_a = rendered
            .lines()
            .find(|l| {
                l.starts_with(&format!("{M_REQUEST_DURATION}_count"))
                    && l.contains("model=\"my-gpt4\"")
            })
            .expect("duration count for my-gpt4 must render");
        assert!(dur_count_a.ends_with(" 2") && dur_count_a.contains("status=\"200\""));
        let dur_count_b = rendered
            .lines()
            .find(|l| {
                l.starts_with(&format!("{M_REQUEST_DURATION}_count"))
                    && l.contains("model=\"other-model\"")
            })
            .expect("duration count for other-model must render");
        assert!(dur_count_b.ends_with(" 1") && dur_count_b.contains("status=\"404\""));
        for line in rendered.lines() {
            if line.starts_with(M_REQUEST_DURATION) {
                assert!(
                    !line.contains("outcome="),
                    "outcome must stay off durations: {line}"
                );
            }
        }
    }

    // ── Vary-one-label key-drift guards ────────────────────────────────
    //
    // The worker cache duplicates each family's label list in two places:
    // the key builder and the register closure. A label present in the
    // register closure but DROPPED from the key builder aliases every
    // pair of label sets that differ only in that label — silently, with
    // correct-looking output for single-label-set traffic. These tests
    // emit a base label set twice plus one variant per label differing
    // ONLY in that label, then assert one rendered series per label set:
    // any single dropped key label folds its variant into the base series
    // and fails both the count and the base-total assertion.
    // (An independent audit demonstrated by mutation that multi-dimension
    // pairs do NOT catch single-label drops; each variant here differs in
    // exactly one.)

    /// Rendered value lines for one exact metric name (brace-delimited,
    /// so `aisix_x` never matches `aisix_x_count` and vice versa).
    fn series_lines<'r>(rendered: &'r str, metric: &str) -> Vec<&'r str> {
        rendered
            .lines()
            .filter(|l| l.starts_with(metric) && l.as_bytes().get(metric.len()) == Some(&b'{'))
            .collect()
    }

    #[track_caller]
    fn assert_one_series_per_label_set(rendered: &str, metric: &str, expected: usize) {
        let series = series_lines(rendered, metric);
        assert_eq!(
            series.len(),
            expected,
            "{metric}: a dropped key label folds a variant into the base series\n{rendered}"
        );
        assert_eq!(
            series.iter().filter(|l| l.ends_with(" 2")).count(),
            1,
            "{metric}: exactly the base series must show both base emits\n{rendered}"
        );
    }

    #[test]
    fn request_series_key_covers_every_label() {
        let base = RequestLabels::default();
        let variants = [
            RequestLabels {
                endpoint: "/v1/messages",
                ..base
            },
            RequestLabels {
                inbound_protocol: "anthropic",
                ..base
            },
            RequestLabels {
                provider: "p2",
                ..base
            },
            RequestLabels {
                model: "m2",
                ..base
            },
            RequestLabels {
                upstream_model: "um2",
                ..base
            },
            RequestLabels {
                provider_key_id: "pk2",
                ..base
            },
            RequestLabels {
                provider_key_name: "pkn2",
                ..base
            },
            RequestLabels {
                api_key_id: "ak2",
                ..base
            },
            RequestLabels {
                team_id: "t2",
                ..base
            },
            RequestLabels {
                user_id: "u2",
                ..base
            },
            RequestLabels {
                user_name: "n2",
                ..base
            },
            RequestLabels {
                stream: true,
                ..base
            },
            RequestLabels {
                is_fallback: true,
                ..base
            },
            RequestLabels {
                status: 201,
                ..base
            },
            RequestLabels {
                outcome: RequestOutcome::ClientError,
                ..base
            },
        ];
        let m = Metrics::new(false);
        m.record_proxy_and_llm_request(base, Duration::from_millis(1));
        m.record_proxy_and_llm_request(base, Duration::from_millis(1));
        for v in &variants {
            m.record_proxy_and_llm_request(*v, Duration::from_millis(1));
        }
        assert_one_series_per_label_set(&m.render(), M_PROXY_REQUESTS_TOTAL, 1 + variants.len());
    }

    #[test]
    fn usage_series_key_covers_every_label() {
        let base = UsageLabels::default();
        let variants = [
            UsageLabels {
                endpoint: "/v1/messages",
                ..base
            },
            UsageLabels {
                inbound_protocol: "anthropic",
                ..base
            },
            UsageLabels {
                provider: "p2",
                ..base
            },
            UsageLabels {
                model: "m2",
                ..base
            },
            UsageLabels {
                upstream_model: "um2",
                ..base
            },
            UsageLabels {
                provider_key_id: "pk2",
                ..base
            },
            UsageLabels {
                provider_key_name: "pkn2",
                ..base
            },
            UsageLabels {
                api_key_id: "ak2",
                ..base
            },
            UsageLabels {
                team_id: "t2",
                ..base
            },
            UsageLabels {
                user_id: "u2",
                ..base
            },
            UsageLabels {
                user_name: "n2",
                ..base
            },
        ];
        let m = Metrics::new(false);
        let one_token = LlmUsage {
            input_tokens: 1,
            ..LlmUsage::default()
        };
        m.record_llm_usage(base, one_token);
        m.record_llm_usage(base, one_token);
        for v in &variants {
            m.record_llm_usage(*v, one_token);
        }
        assert_one_series_per_label_set(&m.render(), M_LLM_INPUT_TOKENS_TOTAL, 1 + variants.len());
    }

    #[test]
    fn legacy_request_key_covers_every_label() {
        let m = Metrics::new(false);
        let emit = |provider, model, status, outcome| {
            m.record_request(provider, model, status, outcome, Duration::from_millis(1));
        };
        emit("openai", "m", 200, RequestOutcome::Success);
        emit("openai", "m", 200, RequestOutcome::Success);
        emit("p2", "m", 200, RequestOutcome::Success);
        emit("openai", "m2", 200, RequestOutcome::Success);
        emit("openai", "m", 201, RequestOutcome::Success);
        emit("openai", "m", 200, RequestOutcome::ClientError);

        let rendered = m.render();
        assert_one_series_per_label_set(&rendered, M_REQUESTS_TOTAL, 5);
        // The duration histogram keys on (provider, model, status) only:
        // the outcome variant lands on the base duration series, so the
        // base count is 3 across 4 series. A status/model/provider drop
        // from the HISTOGRAM key collapses its series count below 4.
        let dur = series_lines(&rendered, &format!("{M_REQUEST_DURATION}_count"));
        assert_eq!(dur.len(), 4, "{rendered}");
        assert_eq!(
            dur.iter().filter(|l| l.ends_with(" 3")).count(),
            1,
            "{rendered}"
        );
    }

    /// `clear_budget_gauges` runs on EVERY request whose decision carries
    /// no budget — which is every request on a deployment with budgets
    /// switched off, the default. Retracting the amounts here would mint
    /// four series per api key for a feature nobody enabled, so it must
    /// touch the presence flag and nothing else.
    #[test]
    fn a_budgetless_key_mints_only_the_presence_flag() {
        let m = Metrics::new(false);
        m.clear_budget_gauges(BudgetLabels {
            api_key_id: "ak-1",
            team_id: "t",
            user_id: "u",
            user_name: "alice",
        });
        let rendered = m.render();
        assert_eq!(
            series_lines(&rendered, M_BUDGET_DETAILS_PRESENT).len(),
            1,
            "{rendered}"
        );
        for metric in [
            M_BUDGET_LIMIT_USD,
            M_BUDGET_SPENT_USD,
            M_BUDGET_REMAINING_USD,
            M_BUDGET_RESET_SECONDS,
        ] {
            assert!(
                series_lines(&rendered, metric).is_empty(),
                "{metric} must not exist for a key that never had a budget:\n{rendered}"
            );
        }
    }

    /// Retiring a budgetless key must not mint the amounts it never had.
    ///
    /// `metrics::gauge!` REGISTERS on first use, so retiring a family's
    /// full list creates the members that label set never carried. Budgets
    /// are off by default, which puts every api key on the
    /// `clear_budget_gauges` path — so retiring blind would mint four
    /// series per deleted key for a feature nobody enabled, the exact
    /// cardinality `clear_budget_gauges` avoids on the write side.
    #[test]
    fn retiring_a_budgetless_key_mints_no_amount_series() {
        let m = Metrics::new(false);
        let labels = BudgetLabels {
            api_key_id: "ak-1",
            team_id: "t",
            user_id: "u",
            user_name: "alice",
        };
        // The shape of a deployment with budgets switched off: the flag
        // and nothing else, on every request.
        m.clear_budget_gauges(labels);
        assert_eq!(m.retire_stale_gauges(|_| false), 1);

        let rendered = m.render();
        assert!(
            series_lines(&rendered, M_BUDGET_DETAILS_PRESENT)
                .pop()
                .expect("the flag was registered, so it retires")
                .ends_with(" 0"),
            "{rendered}"
        );
        for metric in [
            M_BUDGET_LIMIT_USD,
            M_BUDGET_SPENT_USD,
            M_BUDGET_REMAINING_USD,
            M_BUDGET_RESET_SECONDS,
        ] {
            assert!(
                series_lines(&rendered, metric).is_empty(),
                "retirement created {metric}, which this key never had:\n{rendered}"
            );
        }
    }

    /// The amounts appear later than the flag — a key gets its presence
    /// flag on the request that finds no budget, and its amounts only once
    /// one exists — so an already-tracked label set has to learn about a
    /// metric it has not seen, or retirement would leave the amounts frozen.
    #[test]
    fn an_amount_registered_after_the_flag_still_retires() {
        let m = Metrics::new(false);
        let labels = BudgetLabels {
            api_key_id: "ak-1",
            team_id: "t",
            user_id: "u",
            user_name: "alice",
        };
        m.clear_budget_gauges(labels);
        m.set_budget_gauges(
            labels,
            BudgetGauges {
                spent_usd: Some(40.0),
                ..BudgetGauges::default()
            },
        );
        assert_eq!(m.retire_stale_gauges(|_| false), 1);

        let rendered = m.render();
        assert!(
            series_lines(&rendered, M_BUDGET_SPENT_USD)
                .pop()
                .expect("renders")
                .ends_with(" NaN"),
            "{rendered}"
        );
        // Still nothing for the amounts that were never written.
        assert!(
            series_lines(&rendered, M_BUDGET_LIMIT_USD).is_empty(),
            "{rendered}"
        );
    }

    /// A gauge whose key is gone must stop claiming to describe the
    /// present. Before retirement the recorder has no idle timeout, so the
    /// deleted key's sample stayed frozen with `details_present=1` —
    /// enough to latch the documented "over 90%" alert forever.
    #[test]
    fn a_deleted_key_retires_its_budget_series() {
        let m = Metrics::new(false);
        let labels = BudgetLabels {
            api_key_id: "ak-gone",
            team_id: "t",
            user_id: "u",
            user_name: "alice",
        };
        m.set_budget_gauges(
            labels,
            BudgetGauges {
                limit_usd: Some(100.0),
                spent_usd: Some(95.0),
                remaining_usd: Some(5.0),
                reset_seconds: Some(60),
            },
        );
        assert!(m.render().contains("aisix_budget_details_present"));

        // Nothing is live: the key was deleted.
        assert_eq!(m.retire_stale_gauges(|_| false), 1);

        let rendered = m.render();
        let value = |metric: &str| -> String {
            series_lines(&rendered, metric)
                .pop()
                .unwrap_or_else(|| panic!("{metric} must still render\n{rendered}"))
                .rsplit(' ')
                .next()
                .expect("a value")
                .to_string()
        };
        // The presence flag is the one the documented guard reads, and its
        // cleared value is 0 — that is what makes the alert stop matching.
        assert_eq!(value(M_BUDGET_DETAILS_PRESENT), "0");
        // The quantities go to NaN, NOT 0: `spent 0 / limit 0` is a claim
        // about a budget, and a `remaining_usd 0` would read as "exhausted".
        for metric in [
            M_BUDGET_LIMIT_USD,
            M_BUDGET_SPENT_USD,
            M_BUDGET_REMAINING_USD,
            M_BUDGET_RESET_SECONDS,
        ] {
            assert_eq!(value(metric), "NaN", "{metric} must retire to NaN");
        }
    }

    /// `aisix_ratelimit_remaining_requests 0` means "quota exhausted".
    /// Retiring to zero would replace a stale claim with a false one, so
    /// this pins the marker rather than trusting the constant.
    #[test]
    fn retirement_never_asserts_a_meaningful_zero() {
        let m = Metrics::new(false);
        m.set_rate_limit_remaining("ak-gone", "gpt-4o", Some(7), Some(900));
        assert_eq!(m.retire_stale_gauges(|_| false), 1);

        let rendered = m.render();
        for metric in [M_RATELIMIT_REMAINING_REQUESTS, M_RATELIMIT_REMAINING_TOKENS] {
            let line = series_lines(&rendered, metric)
                .pop()
                .unwrap_or_else(|| panic!("{metric} must render\n{rendered}"));
            assert!(
                line.ends_with(" NaN"),
                "{metric} must retire to NaN, never to a value that means \
                 something: {line}"
            );
        }
    }

    /// `aisix_deployment_state` is deliberately NOT swept: it is written
    /// only on a health TRANSITION, so retiring it would blank a live
    /// model's state with nothing to rewrite it until the next flip.
    #[test]
    fn deployment_state_is_not_retirable() {
        let m = Metrics::new(false);
        m.set_deployment_state(
            DeploymentLabels {
                provider: "openai",
                model: "gpt-4o",
                upstream_model: "gpt-4o",
                provider_key_id: "pk-gone",
            },
            DeploymentState::Down,
        );
        // Nothing is live, and it still must not be touched.
        assert_eq!(m.retire_stale_gauges(|_| false), 0);
        assert!(
            series_lines(&m.render(), M_DEPLOYMENT_STATE)
                .pop()
                .expect("renders")
                .ends_with(" 2"),
            "a health-transition gauge must keep its last reported state"
        );
    }

    /// Retirement is per label SET, not per key: rebinding a key to another
    /// team strands the old series under the same `api_key_id`, and only
    /// that one may be retired.
    #[test]
    fn rebinding_retires_the_old_label_set_and_leaves_the_new_one() {
        let m = Metrics::new(false);
        let spend = |v| BudgetGauges {
            spent_usd: Some(v),
            ..BudgetGauges::default()
        };
        let before = BudgetLabels {
            api_key_id: "ak-1",
            team_id: "team-old",
            user_id: "u",
            user_name: "alice",
        };
        let after = BudgetLabels {
            team_id: "team-new",
            ..before
        };
        m.set_budget_gauges(before, spend(40.0));
        m.set_budget_gauges(after, spend(45.0));

        // Only the post-rebind label set is current.
        let retired = m.retire_stale_gauges(|series| match series {
            LiveGaugeSeries::Budget { team_id, .. } => team_id == "team-new",
            _ => true,
        });
        assert_eq!(retired, 1);

        let rendered = m.render();
        let line = |team: &str| -> String {
            series_lines(&rendered, M_BUDGET_SPENT_USD)
                .into_iter()
                .find(|l| l.contains(&format!("team_id=\"{team}\"")))
                .unwrap_or_else(|| panic!("team {team} must render\n{rendered}"))
                .to_string()
        };
        assert!(line("team-old").ends_with(" NaN"), "{}", line("team-old"));
        assert!(line("team-new").ends_with(" 45"), "{}", line("team-new"));
    }

    /// The sweep must not touch a series whose key is still configured —
    /// a wrongly retired gauge hides live state, which is worse than the
    /// stale sample it was meant to clean up.
    #[test]
    fn a_live_key_is_never_retired() {
        let m = Metrics::new(false);
        m.set_budget_gauges(
            BudgetLabels {
                api_key_id: "ak-live",
                team_id: "t",
                user_id: "u",
                user_name: "alice",
            },
            BudgetGauges {
                spent_usd: Some(12.0),
                ..BudgetGauges::default()
            },
        );
        assert_eq!(m.retire_stale_gauges(|_| true), 0);
        let rendered = m.render();
        assert!(
            series_lines(&rendered, M_BUDGET_SPENT_USD)
                .pop()
                .expect("renders")
                .ends_with(" 12"),
            "{rendered}"
        );
    }

    /// A retired label set is DROPPED from the registry, not kept and
    /// re-marked forever. Keeping it would let dead entries fill
    /// `RETIRABLE_CAPACITY` over a long-running process and silently stop
    /// new series from being tracked at all — the very bug this exists to
    /// fix — and would re-emit the same NaN on every tick.
    #[test]
    fn a_retired_series_is_dropped_from_the_registry() {
        let m = Metrics::new(false);
        let labels = BudgetLabels {
            api_key_id: "ak-1",
            team_id: "t",
            user_id: "u",
            user_name: "alice",
        };
        m.set_budget_gauges(
            labels,
            BudgetGauges {
                spent_usd: Some(40.0),
                ..BudgetGauges::default()
            },
        );
        assert_eq!(m.retire_stale_gauges(|_| false), 1);
        // Nothing left to retire: the entry is gone, so the sweep does no
        // work on it again.
        assert_eq!(m.retire_stale_gauges(|_| false), 0);
    }

    /// Rebinding produces a NEW label set, which registers and is
    /// retirable in its own right — so dropping the old entry does not
    /// leave the key untracked. (An identical label set coming back would
    /// need a deleted id to be reissued, which uuids do not do.)
    #[test]
    fn the_label_set_a_rebind_creates_is_tracked_too() {
        let m = Metrics::new(false);
        let spend = |v| BudgetGauges {
            spent_usd: Some(v),
            ..BudgetGauges::default()
        };
        let before = BudgetLabels {
            api_key_id: "ak-1",
            team_id: "team-old",
            user_id: "u",
            user_name: "alice",
        };
        m.set_budget_gauges(before, spend(40.0));
        assert_eq!(m.retire_stale_gauges(|_| false), 1);

        let after = BudgetLabels {
            team_id: "team-new",
            ..before
        };
        m.set_budget_gauges(after, spend(45.0));
        assert_eq!(
            m.retire_stale_gauges(|_| false),
            1,
            "the post-rebind label set must be tracked on its own registration"
        );
    }

    #[test]
    fn budget_gauge_key_covers_every_label() {
        let m = Metrics::new(false);
        let base = BudgetLabels {
            api_key_id: "ak",
            team_id: "t",
            user_id: "u",
            user_name: "n",
        };
        let variants = [
            BudgetLabels {
                api_key_id: "ak2",
                ..base
            },
            BudgetLabels {
                team_id: "t2",
                ..base
            },
            BudgetLabels {
                user_id: "u2",
                ..base
            },
            // A rename keeps the id and changes only the name. Off the
            // worker key, the gauge would keep reporting this member
            // under their old name for as long as the worker lives.
            BudgetLabels {
                user_name: "n2",
                ..base
            },
        ];
        let spend = |v| BudgetGauges {
            spent_usd: Some(v),
            ..BudgetGauges::default()
        };
        m.set_budget_gauges(base, spend(1.0));
        for (i, v) in variants.iter().enumerate() {
            m.set_budget_gauges(*v, spend(2.0 + i as f64));
        }
        let rendered = m.render();
        let series = series_lines(&rendered, M_BUDGET_SPENT_USD);
        assert_eq!(series.len(), 1 + variants.len(), "{rendered}");
        // Gauges overwrite rather than sum: a dropped key label makes the
        // LAST variant's value land on the base series instead.
        assert!(
            series.iter().any(|l| l.contains("api_key_id=\"ak\"")
                && l.contains("team_id=\"t\"")
                && l.ends_with(" 1")),
            "{rendered}"
        );
    }

    #[test]
    fn llm_ttft_key_covers_every_label() {
        let base = UsageLabels::default();
        let variants = [
            UsageLabels {
                endpoint: "/v1/messages",
                ..base
            },
            UsageLabels {
                inbound_protocol: "anthropic",
                ..base
            },
            UsageLabels {
                provider: "p2",
                ..base
            },
            UsageLabels {
                model: "m2",
                ..base
            },
            UsageLabels {
                upstream_model: "um2",
                ..base
            },
            UsageLabels {
                provider_key_id: "pk2",
                ..base
            },
            UsageLabels {
                provider_key_name: "pkn2",
                ..base
            },
            UsageLabels {
                api_key_id: "ak2",
                ..base
            },
            UsageLabels {
                team_id: "t2",
                ..base
            },
            UsageLabels {
                user_id: "u2",
                ..base
            },
            UsageLabels {
                user_name: "n2",
                ..base
            },
        ];
        let m = Metrics::new(false);
        m.record_time_to_first_token(base, Duration::from_millis(1));
        m.record_time_to_first_token(base, Duration::from_millis(1));
        for v in &variants {
            m.record_time_to_first_token(*v, Duration::from_millis(1));
        }
        assert_one_series_per_label_set(
            &m.render(),
            &format!("{M_LLM_TTFT}_count"),
            1 + variants.len(),
        );
    }

    #[test]
    fn latency_histogram_key_covers_every_label() {
        let base = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "m",
            provider: "p",
            status: 200,
            streaming: false,
        };
        let variants = [
            LatencyLabels {
                endpoint: "/v1/messages",
                ..base
            },
            LatencyLabels {
                model: "m2",
                ..base
            },
            LatencyLabels {
                provider: "p2",
                ..base
            },
            // The key uses the status CLASS, so the variant must change
            // the bucket, not just the code.
            LatencyLabels {
                status: 404,
                ..base
            },
            LatencyLabels {
                streaming: true,
                ..base
            },
        ];
        let m = Metrics::new(false);
        m.record_request_e2e_latency(base, Duration::from_millis(1));
        m.record_request_e2e_latency(base, Duration::from_millis(1));
        for v in &variants {
            m.record_request_e2e_latency(*v, Duration::from_millis(1));
        }
        assert_one_series_per_label_set(
            &m.render(),
            &format!("{M_REQUEST_E2E_LATENCY_SECONDS}_count"),
            1 + variants.len(),
        );
    }

    #[test]
    fn auth_decision_key_covers_every_label() {
        let m = Metrics::new(false);
        m.record_auth_decision("api_key", true, "");
        m.record_auth_decision("api_key", true, "");
        m.record_auth_decision("jwt", true, "");
        m.record_auth_decision("api_key", false, "");
        m.record_auth_decision("api_key", true, "key_expired");
        assert_one_series_per_label_set(&m.render(), M_AUTH_DECISIONS_TOTAL, 4);
    }

    #[test]
    fn guardrail_latency_key_covers_every_label() {
        let base = aisix_core::GuardrailExecution {
            guardrail_name: "g",
            kind: "keyword",
            phase: "input",
            result: "allowed",
            error_type: None,
            elapsed: Duration::from_millis(1),
        };
        let variants = [
            aisix_core::GuardrailExecution {
                guardrail_name: "g2",
                ..base
            },
            aisix_core::GuardrailExecution {
                kind: "pii",
                ..base
            },
            aisix_core::GuardrailExecution {
                phase: "output",
                ..base
            },
            aisix_core::GuardrailExecution {
                result: "blocked",
                ..base
            },
            aisix_core::GuardrailExecution {
                error_type: Some("timeout"),
                ..base
            },
        ];
        let m = Metrics::new(false);
        m.record_guardrail_execution(&base);
        m.record_guardrail_execution(&base);
        for v in &variants {
            m.record_guardrail_execution(v);
        }
        assert_one_series_per_label_set(
            &m.render(),
            &format!("{M_GUARDRAIL_LATENCY_SECONDS}_count"),
            1 + variants.len(),
        );
    }

    #[test]
    fn ratelimit_rejection_key_covers_every_label() {
        let m = Metrics::new(false);
        m.record_ratelimit_rejection("requests", "api_key", None);
        m.record_ratelimit_rejection("requests", "api_key", None);
        m.record_ratelimit_rejection("tokens", "api_key", None);
        m.record_ratelimit_rejection("requests", "model", None);
        m.record_ratelimit_rejection("requests", "api_key", Some("p1"));
        assert_one_series_per_label_set(&m.render(), M_RATELIMIT_REJECTIONS, 4);
    }

    #[test]
    fn tokens_by_client_key_covers_every_label() {
        let m = Metrics::new(false);
        m.record_llm_tokens_by_client("cli", "m", 1, 0, 0);
        m.record_llm_tokens_by_client("cli", "m", 1, 0, 0);
        m.record_llm_tokens_by_client("cli2", "m", 1, 0, 0);
        m.record_llm_tokens_by_client("cli", "m2", 1, 0, 0);
        assert_one_series_per_label_set(&m.render(), M_LLM_TOKENS_BY_CLIENT_TOTAL, 3);
    }

    #[test]
    fn consumed_tokens_key_covers_every_label() {
        let m = Metrics::new(false);
        m.record_tokens("p", "m", 1);
        m.record_tokens("p", "m", 1);
        m.record_tokens("p2", "m", 1);
        m.record_tokens("p", "m2", 1);
        assert_one_series_per_label_set(&m.render(), M_TOKENS_CONSUMED, 3);
    }

    const PK_A: UsageEventLabels<'static> = UsageEventLabels {
        model: "gpt-4o",
        provider_key_id: "pk-1",
        provider_key_name: "openai-prod",
        user_id: "member-1",
        user_name: "alice",
        upstream_protocol: "openai",
    };

    #[test]
    fn usage_event_emit_key_covers_every_label() {
        let m = Metrics::new(false);
        m.record_usage_event_emit("chat", 200, "openai", PK_A);
        m.record_usage_event_emit("chat", 200, "openai", PK_A);
        m.record_usage_event_emit("embeddings", 200, "openai", PK_A);
        m.record_usage_event_emit("chat", 404, "openai", PK_A);
        m.record_usage_event_emit("chat", 200, "anthropic", PK_A);
        // AISIX-Cloud#1317: the attribution dimensions belong to the key
        // too. Leaving one out of it folds two models' — or two keys' —
        // emissions into whichever series was cached first.
        m.record_usage_event_emit(
            "chat",
            200,
            "openai",
            UsageEventLabels {
                model: "gpt-4o-mini",
                ..PK_A
            },
        );
        m.record_usage_event_emit(
            "chat",
            200,
            "openai",
            UsageEventLabels {
                provider_key_id: "pk-2",
                provider_key_name: "openai-backup",
                ..PK_A
            },
        );
        // AISIX-Cloud#1403: and so does the upstream protocol. A key
        // repointed at an Anthropic-shape upstream keeps its id, so
        // leaving this out of the cache key would report the new traffic
        // under the old protocol for as long as the worker lives.
        m.record_usage_event_emit(
            "chat",
            200,
            "openai",
            UsageEventLabels {
                upstream_protocol: "anthropic",
                ..PK_A
            },
        );
        // AISIX-Cloud#1455: a renamed member keeps their id, so the name
        // has to be part of the key or this counter reports them under
        // the old name for the worker's lifetime.
        m.record_usage_event_emit(
            "chat",
            200,
            "openai",
            UsageEventLabels {
                user_name: "alice-renamed",
                ..PK_A
            },
        );
        assert_one_series_per_label_set(&m.render(), M_USAGE_EVENT_EMITS_TOTAL, 8);
    }

    /// AISIX-Cloud#1403: the label survives onto the exposition, with the
    /// UPSTREAM's protocol rather than the caller's, and `unknown` where
    /// no upstream was selected.
    ///
    /// The family this test exists for is the one 0.11.0-rc.2 shipped
    /// without the label: it had the resolved ProviderKey right beside
    /// `provider_key_id` and emitted everything but its protocol.
    #[test]
    fn usage_event_emit_reports_the_upstream_protocol_not_the_callers() {
        let m = Metrics::new(false);
        // An Anthropic-protocol caller served by an OpenAI-shape upstream.
        m.record_usage_event_emit("messages", 200, "anthropic", PK_A);
        // A request that never reached a key — a pre-dispatch rejection,
        // or a tunnel with no LLM upstream at all.
        m.record_usage_event_emit("chat", 401, "openai", UsageEventLabels::default());
        let rendered = m.render();

        assert!(
            rendered.contains(
                "handler=\"messages\",status_code=\"2xx\",status=\"200\",\
                 inbound_protocol=\"anthropic\",upstream_protocol=\"openai\""
            ),
            "cross-protocol sample must report the upstream's protocol:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "handler=\"chat\",status_code=\"4xx\",status=\"401\",\
                 inbound_protocol=\"openai\",upstream_protocol=\"unknown\""
            ),
            "an unresolved upstream must read `unknown`, never borrow the \
             inbound protocol:\n{rendered}"
        );
        assert_upstream_protocol_rides_along(&rendered);
    }

    #[test]
    fn usage_event_drop_key_covers_every_label() {
        let m = Metrics::new(false);
        m.record_usage_event_drop("sink_full", PK_A);
        m.record_usage_event_drop("sink_full", PK_A);
        m.record_usage_event_drop("sink_closed", PK_A);
        m.record_usage_event_drop(
            "sink_full",
            UsageEventLabels {
                model: "gpt-4o-mini",
                ..PK_A
            },
        );
        m.record_usage_event_drop(
            "sink_full",
            UsageEventLabels {
                user_name: "alice-renamed",
                ..PK_A
            },
        );
        assert_one_series_per_label_set(&m.render(), M_USAGE_EVENT_DROPS_TOTAL, 4);
    }

    /// AISIX-Cloud#1317: the two counters are read as one pair —
    /// `emitted == delivered + dropped`. A per-model or per-key slice of
    /// that invariant only exists if BOTH sides carry the same attribution,
    /// which is why `try_emit` hands one label set to both.
    ///
    /// Derived, not a list of label names: the drop sample must carry
    /// everything the emit sample does EXCEPT the three dimensions
    /// `record_usage_event_emit` takes as its own arguments. So a field
    /// added to [`UsageEventLabels`] and wired into the emit counter fails
    /// this test until it reaches the drop counter too — which is how
    /// `upstream_protocol` reached only one side in the first place
    /// (AISIX-Cloud#1403).
    #[test]
    fn emits_and_drops_carry_the_same_attribution_labels() {
        let m = Metrics::new(false);
        m.record_usage_event_emit("chat", 200, "openai", PK_A);
        m.record_usage_event_drop("sink_full", PK_A);
        let rendered = m.render();

        let labels_of = |metric: &str| -> BTreeSet<String> {
            let line = series_lines(&rendered, metric)
                .pop()
                .unwrap_or_else(|| panic!("{metric} must render\n{rendered}"));
            line.split_once('{')
                .and_then(|(_, rest)| rest.rsplit_once('}'))
                .expect("a labelled sample")
                .0
                .split(',')
                .filter_map(|pair| pair.split_once('=').map(|(k, _)| k.to_string()))
                .collect()
        };
        // The emit counter's own arguments — a surface of the emitting
        // handler, not of the event's attribution.
        let emit_only: BTreeSet<String> = ["handler", "status_code", "status", "inbound_protocol"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let attribution: BTreeSet<String> = labels_of(M_USAGE_EVENT_EMITS_TOTAL)
            .difference(&emit_only)
            .cloned()
            .collect();
        assert!(
            attribution.contains("model")
                && attribution.contains("provider_key_id")
                && attribution.contains("user_id"),
            "the attribution set looks wrong: {attribution:?}"
        );

        let dropped = labels_of(M_USAGE_EVENT_DROPS_TOTAL);
        let missing: Vec<&String> = attribution.difference(&dropped).collect();
        assert!(
            missing.is_empty(),
            "the drop counter is missing attribution the emit counter \
             carries: {missing:?}. Both sides take one `UsageEventLabels`, \
             so a dimension on only one of them makes that slice of \
             `emitted == delivered + dropped` unanswerable.",
        );
    }

    #[test]
    fn in_flight_gauge_key_covers_every_label() {
        let m = Metrics::new(false);
        m.increment_proxy_in_flight("/v1/chat/completions", "openai");
        m.increment_proxy_in_flight("/v1/messages", "openai");
        m.increment_proxy_in_flight("/v1/chat/completions", "anthropic");
        let rendered = m.render();
        assert_eq!(
            series_lines(&rendered, M_PROXY_IN_FLIGHT).len(),
            3,
            "a dropped key label folds distinct endpoints/protocols into one gauge\n{rendered}"
        );
    }

    #[test]
    fn in_flight_gauge_clamps_an_unmatched_decrement_at_zero() {
        let m = Metrics::new(false);
        m.decrement_proxy_in_flight("/v1/chat/completions", "openai");
        let rendered = m.render();
        let line = rendered
            .lines()
            .find(|l| l.starts_with(M_PROXY_IN_FLIGHT))
            .expect("in-flight gauge must render");
        assert!(
            line.ends_with(" 0"),
            "unmatched decrement must clamp: {line}"
        );
    }

    /// Pins the slot predicate on BOTH fields: a regression comparing only
    /// `endpoint` would route the anthropic edges through the openai slot,
    /// corrupting VALUES while the series count (guarded above) stays 3.
    #[test]
    fn in_flight_slots_stay_distinct_per_endpoint_and_protocol() {
        let m = Metrics::new(false);
        m.increment_proxy_in_flight("/v1/chat/completions", "openai");
        m.increment_proxy_in_flight("/v1/chat/completions", "anthropic");
        m.increment_proxy_in_flight("/v1/messages", "anthropic");
        m.decrement_proxy_in_flight("/v1/chat/completions", "anthropic");

        let rendered = m.render();
        let value = |endpoint: &str, protocol: &str| {
            rendered
                .lines()
                .find(|l| {
                    l.starts_with(M_PROXY_IN_FLIGHT)
                        && l.contains(&format!("endpoint=\"{endpoint}\""))
                        && l.contains(&format!("inbound_protocol=\"{protocol}\""))
                })
                .and_then(|l| l.rsplit(' ').next())
                .map(str::to_owned)
        };
        assert_eq!(
            value("/v1/chat/completions", "openai").as_deref(),
            Some("1")
        );
        assert_eq!(
            value("/v1/chat/completions", "anthropic").as_deref(),
            Some("0")
        );
        assert_eq!(value("/v1/messages", "anthropic").as_deref(), Some("1"));
    }

    #[test]
    fn concurrent_request_series_misses_register_once_and_record_every_call() {
        const THREADS: usize = 16;

        let metrics = Metrics::new(false);
        let barrier = Arc::new(std::sync::Barrier::new(THREADS));
        let threads = (0..THREADS)
            .map(|_| {
                let metrics = metrics.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    metrics.record_proxy_and_llm_request(
                        RequestLabels::default(),
                        Duration::from_millis(1),
                    );
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("metric recording thread panicked");
        }

        // Each thread registers into its own worker cache, but every
        // registration for the same labels resolves to the SAME registry
        // series — the sums below are the contract.
        let rendered = metrics.render();
        for metric in [
            M_PROXY_REQUESTS_TOTAL,
            M_LLM_REQUESTS_TOTAL,
            M_PROXY_FAILED_REQUESTS_TOTAL,
        ] {
            assert!(
                rendered
                    .lines()
                    .any(|line| line.starts_with(metric) && line.ends_with(" 16")),
                "{metric} did not record every concurrent call:\n{rendered}"
            );
        }
        for metric in [M_PROXY_REQUEST_DURATION, M_LLM_REQUEST_DURATION] {
            let count = format!("{metric}_count");
            assert!(
                rendered
                    .lines()
                    .any(|line| line.starts_with(&count) && line.ends_with(" 16")),
                "{metric} did not record every concurrent call:\n{rendered}"
            );
        }
    }

    #[test]
    fn usage_series_cache_is_lazy_and_reuses_handles() {
        let metrics = Metrics::new(false);
        let labels = UsageLabels {
            model: "cached-usage-model",
            ..UsageLabels::default()
        };

        metrics.record_llm_usage(
            labels,
            LlmUsage {
                input_tokens: 5,
                ..LlmUsage::default()
            },
        );
        assert_eq!(
            Metrics::worker_usage_series_len(),
            1,
            "the first non-zero usage sample must create one cached label set"
        );
        let rendered = metrics.render();
        assert!(rendered.lines().any(|line| {
            line.starts_with(M_LLM_INPUT_TOKENS_TOTAL)
                && line.contains("model=\"cached-usage-model\"")
                && line.ends_with(" 5")
        }));
        for absent in [
            M_LLM_OUTPUT_TOKENS_TOTAL,
            M_LLM_TOTAL_TOKENS_TOTAL,
            M_LLM_SPEND_MICRO_USD_TOTAL,
        ] {
            assert!(
                !rendered.contains(absent),
                "a zero-valued usage dimension must not register {absent}"
            );
        }

        metrics.record_llm_usage(
            labels,
            LlmUsage {
                output_tokens: 7,
                total_tokens: 7,
                spend_usd: 0.001,
                ..LlmUsage::default()
            },
        );
        assert_eq!(
            Metrics::worker_usage_series_len(),
            1,
            "repeated labels must reuse the cached usage handles"
        );
        let rendered = metrics.render();
        for (metric, value) in [
            (M_LLM_INPUT_TOKENS_TOTAL, 5),
            (M_LLM_OUTPUT_TOKENS_TOTAL, 7),
            (M_LLM_TOTAL_TOKENS_TOTAL, 7),
            (M_LLM_SPEND_MICRO_USD_TOTAL, 1000),
        ] {
            assert!(
                rendered.lines().any(|line| {
                    line.starts_with(metric)
                        && line.contains("model=\"cached-usage-model\"")
                        && line.ends_with(&format!(" {value}"))
                }),
                "{metric} did not retain the expected value:\n{rendered}"
            );
        }
    }

    #[test]
    fn concurrent_usage_series_misses_register_once_and_record_every_call() {
        const THREADS: usize = 16;

        let metrics = Metrics::new(false);
        let barrier = Arc::new(std::sync::Barrier::new(THREADS));
        let threads = (0..THREADS)
            .map(|_| {
                let metrics = metrics.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    metrics.record_llm_usage(
                        UsageLabels::default(),
                        LlmUsage {
                            input_tokens: 1,
                            output_tokens: 2,
                            total_tokens: 3,
                            spend_usd: 0.000001,
                            ..Default::default()
                        },
                    );
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("metric recording thread panicked");
        }

        // Per-thread caches; the shared-series sums below are the contract.
        let rendered = metrics.render();
        for (metric, value) in [
            (M_LLM_INPUT_TOKENS_TOTAL, THREADS),
            (M_LLM_OUTPUT_TOKENS_TOTAL, THREADS * 2),
            (M_LLM_TOTAL_TOKENS_TOTAL, THREADS * 3),
            (M_LLM_SPEND_MICRO_USD_TOTAL, THREADS),
        ] {
            assert!(
                rendered
                    .lines()
                    .any(|line| line.starts_with(metric) && line.ends_with(&format!(" {value}"))),
                "{metric} did not record every concurrent call:\n{rendered}"
            );
        }
    }

    #[test]
    fn usage_series_handle_cache_is_bounded() {
        let metrics = Metrics::new(false);
        for index in 0..=WORKER_CACHE_CAPACITY {
            let model = format!("usage-model-{index}");
            metrics.record_llm_usage(
                UsageLabels {
                    model: &model,
                    ..UsageLabels::default()
                },
                LlmUsage {
                    input_tokens: 1,
                    ..LlmUsage::default()
                },
            );
        }

        assert_eq!(
            Metrics::worker_usage_series_len(),
            WORKER_CACHE_CAPACITY,
            "usage series handle cache must stay at its fixed capacity"
        );
    }

    #[test]
    fn individual_request_recorders_do_not_create_unrelated_metric_families() {
        let proxy_only = Metrics::new(false);
        proxy_only.record_proxy_request(
            RequestLabels {
                outcome: RequestOutcome::Success,
                ..RequestLabels::default()
            },
            Duration::from_millis(1),
        );
        let rendered = proxy_only.render();
        assert!(!rendered.contains(M_LLM_REQUESTS_TOTAL));
        assert!(!rendered.contains(M_LLM_REQUEST_DURATION));

        let llm_only = Metrics::new(false);
        llm_only.record_llm_request(RequestLabels::default(), Duration::from_millis(1));
        let rendered = llm_only.render();
        assert!(!rendered.contains(M_PROXY_REQUESTS_TOTAL));
        assert!(!rendered.contains(M_PROXY_FAILED_REQUESTS_TOTAL));
        assert!(!rendered.contains(M_PROXY_REQUEST_DURATION));
    }

    #[test]
    fn tokens_by_client_records_bounded_client_type() {
        let m = Metrics::new(false);
        // The caller's canonical total is cache-inclusive, so it can exceed
        // input+output: 155 = 100 + 40 + 15 cache tokens (#1002).
        m.record_llm_tokens_by_client("openai-python", "gpt-4o", 100, 40, 155);
        m.record_llm_tokens_by_client("openai-python", "gpt-4o", 10, 0, 10);
        // All-zero is a no-op (keeps the series sparse).
        m.record_llm_tokens_by_client("curl", "gpt-4o", 0, 0, 0);
        let rendered = m.render();
        assert!(rendered.contains(M_LLM_TOKENS_BY_CLIENT_TOTAL));
        assert!(rendered.contains("client_type=\"openai-python\""));
        assert!(rendered.contains("token_type=\"input\""));
        assert!(rendered.contains("token_type=\"output\""));
        assert!(rendered.contains("token_type=\"total\""));
        // input=110, output=40, total=165 — the total series counts the 15
        // cache tokens the input series omits (165 > 110 + 40).
        assert!(rendered
            .lines()
            .any(|l| l.starts_with("aisix_llm_tokens_by_client_total{")
                && l.contains("token_type=\"total\"")
                && l.contains("model=\"gpt-4o\"")
                && l.trim_end().ends_with(" 165")));
        // The all-zero curl call recorded nothing.
        assert!(!rendered.contains("client_type=\"curl\""));
    }

    #[test]
    fn tokens_by_client_splits_series_per_model() {
        // AISIX-Cloud#1044: one client type spending on two models must
        // produce two independent series per token_type, and every series
        // must carry the model label.
        let m = Metrics::new(false);
        m.record_llm_tokens_by_client("claude-code", "claude-sonnet", 100, 60, 160);
        m.record_llm_tokens_by_client("claude-code", "claude-haiku", 30, 10, 40);
        let rendered = m.render();
        let series: Vec<&str> = rendered
            .lines()
            .filter(|l| l.starts_with("aisix_llm_tokens_by_client_total{"))
            .collect();
        // 2 models × 3 token types, all under the same client_type.
        assert_eq!(series.len(), 6);
        assert!(series
            .iter()
            .all(|l| l.contains("client_type=\"claude-code\"") && l.contains("model=")));
        let value_of = |model: &str, token_type: &str| {
            series
                .iter()
                .find(|l| {
                    l.contains(&format!("model=\"{model}\""))
                        && l.contains(&format!("token_type=\"{token_type}\""))
                })
                .and_then(|l| l.trim_end().rsplit(' ').next())
                .map(|v| v.parse::<u64>().unwrap())
        };
        assert_eq!(value_of("claude-sonnet", "input"), Some(100));
        assert_eq!(value_of("claude-sonnet", "output"), Some(60));
        assert_eq!(value_of("claude-sonnet", "total"), Some(160));
        assert_eq!(value_of("claude-haiku", "input"), Some(30));
        assert_eq!(value_of("claude-haiku", "output"), Some(10));
        assert_eq!(value_of("claude-haiku", "total"), Some(40));
    }

    #[test]
    fn client_type_from_user_agent_normalises_to_allowlist() {
        // Known SDKs/tools normalise to a stable bounded label.
        assert_eq!(
            client_type_from_user_agent("OpenAI/Python 1.30.1"),
            "openai-python"
        );
        assert_eq!(
            client_type_from_user_agent("openai-node/4.20.0"),
            "openai-node"
        );
        assert_eq!(
            client_type_from_user_agent("claude-cli/1.2.3"),
            "claude-code"
        );
        assert_eq!(client_type_from_user_agent("curl/8.4.0"), "curl");
        // Version differences collapse to the SAME bounded type — no
        // per-version cardinality blowup.
        assert_eq!(
            client_type_from_user_agent("OpenAI/Python 1.0.0"),
            client_type_from_user_agent("OpenAI/Python 2.99.9"),
        );
        // Empty → unknown; unrecognised → other (the only unbounded inputs
        // both collapse into bounded buckets).
        assert_eq!(client_type_from_user_agent(""), "unknown");
        assert_eq!(client_type_from_user_agent("   "), "unknown");
        assert_eq!(
            client_type_from_user_agent("SomeRandomBespokeClient/9.9"),
            "other"
        );
    }

    /// AISIX-Cloud#1045: coding clients added from source-verified UA
    /// samples (real formats quoted from each product's provider code —
    /// see the issue's evidence table).
    #[test]
    fn client_type_recognises_coding_clients_1045() {
        // Cline v3.56+ sends `Cline/<ver>` on both BYO paths (PR #8872).
        assert_eq!(client_type_from_user_agent("Cline/3.89.2"), "cline");
        // Roo Code OpenAI-compatible path (DEFAULT_HEADERS since PR #5492)
        // and its `roo-code/<ver> (<os>; <arch>)` native-path variant.
        assert_eq!(client_type_from_user_agent("RooCode/3.54.0"), "roo-code");
        assert_eq!(
            client_type_from_user_agent("roo-code/3.54.0 (darwin 23.5.0; arm64) node/20.19.0"),
            "roo-code"
        );
        // Kilo Code ≤5.16.2 (legacy Roo fork lineage).
        assert_eq!(client_type_from_user_agent("Kilo-Code/5.16.2"), "kilocode");
        // Zoo Code — the community continuation of archived Roo Code;
        // marketplace builds carry large patch numbers.
        assert_eq!(
            client_type_from_user_agent("ZooCode/3.71.100268"),
            "zoo-code"
        );
        // Vercel AI SDK default UA — the bucket for AI-SDK-based tools
        // that don't override it (Cline 4.x on node, Kilo 7.x on bun).
        assert_eq!(
            client_type_from_user_agent(
                "ai/6.0.144 ai-sdk/provider-utils/4.0.22 runtime/node.js/26"
            ),
            "vercel-ai-sdk"
        );
        assert_eq!(
            client_type_from_user_agent(
                "ai/6.0.168 ai-sdk/provider-utils/4.0.29 runtime/bun/1.3.6"
            ),
            "vercel-ai-sdk"
        );
        // VS Code Copilot Chat BYOK (nodeFetcher.ts default UA).
        assert_eq!(
            client_type_from_user_agent("GitHubCopilotChat/0.44.0"),
            "github-copilot"
        );
        // Cursor's backend presents a fixed version segment.
        assert_eq!(client_type_from_user_agent("Cursor/1.0"), "cursor");
        // Copilot CLI BYOK exposes only the SDK UA — classified as the
        // SDK, not as Copilot (identification limit recorded in #1045).
        assert_eq!(
            client_type_from_user_agent("OpenAI/JS 5.20.1"),
            "openai-node"
        );
        // opencode prefixes the AI-SDK UA — the product token must win
        // over the `ai-sdk/provider-utils` bucket also present in the UA.
        assert_eq!(
            client_type_from_user_agent(
                "opencode/1.18.3 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
            ),
            "opencode"
        );
        // Qwen Code, OpenAI-compatible path (live-captured format).
        assert_eq!(
            client_type_from_user_agent("QwenCode/0.20.0 (linux; x64)"),
            "qwen-code"
        );
        // Qwen Code's Anthropic path masquerades as Claude Code toward
        // gateways — a KNOWN collision: it lands in `claude-code`.
        assert_eq!(
            client_type_from_user_agent("claude-cli/0.20.0 (external, cli)"),
            "claude-code"
        );
        // Gemini CLI (Gemini-protocol only today; UA still recognised).
        assert_eq!(
            client_type_from_user_agent(
                "GeminiCLI-tui/0.51.0/gemini-3.1-pro-preview (linux; x64; terminal)"
            ),
            "gemini-cli"
        );
        // Crush and Zed set product UAs on their shared HTTP clients.
        assert_eq!(
            client_type_from_user_agent("Charm-Crush/1.0.0 (https://charm.land/crush)"),
            "crush"
        );
        assert_eq!(
            client_type_from_user_agent("Zed/0.198.0 (linux; x86_64)"),
            "zed"
        );
    }

    /// AISIX-Cloud#1045: operator rules outrank built-ins, first match
    /// wins, non-matches fall back to the built-in table, and empty UA
    /// stays `unknown` even under a match-anything rule.
    #[test]
    fn classifier_custom_rules_first_match_then_builtin_fallback() {
        let rule = |pattern: &str, client: &str| aisix_core::ClientTypeRule {
            pattern: pattern.into(),
            client: client.into(),
        };
        let c = ClientTypeClassifier::compile(&[
            rule("^internal-agent/", "internal-agent"),
            // Overlaps the rule above — order decides (first match wins).
            rule("internal", "internal-other"),
            // Re-buckets a UA the built-in table would call "node".
            rule("billing-batcher", "billing-batcher"),
            rule(".*", "catch-all"),
        ])
        .expect("valid rules");

        assert_eq!(c.classify("internal-agent/2.1"), "internal-agent");
        assert_eq!(c.classify("acme-internal-tool/1.0"), "internal-other");
        // Case-insensitive by default.
        assert_eq!(c.classify("Internal-Agent/9.9"), "internal-agent");
        // axios UA would be built-in "node"; the custom rule outranks it.
        assert_eq!(
            c.classify("billing-batcher/3.0 axios/1.6.0"),
            "billing-batcher"
        );
        // Empty/whitespace UA never reaches custom rules — even ".*".
        assert_eq!(c.classify(""), "unknown");
        assert_eq!(c.classify("   "), "unknown");
        // The ".*" rule shadows the built-in fallback for everything else.
        assert_eq!(c.classify("curl/8.4.0"), "catch-all");

        // Without a catch-all, non-matching UAs use the built-in table.
        let c = ClientTypeClassifier::compile(&[rule("^internal-agent/", "internal-agent")])
            .expect("valid rules");
        assert_eq!(c.classify("curl/8.4.0"), "curl");
        assert_eq!(c.classify("SomeRandomBespokeClient/9.9"), "other");

        // Built-ins only (no config) — same behaviour as the free function.
        let c = ClientTypeClassifier::builtin();
        assert_eq!(c.classify("claude-cli/1.2.3"), "claude-code");
        assert_eq!(c.classify(""), "unknown");
    }

    /// AISIX-Cloud#1045: invalid rule sets are rejected at compile (boot)
    /// time — count cap, pattern syntax/length, label charset/length.
    #[test]
    fn classifier_rejects_invalid_rule_sets() {
        let rule = |pattern: &str, client: &str| aisix_core::ClientTypeRule {
            pattern: pattern.into(),
            client: client.into(),
        };
        // Broken regex syntax.
        assert!(ClientTypeClassifier::compile(&[rule("([unclosed", "x")])
            .unwrap_err()
            .contains("invalid pattern"));
        // Empty and oversized patterns.
        assert!(ClientTypeClassifier::compile(&[rule("", "x")]).is_err());
        let oversized = "a".repeat(ClientTypeClassifier::MAX_PATTERN_LEN + 1);
        assert!(ClientTypeClassifier::compile(&[rule(&oversized, "x")]).is_err());
        // Label charset: uppercase, leading dash, spaces, empty, too long.
        for bad in ["Upper", "-lead", "has space", "", "田"] {
            assert!(
                ClientTypeClassifier::compile(&[rule("ok", bad)]).is_err(),
                "label {bad:?} should be rejected"
            );
        }
        let long_label = "a".repeat(ClientTypeClassifier::MAX_CLIENT_LEN + 1);
        assert!(ClientTypeClassifier::compile(&[rule("ok", &long_label)]).is_err());
        // Valid edge labels pass.
        assert!(ClientTypeClassifier::compile(&[rule("ok", "0-tool_v2.beta")]).is_ok());
        // Rule-count cap.
        let too_many: Vec<_> = (0..=ClientTypeClassifier::MAX_RULES)
            .map(|i| rule(&format!("tool-{i}"), "tool"))
            .collect();
        assert!(ClientTypeClassifier::compile(&too_many)
            .unwrap_err()
            .contains("exceed"));
    }

    #[test]
    fn zero_llm_usage_does_not_emit_samples() {
        let m = Metrics::new(false);
        m.record_llm_usage(UsageLabels::default(), LlmUsage::default());
        let rendered = m.render();
        assert!(!rendered.contains(M_LLM_INPUT_TOKENS_TOTAL));
        assert!(!rendered.contains(M_LLM_TOTAL_TOKENS_TOTAL));
    }

    #[test]
    fn in_flight_gauge_returns_to_zero() {
        let m = Metrics::new(false);
        m.increment_proxy_in_flight("/v1/chat/completions", "openai");
        m.decrement_proxy_in_flight("/v1/chat/completions", "openai");
        let rendered = m.render();
        assert!(rendered.contains(M_PROXY_IN_FLIGHT));
        assert!(
            rendered.contains(" 0"),
            "expected gauge to return to zero:\n{rendered}"
        );
    }

    /// AISIX-Cloud#1011: the two SLO series must render as REAL bucketed
    /// histograms (`_bucket{le=…}` + `_sum`/`_count`) — the property that
    /// makes `histogram_quantile()` and cross-instance aggregation work.
    /// Every other `histogram!` series stays a summary (no buckets), so a
    /// bucket-config regression is invisible without this pin.
    #[test]
    fn slo_latency_series_render_as_bucketed_histograms() {
        let m = Metrics::new_with_env_id("env-42");
        let labels = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "gpt-4o",
            provider: "openai",
            status: 200,
            streaming: false,
        };
        m.record_request_e2e_latency(labels, Duration::from_millis(1500));
        m.record_request_ttft(
            LatencyLabels {
                streaming: true,
                ..labels
            },
            Duration::from_millis(80),
        );
        let out = m.render();

        // Real histogram exposition: le-bucketed series + sum/count.
        assert!(
            out.contains("aisix_request_e2e_latency_seconds_bucket"),
            "e2e series must expose _bucket lines:\n{out}"
        );
        assert!(out.contains("aisix_request_e2e_latency_seconds_sum"));
        assert!(out.contains("aisix_request_e2e_latency_seconds_count"));
        assert!(out.contains("aisix_request_ttft_seconds_bucket"));
        assert!(out.contains("le=\"2\""), "configured bucket edges present");

        // The label contract: constant env_id, bucketed status, bounded
        // dims — and none of the per-key/per-user dimensions.
        assert!(out.contains("env_id=\"env-42\""));
        assert!(out.contains("status_class=\"2xx\""));
        assert!(out.contains("streaming=\"false\""));
        assert!(out.contains("streaming=\"true\""));
        // Substring matches, so each NAME label has to be listed beside its
        // id: `contains("user_id")` does not see `user_name`, and these
        // histograms are the one family the id/name pair must NOT reach.
        for high_card in [
            "api_key_id",
            "user_id",
            "user_name",
            "team_id",
            "provider_key_id",
            "provider_key_name",
        ] {
            for line in out.lines().filter(|l| l.contains("aisix_request_")) {
                assert!(
                    !line.contains(high_card),
                    "SLO histogram must not carry {high_card}: {line}"
                );
            }
        }
    }

    /// A 1.5s observation lands in the 2.0 bucket but not the 1.0 bucket —
    /// pins that the configured edges actually apply (a default-bucket
    /// fallback would place them differently or render a summary).
    #[test]
    fn slo_latency_observation_lands_in_the_right_bucket() {
        let m = Metrics::new_with_env_id("");
        m.record_request_e2e_latency(
            LatencyLabels {
                endpoint: "/v1/messages",
                model: "m",
                provider: "anthropic",
                status: 502,
                streaming: false,
            },
            Duration::from_millis(1500),
        );
        let out = m.render();
        let bucket_val = |le: &str| -> u64 {
            out.lines()
                .find(|l| {
                    l.starts_with("aisix_request_e2e_latency_seconds_bucket")
                        && l.contains(&format!("le=\"{le}\""))
                })
                .and_then(|l| l.rsplit(' ').next())
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("no bucket le={le} in:\n{out}"))
        };
        assert_eq!(bucket_val("1"), 0, "1.5s must not land in le=1");
        assert_eq!(bucket_val("2"), 1, "1.5s must land in le=2");
        // Empty env_id collapses to the missing-dimension convention.
        assert!(out.contains("env_id=\"unknown\""));
        assert!(out.contains("status_class=\"5xx\""));
    }

    /// Zero TTFT (never measured) is skipped, and the legacy duration
    /// series keep their summary exposition — no `_bucket` lines appear
    /// for them even after the SLO buckets are configured.
    #[test]
    fn slo_ttft_skips_zero_and_legacy_series_stay_summaries() {
        let m = Metrics::new_with_env_id("e");
        let labels = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "m",
            provider: "openai",
            status: 200,
            streaming: true,
        };
        m.record_request_ttft(labels, Duration::ZERO);
        assert!(
            !m.render().contains("aisix_request_ttft_seconds"),
            "zero TTFT must not be observed"
        );

        m.record_request(
            "openai",
            "m",
            200,
            RequestOutcome::Success,
            Duration::from_millis(100),
        );
        let out = m.render();
        assert!(
            !out.contains("aisix_request_duration_seconds_bucket"),
            "legacy duration series must stay a summary (quantiles), got:\n{out}"
        );
        assert!(out.contains("aisix_request_duration_seconds"));
    }

    /// Every `le` value exposed for `metric`, in exposition order.
    fn rendered_edges(out: &str, metric: &str) -> Vec<String> {
        let prefix = format!("{metric}_bucket");
        out.lines()
            .filter(|l| l.starts_with(&prefix))
            .filter_map(|l| l.split("le=\"").nth(1))
            .filter_map(|rest| rest.split('"').next())
            .map(str::to_owned)
            .collect()
    }

    /// AISIX-Cloud#1226: TTFT no longer borrows the end-to-end latency
    /// edges. The two sets are pinned here because they are a public
    /// metric contract — a dashboard that hardcodes an `le` breaks when
    /// they move, so moving them must be a deliberate edit to this list.
    #[test]
    fn ttft_and_e2e_expose_their_own_default_bucket_edges() {
        let m = Metrics::new_with_env_id("env-1");
        let labels = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "gpt-4o",
            provider: "openai",
            status: 200,
            streaming: true,
        };
        m.record_request_e2e_latency(labels, Duration::from_millis(1500));
        m.record_request_ttft(labels, Duration::from_millis(1500));
        let out = m.render();

        assert_eq!(
            rendered_edges(&out, "aisix_request_e2e_latency_seconds"),
            [
                "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", "30",
                "60", "120", "300", "420", "600", "+Inf",
            ],
        );
        assert_eq!(
            rendered_edges(&out, "aisix_request_ttft_seconds"),
            [
                "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", "30", "60", "120", "300",
                "+Inf",
            ],
        );
    }

    /// The observation-placement contract for TTFT, the metric #1226 is
    /// about: 1.5s lands in `le=2` and not in the edge below it.
    #[test]
    fn ttft_observation_lands_in_the_right_bucket() {
        let m = Metrics::new_with_env_id("env-1");
        m.record_request_ttft(
            LatencyLabels {
                endpoint: "/v1/messages",
                model: "claude",
                provider: "anthropic",
                status: 200,
                streaming: true,
            },
            Duration::from_millis(1500),
        );
        let out = m.render();
        let bucket_val = |le: &str| -> u64 {
            out.lines()
                .find(|l| {
                    l.starts_with("aisix_request_ttft_seconds_bucket")
                        && l.contains(&format!("le=\"{le}\""))
                })
                .and_then(|l| l.rsplit(' ').next())
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("no bucket le={le} in:\n{out}"))
        };
        assert_eq!(bucket_val("1"), 0, "1.5s must not land in le=1");
        assert_eq!(bucket_val("2"), 1, "1.5s must land in le=2");
        assert_eq!(bucket_val("+Inf"), 1);
    }

    /// An override replaces only the metric it names; the other two keep
    /// their defaults. This is the whole point of the per-metric shape —
    /// tuning TTFT must not silently re-cut the e2e or guardrail series.
    #[test]
    fn bucket_override_applies_to_only_the_named_metric() {
        let buckets = HistogramBuckets::from_config(&aisix_core::HistogramBucketsConfig {
            request_ttft: Some(vec![0.5, 3.0]),
            ..Default::default()
        })
        .expect("valid override");
        let m = Metrics::new_with_buckets("env-1", &buckets);
        let labels = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "gpt-4o",
            provider: "openai",
            status: 200,
            streaming: true,
        };
        m.record_request_e2e_latency(labels, Duration::from_millis(1500));
        m.record_request_ttft(labels, Duration::from_millis(1500));
        m.record_guardrail_execution(&aisix_core::GuardrailExecution {
            guardrail_name: "g",
            kind: "keyword",
            phase: "input",
            result: "passed",
            error_type: None,
            elapsed: Duration::from_millis(3),
        });
        let out = m.render();

        assert_eq!(
            rendered_edges(&out, "aisix_request_ttft_seconds"),
            ["0.5", "3", "+Inf"],
        );
        assert_eq!(
            rendered_edges(&out, "aisix_request_e2e_latency_seconds").first(),
            Some(&"0.005".to_string()),
            "e2e keeps its default edges",
        );
        assert_eq!(
            rendered_edges(&out, "aisix_guardrail_latency_seconds").first(),
            Some(&"0.001".to_string()),
            "guardrail keeps its default edges",
        );
    }

    /// Unset fields fall back to the built-in defaults rather than to an
    /// empty (summary-rendering) list.
    #[test]
    fn empty_bucket_config_resolves_to_the_defaults() {
        let resolved =
            HistogramBuckets::from_config(&aisix_core::HistogramBucketsConfig::default())
                .expect("empty config is valid");
        assert_eq!(resolved.request_e2e_latency, DEFAULT_E2E_LATENCY_BUCKETS);
        assert_eq!(resolved.request_ttft, DEFAULT_TTFT_BUCKETS);
        assert_eq!(
            resolved.guardrail_latency,
            DEFAULT_GUARDRAIL_LATENCY_BUCKETS
        );
    }

    /// Boot-fatal validation: every shape Prometheus cannot express, or
    /// that would silently produce a broken exposition, is rejected with
    /// the offending field named.
    #[test]
    fn invalid_bucket_overrides_are_rejected() {
        let cases: [(Vec<f64>, &str); 6] = [
            (vec![], "at least one"),
            (vec![0.1, 0.1], "strictly ascend"),
            (vec![1.0, 0.5], "strictly ascend"),
            (vec![0.0, 1.0], "finite positive"),
            (vec![-1.0, 1.0], "finite positive"),
            (vec![0.1, f64::INFINITY], "finite positive"),
        ];
        for (edges, needle) in cases {
            let err = HistogramBuckets::from_config(&aisix_core::HistogramBucketsConfig {
                request_ttft: Some(edges.clone()),
                ..Default::default()
            })
            .expect_err(&format!("{edges:?} must be rejected"));
            assert!(
                err.contains(needle) && err.contains("buckets.request_ttft"),
                "{edges:?} → {err:?} must name the field and mention {needle:?}",
            );
        }

        let too_many: Vec<f64> = (1..=HistogramBuckets::MAX_EDGES + 1)
            .map(|i| i as f64)
            .collect();
        let err = HistogramBuckets::from_config(&aisix_core::HistogramBucketsConfig {
            guardrail_latency: Some(too_many),
            ..Default::default()
        })
        .expect_err("over-long list must be rejected");
        assert!(err.contains("exceed the limit of 64"), "{err}");
    }

    fn config_metrics_view(source_kind: aisix_core::SourceKind) -> aisix_core::ConfigMetricsView {
        aisix_core::ConfigMetricsView {
            source_kind,
            last_reload_successful: true,
            last_reload_success_ts: Some(1_760_000_000),
            reloads_total: 3,
            reload_failures: std::collections::BTreeMap::new(),
            rejected_by_kind: std::collections::BTreeMap::new(),
            partially_compatible_by_kind: std::collections::BTreeMap::new(),
            stale_served_by_kind: std::collections::BTreeMap::new(),
            observed_revision: Some(42),
            applied_revision: Some(42),
            config_hash: Some("abc123".into()),
            connected: Some(true),
        }
    }

    #[test]
    fn config_status_sync_renders_all_series_in_etcd_mode() {
        let m = Metrics::new(false);
        let mut view = config_metrics_view(aisix_core::SourceKind::Etcd);
        view.last_reload_successful = false;
        view.reload_failures.insert("validate", 2);
        view.rejected_by_kind.insert("models".to_string(), 1);
        view.partially_compatible_by_kind
            .insert("api_keys".to_string(), 3);
        view.stale_served_by_kind.insert("models".to_string(), 1);
        m.sync_config_status(&view);
        let out = m.render();

        assert!(out.contains(&format!("{M_CONFIG_LAST_RELOAD_SUCCESSFUL} 0")));
        assert!(out.contains(M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP));
        assert!(out.contains(&format!("{M_CONFIG_RELOADS_TOTAL} 3")));
        assert!(out.contains(&format!(
            "{M_CONFIG_RELOAD_FAILURES_TOTAL}{{reason=\"validate\"}} 2"
        )));
        assert!(out.contains(&format!(
            "{M_CONFIG_REJECTED_RESOURCES}{{kind=\"models\"}} 1"
        )));
        assert!(out.contains(&format!(
            "{M_CONFIG_PARTIALLY_COMPATIBLE_RESOURCES}{{kind=\"api_keys\"}} 3"
        )));
        assert!(out.contains(&format!(
            "{M_CONFIG_STALE_SERVED_RESOURCES}{{kind=\"models\"}} 1"
        )));
        assert!(out.contains(&format!("{M_CONFIG_OBSERVED_REVISION} 42")));
        assert!(out.contains(&format!("{M_CONFIG_APPLIED_REVISION} 42")));
        assert!(out.contains(&format!("{M_CONFIG_HASH_INFO}{{hash=\"abc123\"}} 1")));
        assert!(out.contains(&format!("{M_CONFIG_SOURCE_CONNECTED} 1")));
    }

    #[test]
    fn config_status_sync_omits_etcd_only_series_in_file_mode() {
        let m = Metrics::new(false);
        let view = config_metrics_view(aisix_core::SourceKind::File);
        m.sync_config_status(&view);
        let out = m.render();
        // Source-agnostic series still present.
        assert!(out.contains(M_CONFIG_LAST_RELOAD_SUCCESSFUL));
        assert!(out.contains(M_CONFIG_RELOADS_TOTAL));
        // Etcd-only series absent in file mode.
        assert!(!out.contains(M_CONFIG_OBSERVED_REVISION));
        assert!(!out.contains(M_CONFIG_APPLIED_REVISION));
        assert!(!out.contains(M_CONFIG_SOURCE_CONNECTED));
    }

    #[test]
    fn config_status_sync_zeroes_stale_hash_and_rejected_labels() {
        let m = Metrics::new(false);
        let mut first = config_metrics_view(aisix_core::SourceKind::Etcd);
        first.config_hash = Some("hash-A".into());
        first.rejected_by_kind.insert("models".to_string(), 2);
        first
            .partially_compatible_by_kind
            .insert("api_keys".to_string(), 1);
        first.stale_served_by_kind.insert("models".to_string(), 1);
        m.sync_config_status(&first);

        // The applied config changes and the models rejection clears.
        let mut second = config_metrics_view(aisix_core::SourceKind::Etcd);
        second.config_hash = Some("hash-B".into());
        // rejected_by_kind empty now.
        m.sync_config_status(&second);

        let out = m.render();
        // Exactly one live hash sample: old zeroed, new is 1.
        assert!(out.contains(&format!("{M_CONFIG_HASH_INFO}{{hash=\"hash-A\"}} 0")));
        assert!(out.contains(&format!("{M_CONFIG_HASH_INFO}{{hash=\"hash-B\"}} 1")));
        // The cleared kind is zeroed, not left at its stale count.
        assert!(out.contains(&format!(
            "{M_CONFIG_REJECTED_RESOURCES}{{kind=\"models\"}} 0"
        )));
        // Same zeroing discipline for the partially-compatible gauge.
        assert!(out.contains(&format!(
            "{M_CONFIG_PARTIALLY_COMPATIBLE_RESOURCES}{{kind=\"api_keys\"}} 0"
        )));
        // And for the stale-served gauge (#871).
        assert!(out.contains(&format!(
            "{M_CONFIG_STALE_SERVED_RESOURCES}{{kind=\"models\"}} 0"
        )));
    }

    /// Issue #408 audit MEDIUM-2: pin every boundary of
    /// `status_bucket` so an off-by-one (e.g. `200..299` excluding
    /// 299) would surface as a CI failure rather than slipping
    /// past as silent re-labelling. Covers all 5 buckets including
    /// the dead-code `3xx` / `other` arms which have no live caller
    /// today.
    #[test]
    fn status_bucket_boundaries_are_inclusive() {
        // 2xx
        assert_eq!(status_bucket(200), "2xx");
        assert_eq!(status_bucket(299), "2xx");
        // 3xx
        assert_eq!(status_bucket(300), "3xx");
        assert_eq!(status_bucket(399), "3xx");
        // 4xx
        assert_eq!(status_bucket(400), "4xx");
        assert_eq!(status_bucket(499), "4xx");
        // 5xx
        assert_eq!(status_bucket(500), "5xx");
        assert_eq!(status_bucket(599), "5xx");
        // out-of-range → other
        assert_eq!(status_bucket(199), "other");
        assert_eq!(status_bucket(600), "other");
        assert_eq!(status_bucket(0), "other");
    }
}
