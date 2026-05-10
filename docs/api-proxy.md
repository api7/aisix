# Proxy API

The Proxy API is the public, OpenAI-compatible HTTP surface that
caller traffic targets. It is served on the proxy listener (default
`:3000`). For the operator CRUD surface see
[`api-admin.md`](./api-admin.md).

> **Compatibility goal**: any client SDK that targets OpenAI
> (`openai` Python, `openai-node`, `openai-go`, `instructor`, the
> Anthropic SDK against `/v1/messages`, etc.) should work unchanged
> by repointing `base_url` at aisix.

## 1. Authentication

Every endpoint requires a caller API key, presented as either
`Authorization: Bearer <key>` (preferred) or `x-api-key: <key>`
(fallback for clients that can't set `Authorization`). A bare
`Authorization: <key>` without the `Bearer` scheme is rejected.
The key must exist in the `apikeys` table of the current snapshot.
See [architecture.md §3](./architecture.md#3-configuration-data-plane).

```http
Authorization: Bearer sk-aisix-…
```

Authorization is a *separate* check: the resolved `ApiKey` must list
the requested Model in its `allowed_models` array (or contain the
`"*"` wildcard).

## 2. Error envelope

Errors follow the OpenAI shape so SDK error handlers light up:

```json
{
  "error": {
    "message": "model 'mygpt' not found",
    "type": "model_not_found"
  }
}
```

The `param` and `code` fields are present on the wire only when set —
unset values are omitted from the envelope.

| Status | `type` | `code` | When |
|---|---|---|---|
| 400 | `invalid_request_error` | — | Malformed body, missing `model`, etc. |
| 401 | `invalid_api_key` | — | Missing, malformed, or unknown bearer/`x-api-key` |
| 403 | `permission_denied` | — | Key valid but Model not in `allowed_models` |
| 404 | `model_not_found` | — | `req.model` does not resolve in the snapshot |
| 413 | (axum default) | — | Body exceeds axum's built-in `Json<…>` extractor limit (2 MiB). The `proxy.request_body_limit_bytes` config field is currently unused — see [#193](https://github.com/api7/ai-gateway/issues/193) for the wiring follow-up. |
| 422 | `content_filter` | — | Guardrail rejected request or response content |
| 429 | `rate_limit_exceeded` | — | RPM/TPM/concurrency cap engaged (all three quotas surface here — the gateway does not split concurrency into a separate code) |
| 429 | `billing_error` | `budget_exceeded` | Per-key USD budget exhausted |
| 502 | (per-bridge) | — | Upstream returned 5xx or invalid wire format; `type` comes from the bridge — see [§6](#6-provider-specific-notes) |
| 503 | `provider_unavailable` | — | No bridge registered for the resolved Model's provider |
| 504 | `timeout` | — | Upstream call exceeded its deadline. Surfaced by `BridgeError::Timeout` and mapped through the bridge's `error_type()`. |

Rate-limit rejections carry a `Retry-After: <seconds>` header. Budget
rejections do not — the operator is the one who lifts the cap, so
there is no deterministic retry interval to advertise.

The `code` field is populated only where listed above; otherwise it
is omitted from the envelope.

## 3. Response headers

Header coverage today is uneven across endpoints — converging on a
single canonical request-id header is tracked as a follow-up code
change.

| Header | Meaning | Where emitted today |
|---|---|---|
| `x-aisix-request-id` | Server-issued request UUID. Echo this when filing support tickets. | `/v1/messages`, `/v1/responses`, `/v1/rerank`, `/v1/audio/*`, `/passthrough/*` |
| `x-aisix-call-id` | Same intent as `x-aisix-request-id`; will be retired in favour of it. | `/v1/chat/completions` only |
| `x-aisix-cache` | `hit` if the response came from cache, `miss` otherwise. Absent for streaming responses. | every endpoint that goes through the cache layer |
| `x-ratelimit-limit-{requests,tokens,concurrent}` | Configured caps. | every endpoint that ran the rate-limit middleware |
| `x-ratelimit-remaining-{requests,tokens,concurrent}` | Live counters at end of request. | same as above |
| `x-ratelimit-reset-{requests,tokens}` | Unix timestamp when the window resets. | same as above |
| `Retry-After` | On 429 rate-limit responses only. | proxy 429 path |

`/v1/completions`, `/v1/embeddings`, `/v1/images/generations`, and
`/v1/models` do not currently emit any request-id header. Treat the
absence as a known gap rather than a contract.

## 4. Endpoints

### 4.1 `GET /v1/models`

Returns the OpenAI list shape, filtered to the Models the calling key
is allowed to access. Wildcard `*` keys see every Model.

```bash
curl -H "Authorization: Bearer sk-aisix-…" \
  http://localhost:3000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "my-gpt4", "object": "model", "created": 0, "owned_by": "openai"}
  ]
}
```

### 4.2 `POST /v1/chat/completions`

OpenAI-compatible chat completion. Both streaming and non-streaming
work identically to upstream OpenAI.

**Non-streaming**

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-aisix-…" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-gpt4",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

**Streaming** — set `"stream": true`. The response is `text/event-stream`
with one `data: {chunk-json}` per delta and a final `data: [DONE]`.
Set `"stream_options": {"include_usage": true}` to receive a final
`usage` chunk before `[DONE]`. Callers that need accurate token
totals on streamed completions should set this explicitly — the
gateway forwards the request body to the upstream verbatim and does
not auto-inject the field.

**Tool calls**, **JSON mode**, **vision content blocks**, and
**function-style tool definitions** all pass through unchanged.

**Caching** — non-streaming requests with the same fingerprint
(model + messages + temperature + top_p + max_tokens, plus the
`tools` / `tool_choice` / `response_format` / `seed` / `stop` /
`presence_penalty` / `frequency_penalty` request fields when set)
hit the cache when a `CachePolicy` resource matches. Streaming
responses are never cached. Cache behavior is configured operator-
side via `/admin/v1/cache_policies`; there is no per-request
`Cache-Control` override today.

### 4.3 `POST /v1/completions`

Legacy OpenAI Completions (text-in / text-out). Same auth, error,
and header semantics as chat.

### 4.4 `POST /v1/embeddings`

Forwards to the configured Model's `/v1/embeddings` endpoint.
`input` may be a single string or an array; both pass through.

```json
{"model": "my-embeddings", "input": ["foo", "bar"]}
```

### 4.5 `POST /v1/messages` (Anthropic Messages — any upstream)

Native Anthropic Messages API path. Works against any configured
upstream — Anthropic, OpenAI, Gemini, DeepSeek — based on the
resolved `Model.provider`:

- **Anthropic upstream** — byte-for-byte passthrough. The body is
  forwarded with the `model` field rewritten to the upstream
  Anthropic model id. SSE streamed verbatim. This preserves features
  the gateway-internal `ChatFormat` can't lossily round-trip
  (`cache_control`, thinking blocks, image blocks, tool_use blocks).
- **Non-Anthropic upstream** — the gateway parses the Anthropic body
  into its internal `ChatFormat` (folding `system` into a leading
  system message, concatenating text content blocks), dispatches
  through the `Hub` to the matching `Bridge`, and re-encodes the
  bridge's response as Anthropic JSON or Anthropic SSE
  (`message_start` / `content_block_start` / `content_block_delta` /
  `content_block_stop` / `message_delta` / `message_stop`). The
  `model` field in the response echoes the operator alias the client
  sent, not the underlying upstream id.

Today's translation supports text content blocks. Tool_use, image,
and thinking blocks are scoped to a follow-up; current behavior is
to skip non-text blocks silently on the inbound parse.

### 4.6 `POST /v1/responses` (OpenAI Responses)

Native OpenAI Responses API. OpenAI Models only — non-OpenAI providers
return 400.

### 4.7 `POST /v1/rerank`

Cohere-style rerank. Routed to `{base}/v1/rerank`. The Model's
provider supplies the API key; the request body is forwarded
verbatim after rewriting the `model` field.

### 4.8 `POST /v1/audio/transcriptions` / `translations` / `speech`

Multipart file upload + JSON pass-through. `audio/speech` returns a
binary audio stream; the others return JSON with text + segments.

### 4.9 `POST /v1/images/generations`

OpenAI Images API. Forwarded with the `model` field rewritten.

### 4.10 `ANY /passthrough/{provider}/*rest`

Lowest-overhead escape hatch: aisix injects the configured provider
API key (and Anthropic's `x-api-key` + `anthropic-version` headers
when applicable) and forwards the request verbatim. Useful for
provider endpoints we haven't yet wrapped natively (e.g. OpenAI's
batches API, files, fine-tuning).

```bash
curl -X POST http://localhost:3000/passthrough/openai/v1/batches \
  -H "Authorization: Bearer sk-aisix-…" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

The provider segment must match a configured Model's provider prefix
(`openai`, `anthropic`, `gemini`, `deepseek`). aisix picks the first
Model with that prefix and uses its credentials.

## 5. Streaming protocol details

aisix preserves the upstream SSE wire format byte-for-byte where
possible:

- One `data:` line per chunk, terminated by `\n\n`.
- A keepalive comment (`: ping`) is emitted every 15 s of idle time
  on `/v1/chat/completions` streams to prevent intermediate proxies
  from dropping the connection. `/v1/messages` and `/v1/responses`
  do not currently emit keepalives.
- The terminal `data: [DONE]` is always sent on a clean upstream
  finish, even if the upstream omitted it.
- If the upstream stream terminates abnormally, aisix sends a final
  error chunk and closes the response without `[DONE]`. Client SDKs
  that interpret missing `[DONE]` as an error will surface the right
  error class.

## 6. Provider-specific notes

| Provider | Native endpoint | OpenAI-translated endpoint | Notes |
|---|---|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/responses` | (no translation needed) | Request body forwarded verbatim |
| Anthropic | `/v1/messages` | `/v1/chat/completions` (text content blocks) | Anthropic→Anthropic is a byte-for-byte passthrough that preserves `cache_control`, thinking, image, and tool_use blocks. Cross-provider translation today covers **text content blocks only** — tool_use, image, and thinking blocks are silently dropped on the inbound parse and are scheduled for a follow-up. See §4.5. |
| Gemini | `/v1/chat/completions` (OpenAI-compat endpoint) | (same) | Uses Gemini's OpenAI-compatible base URL with `x-goog-api-key` auth |
| DeepSeek | `/v1/chat/completions` (OpenAI-compat endpoint) | (same) | Uses Bearer auth, OpenAI-compatible payloads |

`/v1/messages` is symmetric to `/v1/chat/completions` for the
**inbound** axis: an Anthropic-SDK client can target an OpenAI /
Gemini / DeepSeek upstream and the gateway translates both
directions (text content blocks today; tool_use / image / thinking
blocks land in a follow-up).

For `gemini` and `deepseek` there is no separate "native" endpoint —
both providers expose OpenAI-compatible APIs and the Bridge is a thin
auth + base-URL wrapper.

## 7. Worked example: OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="sk-aisix-…",
)

# Non-streaming
resp = client.chat.completions.create(
    model="my-gpt4",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="my-gpt4",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## 8. Worked example: Anthropic SDK against `/v1/messages`

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:3000",
    api_key="sk-aisix-…",
)

msg = client.messages.create(
    model="my-claude",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content[0].text)
```

The Anthropic SDK appends `/v1/messages` itself, hence the
`base_url` does *not* include `/v1`.

## 9. Versioning

The proxy surface is versioned at the **path** level (`/v1/...`).
Within a major version aisix follows OpenAI's compatibility rules:

- New optional request fields are accepted and forwarded.
- New optional response fields appear without breaking existing
  parsers.
- Removed fields stay accepted (but ignored) for at least one minor
  version.

When OpenAI ships a v2 surface, aisix will mount it under `/v2/...`
in parallel for at least one release.

## 10. See also

- [`architecture.md`](./architecture.md) — how the data and request
  paths fit together internally.
- [`api-admin.md`](./api-admin.md) — operator CRUD surface.
- The auto-generated OpenAPI spec lives at `/admin/openapi.json` on
  the admin listener (with a Scalar UI at `/admin/openapi-scalar`).
  It is the canonical machine-readable contract; this document is
  the human-readable companion.
