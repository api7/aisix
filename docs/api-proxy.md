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
`Authorization: Bearer <key>` (preferred) or `Authorization: <key>`
(bare-key fallback for legacy SDKs). The key must exist in the
`apikeys` table of the current snapshot. See
[architecture.md §3](./architecture.md#3-configuration-data-plane).

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
    "type": "model_not_found",
    "param": null,
    "code": null
  }
}
```

| Status | `type` | When |
|---|---|---|
| 400 | `invalid_request_error` | Malformed body, missing `model`, etc. |
| 401 | `authentication_error` | Missing or unknown bearer key |
| 403 | `model_access_forbidden` | Key valid but Model not in `allowed_models` |
| 404 | `model_not_found` | `req.model` does not resolve in the snapshot |
| 413 | `request_too_large` | Body exceeds `proxy.request_body_limit_bytes` (default 10 MB) |
| 422 | `invalid_request_error` | Schema-valid JSON but semantically wrong (e.g. empty `messages`) |
| 429 | `rate_limit_exceeded` / `concurrency_limit_exceeded` / `budget_exceeded` | RPM/TPM/concurrency/budget cap |
| 502 | `provider_error` | Upstream returned 5xx or invalid wire format |
| 503 | `service_unavailable` | No bridge registered for the resolved Model's provider |
| 504 | `request_timeout` | Upstream exceeded `Model.timeout` ms |

For rate-limit and budget errors the response also carries
`Retry-After: <seconds>` (rate limit) or `Retry-After-Seconds-Header`
(budget) headers when known.

## 3. Response headers (every endpoint)

| Header | Meaning |
|---|---|
| `x-aisix-call-id` | Server-issued request UUID. Echo this when filing support tickets. |
| `x-aisix-cache` | `hit` if the response came from cache, `miss` otherwise. Absent for streaming responses. |
| `x-ratelimit-limit-{requests,tokens,concurrent}` | Configured caps. |
| `x-ratelimit-remaining-{requests,tokens,concurrent}` | Live counters at end of request. |
| `x-ratelimit-reset-{requests,tokens}` | Unix timestamp when the window resets. |
| `Retry-After` | On 429 rate-limit responses only. |

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
`usage` chunk before `[DONE]`. aisix injects this automatically when
the request omits the field, so client SDKs get accurate token totals
even in streaming mode.

**Tool calls**, **JSON mode**, **vision content blocks**, and
**function-style tool definitions** all pass through unchanged.

**Caching** — non-streaming requests with the same fingerprint
(model + messages + temperature + top_p + max_tokens) hit the cache.
Override per request with `Cache-Control` header values:

| Header value | Effect |
|---|---|
| `no-store` | Skip cache lookup AND skip storing the response |
| `no-cache` | Skip lookup but still store on success |
| `s-maxage=N` | Override TTL for this entry |

### 4.3 `POST /v1/completions`

Legacy OpenAI Completions (text-in / text-out). Same auth, error,
and header semantics as chat.

### 4.4 `POST /v1/embeddings`

Forwards to the configured Model's `/v1/embeddings` endpoint.
`input` may be a single string or an array; both pass through.

```json
{"model": "my-embeddings", "input": ["foo", "bar"]}
```

### 4.5 `POST /v1/messages` (Anthropic native)

Native Anthropic Messages API path. The body is forwarded with the
`model` field rewritten to the upstream Anthropic model id. Use this
when your client already speaks Anthropic and you want zero
translation overhead.

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
  to prevent intermediate proxies from dropping the connection.
- The terminal `data: [DONE]` is always sent on a clean upstream
  finish, even if the upstream omitted it.
- If the upstream stream terminates abnormally, aisix sends a final
  error chunk and closes the response without `[DONE]`. Client SDKs
  that interpret missing `[DONE]` as an error will surface the right
  error class.

## 6. Provider-specific notes

| Provider | Native endpoint | OpenAI-translated endpoint | Notes |
|---|---|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/responses` | (no translation needed) | aisix auto-injects `stream_options.include_usage = true` |
| Anthropic | `/v1/messages` | `/v1/chat/completions` (full translation) | The Hub maps content blocks ↔ messages, tool_use ↔ tool_calls, system extraction, cache_control passthrough, stop_reason normalisation |
| Gemini | `/v1/chat/completions` (OpenAI-compat endpoint) | (same) | Uses Gemini's OpenAI-compatible base URL with `x-goog-api-key` auth |
| DeepSeek | `/v1/chat/completions` (OpenAI-compat endpoint) | (same) | Uses Bearer auth, OpenAI-compatible payloads |

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
- The auto-generated OpenAPI spec lives at `/openapi` on the admin
  listener. It is the canonical machine-readable contract; this
  document is the human-readable companion.
