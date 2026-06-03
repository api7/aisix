---
title: Errors and Retries
description: Understand proxy error envelopes, upstream error mapping, and retry behavior in AISIX AI Gateway.
sidebar_position: 24
---

AISIX AI Gateway returns protocol-shaped errors to callers. Most proxy endpoints
use an OpenAI-compatible error envelope; Anthropic Messages uses an
Anthropic-shaped envelope.

This guide explains whether a caller should fix the request, change
configuration, back off, or retry.

## Proxy error envelope

Most proxy endpoints return an OpenAI-compatible body:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_api_key",
    "param": null,
    "code": null
  }
}
```

`param` and `code` are omitted when they are not set. Client code should not
assume those fields are always present.

`POST /v1/messages` and `POST /v1/messages/count_tokens` use the Anthropic error
shape instead. See [Anthropic-style Messages API](anthropic-messages.md#error-shape).

`ANY /passthrough/:provider/*rest` keeps the upstream status and body after proxy
authentication and provider resolution complete. See [Provider passthrough](passthrough.md).

The admin API is a separate operator surface and uses `{"error_msg": "..."}`.
Do not treat admin and proxy errors as the same contract.

## Gateway-generated failures

Gateway-generated errors use a stable `error.type` taxonomy.

Request and configuration problems are not retryable without a change:

- `400 invalid_request_error` for malformed payloads or invalid endpoint usage.
- `401 invalid_api_key` for a missing, malformed, or unknown caller API key.
- `403 permission_denied` when a valid key is not allowed to use the resolved
  model.
- `404 model_not_found` when the model alias does not exist in the current
  snapshot.
- `413 invalid_request_error` when the request body exceeds
  `proxy.request_body_limit_bytes`.

Gateway policy and runtime state can produce retryable or conditionally
retryable failures:

- `422 content_filter` when a guardrail blocks request or response content.
- `429 rate_limit_exceeded` when a rate limit rejects the request.
- `429 billing_error` with `code: "budget_exceeded"` when a managed budget check
  rejects the request.
- `503 provider_unavailable` when no provider bridge is registered for the
  resolved provider on the direct-dispatch path.
- `503 all_candidates_unavailable` when every routing target is filtered out by
  runtime state and the routing model uses `on_all_filtered: fail`.

`all_candidates_unavailable` includes `Retry-After: 30`. See
[Routing and failover](../configuration/routing-and-failover.md#all-targets-filtered-policy).

## Budget errors

Budget denials are the one gateway-generated path that sets a stable
`error.code`:

```json
{
  "error": {
    "message": "budget exceeded for ApiKey \"<id>\"",
    "type": "billing_error",
    "code": "budget_exceeded"
  }
}
```

When the managed control plane returns structured budget detail, the OpenAI
envelope can also include fields such as `scope`, `scope_ref`, `limit_usd`,
`spent_usd`, `period`, `period_resets_at`, and `retry_after_seconds`.

See [Budgets](../configuration/budgets.md).

## Upstream errors

Upstream-originated errors are rendered differently from gateway-generated
errors.

For upstream `4xx` responses, AISIX preserves the client-visible failure class
but normalizes the OpenAI-shaped `error.type` to `upstream_error`. When the
upstream protocol exposes a useful retry or recovery code, AISIX puts that value
in `error.code`.

For example, an Anthropic upstream `rate_limit_error`, a Bedrock
`ThrottlingException`, or a Vertex `RESOURCE_EXHAUSTED` response can become an
OpenAI-shaped response with:

```json
{
  "error": {
    "message": "...",
    "type": "upstream_error",
    "code": "rate_limit_exceeded"
  }
}
```

For upstream `5xx` responses, AISIX returns `502` and suppresses upstream error
details that may contain provider-internal information. Operators can inspect
gateway logs for the upstream body.

## Retry-After

The proxy may return `Retry-After` for rate-limit-style failures, budget
failures, and routing candidates that are temporarily unavailable.

Use `Retry-After` as the first retry signal when it is present. If your client
also has automatic retry logic, prefer the server-provided delay.

## Endpoint-specific notes

- `/v1/embeddings`, `/v1/completions`, and `/v1/images/generations` can return
  `501 not_implemented` when the resolved provider does not support that
  endpoint.
- `/v1/images/generations` and `/v1/responses` return `400` when the resolved
  model is not an OpenAI provider.
- `/v1/rerank` returns `400` unless the resolved model provider is `openai`,
  `cohere`, or `jina`.
- `/v1/audio/*` forwards to the resolved provider base URL and returns upstream
  failures when the provider does not expose the requested OpenAI-style audio
  route.
- `/passthrough/:provider/*rest` follows its own upstream status-and-body relay
  behavior after proxy auth and provider resolution.

## Retry guidance

Treat `400`, `401`, `403`, and `404` as request or configuration bugs. Do not
retry them without changing the request, key, model, or configuration.

Treat `429` as backoff-and-retry territory. Honor `Retry-After` when it is
present.

Treat `502` as an upstream or transient provider class. Retry cautiously and
consider idempotency, streaming behavior, and client timeout budgets.

Treat `501` as a capability mismatch. Choose a different provider, adapter, or
endpoint.

## Troubleshooting

### The same request sometimes returns `429`

Inspect caller-key rate limits, model limits, matching rate-limit policies, and
managed budget checks.

### The same request returns `502` only for one upstream-backed model

That usually points to upstream instability, provider-path issues, or a provider
endpoint mismatch rather than caller authentication.

### Upstream errors all show `type: "upstream_error"`

That is expected for OpenAI-shaped proxy responses. Use `error.code`, HTTP
status, and operator logs for more specific retry or diagnosis decisions.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Provider passthrough](passthrough.md)
- [Headers and error codes](../reference/headers-and-error-codes.md)
