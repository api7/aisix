---
title: Errors and Retries
description: Understand proxy error envelopes, upstream error mapping, and retry behavior in AISIX AI Gateway.
sidebar_position: 24
toc_max_heading_level: 2
---

AISIX AI Gateway returns protocol-specific errors to callers. Most proxy endpoints use an OpenAI-compatible error envelope; Anthropic Messages uses an Anthropic-style envelope.

The error type, status code, and retry headers indicate whether a caller should fix the request, change configuration, back off, or retry.

## Error Response Formats

AISIX AI Gateway separates proxy errors, passthrough responses, and admin API errors. Check which API family returned the response before handling the response body.

### Proxy Error Envelope

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

`param` and `code` are omitted when they are not set. Client code should not assume those fields are always present.

`POST /v1/messages` and `POST /v1/messages/count_tokens` use the Anthropic error format instead. See [Anthropic-style Messages API](anthropic-messages.md#error-format).

`ANY /passthrough/:provider/*rest` keeps the upstream status and body after proxy authentication and provider resolution complete. See [Provider passthrough](passthrough.md).

The admin API is separate from the proxy API and uses `{"error_msg": "..."}`. Do not treat admin and proxy errors as the same response format.

### Upstream Errors

Upstream-originated errors are rendered differently from gateway-generated errors.

For upstream `4xx` responses, AISIX preserves the client-visible failure class but normalizes the OpenAI-compatible `error.type` to `upstream_error`. When the upstream protocol exposes a useful retry or recovery code, AISIX puts that value in `error.code`.

For example, an Anthropic upstream `rate_limit_error`, a Bedrock `ThrottlingException`, or a Vertex `RESOURCE_EXHAUSTED` response can become an OpenAI-compatible response with:

```json
{
  "error": {
    "message": "...",
    "type": "upstream_error",
    "code": "rate_limit_exceeded"
  }
}
```

For upstream `5xx` responses, AISIX returns `502` and suppresses upstream error details that may contain provider account or infrastructure information. Use gateway logs when you need the upstream body for debugging.

### Endpoint-Specific Errors

Some endpoint families have capability-specific failure behavior. Embeddings, completions, and image generation can return `501 not_implemented` when the resolved provider does not support the endpoint. Image generation and Responses return `400` when the resolved model is not an OpenAI provider.

Rerank returns `400` unless the resolved model provider is `openai`, `cohere`, or `jina`. Audio routes forward to the resolved provider base URL and return upstream failures when that provider does not expose the requested OpenAI-style audio route. Provider passthrough follows its own upstream status-and-body relay behavior after proxy authentication and provider resolution.

## Gateway-Generated Failures

Gateway-generated errors use a stable `error.type` taxonomy.

Request and configuration problems are not retryable without a change:

| Status and type | Meaning |
| --- | --- |
| `400 invalid_request_error` | Malformed payload or invalid endpoint usage. |
| `401 invalid_api_key` | Missing, malformed, or unknown caller API key. |
| `403 permission_denied` | Valid key is not allowed to use the resolved model. |
| `404 model_not_found` | Model alias is not available in the proxy configuration. |
| `413 invalid_request_error` | Request body exceeds `proxy.request_body_limit_bytes`. |

Gateway policy and runtime state can produce retryable or conditionally retryable failures:

| Status and type | Meaning |
| --- | --- |
| `422 content_filter` | Guardrail blocks request or response content. |
| `429 rate_limit_exceeded` | Rate limit rejects the request. |
| `429 billing_error` with `code: "budget_exceeded"` | Managed budget check rejects the request. |
| `503 provider_unavailable` | No provider adapter is available for the resolved provider on the direct model path. |
| `503 all_candidates_unavailable` | Every routing target is filtered out by runtime state and the routing model uses `on_all_filtered: fail`. |

`all_candidates_unavailable` includes `Retry-After: 30`. See [Routing and failover](../configuration/routing-and-failover.md#all-targets-filtered-policy).

### Budget Errors

Budget denials are the one gateway-generated path that sets a stable `error.code`:

```json
{
  "error": {
    "message": "budget exceeded for ApiKey \"<id>\"",
    "type": "billing_error",
    "code": "budget_exceeded"
  }
}
```

When the managed control plane returns structured budget detail, the OpenAI envelope can also include fields such as `scope`, `scope_ref`, `limit_usd`, `spent_usd`, `period`, `period_resets_at`, and `retry_after_seconds`.

See [Budgets](../configuration/budgets.md).

## Retry Behavior

The proxy may return `Retry-After` for rate-limit-style failures, budget failures, and routing candidates that are temporarily unavailable.

Use `Retry-After` as the first retry signal when it is present. If your client also has automatic retry logic, prefer the server-provided delay.

Treat `400`, `401`, `403`, and `404` as request or configuration bugs. Do not retry them without changing the request, key, model, or configuration.

Treat `429` as backoff-and-retry territory. Honor `Retry-After` when it is present.

Treat `502` as an upstream or transient provider class. Retry cautiously and consider idempotency, streaming behavior, and client timeout budgets.

Treat `501` as a capability mismatch. Choose a different provider, adapter, or endpoint.

## Troubleshooting

Use these checks when error behavior varies across retries or provider responses.

### The Same Request Sometimes Returns `429`

Inspect caller-key rate limits, model limits, matching rate-limit policies, and managed budget checks.

### The Same Request Returns `502` Only for One Upstream-Backed Model

That usually points to upstream instability, provider-path issues, or a provider endpoint mismatch rather than caller authentication.

### Upstream Errors All Use the Upstream Error Type

OpenAI-compatible proxy responses use `upstream_error` for upstream-originated failures. Use `error.code`, HTTP status, and gateway logs for more specific retry or diagnosis decisions.

## Related Reading

For related endpoint and error references, see [OpenAI-compatible API](openai-compatible-api.md), [Provider passthrough](passthrough.md), and [Headers and error codes](../reference/headers-and-error-codes.md).
