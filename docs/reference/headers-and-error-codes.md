---
title: Headers and Error Codes
description: Reference for current AISIX AI Gateway response headers, auth headers, and error-code boundaries.
sidebar_position: 63
---

This reference explains how to interpret gateway responses in logs,
clients, and automation.

AISIX has more than one response contract. Start by identifying the surface that returned the response, then use the matching header and error guidance.

## Start with the response contract

Use the request path to decide which contract applies.

| Response came from | Error envelope | Read |
| --- | --- | --- |
| OpenAI-compatible proxy routes such as `/v1/chat/completions`, `/v1/embeddings`, `/v1/responses`, audio, images, and rerank | `{"error": {...}}` | [OpenAI-style proxy errors](#openai-style-proxy-errors) |
| Anthropic-style proxy routes such as `/v1/messages` and `/v1/messages/count_tokens` | `{"type":"error","error": {...}}` | [Anthropic-style proxy errors](#anthropic-style-proxy-errors) |
| Provider passthrough routes under `/passthrough/:provider/*rest` | Upstream provider status and body | [Passthrough errors](#passthrough-errors) |
| Standalone Admin API routes under `/admin/*` | `{"error_msg":"..."}` | [Admin error envelope](#admin-error-envelope) |

## Proxy response headers

Operational headers vary by endpoint. Do not treat every header as universal across every `/v1/*` route.

| Header | When to use it |
| --- | --- |
| <nobr><code>x-aisix-call-id</code></nobr> | Appears on chat-completions responses. Use it to correlate one gateway call. |
| <nobr><code>x-aisix-request-id</code></nobr> | Appears on direct passthrough-style endpoints such as messages, responses, rerank, audio, and passthrough. Use it to correlate the proxied request path. |
| <nobr><code>x-aisix-served-by</code></nobr> | Appears on successful chat-completions routing responses. Use it to identify the direct model target that served the request. |
| <nobr><code>x-aisix-cache</code></nobr> | Appears on chat cache hit or miss paths. Use it to check whether the gateway served the response from cache. |
| <nobr><code>x-ratelimit-*</code></nobr> | Appears on successful chat-completions responses when the caller API key has rate limits configured. Use it to inspect request, token, and concurrent limit state where applicable. |
| <nobr><code>Retry-After</code></nobr> | Appears on rate-limit, budget, and all-candidates-unavailable rejections when the gateway has a retry hint. Use it to tell callers when to retry. |

## OpenAI-style proxy errors

Gateway-generated OpenAI-compatible proxy errors use this envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error"
  }
}
```

The `param` and `code` fields are omitted when AISIX has no value for them. Budget denials can include additional budget fields inside the `error` object.

Common gateway-generated `error.type` values are:

| Error type | Typical status | Meaning |
| --- | --- | --- |
| `invalid_api_key` | `401` | Caller authentication is missing or invalid. |
| `permission_denied` | `403` | The caller key is valid but cannot use the requested model. |
| `model_not_found` | `404` | The requested model alias is not configured. |
| `invalid_request_error` | `400` or `413` | The request body or endpoint usage is invalid. Oversized OpenAI-style requests keep this error type while returning `413`. |
| `provider_unavailable` | `503` | The selected upstream provider bridge cannot complete the request. |
| `all_candidates_unavailable` | `503` | Every routing candidate was filtered out or unavailable. |
| `content_filter` | `422` | The request or response was blocked by policy. |
| `billing_error` | `429` | The request was rejected by billing or budget state. |
| `rate_limit_exceeded` | `429` | The request exceeded a configured rate limit. |
| `not_implemented` | `501` | The resolved provider bridge does not implement the requested endpoint. |
| `upstream_error` | Varies, often `502` for upstream server-side failures | The upstream provider returned an error that AISIX rendered through the proxy error contract. |

### How upstream errors are rendered

For upstream provider errors on OpenAI-style routes, AISIX separates client-safe information from operator-only detail.

Upstream `4xx` responses keep the client-visible HTTP class and are rendered through the proxy error contract. Where AISIX can parse the upstream provider error, it preserves or translates provider error semantics into the OpenAI-style envelope.

Upstream `5xx` responses generally collapse into `502` through the bridge mapping path. AISIX does not expose raw upstream `5xx` bodies because they can contain provider-internal detail.

## Anthropic-style proxy errors

`POST /v1/messages` and `POST /v1/messages/count_tokens` use the Anthropic-style error envelope:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "..."
  }
}
```

The nested `error.type` follows Anthropic SDK-compatible status mappings:

| Status | Anthropic `error.type` |
| --- | --- |
| `400` or `422` | `invalid_request_error` |
| `401` | `authentication_error` |
| `403` | `permission_error` |
| `404` | `not_found_error` |
| `413` | `request_too_large` |
| `429` | `rate_limit_error` |
| `503` | `overloaded_error` |
| Other status codes | `api_error` |

See [Anthropic messages](../integration/anthropic-messages.md#error-shape) for examples.

## Passthrough errors

`ANY /passthrough/:provider/*rest` forwards the upstream provider's status code and body unchanged after proxy authentication and provider resolution. See [Provider passthrough](../integration/passthrough.md).

## Proxy status boundaries

Use the error type first when the envelope includes one. The status code tells you the broad boundary, but the error type usually identifies the more precise gateway condition.

| Status | Meaning |
| --- | --- |
| `400` | The request is invalid. |
| `401` | Caller authentication is missing or invalid. |
| `403` | The model is not allowed for the key. |
| `404` | The model alias was not found. |
| `413` | The request body exceeds the configured proxy request body limit. |
| `422` | Content was blocked by policy. |
| `429` | The request hit a rate limit or budget rejection. |
| `501` | The resolved provider bridge does not implement the requested endpoint. |
| `502` | The upstream provider returned a server-side failure or the bridge mapped an upstream failure into the proxy contract. |
| `503` | The provider bridge is unavailable, or every routing candidate was filtered out by runtime status. |

## Admin error envelope

The admin API uses this envelope:

```json
{
  "error_msg": "..."
}
```

Admin status boundaries are:

| Status | Meaning |
| --- | --- |
| `400` | The admin payload is invalid. |
| `401` | Admin authentication is missing or invalid. |
| `404` | The resource was not found. |
| `409` | A resource conflict, such as a duplicate unique field. |
| `500` | A gateway-side admin API failure. |

## Next steps

- [Proxy API reference](proxy-api-reference.md)
- [Admin API reference](/ai-gateway/reference/admin-api)
- [OpenAI-compatible API](../integration/openai-compatible-api.md)
