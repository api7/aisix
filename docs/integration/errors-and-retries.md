---
title: Errors And Retries
description: Understand the shared proxy error envelope, endpoint-specific status boundaries, and retry behavior on AISIX AI Gateway.
sidebar_position: 30
---

AISIX AI Gateway uses a shared proxy error envelope across its client-facing proxy endpoints.

## Error Envelope

The proxy returns an OpenAI-compatible error body:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

In practice, `param` and `code` may be omitted when they are not set.

## Common Status Boundaries

- `400` invalid request payload or endpoint-specific invalid usage
- `401` missing or invalid caller API key
- `403` valid key, but model access is denied
- `404` model alias not found
- `422` content blocked by policy
- `429` rate limit or budget rejection
- `503` no provider bridge registered for the resolved provider

## Upstream Error Mapping

When the upstream returns `4xx`, that client-visible error class is preserved through the proxy mapping.

When the upstream returns `5xx`, the proxy collapses that class to `502`.

## Retry-After

For rate-limit-style rejections, the proxy may return a `Retry-After` header.

Use that header as the first retry signal when present.

## Endpoint-Specific Notes

- `/v1/embeddings`, `/v1/completions`, and `/v1/images/generations` can return `501` with error type `not_implemented` when the resolved provider does not support that endpoint
- `/v1/responses` returns `400` when the resolved model is not an OpenAI provider
- `/passthrough/:provider/*rest` follows its own raw upstream status behavior after proxy auth and provider resolution

## Retry Guidance

Safe retry behavior depends on the failure class:

- retry `429` using backoff and `Retry-After` when present
- retry transient transport or `502` errors carefully with idempotency in mind
- do not retry `400`, `401`, `403`, or `404` without changing the request or configuration

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Provider Passthrough](passthrough.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
