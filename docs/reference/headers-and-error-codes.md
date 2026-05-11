---
title: Headers And Error Codes
description: Reference for current AISIX AI Gateway response headers, auth headers, and error-code boundaries.
sidebar_position: 63
---

## Proxy Response Headers

Current operational headers vary by endpoint:

- `x-aisix-call-id` on chat-completions responses
- `x-aisix-request-id` on several direct passthrough-style endpoints such as messages, responses, rerank, audio, and passthrough
- `x-aisix-cache` on chat cache hit or miss paths
- `Retry-After` on rate-limit-style rejections when applicable

`x-aisix-cache` is currently used on chat cache hit or miss paths.

## Proxy Error Types

Current proxy error `type` values include:

- `invalid_api_key`
- `permission_denied`
- `model_not_found`
- `invalid_request_error`
- `provider_unavailable`
- `content_filter`
- `billing_error`
- `rate_limit_exceeded`

## Proxy Status Boundaries

- `400` invalid request
- `401` missing or invalid caller auth
- `403` model not allowed for the key
- `404` model alias not found
- `422` content blocked by policy
- `429` rate limit or budget rejection
- `503` provider bridge unavailable

## Admin Error Envelope

The admin API uses:

```json
{
  "error_msg": "..."
}
```

Current admin status boundaries include:

- `400`
- `401`
- `404`
- `409`
- `500`

## Related Pages

- [Proxy API Reference](proxy-api-reference.md)
- [Admin API Reference](admin-api-reference.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
