---
title: Rate Limits
description: Configure per-key request, token, and concurrency limits in AISIX AI Gateway.
sidebar_position: 36
---

AISIX AI Gateway supports rate-limit fields on resources, but current runtime enforcement is centered on the authenticated API key.

## Current Rate-Limit Fields

- `tpm`: tokens per minute
- `tpd`: tokens per day
- `rpm`: requests per minute
- `rpd`: requests per day
- `concurrency`: maximum in-flight requests

All fields are optional. Missing fields mean no limit on that dimension.

## Current Enforcement Boundary

Current enforcement uses the API key's `rate_limit` object.

Example:

```json title="ApiKey rate limits"
{
  "key_hash": "YOUR_CALLER_KEY_HASH",
  "allowed_models": ["gpt-4o-prod"],
  "rate_limit": {
    "rpm": 60,
    "tpm": 100000,
    "concurrency": 5
  }
}
```

The shared quota gate now applies rate-limit checks across the current LLM endpoint set, not only `POST /v1/chat/completions`.

## Response Behavior

When the request is blocked by rate limiting, the proxy returns `429`.

For rate-limit-style rejections that have a retry window, the proxy can also emit `Retry-After`.

Successful non-streaming chat responses also include current `x-ratelimit-*` headers based on the post-dispatch limiter state.

## Important Caveat

`Model.rate_limit` exists in the current schema and admin surface, but the current enforcement path reads limits from the authenticated API key.

Document and operate against `ApiKey.rate_limit` as the reliable current control.

## Related Pages

- [API Keys](api-keys.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
