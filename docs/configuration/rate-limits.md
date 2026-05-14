---
title: Rate Limits
description: Configure per-key request, token, and concurrency limits in AISIX AI Gateway.
sidebar_position: 36
---

AISIX AI Gateway supports rate-limit fields on resources, but current runtime enforcement is centered on the authenticated API key.

Use this page to decide where to put your real limits today and what caller-visible behavior to expect when they trigger.

## Current Rate-Limit Fields

- `tpm`: tokens per minute
- `tpd`: tokens per day
- `rpm`: requests per minute
- `rpd`: requests per day
- `concurrency`: maximum in-flight requests

All fields are optional. Missing fields mean no limit on that dimension.

In practice, most deployments start with:

- `rpm` for request burst control
- `concurrency` for in-flight protection
- optional token limits where usage-based control matters

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

That means rate limits are no longer just a chat-completions concern.

## Response Behavior

When the request is blocked by rate limiting, the proxy returns `429`.

For rate-limit-style rejections that have a retry window, the proxy can also emit `Retry-After`.

Successful non-streaming chat responses also include current `x-ratelimit-*` headers based on the post-dispatch limiter state.

Those headers are useful for debugging and for client-side adaptive throttling.

## Important Caveat

`Model.rate_limit` exists in the current schema and admin surface, but the current enforcement path reads limits from the authenticated API key.

Document and operate against `ApiKey.rate_limit` as the reliable current control.

## Operator Guidance

- put caller-facing safety limits on API keys
- use concurrency limits to protect shared upstream capacity
- treat model-level limit fields as schema surface, not as the current primary enforcement tool

## Troubleshooting

### A caller sees `429` unexpectedly

Inspect the API key's `rate_limit` object before looking at model rows.

### Limits appear to work for chat but not other endpoints

That should not be assumed. The current quota gate is broader than chat-only behavior.

## Related Pages

- [API Keys](api-keys.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
