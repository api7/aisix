---
slug: /aisix/guides/rate-limiting
title: Rate Limiting
description: Control costs and prevent abuse with AISIX's powerful rate limiting.
---

Rate limiting is a critical feature for managing AI service usage, controlling costs, and protecting your upstream services from abuse. AISIX provides a flexible rate limiting engine that can be applied to both API Keys and Models.

## How Rate Limiting Works

The `RateLimitHook` is a default hook that enforces rate limits. It operates in both the `pre_call` and `post_call` stages:

-   **`pre_call`**: Before forwarding a request, the hook checks if it would exceed the configured **requests-per-minute/day** limit. If so, the request is rejected with a `429 Too Many Requests` error.
-   **`post_call`**: After a successful response is received, the hook inspects the token usage and updates the **tokens-per-minute/day** counters.

## Configuring Rate Limits

Rate limits can be defined in the `rate_limit` field of both `ApiKey` and `Model` entities for granular control.

### Rate Limit Metrics

You can set limits based on five metrics:

| Metric | Description |
| :--- | :--- |
| `rpm` | Requests Per Minute |
| `rpd` | Requests Per Day |
| `tpm` | Tokens Per Minute |
| `tpd` | Tokens Per Day |
| `concurrency` | Request Concurrency |

### Example Configuration

Here is how to configure a rate limit on a `Model` entity. The same structure applies to `ApiKey` entities.

```bash
# Create a model with a rate limit
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer your-strong-admin-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "limited-model",
    "model": "openai/gpt-4.1-mini",
    "provider_config": { "api_key": "..." },
    "rate_limit": {
      "rpm": 100,
      "tpm": 10000,
      "concurrency": 10
    }
  }'
```

This configuration limits the `limited-model` to **100 requests per minute**, **10,000 tokens per minute**, and **10 concurrent requests**.

### Independent Limits

When rate limits are configured on both an `ApiKey` and a `Model` for a request, they are **evaluated independently**. Both limits must be satisfied for the request to proceed.

-   The API Key's limit controls the usage for that client.
-   The Model's limit controls the aggregate usage for that model.

If either limit is exceeded, the request is rejected with a `429` error, and the error message indicates which entity and metric caused the rejection.

:::note[Quota Timing]
Request-based quotas (RPM/RPD) on an API Key are consumed during the `pre_call` stage before model-level checks run. If a subsequent model pre-check rejects the request, the API Key quota has already been decremented.
:::

## Rate Limit Headers

For every request that passes through the `RateLimitHook`, AISIX adds HTTP headers to the response. These headers give the client real-time visibility into their rate limit status, allowing them to self-regulate their request rate.

### Request-Based Limit Headers

| Header | Description |
| :--- | :--- |
| `x-ratelimit-limit-requests` | The total request limit for the current window. |
| `x-ratelimit-remaining-requests` | The number of requests remaining in the current window. |
| `x-ratelimit-reset-requests` | The time remaining until the request limit window resets, in a human-readable format (e.g., `55s`). |

### Concurrency Limit Headers

| Header | Description |
| :--- | :--- |
| `x-ratelimit-limit-concurrent` | The maximum number of concurrent requests allowed. |
| `x-ratelimit-remaining-concurrent` | The number of available concurrent request slots. |

### Token-Based Limit Headers

| Header | Description |
| :--- | :--- |
| `x-ratelimit-limit-tokens` | The total token limit for the current window. |
| `x-ratelimit-remaining-tokens` | The number of tokens remaining in the current window. |
| `x-ratelimit-reset-tokens` | The time remaining until the token limit window resets. |

When limits are on both the API Key and the Model, the headers reflect the **strictest** of the two limits.

## Error Response

When a rate limit is exceeded, AISIX returns a `429 Too Many Requests` error with a JSON body with details:

```json
{
  "error": {
    "message": "Rate limit exceeded for API key ID: my-app. Limited on rpm, current limit: 100, remaining: 0",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

The response also includes a `Retry-After` header indicating how many seconds the client should wait before making another request.
