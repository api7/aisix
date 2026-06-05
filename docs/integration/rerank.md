---
title: Rerank
description: Learn how AISIX AI Gateway proxies rerank requests for OpenAI, Cohere, and Jina providers.
sidebar_position: 29
toc_max_heading_level: 2
---

AISIX AI Gateway exposes `POST /v1/rerank` as a rerank proxy endpoint for OpenAI, Cohere, and Jina-style rerank providers.

Use this endpoint when rerank calls should use the same caller keys and model aliases as the rest of the gateway.

## Send a Rerank Request

Send the rerank request through the gateway proxy:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/rerank" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-prod",
    "query": "gateway docs",
    "documents": ["doc a", "doc b", "doc c"]
  }'
```

## Provider and Gateway Behavior

For rerank requests, AISIX authenticates the caller key, resolves the AISIX model alias, checks `allowed_models`, rewrites `model` to the upstream provider model id, and forwards the remaining request fields without changing them.

The gateway builds the upstream target with `/v1/rerank`. It tolerates the common case where `api_base` already ends in `/v1`, so you can configure either a provider host or the API root from the provider's reference.

For rerank-capable models, confirm `ProviderKey.api_base` before debugging caller authentication or model allowlists.

The gateway accepts rerank requests only when the resolved model's `provider` is `openai`, `cohere`, or `jina`.

Requests for Anthropic, Gemini, DeepSeek, Bedrock, Vertex, Azure OpenAI, and other providers are rejected with `400` before the provider request.

Voyage AI is not in this provider set even though it exposes a rerank API. Its request and response fields differ from the OpenAI/Cohere/Jina format, so it needs a dedicated adapter before it can be treated as compatible.

For Cohere and Jina, configure the provider key `api_base` for the API root in the provider's reference. If the base URL is wrong, rerank failures are usually configuration mistakes rather than caller-auth issues.

Successful rerank responses are relayed as upstream bytes. The gateway parses the body only to extract usage for telemetry when the provider returns a recognized usage format.

## Troubleshooting

Use these checks when rerank requests reach an unexpected provider route.

### The Request Returns an Upstream `404`

Check the rerank provider base URL first. The gateway targets the provider's `/v1/rerank` route and de-duplicates common `/v1` paste variants, but it does not guess vendor-specific path prefixes beyond that.

### The Request Returns `400` Before Reaching the Provider

Check the model's `provider`. The gateway accepts only `openai`, `cohere`, and `jina` on `/v1/rerank`.

## Related Reading

Configure upstream credentials and base URLs with [Provider keys](../configuration/provider-keys.md), and check provider support in [Provider compatibility](../reference/provider-compatibility.md). For proxy errors and the rerank route, see [Errors and retries](errors-and-retries.md) and [Proxy API reference](../reference/proxy-api-reference.md).
