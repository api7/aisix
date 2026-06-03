---
title: Rerank
description: Learn how AISIX AI Gateway proxies rerank requests for OpenAI, Cohere, and Jina providers.
sidebar_position: 29
---

AISIX AI Gateway exposes `POST /v1/rerank` as a rerank proxy endpoint for OpenAI, Cohere, and Jina-style rerank providers.

Use this endpoint when you want to keep rerank calls behind the same caller-key and model-alias contract as the rest of the gateway.

## Gateway behavior

For rerank requests, the gateway:

1. authenticates the caller key
2. resolves the AISIX model alias
3. checks `allowed_models`
4. rewrites `model` to the upstream provider model id
5. forwards the remaining request body verbatim

The gateway builds the upstream target with `/v1/rerank`. It tolerates the common case where `api_base` already ends in `/v1`, so operators can paste either a bare provider host or the provider's documented API root.

That makes the `ProviderKey.api_base` value especially important for rerank-capable models.

## Current provider boundary

The gateway accepts rerank requests only when the resolved model's `provider` is:

- `openai`
- `cohere`
- `jina`

Requests for Anthropic, Gemini, DeepSeek, Bedrock, Vertex, Azure OpenAI, and other providers are rejected with `400` before upstream dispatch.

Voyage AI is not in this set today even though it exposes a rerank API. Its request and response fields differ from the OpenAI/Cohere/Jina shape, so it needs a dedicated adapter before it can be treated as compatible.

For Cohere and Jina, configure the provider key `api_base` for the provider's documented API root. If the base URL is wrong, rerank failures are usually configuration mistakes rather than caller-auth issues.

Successful rerank responses are relayed as upstream bytes. The gateway parses the body only to extract usage for telemetry when the provider returns a recognized usage shape.

## Example

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/rerank \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-prod",
    "query": "gateway docs",
    "documents": ["doc a", "doc b", "doc c"]
  }'
```

## Troubleshooting

### The request returns an upstream `404`

Check the rerank provider base URL first. The gateway targets the provider's `/v1/rerank` route and de-duplicates common `/v1` paste variants, but it does not guess vendor-specific path prefixes beyond that.

### The request returns `400` before reaching the provider

Check the model's `provider`. The current gateway boundary only admits `openai`, `cohere`, and `jina` on `/v1/rerank`.

## Next steps

- [Provider keys](../configuration/provider-keys.md)
- [Provider compatibility](../reference/provider-compatibility.md)
- [Errors and retries](errors-and-retries.md)
- [Proxy API reference](../reference/proxy-api-reference.md)
