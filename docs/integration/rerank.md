---
title: Rerank
description: Learn how AISIX AI Gateway proxies rerank requests and how to configure the upstream base URL boundary for rerank providers.
sidebar_position: 28
---

AISIX AI Gateway exposes `POST /v1/rerank` as a rerank proxy endpoint.

## Gateway Behavior

For rerank requests, the gateway:

1. authenticates the caller key
2. resolves the AISIX model alias
3. checks `allowed_models`
4. rewrites `model` to the upstream provider model id
5. forwards the remaining request body verbatim

The gateway builds the upstream target by appending `/v1/rerank` to the configured rerank base URL.

## Configuration Boundary

This endpoint is intended for providers that expose a native rerank surface.

In practice, configure the provider key `api_base` for the provider's rerank endpoint root.

## Example

```bash title="Send a rerank request"
curl -sS -X POST http://127.0.0.1:3000/v1/rerank \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-prod",
    "query": "gateway docs",
    "documents": ["doc a", "doc b", "doc c"]
  }'
```

## Related Pages

- [Provider Keys](../configuration/provider-keys.md)
- [Errors And Retries](errors-and-retries.md)
- [Proxy API Reference](../reference/proxy-api-reference.md)
