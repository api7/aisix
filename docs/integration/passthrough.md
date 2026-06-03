---
title: Provider Passthrough
description: Use the raw provider passthrough route when you need an upstream endpoint that AISIX AI Gateway does not natively model.
sidebar_position: 30
---

AISIX AI Gateway exposes `ANY /passthrough/:provider/*rest` as a raw provider passthrough route.

Use this when you need provider-specific endpoints that the gateway does not currently model directly.

This is the escape hatch, not the preferred first choice.

## Current behavior

The passthrough route:

- accepts any HTTP method
- forwards the request body and safe headers to the upstream provider
- strips hop-by-hop headers and the provider key's configured `strip_headers`
- injects provider authentication from the selected provider key
- preserves the query string
- relays upstream status, response body, and safe response headers

Compared with first-class routes, passthrough does much less normalization on your behalf.

## Provider resolution

The `:provider` segment is used to find a configured model whose `provider` matches that value and that the caller key is allowed to access. The gateway uses that model to borrow the provider key and base URL for the passthrough request.

This route is provider-scoped, not model-scoped.

That distinction matters because the route is not choosing a specific model alias the way `/v1/chat/completions` does.

If the selected provider key does not set `api_base`, the gateway uses a known default only for providers with built-in defaults such as OpenAI, Anthropic, Google, and DeepSeek. For other providers, configure `api_base` explicitly.

## Important authorization boundary

Standard proxy authentication still applies. The caller key must be allowed to access at least one configured model for the requested provider before AISIX lends that provider key through passthrough.

Passthrough is still less precise than first-class routes because the path does not name a model alias. If you need strict model-level behavior for a specific model, prefer the gateway's first-class modeled endpoints where possible.

## Example

```shell
curl -sS -X GET "http://127.0.0.1:3000/passthrough/openai/v1/fine_tuning/jobs" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY"
```

## When to use passthrough

- provider-specific APIs not yet exposed as first-class gateway routes
- exploratory integration work
- temporary access while waiting for a native gateway endpoint

Avoid it when:

- you need model-level authorization semantics
- you want the gateway to normalize request or response shapes for you
- a first-class route already exists for the capability you need

## Troubleshooting

### The call authenticates but hits the wrong upstream base

Check which accessible configured model for that provider is being used to borrow the provider key and base URL.

### The request returns `403`

The caller key is valid, but it is not allowed to access any configured model for the requested provider.

### The call returns `400` with no default base URL

Set `api_base` on the provider key. Passthrough does not know defaults for every provider label.

### The route works but bypasses the model-level behavior you expected

That is expected. Passthrough is intentionally thinner than first-class modeled routes.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Provider keys](../configuration/provider-keys.md)
- [Provider compatibility](../reference/provider-compatibility.md)
- [Errors and retries](errors-and-retries.md)
- [Proxy API reference](../reference/proxy-api-reference.md)
