---
title: Provider Passthrough
description: Use the provider passthrough route when you need an upstream endpoint that AISIX AI Gateway does not natively model.
sidebar_position: 30
toc_max_heading_level: 2
---

AISIX AI Gateway provides `ANY /passthrough/:provider/*rest` for provider passthrough requests.

Passthrough is available for provider-specific endpoints that the gateway does not model directly. It is a fallback path for narrow cases, not the preferred path for AI traffic already covered by first-class gateway routes.

## Passthrough Behavior

The passthrough route accepts any HTTP method, preserves the query string, injects provider authentication from the selected provider key, and relays the upstream status, response body, and safe response headers.

The gateway forwards the request body and safe headers to the upstream provider. It strips hop-by-hop headers and the provider key's configured `strip_headers` before sending the request.

Compared with first-class routes, passthrough applies fewer request and response transformations for the caller.

The `:provider` segment is used to find a configured model whose `provider` matches that value and that the caller key is allowed to access. The gateway uses that model to select the provider key and base URL for the passthrough request.

This route is provider-scoped, not model-scoped. It does not choose a specific model alias the way `/v1/chat/completions` does.

If the selected provider key does not set `api_base`, the gateway uses a known default only for providers with built-in defaults such as OpenAI, Anthropic, Google, and DeepSeek. For other providers, configure `api_base` explicitly.

Standard proxy authentication still applies. The caller key must be allowed to access at least one configured model for the requested provider before AISIX uses that provider key for passthrough.

Passthrough is still less precise than first-class routes because the path does not name a model alias. If you need strict model-level behavior for a specific model, prefer the gateway's first-class modeled endpoints where possible.

## Send a Passthrough Request

Send passthrough traffic by naming the upstream provider in the path:

```shell
curl -sS -X GET "http://127.0.0.1:3000/passthrough/openai/v1/fine_tuning/jobs" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY"
```

## Use Cases

Passthrough is suitable for provider-specific APIs that are not exposed as first-class gateway routes, exploratory integration work, or temporary access while evaluating whether a native gateway endpoint is required.

Avoid passthrough when you need model-level authorization semantics, want the gateway to normalize request or response formats, or already have a first-class route for the capability.

## Troubleshooting

Use these checks when passthrough authentication succeeds but the upstream target is not the one you expected.

### The Call Authenticates but Hits the Wrong Upstream Base

Check which accessible model for that provider is being used to select the provider key and base URL.

### The Request Returns `403`

The caller key is valid, but it is not allowed to access any configured model for the requested provider.

### The Call Returns `400` with No Default Base URL

Set `api_base` on the provider key. Passthrough does not know defaults for every provider label.

### The Route Works but Bypasses the Model-Level Behavior You Expected

Passthrough applies fewer gateway transformations than first-class modeled routes. Use a modeled endpoint when model-level behavior is required.

## Related Reading

Use first-class routes in [OpenAI-compatible API](openai-compatible-api.md) when possible. Configure upstream credentials and header stripping with [Provider keys](../configuration/provider-keys.md), and check [Provider compatibility](../reference/provider-compatibility.md) before relying on passthrough. For errors and proxy API families, see [Errors and retries](errors-and-retries.md) and [Proxy API reference](../reference/proxy-api-reference.md).
