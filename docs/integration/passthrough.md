---
title: Provider Passthrough
description: Use the raw provider passthrough route when you need an upstream endpoint that AISIX AI Gateway does not natively model.
sidebar_position: 29
---

AISIX AI Gateway exposes `ANY /passthrough/:provider/*rest` as a raw provider passthrough route.

Use this when you need provider-specific endpoints that the gateway does not currently model directly.

This is the escape hatch, not the preferred first choice.

## Current Behavior

The passthrough route:

- accepts any HTTP method
- forwards the request body and most headers to the upstream provider
- strips the incoming proxy auth header
- injects provider authentication from the configured provider key
- preserves the query string

Compared with first-class routes, passthrough does much less normalization on your behalf.

## Provider Resolution

The `:provider` segment is used to select the first configured model for that provider so the gateway can borrow its provider key and base URL.

This route is provider-scoped, not model-scoped.

That distinction matters because the route is not choosing a specific model alias the way `/v1/chat/completions` does.

## Important Authorization Boundary

Standard proxy authentication still applies, but this route does not enforce per-model authorization beyond validating the proxy API key itself.

If you need strict model-level access control, prefer the gateway's first-class modeled endpoints where possible.

## Example

```bash title="Call a provider-specific passthrough route"
curl -sS -X GET "http://127.0.0.1:3000/passthrough/openai/fine_tuning/jobs" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY"
```

## When To Use Passthrough

- provider-specific APIs not yet exposed as first-class gateway routes
- exploratory integration work
- temporary access while waiting for a native gateway endpoint

Avoid it when:

- you need model-level authorization semantics
- you want the gateway to normalize request or response shapes for you
- a first-class route already exists for the capability you need

## Troubleshooting

### The call authenticates but hits the wrong upstream base

Check which configured provider model is being used to borrow the provider key and base URL.

### The route works but bypasses the model-level behavior you expected

That is expected. Passthrough is intentionally thinner than first-class modeled routes.

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
- [Roadmap](../roadmap.md)
