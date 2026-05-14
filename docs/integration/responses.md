---
title: Responses API
description: Learn how AISIX AI Gateway handles the OpenAI Responses API and its current provider boundary.
sidebar_position: 25
---

AISIX AI Gateway exposes `POST /v1/responses` as a proxy for the OpenAI Responses API.

Use this endpoint only when you specifically want the Responses API surface rather than chat completions.

## Current Provider Boundary

This endpoint is currently available only for models whose configured provider is `openai`.

If the resolved model points to any non-OpenAI provider, the gateway returns `400`.

This is a stricter provider boundary than `/v1/chat/completions`.

## Gateway Behavior

For supported models, the gateway:

1. authenticates and authorizes the caller key
2. verifies the model is an OpenAI provider
3. rewrites `model` to the upstream provider model id
4. forwards the request body to the upstream `/v1/responses` endpoint
5. returns JSON or streaming SSE depending on the request

The gateway is acting as a thin proxy here rather than a cross-provider compatibility layer.

## Example

```bash title="Call the Responses API"
curl -sS -X POST http://127.0.0.1:3000/v1/responses \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "input": "Say hello from AISIX."
  }'
```

## When To Use Responses Instead Of Chat Completions

- use `/v1/responses` when your application is already standardized on that OpenAI API surface
- use `/v1/chat/completions` when you want the broadest current compatibility across provider-backed models

## Troubleshooting

### The same alias works for chat completions but not for responses

That usually means the alias resolves to a non-OpenAI provider.

## Related Pages

- [Streaming](streaming.md)
- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
