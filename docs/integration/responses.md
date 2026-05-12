---
title: Responses API
description: Learn how AISIX AI Gateway handles the OpenAI Responses API and its current provider boundary.
sidebar_position: 25
---

AISIX AI Gateway exposes `POST /v1/responses` as a proxy for the OpenAI Responses API.

## Current Provider Boundary

This endpoint is currently available only for models whose configured provider is `openai`.

If the resolved model points to any non-OpenAI provider, the gateway returns `400`.

## Gateway Behavior

For supported models, the gateway:

1. authenticates and authorizes the caller key
2. verifies the model is an OpenAI provider
3. rewrites `model` to the upstream provider model id
4. forwards the request body to the upstream `/v1/responses` endpoint
5. returns JSON or streaming SSE depending on the request

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

## Related Pages

- [Streaming](streaming.md)
- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
