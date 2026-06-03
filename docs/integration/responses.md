---
title: Responses API
description: Learn how AISIX AI Gateway handles the OpenAI Responses API and its current provider boundary.
sidebar_position: 26
---

AISIX AI Gateway exposes `POST /v1/responses` as a proxy for the OpenAI Responses API.

Use this endpoint only when your application specifically depends on the OpenAI
Responses API surface. If you want the broadest current model/provider
compatibility, use [OpenAI-compatible chat completions](openai-compatible-api.md)
instead.

## Current provider boundary

This endpoint is currently available only for models whose configured provider is `openai`.

If the resolved model points to any non-OpenAI provider, the gateway returns `400`.

This is stricter than using the `openai` adapter. An OpenAI-compatible vendor can
work on `/v1/chat/completions` and still be rejected on `/v1/responses` if the
model's `provider` is not `openai`.

## Gateway behavior

For supported models, the gateway:

1. authenticates and authorizes the caller key
2. verifies the model is an OpenAI provider
3. rewrites `model` to the upstream provider model id
4. forwards the request body to the upstream `/v1/responses` endpoint
5. returns JSON or streaming SSE depending on the request

The gateway is acting as a thin proxy here rather than a cross-provider compatibility layer.

For non-streaming successful responses, the gateway records usage when the upstream response includes the Responses API `usage` block. Streaming responses are passed through as SSE; the gateway does not parse the stream for usage on this path today.

## Example

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/responses \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "input": "Say hello from AISIX."
  }'
```

## When to use Responses instead of chat completions

- use `/v1/responses` when your application is already standardized on that OpenAI API surface
- use `/v1/chat/completions` when you want the broadest current compatibility across provider-backed models

## Troubleshooting

### The same alias works for chat completions but not for responses

That usually means the alias resolves to a non-OpenAI provider.

### Streaming works but usage is not visible in gateway analytics

That is a current boundary of this endpoint. Use non-streaming `/v1/responses` when gateway-side usage attribution for this route is required.

## Next steps

- [Streaming](streaming.md)
- [OpenAI-compatible API](openai-compatible-api.md)
- [Errors and retries](errors-and-retries.md)
