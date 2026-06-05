---
title: Responses API
description: Learn how AISIX AI Gateway handles the OpenAI Responses API and provider support.
sidebar_position: 26
toc_max_heading_level: 2
---

AISIX AI Gateway exposes `POST /v1/responses` as a proxy for the OpenAI Responses API.

Use this endpoint only when your application specifically depends on the OpenAI Responses API. If you want the broadest model and provider compatibility, use [OpenAI-compatible chat completions](openai-compatible-api.md) instead.

## Send a Responses Request

Send the Responses API request through the gateway proxy:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/responses" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "input": "Say hello from AISIX."
  }'
```

Use `/v1/responses` when your application is already standardized on that OpenAI API. Use `/v1/chat/completions` when you want the broadest compatibility across provider-backed models.

## Provider and Gateway Behavior

This endpoint is available only for models whose configured provider is `openai`.

If the resolved model points to any non-OpenAI provider, the gateway returns `400`.

This is stricter than using the `openai` adapter. An OpenAI-compatible vendor can work on `/v1/chat/completions` and still be rejected on `/v1/responses` if the model's `provider` is not `openai`.

For supported models, AISIX authenticates and authorizes the caller key, verifies that the resolved model is an OpenAI provider, rewrites `model` to the upstream provider model id, and forwards the body to the upstream `/v1/responses` endpoint. The response returns as JSON or streaming SSE depending on the request.

This path is a provider-specific proxy, not a cross-provider compatibility layer.

For non-streaming successful responses, the gateway records usage when the upstream response includes the Responses API `usage` block. Streaming responses are passed through as SSE; the gateway does not parse the stream for usage on this path.

## Troubleshooting

Use these checks when the same model alias behaves differently across proxy endpoints.

### The Same Alias Works for Chat Completions but Not for Responses

That usually means the alias resolves to a non-OpenAI provider.

### Streaming Works but Usage Is Not Visible in Gateway Analytics

Streaming `/v1/responses` is passed through without stream parsing on this path. Use non-streaming `/v1/responses` when gateway-side usage attribution for this route is required.

## Related Reading

See [Streaming](streaming.md) for SSE behavior, [OpenAI-compatible API](openai-compatible-api.md) for the broader chat-completions path, and [Errors and retries](errors-and-retries.md) for proxy errors and retry behavior.
