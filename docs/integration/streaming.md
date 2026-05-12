---
title: Streaming
description: Understand streaming behavior on the AISIX AI Gateway proxy surface, including OpenAI-style and Anthropic-style streaming paths.
sidebar_position: 21
---

AISIX AI Gateway supports streaming on its client-facing proxy surface.

The stable streaming entry points today are:

- `POST /v1/chat/completions` with `stream: true`
- `POST /v1/messages` with `stream: true`
- `POST /v1/responses` when the target model is an OpenAI model

## OpenAI-Style Streaming

For `/v1/chat/completions`, the gateway returns OpenAI-style SSE chunks.

This is the main streaming path used by OpenAI-compatible SDKs and clients.

Example request:

```bash title="Stream chat completions"
curl -N -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Stream a short greeting."}
    ]
  }'
```

## Anthropic-Style Streaming

For `/v1/messages`, the gateway returns Anthropic-style SSE events.

Current behavior depends on the resolved model provider:

- Anthropic upstream: upstream SSE is passed through
- non-Anthropic upstream: the gateway translates internal chat chunks into Anthropic event types such as `message_start`, `content_block_*`, `message_delta`, and `message_stop`

## Responses API Streaming

`POST /v1/responses` supports both JSON and streaming SSE, but only for models whose configured provider is `openai`.

Non-OpenAI models receive `400` on this endpoint.

## Current Reliability Boundary

The current e2e contract pins one important client-visible property:

- if a client aborts a stream mid-response, the gateway should remain healthy and continue serving later requests

:::note
The current docs do not promise partial upstream chunks when the upstream disconnects mid-stream. That path is not yet the stable documented contract.
:::

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Anthropic Messages](anthropic-messages.md)
- [Responses API](responses.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
