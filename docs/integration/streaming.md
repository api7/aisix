---
title: Streaming
description: Understand streaming behavior on the AISIX AI Gateway proxy surface, including OpenAI-style and Anthropic-style streaming paths.
sidebar_position: 21
---

AISIX AI Gateway supports streaming on its client-facing proxy surface.

Use this page when you need to answer three practical questions:

- which endpoints stream today
- what wire shape the client should expect
- what reliability guarantees are actually documented today

The stable streaming entry points today are:

- `POST /v1/chat/completions` with `stream: true`
- `POST /v1/messages` with `stream: true`
- `POST /v1/responses` when the target model is an OpenAI model

## OpenAI-Style Streaming

For `/v1/chat/completions`, the gateway returns OpenAI-style SSE chunks.

This is the main streaming path used by OpenAI-compatible SDKs and clients.

Typical consumers include:

- the official OpenAI SDKs
- server-side event consumers
- browser or backend code that incrementally renders assistant output

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

The client should expect standard OpenAI-style SSE chunks followed by the usual stream completion semantics from its SDK or SSE parser.

## Anthropic-Style Streaming

For `/v1/messages`, the gateway returns Anthropic-style SSE events.

Current behavior depends on the resolved model provider:

- Anthropic upstream: upstream SSE is passed through
- non-Anthropic upstream: the gateway translates internal chat chunks into Anthropic event types such as `message_start`, `content_block_*`, `message_delta`, and `message_stop`

Use this path when your client already expects Anthropic-style streaming events and you do not want to change that caller-side contract.

## Responses API Streaming

`POST /v1/responses` supports both JSON and streaming SSE, but only for models whose configured provider is `openai`.

Non-OpenAI models receive `400` on this endpoint.

That means `responses` is not a general-purpose multi-provider streaming entry point today.

## Current Reliability Boundary

The current e2e contract pins one important client-visible property:

- if a client aborts a stream mid-response, the gateway should remain healthy and continue serving later requests

:::note
The current docs do not promise partial upstream chunks when the upstream disconnects mid-stream. That path is not yet the stable documented contract.
:::

## When To Use Which Streaming Path

- use `/v1/chat/completions` for the default OpenAI-compatible streaming contract
- use `/v1/messages` when your client is already built around Anthropic-style events
- use `/v1/responses` only when you specifically need the OpenAI Responses API and the resolved provider is OpenAI

## Troubleshooting

### The client hangs waiting for chunks

Check that the request actually includes `stream: true` and that your client is using a streaming-aware API path.

### The stream fails with `400` on `/v1/responses`

The resolved model is likely not an OpenAI provider.

### The stream is interrupted and later requests fail

That would contradict the currently pinned liveness contract and should be treated as a product bug rather than normal behavior.

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Anthropic Messages](anthropic-messages.md)
- [Responses API](responses.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
