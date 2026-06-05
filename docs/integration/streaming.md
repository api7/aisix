---
title: Streaming
description: Understand streaming behavior on AISIX AI Gateway, including OpenAI-style and Anthropic-style streaming paths.
sidebar_position: 22
toc_max_heading_level: 2
---

AISIX AI Gateway supports streaming on its client-facing proxy API. Streaming support depends on the endpoint family, resolved provider, and response format the client expects.

## Streaming Support

Use `/v1/chat/completions` for the default OpenAI-compatible streaming path. Use `/v1/messages` when your client is already built around Anthropic-style events. Anthropic upstreams stream natively on this path; non-Anthropic upstreams stream through translation.

Use `/v1/responses` only when you specifically need the OpenAI Responses API and the resolved provider is OpenAI.

## Streaming Behavior

AISIX forwards streaming responses according to the selected provider path and response format.

### OpenAI-Style Streaming

For `/v1/chat/completions`, the gateway returns OpenAI-style SSE chunks.

This is the main streaming path used by OpenAI-compatible SDKs, SSE consumers, and applications that incrementally render assistant output.

Example request:

```shell
curl -N -X POST "http://127.0.0.1:3000/v1/chat/completions" \
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

The client receives standard OpenAI-style SSE chunks followed by the stream completion semantics from its SDK or SSE parser.

### Anthropic-Style Streaming

For `/v1/messages`, the gateway returns Anthropic-style SSE events.

When the resolved model provider is Anthropic, upstream SSE is passed through. For non-Anthropic upstreams, gateway chat chunks are translated into Anthropic event types such as `message_start`, `content_block_*`, `message_delta`, and `message_stop`.

Use this path when your client already expects Anthropic-style streaming events and you do not want to change client-side parsing.

### Responses API Streaming

`POST /v1/responses` supports both JSON and streaming SSE, but only for models whose configured provider is `openai`.

Non-OpenAI models receive `400` on this endpoint.

That means `responses` is not a general-purpose multi-provider streaming entry point.

## Interrupted Streams

If a client aborts a stream mid-response, the gateway remains healthy and continues serving later requests.

:::note
Do not rely on partial upstream chunks when the upstream disconnects mid-stream unless the endpoint reference states that behavior.
:::

## Troubleshooting

Use these checks when streaming clients wait for chunks or completion markers that never arrive.

### The Client Hangs Waiting for Chunks

Check that the request actually includes `stream: true` and that your client is using a streaming-aware API path.

### `/v1/responses` Streaming Returns `400`

The resolved model is likely not an OpenAI provider.

### The Stream Is Interrupted and Later Requests Fail

Inspect gateway logs and health endpoints. Later requests continue to be served after an interrupted stream.

## Related Reading

[OpenAI-compatible API](openai-compatible-api.md) covers OpenAI-style chat streaming, [Anthropic-style Messages API](anthropic-messages.md) covers Anthropic-style streaming events, and [Responses API](responses.md) covers OpenAI Responses API streaming on OpenAI-backed models. For stream errors and response headers, see [Headers and error codes](../reference/headers-and-error-codes.md).
