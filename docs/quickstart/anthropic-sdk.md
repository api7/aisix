---
title: Anthropic SDK Quickstart
description: Configure an Anthropic-compatible client against AISIX AI Gateway and the /v1/messages endpoint.
sidebar_position: 13
---

This quickstart shows how to call AISIX AI Gateway through the Anthropic-style `POST /v1/messages` surface.

Use this page when you want Claude SDK style request and response shapes while still routing through AISIX models and policies.

Use it when:

- your application already uses Anthropic or Claude-style clients
- you want to keep Anthropic-style `messages` requests at the client edge
- you still want AISIX to enforce model aliases, credentials, and policy boundaries

## Before You Start

You should already have:

- a running gateway
- a provider key
- a model alias
- a caller-facing API key

If not, start with [First Model, First Key, First Request](first-model-first-key-first-request.md).

## Gateway Contract

`POST /v1/messages` has two current execution paths:

- Anthropic upstream models: the gateway forwards the Anthropic request to `{api_base}/v1/messages`
- non-Anthropic upstream models: the gateway translates the Anthropic-style body through the internal chat format and returns Anthropic-style JSON or SSE

:::note
The non-Anthropic translation path is currently conservative. Text content blocks are the stable path today. Tool use, thinking blocks, and image blocks on that path are follow-up work.
:::

## Minimal Example

Use your Anthropic-compatible client with the gateway base URL and your AISIX caller key.

```python title="anthropic-sdk-example.py"
from anthropic import Anthropic

client = Anthropic(
    api_key="sk-demo-caller",
    base_url="http://127.0.0.1:3000",
)

message = client.messages.create(
    model="claude-prod",
    max_tokens=128,
    messages=[
        {"role": "user", "content": "Say hello from AISIX."}
    ],
)

print(message.content)
```

## Expected Result

If the gateway can resolve `claude-prod` and the upstream is reachable, the client receives an Anthropic-style message response from the gateway.

At the client edge, the contract remains Anthropic-shaped:

- request goes to `POST /v1/messages`
- `model` is the AISIX model alias
- `messages` and `max_tokens` follow Anthropic-style fields

At the gateway layer, AISIX still handles:

- caller API key authentication
- alias resolution
- `allowed_models` authorization
- upstream credential injection

## Request Shape

The Anthropic-style entry point expects Anthropic-style fields such as:

- `model`
- `messages`
- `max_tokens`
- `stream`

The gateway still authenticates with the AISIX caller key and still resolves `model` as an AISIX model alias.

That means `claude-prod` in the example is an AISIX-managed alias, not necessarily the upstream provider model identifier.

## Streaming

Streaming is supported on the same endpoint.

When the resolved model points to Anthropic, the gateway relays Anthropic SSE events from upstream.

When the resolved model points to a non-Anthropic provider, the gateway emits Anthropic-style SSE events from the translated internal response stream.

This lets Anthropic-style clients keep the same entry point even when the upstream provider differs, but the richest feature coverage remains strongest when the resolved provider is Anthropic.

## Current Boundary

Use this endpoint when you specifically need Anthropic request and response shapes.

If your application already uses OpenAI SDKs, the simpler default remains [OpenAI-Compatible API](../integration/openai-compatible-api.md).

If you depend on advanced Anthropic-only content block behavior, prefer models whose configured provider is actually Anthropic.

## Verification Notes

- `401` means the AISIX caller API key is missing or invalid
- `403` means the key cannot access the requested model alias
- `404` means the model alias is not present in the current snapshot
- errors on `/v1/messages` use the Anthropic-shape envelope `{type:"error", error:{type, message}}` — see [Anthropic Messages — Error Shape](../integration/anthropic-messages.md#error-shape) for the gateway's emitted type strings and how they map to Anthropic's canonical set

## Troubleshooting

### The client gets `404`

Check that `model` is the AISIX alias, not just the upstream Anthropic model id.

### The client gets `403`

The caller key is valid, but it is not authorized for that model alias.

### Anthropic-style requests work only partially on non-Anthropic models

That is expected today. The non-Anthropic translation path is currently strongest for text-oriented behavior.

## Related Pages

- [Anthropic Messages](../integration/anthropic-messages.md)
- [Streaming](../integration/streaming.md)
- [First Model, First Key, First Request](first-model-first-key-first-request.md)
- [OpenAI Client To Anthropic Upstream](../tutorials/openai-client-to-anthropic-upstream.md)
