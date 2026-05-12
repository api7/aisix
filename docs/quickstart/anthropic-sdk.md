---
title: Anthropic SDK Quickstart
description: Configure an Anthropic-compatible client against AISIX AI Gateway and the /v1/messages endpoint.
sidebar_position: 13
---

This quickstart shows how to call AISIX AI Gateway through the Anthropic-style `POST /v1/messages` surface.

Use this page when you want Claude SDK style request and response shapes while still routing through AISIX models and policies.

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

## Request Shape

The Anthropic-style entry point expects Anthropic-style fields such as:

- `model`
- `messages`
- `max_tokens`
- `stream`

The gateway still authenticates with the AISIX caller key and still resolves `model` as an AISIX model alias.

## Streaming

Streaming is supported on the same endpoint.

When the resolved model points to Anthropic, the gateway relays Anthropic SSE events from upstream.

When the resolved model points to a non-Anthropic provider, the gateway emits Anthropic-style SSE events from the translated internal response stream.

## Current Boundary

Use this endpoint when you specifically need Anthropic request and response shapes.

If your application already uses OpenAI SDKs, the simpler default remains [OpenAI-Compatible API](../integration/openai-compatible-api.md).

## Verification Notes

- `401` means the AISIX caller API key is missing or invalid
- `403` means the key cannot access the requested model alias
- `404` means the model alias is not present in the current snapshot
- errors still use the gateway's OpenAI-compatible proxy error envelope

## Related Pages

- [Anthropic Messages](../integration/anthropic-messages.md)
- [Streaming](../integration/streaming.md)
- [First Model, First Key, First Request](first-model-first-key-first-request.md)
- [OpenAI Client To Anthropic Upstream](../tutorials/openai-client-to-anthropic-upstream.md)
