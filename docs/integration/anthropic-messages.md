---
title: Anthropic Messages
description: Learn how AISIX AI Gateway handles the Anthropic-style /v1/messages endpoint across Anthropic and non-Anthropic upstreams.
sidebar_position: 23
---

AISIX AI Gateway exposes `POST /v1/messages` as an Anthropic-style proxy entry point.

## Two Current Execution Paths

### Anthropic Upstream

When the resolved model provider is `anthropic`, the gateway forwards the request to `{api_base}/v1/messages`.

The gateway:

- injects `x-api-key`
- injects `anthropic-version`
- rewrites `model` to the upstream provider model id
- passes Anthropic SSE through for streaming requests

This path preserves Anthropic-specific request and response details more directly.

### Non-Anthropic Upstream

When the resolved model provider is `openai`, `gemini`, or `deepseek`, the gateway translates the Anthropic-style request into the internal chat format, dispatches through the provider bridge, and then re-encodes the response as Anthropic-style JSON or SSE.

## Current Translation Scope

The current non-Anthropic path is scoped primarily to text content blocks.

Treat these as follow-up work on that path:

- `tool_use` blocks
- thinking blocks
- image blocks

## Authentication And Authorization

This endpoint uses the same proxy API key path as the rest of the gateway:

- authenticate the caller key
- resolve the model alias
- enforce `allowed_models`

## Error Shape

Even on the Anthropic-style endpoint, proxy errors still use the gateway's OpenAI-compatible error envelope so client-side proxy handling stays consistent.

## Related Pages

- [Anthropic SDK Quickstart](../quickstart/anthropic-sdk.md)
- [Streaming](streaming.md)
- [Errors And Retries](errors-and-retries.md)
- [Proxy API Reference](../reference/proxy-api-reference.md)
