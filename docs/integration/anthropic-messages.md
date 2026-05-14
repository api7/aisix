---
title: Anthropic Messages
description: Learn how AISIX AI Gateway handles the Anthropic-style /v1/messages endpoint across Anthropic and non-Anthropic upstreams.
sidebar_position: 23
---

AISIX AI Gateway exposes `POST /v1/messages` as an Anthropic-style proxy entry point.

Use this page when your client already expects Anthropic-style request and response shapes and you want to understand how far that contract currently extends through the gateway.

## Two Current Execution Paths

### Anthropic Upstream

When the resolved model provider is `anthropic`, the gateway forwards the request to `{api_base}/v1/messages`.

The gateway:

- injects `x-api-key`
- injects `anthropic-version`
- rewrites `model` to the upstream provider model id
- passes Anthropic SSE through for streaming requests

This path preserves Anthropic-specific request and response details more directly.

If you rely on Anthropic-specific semantics, this is the safest path.

### Non-Anthropic Upstream

When the resolved model provider is `openai`, `google`, or `deepseek`, the gateway translates the Anthropic-style request into the internal chat format, dispatches through the provider bridge, and then re-encodes the response as Anthropic-style JSON or SSE.

This path is useful for keeping a stable Anthropic-style client edge, but it should not be treated as feature-identical to native Anthropic behavior.

## Current Translation Scope

The current non-Anthropic path is scoped primarily to text content blocks.

Treat these as follow-up work on that path:

- `tool_use` blocks
- thinking blocks
- image blocks

If your application depends on those richer content-block types, prefer a true Anthropic-backed model.

## Authentication And Authorization

This endpoint uses the same proxy API key path as the rest of the gateway:

- authenticate the caller key
- resolve the model alias
- enforce `allowed_models`

The caller still uses the gateway API key, not the upstream Anthropic provider key.

## Error Shape

Even on the Anthropic-style endpoint, proxy errors still use the gateway's OpenAI-compatible error envelope so client-side proxy handling stays consistent.

## When To Use `/v1/messages`

- use it when your application is already Claude-style at the edge
- use it when Anthropic request semantics are more important than OpenAI compatibility
- avoid it when your application is already standardized on OpenAI SDKs and OpenAI-style tool-calling

## Troubleshooting

### The request works on Anthropic-backed models but behaves differently on other providers

That is expected. The non-Anthropic translation path is deliberately narrower than native Anthropic behavior.

## Related Pages

- [Anthropic SDK Quickstart](../quickstart/anthropic-sdk.md)
- [Streaming](streaming.md)
- [Errors And Retries](errors-and-retries.md)
- [Proxy API Reference](../reference/proxy-api-reference.md)
