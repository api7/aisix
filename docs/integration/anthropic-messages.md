---
title: Anthropic-Style Messages API
description: Learn how AISIX AI Gateway handles the Anthropic-style /v1/messages endpoint across Anthropic and non-Anthropic upstreams.
sidebar_position: 21
---

AISIX AI Gateway exposes `POST /v1/messages` as an Anthropic-style proxy entry point.

This guide explains the Anthropic-style endpoint surface for clients that already expect Anthropic request and response shapes. If your application is already built around OpenAI-compatible SDKs, start with [OpenAI-compatible API](openai-compatible-api.md) instead.

## Request pattern

Call the gateway proxy listener with a caller-facing AISIX API key. The `model` value is the AISIX model alias, not necessarily the upstream provider model ID.

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/messages \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-prod",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Say hello from AISIX."}
    ]
  }'
```

For a runnable SDK setup, see [Anthropic SDK quickstart](../quickstart/anthropic-sdk.md).

## Two current execution paths

### Anthropic upstream

When the resolved model provider is `anthropic`, the gateway forwards the request to `{api_base}/v1/messages`.

The gateway:

- injects `x-api-key`
- injects `anthropic-version`
- rewrites `model` to the upstream provider model id
- passes Anthropic SSE through for streaming requests

This path preserves Anthropic-specific request and response details more directly.

If you rely on Anthropic-specific semantics, this is the safest path.

### Non-Anthropic upstream

When the resolved model is not an Anthropic provider, the gateway translates the Anthropic-style request into the gateway chat format, dispatches through the resolved provider key's bridge, and then re-encodes the response as Anthropic-style JSON or SSE.

This can route Anthropic-shaped clients to OpenAI-compatible, Bedrock, Vertex, Azure OpenAI, or other configured adapter families. It is useful for keeping a stable Anthropic-style client edge, but it should not be treated as feature-identical to native Anthropic behavior.

## Current translation scope

The current non-Anthropic path supports the common text and tool-calling bridge:

- text content is translated into the gateway chat format
- top-level Anthropic `tools` are translated into OpenAI-style function tools
- Anthropic `tool_choice` is translated into OpenAI-style tool choice when the shape is recognized
- upstream OpenAI-style `tool_calls` can be rendered back as Anthropic `tool_use` blocks

Treat these as compatibility boundaries on the non-Anthropic path:

- full tool-result round trips
- thinking blocks
- image blocks

If your application depends on those richer content-block types, prefer a true Anthropic-backed model or validate the exact flow in your environment before relying on it.

## Authentication and authorization

This endpoint uses the same proxy API key path as the rest of the gateway:

- authenticate the caller key
- resolve the model alias
- enforce `allowed_models`

The caller still uses the gateway API key, not the upstream Anthropic provider key.

`/v1/messages` resolves direct model aliases and routing aliases. Streaming requests use the first selected target and do not fail over mid-stream. Non-streaming requests can fail over to the next routing target on retryable upstream failures.

`/v1/messages/count_tokens` follows the same auth path and can resolve routing aliases, but it only dispatches to Anthropic-backed targets. Non-Anthropic targets are skipped, and a request with no Anthropic target is rejected at the gateway boundary.

## Error shape

Proxy errors on `/v1/messages` use the Anthropic-shape envelope `{type:"error", error:{type, message}}`. Real Anthropic upstream responses also carry an optional `request_id` field; the gateway omits it.

The gateway emits these `error.type` strings:

- `invalid_request_error` (400, 422)
- `authentication_error` (401)
- `permission_error` (403)
- `not_found_error` (404)
- `request_too_large` (413)
- `rate_limit_error` (429)
- `overloaded_error` (503)
- `api_error` — all other 4xx/5xx (including 402, which Anthropic's canonical spec maps to `billing_error`)

Gateway timeout failures generally surface through the provider bridge rather than as a standalone Anthropic `timeout_error` response.

See Anthropic's [Errors documentation](https://platform.claude.com/docs/en/api/errors) for the canonical type list.

## When to use `/v1/messages`

- use it when your application is already Claude-style at the edge
- use it when Anthropic request semantics are more important than OpenAI compatibility
- avoid it when your application is already standardized on OpenAI SDKs and OpenAI-style tool-calling
- prefer an Anthropic-backed model when you depend on Anthropic-specific content block behavior

## Troubleshooting

### The request works on Anthropic-backed models but behaves differently on other providers

That is expected. The non-Anthropic translation path is deliberately narrower than native Anthropic behavior.

## Next steps

- [Anthropic SDK quickstart](../quickstart/anthropic-sdk.md)
- [Streaming](streaming.md)
- [Errors and retries](errors-and-retries.md)
- [Provider compatibility](../reference/provider-compatibility.md)
- [Proxy API reference](../reference/proxy-api-reference.md)
