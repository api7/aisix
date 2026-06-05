---
title: Anthropic-Style Messages API
description: Learn how AISIX AI Gateway handles the Anthropic-style /v1/messages endpoint across Anthropic and non-Anthropic upstreams.
sidebar_position: 21
toc_max_heading_level: 2
---

AISIX AI Gateway exposes `POST /v1/messages` as an Anthropic-style proxy entry point.

Use this endpoint for clients that already expect Anthropic request and response formats. If your application is already built around OpenAI-compatible SDKs, start with [OpenAI-compatible API](openai-compatible-api.md) instead.

## Send a Messages Request

Call the gateway proxy listener with a caller-facing AISIX API key. The `model` value is the AISIX model alias, not necessarily the upstream provider model ID.

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/messages" \
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

This endpoint uses the same proxy API key path as the rest of the gateway: authenticate the caller key, resolve the model alias, and enforce `allowed_models`. The caller still uses the gateway API key, not the upstream Anthropic provider key.

`/v1/messages` resolves direct model aliases and routing aliases. Streaming requests use the first selected target and do not fail over mid-stream. Non-streaming requests can fail over to the next routing target on retryable upstream failures.

`/v1/messages/count_tokens` follows the same auth path and can resolve routing aliases, but it only uses Anthropic-backed targets. Non-Anthropic targets are skipped, and a request with no Anthropic target is rejected by the gateway.

## Upstream Execution

The following notes explain how Anthropic-style requests are executed after AISIX resolves the caller-facing model alias.

### Anthropic Upstream

When the resolved model provider is `anthropic`, the gateway forwards the request to `{api_base}/v1/messages`.

The gateway injects `x-api-key` and `anthropic-version`, rewrites `model` to the upstream provider model id, and passes Anthropic SSE through for streaming requests. This path preserves Anthropic-specific request and response details more directly. If you rely on Anthropic-specific semantics, this is the safest path.

### Non-Anthropic Upstream

When the resolved model is not an Anthropic provider, the gateway translates the Anthropic-style request into the gateway chat format, sends it through the resolved provider adapter, and then re-encodes the response as Anthropic-style JSON or SSE.

This path can route Anthropic-style clients to OpenAI-compatible, Bedrock, Vertex, Azure OpenAI, or other configured adapter families. It keeps a stable Anthropic-style client API, but it is not feature-identical to native Anthropic behavior.

## Translation Scope

The non-Anthropic path supports common text and tool-calling translation. Text content is translated into the gateway chat format. Top-level Anthropic `tools` become OpenAI-style function tools, and Anthropic `tool_choice` is translated when the format is recognized. When a non-Anthropic upstream returns OpenAI-style `tool_calls`, AISIX can render them back as Anthropic `tool_use` blocks.

Full tool-result round trips, thinking blocks, and image blocks have limited compatibility on the non-Anthropic path.

If your application depends on those richer content-block types, use an Anthropic-backed model or validate the exact flow in your environment before relying on it.

## Error Format

Proxy errors on `/v1/messages` use the Anthropic-style envelope `{type:"error", error:{type, message}}`. Native Anthropic upstream responses can also carry an optional `request_id` field; the gateway omits it.

The gateway emits these `error.type` strings:

| `error.type` | Status |
| --- | --- |
| `invalid_request_error` | `400`, `422` |
| `authentication_error` | `401` |
| `permission_error` | `403` |
| `not_found_error` | `404` |
| `request_too_large` | `413` |
| `rate_limit_error` | `429` |
| `overloaded_error` | `503` |
| `api_error` | All other `4xx` and `5xx`, including `402`, which Anthropic maps to `billing_error`. |

Gateway timeout failures generally return through the provider adapter rather than as a standalone Anthropic `timeout_error` response.

See Anthropic's [Errors documentation](https://platform.claude.com/docs/en/api/errors) for the provider type list.

## When to Use `/v1/messages`

Use `/v1/messages` when your application already uses Anthropic-style clients or when Anthropic request semantics are more important than OpenAI compatibility. If your application is already standardized on OpenAI SDKs and OpenAI-style tool calling, use the OpenAI-compatible path instead. Prefer an Anthropic-backed model when you depend on Anthropic-specific content block behavior.

## Troubleshooting

Use these checks when Anthropic-style requests behave differently across provider families.

### The Request Works on Anthropic-Backed Models but Behaves Differently on Other Providers

The non-Anthropic translation path is narrower than native Anthropic behavior. Use an Anthropic-backed model for Anthropic-specific content block semantics.

## Related Reading

Configure an Anthropic SDK client with [Anthropic SDK quickstart](../quickstart/anthropic-sdk.md). For Anthropic-style streaming, proxy errors, provider support, and gateway behavior, see [Streaming](streaming.md), [Errors and retries](errors-and-retries.md), [Provider compatibility](../reference/provider-compatibility.md), and [Proxy API reference](../reference/proxy-api-reference.md).
