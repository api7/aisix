---
title: Tool Calling
description: Understand current tool-calling behavior on AISIX AI Gateway, including OpenAI-compatible requests and the current Anthropic translation boundary.
sidebar_position: 23
---

AISIX AI Gateway supports tool-calling workflows on the OpenAI-compatible chat-completions path and includes targeted translation for Anthropic-style tool definitions.

This guide explains the current tool-calling behavior for applications that depend on agent loops, function calling, or structured tool execution.

## OpenAI-style tool calling

For `POST /v1/chat/completions`, callers can send OpenAI-style `tools` definitions and receive OpenAI-style `tool_calls` in the assistant response.

This is the default integration path for agent frameworks that already speak OpenAI tool-calling semantics.

That includes frameworks and application code that expect assistant messages to carry OpenAI-style `tool_calls` entries and send follow-up `tool` messages with `tool_call_id`.

Use a request like this to verify that your caller key, model alias, and provider path can carry tool definitions through the gateway:

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris? Use the tool if needed."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

When the upstream model chooses a tool, the response remains OpenAI-shaped. Check for:

- `choices[0].message.tool_calls[]`
- `choices[0].finish_reason` set to `tool_calls`
- a follow-up `tool` message that uses the returned `tool_call_id`

If the model returns plain text instead, the gateway may still be working correctly. Tool use depends on the upstream model, prompt, and `tool_choice` value.

## Anthropic translation boundary

Tool-calling behavior is strongest when the client protocol and upstream provider protocol already match. AISIX also supports several useful cross-protocol translations:

- OpenAI-style requests to Anthropic-backed models translate OpenAI `tools`, `tool_choice`, assistant `tool_calls`, and follow-up `tool` messages into Anthropic Messages API shapes.
- Anthropic-style `/v1/messages` requests to non-Anthropic upstreams translate top-level `tools` and `tool_choice` into OpenAI-style function tools.
- OpenAI-style `tool_calls` returned by a non-Anthropic upstream can be rendered back to Anthropic-style `tool_use` content blocks.

These translations are useful, but they are not a promise of full provider parity. Richer Anthropic content blocks, such as image blocks, thinking blocks, and full tool-result round trips on non-Anthropic upstreams, still need explicit validation for your application.

## What this means for SDK users

If your application already uses OpenAI SDKs or OpenAI-style agent frameworks, the safest current path is to use `/v1/chat/completions` and models whose provider behavior already matches the OpenAI-compatible tool-calling surface you need.

This keeps your agent loop simpler:

- request shape stays OpenAI-style
- response parsing stays OpenAI-style
- fewer translation assumptions sit between the client and the upstream provider

## Current boundary

The verified contract is strongest on the OpenAI-compatible chat-completions entry point.

Anthropic-style `/v1/messages` translation for non-Anthropic upstreams supports top-level tool definitions and translated `tool_use` output, but remains conservative for richer non-text block types.

## Recommended usage

- use provider-native OpenAI-compatible models for the lowest-risk production tool-calling path
- validate cross-provider tool-calling with the exact client, provider, model, and stream mode you plan to run
- use passthrough only when a provider-native endpoint is required and you are willing to own more client-side behavior

## Troubleshooting

### The model returns plain text instead of tool calls

First verify that the provider/model combination you chose is one whose current caller-visible tool-calling behavior you trust in production.

### The same agent loop works with one model but not another

That usually points to provider-specific capability depth rather than a generic SDK issue.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Anthropic-style Messages API](anthropic-messages.md)
- [Streaming](streaming.md)
- [Errors and retries](errors-and-retries.md)
- [Provider compatibility](../reference/provider-compatibility.md)
- [OpenAI client to Anthropic upstream](../tutorials/openai-client-to-anthropic-upstream.md)
