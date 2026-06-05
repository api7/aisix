---
title: Tool Calling
description: Understand tool-calling behavior on AISIX AI Gateway, including OpenAI-compatible requests and Anthropic translation.
sidebar_position: 23
toc_max_heading_level: 2
---

AISIX AI Gateway supports tool-calling workflows on the OpenAI-compatible chat-completions path and includes targeted translation for Anthropic-style tool definitions.

Applications that depend on function calling, structured tool execution, or tool loops can use the gateway while keeping provider credentials and model routing behind AISIX.

## Send a Tool-Calling Request

For `POST /v1/chat/completions`, callers can send OpenAI-style `tools` definitions and receive OpenAI-style `tool_calls` in the assistant response.

This is the default integration path for clients that already use OpenAI tool-calling semantics, including frameworks and application code that expect assistant messages to carry OpenAI-style `tool_calls` entries and send follow-up `tool` messages with `tool_call_id`.

Use a request like this to verify that the caller key, model alias, and provider path can carry tool definitions through the gateway:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
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
          "description": "Get weather for a city.",
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

When the upstream model chooses a tool, the response remains OpenAI-compatible. Check that `choices[0].message.tool_calls[]` contains the selected tool calls and that `choices[0].finish_reason` is `tool_calls`. Follow-up `tool` messages should use the returned `tool_call_id`.

If the model returns plain text instead, the gateway may still be working. Tool use depends on the upstream model, prompt, and `tool_choice` value.

## Protocol Translation

Tool-calling behavior is strongest when the client protocol and upstream provider protocol already match. AISIX also supports targeted cross-protocol translation. OpenAI-style requests to Anthropic-backed models can translate OpenAI `tools`, `tool_choice`, assistant `tool_calls`, and follow-up `tool` messages into Anthropic Messages API structures.

Anthropic-style `/v1/messages` requests to non-Anthropic upstreams can translate top-level `tools` and `tool_choice` into OpenAI-style function tools. When a non-Anthropic upstream returns OpenAI-style `tool_calls`, AISIX can render them back to Anthropic-style `tool_use` content blocks.

These translations are useful, but they are not a promise of full provider parity. Richer Anthropic content blocks, such as image blocks, thinking blocks, and full tool-result round trips on non-Anthropic upstreams, still need explicit validation for your application.

## Choose a Tool-Calling Path

If your application already uses OpenAI SDKs or OpenAI-style tool frameworks, the safest path is to use `/v1/chat/completions` and models whose provider behavior already matches the OpenAI-compatible tool-calling behavior you need.

This keeps the tool loop simpler: the request format stays OpenAI-style, response parsing stays OpenAI-style, and fewer translation assumptions sit between the client and the upstream provider.

Use provider-native OpenAI-compatible models for the lowest-risk production tool-calling path. Validate cross-provider tool calling with the exact client, provider, model, and stream mode you plan to run. Use passthrough only when a provider-native endpoint is required and you are willing to own more client-side behavior.

Anthropic-style `/v1/messages` translation for non-Anthropic upstreams supports top-level tool definitions and translated `tool_use` output. Richer non-text block types still need validation with the exact provider and model.

## Troubleshooting

Use these checks when tool-call requests complete but the model does not return structured tool calls.

### The Model Returns Plain Text Instead of Tool Calls

First verify that the provider and model combination you chose supports the tool-calling behavior your application needs in production.

### The Same Tool Loop Works with One Model but Not Another

Compare the provider, model, stream mode, and tool schema. Tool-calling depth varies by upstream model and provider family.

## Related Reading

[OpenAI-compatible API](openai-compatible-api.md) covers the default OpenAI-style chat path, and [Anthropic-style Messages API](anthropic-messages.md) covers Anthropic-style requests. For streaming, error handling, and provider support, see [Streaming](streaming.md), [Errors and retries](errors-and-retries.md), and [Provider compatibility](../reference/provider-compatibility.md). To route OpenAI-compatible clients to an Anthropic upstream, see [OpenAI client to Anthropic upstream](../tutorials/openai-client-to-anthropic-upstream.md).
