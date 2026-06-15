---
title: Tool Calling
description: Understand current tool-calling behavior on AISIX AI Gateway, including OpenAI-compatible requests and the current Anthropic translation boundary.
sidebar_position: 22
---

AISIX AI Gateway supports tool-calling workflows on the OpenAI-compatible chat-completions path.

Use this page when you need to decide whether the current gateway behavior is strong enough for agent loops, function calling, or structured tool execution.

## OpenAI-Style Tool Calling

For `POST /v1/chat/completions`, callers can send OpenAI-style `tools` definitions and receive OpenAI-style `tool_calls` in the assistant response.

This is the default integration path for agent frameworks that already speak OpenAI tool-calling semantics.

That includes frameworks and internal application code that expect assistant messages to carry OpenAI-style `tool_calls` entries.

## Cross-Provider Boundary

Tool-calling behavior should be treated as strongest on provider-native OpenAI-compatible chat-completions paths.

Cross-provider tool-calling translations, especially OpenAI-style tool calls against Anthropic-backed models, should be treated conservatively until they are tracked and documented as a separate stable contract.

In practice, that means you should avoid assuming cross-provider tool-calling parity unless a provider combination is explicitly documented and tested.

## What This Means For SDK Users

If your application already uses OpenAI SDKs or OpenAI-style agent frameworks, the safest current path is to use models whose provider-native behavior already matches the OpenAI-compatible tool-calling surface you need.

This keeps your agent loop simpler:

- request shape stays OpenAI-style
- response parsing stays OpenAI-style
- fewer translation assumptions sit between the client and the upstream provider

## Current Boundary

The verified contract is strongest on the OpenAI chat-completions entry point.

Anthropic-style `/v1/messages` translation for non-Anthropic upstreams is currently text-first and should be treated conservatively for richer block types.

## Recommended Usage

- use provider-native OpenAI-compatible models for production tool-calling paths
- treat cross-provider tool-calling as a compatibility area that needs explicit validation in your own environment
- use passthrough only when a provider-native endpoint is required and you are willing to own more client-side behavior

## Troubleshooting

### The model returns plain text instead of tool calls

First verify that the provider/model combination you chose is one whose current caller-visible tool-calling behavior you trust in production.

### The same agent loop works with one model but not another

That usually points to provider-specific capability depth rather than a generic SDK issue.

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Anthropic Messages](anthropic-messages.md)
- [Streaming](streaming.md)
- [OpenAI Client To Anthropic Upstream](../tutorials/openai-client-to-anthropic-upstream.md)
