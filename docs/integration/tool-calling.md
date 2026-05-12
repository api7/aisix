---
title: Tool Calling
description: Understand current tool-calling behavior on AISIX AI Gateway, including OpenAI-compatible requests and the current Anthropic translation boundary.
sidebar_position: 22
---

AISIX AI Gateway supports tool-calling workflows on the OpenAI-compatible chat-completions path.

## OpenAI-Style Tool Calling

For `POST /v1/chat/completions`, callers can send OpenAI-style `tools` definitions and receive OpenAI-style `tool_calls` in the assistant response.

This is the default integration path for agent frameworks that already speak OpenAI tool-calling semantics.

## Cross-Provider Boundary

Tool-calling behavior should be treated as strongest on provider-native OpenAI-compatible chat-completions paths.

Cross-provider tool-calling translations, especially OpenAI-style tool calls against Anthropic-backed models, should be treated conservatively until they are tracked and documented as a separate stable contract.

## What This Means For SDK Users

If your application already uses OpenAI SDKs or OpenAI-style agent frameworks, the safest current path is to use models whose provider-native behavior already matches the OpenAI-compatible tool-calling surface you need.

## Current Boundary

The verified contract is strongest on the OpenAI chat-completions entry point.

Anthropic-style `/v1/messages` translation for non-Anthropic upstreams is currently text-first and should be treated conservatively for richer block types.

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Anthropic Messages](anthropic-messages.md)
- [Streaming](streaming.md)
- [OpenAI Client To Anthropic Upstream](../tutorials/openai-client-to-anthropic-upstream.md)
