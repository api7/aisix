---
slug: /core-concepts/provider-abstraction
title: 'Provider Abstraction: Multi-Provider LLM Proxy'
description: 'AISIX acts as a multi-provider LLM proxy with a unified OpenAI-compatible API. Route traffic to OpenAI, Anthropic, Gemini, or DeepSeek by changing a single model prefix.'
keywords: ['multi-provider LLM proxy', 'OpenAI-compatible gateway', 'LLM provider abstraction', 'AI gateway routing', 'OpenAI Anthropic Gemini proxy']
---

AISIX is an open source, multi-provider LLM proxy that eliminates vendor lock-in by exposing a single OpenAI-compatible API regardless of the upstream provider. It solves the challenge of different provider APIs, authentication, and data formats through a **Provider Abstraction** layer — letting teams switch between OpenAI, Anthropic, Google Gemini, and DeepSeek without changing client code.

## A Unified, OpenAI-Compatible API

The core principle is to present a single, consistent, OpenAI-compatible API to the client, regardless of the upstream service. AISIX standardizes on the OpenAI API as its external interface because it is the industry standard.

As a client, you write your code once and can switch between underlying LLMs (e.g., OpenAI, Google Gemini, DeepSeek) by changing the `model` parameter in your request, without other code changes.

## The `Provider` Trait

Internally, this is achieved through a `Provider` trait (similar to an interface). Each supported LLM service implements this trait, which defines standard methods like `chat_completion()` and `chat_completion_stream()`.

When a request comes in, AISIX inspects the `model` field of the configured Model entity (e.g., `gemini/gemini-1.5-pro-latest`). The `gemini` prefix tells the gateway which provider driver to load. The gateway then calls the `chat_completion()` method on the Gemini provider, which is responsible for:

1.  Translating the OpenAI-formatted request to the Google Gemini API format.
2.  Adding authentication credentials from `provider_config`.
3.  Sending the request to the Gemini API endpoint.
4.  Translating the Gemini API response back to the OpenAI format before sending it to the client.

This translation is seamless and transparent.

## Supported Providers

AISIX has built-in support for several LLM providers. The `model` field in your Model configuration must be prefixed with the correct identifier to use the right driver.

| Provider | `model` Prefix | Upstream API Endpoint (Default) |
| :--- | :--- | :--- |
| OpenAI | `openai/` | `https://api.openai.com/v1` |
| Google Gemini | `gemini/` | `https://generativelanguage.googleapis.com/v1beta` (note: uses `/v1beta` endpoint) |
| DeepSeek | `deepseek/` | `https://api.deepseek.com/v1` |
| Anthropic | `anthropic/` | `https://api.anthropic.com/v1` |

For example, to configure a Model for DeepSeek's chat model, your `model` field would be `deepseek/deepseek-chat`.

This provider-based architecture makes AISIX highly extensible. Adding support for a new LLM provider requires creating a new struct that implements the `Provider` trait, including request/response transformation logic (input normalization, output parsing, error mapping) to ensure compatibility with the OpenAI-compatible API surface.

## Related Docs

- [Quick Start](../getting-started/quick-start.md) — Configure your first LLM model and make a proxied request
- [Model Management](../guides/model-management.md) — Full CRUD reference for creating and updating LLM models
- [Observability](../observability.md) — Monitor per-provider LLM latency and token usage with Prometheus
