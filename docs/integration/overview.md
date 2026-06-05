---
title: Client API Overview
sidebar_label: Overview
description: Choose the caller-facing API for AISIX AI Gateway clients.
sidebar_position: 19
---

AISIX AI Gateway gives applications a stable caller-facing API while the gateway owns provider credentials, model resolution, routing, and policy.

After the [Quickstart](../quickstart) confirms that the gateway can serve a request, choose the client API format your application already speaks. These integration docs explain gateway behavior around existing API formats; use the provider API reference for the full provider request and response contract.

## Choose a Client API

If your application already uses OpenAI SDKs or OpenAI-compatible HTTP clients, start with the [OpenAI-compatible API](openai-compatible-api.md). In most cases the application changes the base URL, caller key, and model alias while keeping the same request format.

If your application is Anthropic-native, use the [Anthropic-style Messages API](anthropic-messages.md) so the caller can keep `/v1/messages` requests and token-counting behavior.

For OpenAI-style endpoint families, use the endpoint-specific pages for [embeddings](embeddings.md), [responses](responses.md), [audio](audio.md), [images](images.md), and [rerank](rerank.md). Provider support can differ by endpoint family.

For provider routes that are not modeled directly, use [Provider passthrough](passthrough.md). AISIX authenticates the caller, resolves the provider key, and forwards the provider-native route.

The client sends a gateway-issued caller key and a caller-visible model alias. AISIX resolves the provider key and upstream model behind that alias before it forwards the request.

## Stable Caller Behavior

Applications use AISIX as the API base URL and send gateway-issued caller API keys instead of upstream provider credentials. The `model` value is a caller-visible alias, such as `gpt-4o-prod`, and does not need to match the provider's model or deployment ID.

Gateway-generated errors follow the API family the caller used. OpenAI-compatible routes return OpenAI-compatible errors, while Anthropic-style routes return Anthropic-style errors.

## Authentication and Error Formats

OpenAI-compatible routes use `Authorization: Bearer YOUR_CALLER_API_KEY` and return `{"error": {...}}` for gateway-generated failures. Start with `POST /v1/chat/completions` when checking this path.

Anthropic-style routes support `x-api-key: YOUR_CALLER_API_KEY` for Anthropic SDKs. AISIX also accepts bearer auth on this path. Gateway-generated failures return `{"type":"error","error": {...}}`, and the first route to check is `POST /v1/messages`.

Provider passthrough still authenticates the caller at the gateway, then AISIX injects the provider credential before forwarding. After provider resolution, the caller receives the upstream provider status and body.

For route coverage, headers, and error handling, use [Proxy API reference](../reference/proxy-api-reference.md) and [Headers and error codes](../reference/headers-and-error-codes.md).

## Related Reading

Most clients should start with the [OpenAI-compatible API](openai-compatible-api.md) unless the application is already Anthropic-native. For chat behavior, also review [Streaming](streaming.md), [Tool calling](tool-calling.md), and [Errors and retries](errors-and-retries.md). For provider and endpoint support, see [Provider compatibility](../reference/provider-compatibility.md).
