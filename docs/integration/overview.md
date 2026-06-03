---
title: Client APIs Overview
sidebar_label: Overview
description: Choose the right caller-facing API surface for AISIX AI Gateway clients.
sidebar_position: 19
---

AISIX AI Gateway gives applications a caller-facing API contract while the
gateway owns provider credentials, model resolution, routing, and policy.

This section builds on the [Quickstart](../quickstart) after it proves
that the gateway can serve a request. Start by choosing the client API shape
your application already speaks, then move to endpoint-specific details only
when you need them.

## Start with the client surface you already have

| If your client speaks | Start with | Use it when |
| --- | --- | --- |
| OpenAI-compatible chat or model discovery | [OpenAI-compatible API](openai-compatible-api.md) | Existing OpenAI SDKs or HTTP clients should point at AISIX with minimal change. |
| Anthropic Messages | [Anthropic-style Messages API](anthropic-messages.md) | Existing Anthropic SDK clients should use `/v1/messages` or token counting. |
| OpenAI-style specialized endpoints | Endpoint-specific pages for [embeddings](embeddings.md), [responses](responses.md), [audio](audio.md), [images](images.md), and [rerank](rerank.md) | Your application needs a narrower API family with provider-specific support boundaries. |
| Provider-native APIs | [Provider passthrough](passthrough.md) | AISIX should authenticate, resolve the provider key, and forward a route that is not modeled directly. |

The client sends a gateway-issued caller key and a caller-visible model alias.
AISIX resolves the provider key and upstream model behind that alias before it
forwards the request.

## What stays stable for callers

- Applications use AISIX as the API base URL.
- Applications send caller API keys, not upstream provider credentials.
- Applications use model aliases such as `gpt-4o-prod`, not necessarily the
  provider's model or deployment id.
- Gateway-generated errors follow the API surface the caller used.

## What differs by surface

| Surface | Auth header | Error shape | Good first check |
| --- | --- | --- | --- |
| OpenAI-compatible routes | `Authorization: Bearer YOUR_CALLER_API_KEY` | `{"error": {...}}` | `POST /v1/chat/completions` |
| Anthropic-style routes | `x-api-key: YOUR_CALLER_API_KEY` for Anthropic SDKs; bearer auth is also accepted by AISIX | `{"type":"error","error": {...}}` | `POST /v1/messages` |
| Provider passthrough | Gateway caller auth, then provider auth injected by AISIX | Upstream provider status and body | A provider route under `/passthrough/:provider/*rest` |

For exact mounted routes, current headers, and error boundaries, use
[Proxy API reference](../reference/proxy-api-reference.md) and
[Headers and error codes](../reference/headers-and-error-codes.md).

## Recommended reading order

1. Start with [OpenAI-compatible API](openai-compatible-api.md) unless your
   application is already Anthropic-native.
2. Read [Streaming](streaming.md) and [Tool calling](tool-calling.md) if your
   chat workload depends on those behaviors.
3. Move to endpoint-specific pages when you need embeddings, responses, audio,
   images, rerank, or passthrough.
4. Read [Errors and retries](errors-and-retries.md) before putting a client in
   production.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Anthropic-style Messages API](anthropic-messages.md)
- [Errors and retries](errors-and-retries.md)
- [Provider compatibility](../reference/provider-compatibility.md)
