---
title: Proxy API Reference
description: Reference for AISIX AI Gateway proxy API families and gateway-specific behavior.
sidebar_position: 60
---

AISIX exposes client-facing proxy APIs on the proxy listener. These routes accept the client API format your application already uses, then apply gateway authentication, model resolution, policy checks, and provider dispatch.

Start from the proxy API family that matches your client. For request and response schemas, use the linked integration guide and the upstream API reference for the API format AISIX follows.

:::note Proxy API Schemas
AISIX publishes OpenAPI for the standalone Admin API. The proxy API follows upstream provider request and response formats, so use this reference for gateway behavior and the upstream API reference for provider request formats.
:::

## Choose the Proxy API

Start from the client or API format you already have.

| If your client uses | Send requests to | Read |
| --- | --- | --- |
| OpenAI-compatible chat and model discovery | `/v1/chat/completions`, `/v1/models`, `/v1/completions` | [OpenAI-compatible API](../integration/openai-compatible-api.md) |
| Anthropic Messages API | `/v1/messages`, `/v1/messages/count_tokens` | [Anthropic-style Messages API](../integration/anthropic-messages.md) |
| OpenAI-style endpoint families | Embeddings, responses, images, audio, and rerank routes | [Endpoint guides](#endpoint-guides) |
| Provider-native paths that AISIX does not model | `/passthrough/:provider/*rest` | [Provider passthrough](../integration/passthrough.md) |

Use the matching proxy route when your application needs endpoint-specific behavior. Provider support can differ by adapter, model, and route.

## Gateway Behavior

Where a proxy route follows an upstream provider API format, use the upstream provider API reference for the base request and response schema. Use the AISIX docs for gateway behavior around that provider API, including caller-facing API keys, model aliases, provider target selection, retries, fallback, streaming behavior, cache, guardrails, rate limits, budgets, response headers, and error mapping.

## Endpoint Guides

Endpoint guides cover endpoint-specific behavior and provider support. They are not separate products or setup paths; read them when your application uses that part of the proxy API.

| Page | Read it when |
| --- | --- |
| [Embeddings](../integration/embeddings.md) | You need vector embedding requests through AISIX. |
| [Responses](../integration/responses.md) | You use the OpenAI Responses API format. |
| [Audio](../integration/audio.md) | You need speech or transcription routes. |
| [Images](../integration/images.md) | You need image-generation routes. |
| [Rerank](../integration/rerank.md) | You need reranking routes and provider support details. |
| [Streaming](../integration/streaming.md) | You need SSE behavior, failover behavior, or stream error handling. |
| [Tool calling](../integration/tool-calling.md) | You need tool definitions, tool calls, or translated tool behavior. |

## Authentication

Proxy requests use caller-facing API keys.

Preferred form:

```http
Authorization: Bearer <plaintext-caller-key>
```

Fallback form:

```http
x-api-key: <plaintext-caller-key>
```

The caller key is an AISIX gateway credential. It is not an upstream provider key.

## Route Behavior

`/v1/models` is model discovery for a caller key. It does not expose every callable alias in every case because routing aliases are hidden from discovery.

Routing aliases apply to `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, and `/v1/responses`. Streaming requests use the first selected eligible target and do not fail over mid-stream. Non-streaming requests can fail over to the next eligible routing target on retryable upstream failures.

`/v1/responses` can resolve a routing alias, but it only uses OpenAI-backed targets. If no OpenAI target is available, the gateway rejects the request at that route.

`/v1/messages/count_tokens` can resolve a routing alias, but it only uses Anthropic-backed targets. If no Anthropic target is available, the gateway rejects the request for that route.

`/passthrough/:provider/*rest` is intentionally thinner than first-class modeled routes.

Endpoint support depends on the resolved model's provider and adapter family. See [Provider compatibility](provider-compatibility.md).

## Related Reading

[OpenAI-compatible API](../integration/openai-compatible-api.md) and [Anthropic-style Messages API](../integration/anthropic-messages.md) cover caller-facing endpoint behavior. For error headers and provider support details, see [Headers and error codes](headers-and-error-codes.md) and [Provider compatibility](provider-compatibility.md).
