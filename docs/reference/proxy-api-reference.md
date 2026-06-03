---
title: Proxy API Reference
description: Reference boundaries for the current AISIX AI Gateway proxy surface and client-facing API families.
sidebar_position: 60
---

AISIX exposes client-facing proxy APIs on the proxy listener.

This reference helps you choose the right proxy surface and understand the AISIX-specific compatibility boundaries. For runnable request and response examples, use the linked integration guides.

:::note No generated proxy OpenAPI yet
The AI Gateway repo currently has an OpenAPI source for the standalone Admin API, but not for the proxy API surface. Because most proxy routes intentionally follow upstream provider wire shapes, this page documents AISIX-specific boundaries and links to the integration pages that describe gateway behavior.
:::

## Start with the proxy surface your client needs

Start from the client or API contract you already have.

| If your client uses | Send requests to | Read |
| --- | --- | --- |
| OpenAI-compatible chat and model discovery | `/v1/chat/completions`, `/v1/models`, `/v1/completions` | [OpenAI-compatible API](../integration/openai-compatible-api.md) |
| Anthropic Messages API | `/v1/messages`, `/v1/messages/count_tokens` | [Anthropic-style Messages API](../integration/anthropic-messages.md) |
| Specialized OpenAI-style endpoints | `/v1/embeddings`, `/v1/responses`, `/v1/images/generations`, `/v1/audio/*`, `/v1/rerank` | [Specialized proxy behavior](#specialized-proxy-behavior) |
| Provider-native paths that AISIX does not model | `/passthrough/:provider/*rest` | [Provider passthrough](../integration/passthrough.md) |

Use the matching mounted `/v1/*` route when your application needs endpoint-specific behavior. Provider support can differ by adapter, model, and route.

## What AISIX adds around upstream APIs

Where a proxy route follows an upstream provider wire shape, use the upstream provider API reference for the base request and response schema. Use the AISIX docs for gateway behavior that sits around that provider contract:

- caller-facing API keys
- model aliases and provider target selection
- retries, fallback, and streaming boundaries
- cache, guardrail, rate-limit, and budget enforcement
- AISIX headers and error mapping

## Read the route-specific guides when you need them

These pages cover endpoint-specific behavior and provider support. They are not separate products or setup paths; read them when your application uses that part of the proxy surface.

| Page | Read it when |
| --- | --- |
| [Embeddings](../integration/embeddings.md) | You need vector embedding requests through AISIX. |
| [Responses](../integration/responses.md) | You use the OpenAI Responses API shape. |
| [Audio](../integration/audio.md) | You need speech or transcription routes. |
| [Images](../integration/images.md) | You need image-generation routes. |
| [Rerank](../integration/rerank.md) | You need reranking routes and provider support details. |
| [Streaming](../integration/streaming.md) | You need SSE behavior, failover boundaries, or stream error handling. |
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

## Source of truth

Unlike the Admin API, the proxy API does not currently have a generated OpenAPI document checked into the repo.

Treat this page as the human-readable proxy compatibility guide. The mounted-route list below reflects the current proxy router, but route-specific request and response bodies are documented in the integration pages and in the upstream provider API contracts that AISIX follows.

If a generated proxy OpenAPI source is added later, keep this page as the compatibility guide and let the generated reference carry the exact route and schema contract.

## Important boundaries

`/v1/models` is model discovery for a caller key. It does not expose every callable alias in every case because routing aliases are hidden today.

Routing aliases apply to `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`, and `/v1/responses`. Streaming requests use the first selected eligible target and do not fail over mid-stream. Non-streaming requests can fail over to the next eligible routing target on retryable upstream failures.

`/v1/responses` can resolve a routing alias, but it only dispatches to OpenAI-backed targets. If no OpenAI target is available, the gateway rejects the request at the boundary.

`/v1/messages/count_tokens` can resolve a routing alias, but it only dispatches to Anthropic-backed targets. If no Anthropic target is available, the gateway rejects the request at the boundary.

`/passthrough/:provider/*rest` is intentionally thinner than first-class modeled routes.

Endpoint support depends on the resolved model's provider and adapter family. See [Provider compatibility](provider-compatibility.md).

## Current mounted routes

The proxy router currently mounts these client-facing paths:

```text
GET  /livez
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
POST /v1/images/generations
POST /v1/messages
POST /v1/messages/count_tokens
POST /v1/rerank
POST /v1/responses
POST /v1/audio/transcriptions
POST /v1/audio/translations
POST /v1/audio/speech
ANY  /passthrough/:provider/*rest
```

## Related guides

- [OpenAI-compatible API](../integration/openai-compatible-api.md)
- [Anthropic-style Messages API](../integration/anthropic-messages.md)
- [Headers and error codes](headers-and-error-codes.md)
- [Provider compatibility](provider-compatibility.md)
