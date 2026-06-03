---
title: Provider Compatibility
description: Reference for current proxy endpoint support and provider compatibility boundaries in AISIX AI Gateway.
sidebar_position: 64
---

This reference shows which proxy endpoints can be used with a
provider-backed model.

AISIX has two compatibility layers:

- **Adapter families** decide how chat-style requests are encoded for upstream providers. See [Adapter protocol families](adapters.md).
- **Endpoint gates** decide whether a specific proxy route accepts the resolved model at all.

That distinction matters. A model can work on `/v1/chat/completions` and still be rejected on `/v1/responses`, `/v1/images/generations`, or `/v1/rerank`.

## Start with this quick check

Start here when choosing a route.

| If you need | Use | Current provider boundary |
| --- | --- | --- |
| Broad chat compatibility | `/v1/chat/completions` | OpenAI, Anthropic, Bedrock, Vertex, Azure OpenAI, and OpenAI-compatible providers through their configured adapter. |
| Anthropic-style clients | `/v1/messages` | Anthropic upstreams natively; non-Anthropic upstreams through translation with narrower feature coverage. |
| Streaming chat | `/v1/chat/completions` or `/v1/messages` with `stream: true` | Same provider boundary as the chosen endpoint. Streaming uses the first selected target and does not fail over mid-stream. |
| Embeddings | `/v1/embeddings` | OpenAI-family bridge support. Other bridges return `501 not_implemented` unless they implement embeddings later. |
| OpenAI Responses API | `/v1/responses` | OpenAI provider only. OpenAI-compatible vendors are not enough unless the model's `provider` is `openai`. |
| Image generation | `/v1/images/generations` | OpenAI provider only. |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech` | OpenAI-style upstream audio routes. AISIX forwards the audio shape; it does not translate audio across provider families. |
| Rerank | `/v1/rerank` | OpenAI, Cohere, and Jina provider labels. |
| Provider-native routes | `/passthrough/:provider/*rest` | Any configured provider key, with less gateway normalization. |

## Broad chat routes

`POST /v1/chat/completions` is the broadest proxy route. It accepts OpenAI-shaped caller requests, resolves the model alias, dispatches through the configured provider key, and returns an OpenAI-shaped chat-completions response.

For non-OpenAI upstreams, the provider-facing request is not necessarily OpenAI-shaped. Bedrock, Vertex, Azure OpenAI, and Anthropic-backed models use their own bridge behavior behind the gateway.

Streaming chat uses server-sent events. It follows the same model resolution rules as non-streaming chat, but streaming requests use the first selected target and do not fail over mid-stream.

## Provider-specific routes

Some proxy routes intentionally stay narrow because their upstream API shape is provider-specific.

### OpenAI-only routes

`POST /v1/responses` and `POST /v1/images/generations` require the resolved model to have `provider: "openai"`.

This is stricter than using the `openai` adapter. For example, an OpenAI-compatible vendor can work on `/v1/chat/completions` with `adapter: "openai"` and still be rejected on `/v1/responses` or `/v1/images/generations` if its provider label is not `openai`.

### OpenAI-style forwarding routes

`POST /v1/embeddings` uses the bridge `embed` implementation. The OpenAI bridge implements embeddings today; bridges that keep the default implementation return `501 not_implemented`.

The audio endpoints forward OpenAI-style audio requests to the resolved provider base URL and return the upstream response shape. Use them with upstreams that expose matching OpenAI-style audio routes.

### Rerank

`POST /v1/rerank` bypasses the chat bridge and is keyed on the model's `provider` label. The current accepted provider labels are `openai`, `cohere`, and `jina`.

### Anthropic Messages

`POST /v1/messages` accepts Anthropic models natively and accepts non-Anthropic models through translation. Use this route when the caller is already built around the Anthropic Messages API. For OpenAI-style clients, prefer `/v1/chat/completions`.

## Check these compatibility boundaries

Provider compatibility is not a single yes-or-no question. Check all of these before depending on a path:

**Caller endpoint family**

The caller route determines whether the request enters an OpenAI-compatible, Anthropic-style, rerank, audio, or passthrough path.

**Adapter behind the resolved model**

The adapter determines the upstream wire shape and bridge capability.

**Provider-native versus translated path**

Some routes forward the provider's native shape. Others translate between API families. Translation support is narrower than native forwarding.

**Provider-specific response extensions**

Vendor-specific response extensions beyond the OpenAI envelope are not normalized. Reasoning-style fields can be lifted per key through the `response.reasoning_field` override. See [Provider key schema](provider-key-schema.md#runtime-overrides).

**Usage accounting**

Usage events vary by endpoint and upstream response. For example, non-streaming Responses API requests emit usage when the upstream response includes a recognized `usage` block; streaming Responses API requests are passed through without stream parsing on this path today.

## Featured and community catalog providers

In AISIX Cloud, the catalog distinguishes **featured** providers from community providers. Featured status affects discovery and dashboard presentation only.

Both featured and community providers resolve to one of the adapter families and run through the same bridge path. The self-hosted gateway has no provider catalog and no featured concept; configure `provider`, `adapter`, and `api_base` on each provider key yourself. See [Adapter protocol families](adapters.md#catalog-and-bring-your-own-providers).

## How to use this reference

Start with the integration doc for the caller endpoint family. Then check this page for the endpoint gate. Finally, use [Feature Status](../overview/feature-matrix.md) for the current product boundary.

## Next steps

- [Adapter protocol families](adapters.md) — the five families and how a model resolves to a bridge.
- [Proxy API reference](proxy-api-reference.md)
- [Feature Status](../overview/feature-matrix.md)
- [OpenAI-compatible API](../integration/openai-compatible-api.md)
