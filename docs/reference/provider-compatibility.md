---
title: Provider Compatibility
description: Reference for proxy endpoint support and provider compatibility in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 64
---

This reference shows which proxy endpoints can be used with a provider-backed model.

AISIX has two compatibility layers:

| Layer | Purpose |
| --- | --- |
| Adapter families | Decide how chat-style requests are encoded for upstream providers. See [Adapter protocol families](adapters.md). |
| Endpoint gates | Decide whether a specific proxy route accepts the resolved model at all. |

That distinction matters. A model can work on `/v1/chat/completions` and still be rejected on `/v1/responses`, `/v1/images/generations`, or `/v1/rerank`.

## Choose an Endpoint

Use the caller's API format and provider support requirements to choose a route.

| If you need | Use | Provider support |
| --- | --- | --- |
| Broad chat compatibility | `/v1/chat/completions` | OpenAI, Anthropic, Bedrock, Vertex, Azure OpenAI, and OpenAI-compatible providers through their configured adapter. |
| Anthropic-style clients | `/v1/messages` | Anthropic upstreams natively; non-Anthropic upstreams through translation with narrower feature coverage. |
| Streaming chat | `/v1/chat/completions` or `/v1/messages` with `stream: true` | Same provider support as the chosen endpoint. Streaming uses the first selected target and does not fail over mid-stream. |
| Embeddings | `/v1/embeddings` | OpenAI-family adapter support. Other adapters return `501 not_implemented` unless they add embeddings support later. |
| OpenAI Responses API | `/v1/responses` | OpenAI provider only. OpenAI-compatible vendors are not enough unless the model's `provider` is `openai`. |
| Image generation | `/v1/images/generations` | OpenAI provider only. |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech` | OpenAI-style upstream audio routes. AISIX forwards the audio format; it does not translate audio across provider families. |
| Rerank | `/v1/rerank` | OpenAI, Cohere, and Jina provider labels. |
| Provider-native routes | `/passthrough/:provider/*rest` | Any configured provider key, with less gateway normalization. |

## Broad Chat Routes

`POST /v1/chat/completions` is the broadest proxy route. It accepts OpenAI-compatible caller requests, resolves the model alias, uses the configured provider key, and returns an OpenAI-compatible chat-completions response.

For non-OpenAI upstreams, the provider-facing request is not necessarily OpenAI-compatible. Bedrock, Vertex, Azure OpenAI, and Anthropic-backed models use provider-specific adapter behavior behind the gateway.

Streaming chat uses server-sent events. It follows the same model resolution rules as non-streaming chat, but streaming requests use the first selected target and do not fail over mid-stream.

## Provider-Specific Routes

Some proxy routes intentionally stay narrow because their upstream API format is provider-specific.

### OpenAI-Only Routes

`POST /v1/responses` and `POST /v1/images/generations` require the resolved model to have `provider: "openai"`.

This is stricter than using the `openai` adapter. For example, an OpenAI-compatible vendor can work on `/v1/chat/completions` with `adapter: "openai"` and still be rejected on `/v1/responses` or `/v1/images/generations` if its provider label is not `openai`.

### OpenAI-Style Forwarding Routes

`POST /v1/embeddings` uses adapter-specific embeddings behavior. The OpenAI adapter supports embeddings; adapters that keep the default behavior return `501 not_implemented`.

The audio endpoints forward OpenAI-style audio requests to the resolved provider base URL and return the upstream response format. Use them with upstreams that expose matching OpenAI-style audio routes.

### Rerank

`POST /v1/rerank` uses a route-specific provider allowlist keyed on the model's `provider` label. Accepted provider labels are `openai`, `cohere`, and `jina`.

### Anthropic Messages

`POST /v1/messages` accepts Anthropic models natively and accepts non-Anthropic models through translation. Use this route when the caller is already built around the Anthropic Messages API. For OpenAI-style clients, prefer `/v1/chat/completions`.

## Compatibility Checks

Provider compatibility is not a single yes-or-no question. Check these details before depending on a path.

| Check | Why it matters |
| --- | --- |
| Caller endpoint family | The caller route determines whether the request enters an OpenAI-compatible, Anthropic-style, rerank, audio, or passthrough path. |
| Adapter behind the resolved model | The adapter determines the upstream request format and provider capability. |
| Provider-native versus translated path | Some routes forward the provider's native format. Others translate between API families. Translation support is narrower than native forwarding. |
| Provider-specific response extensions | Vendor-specific response extensions beyond the OpenAI envelope are not normalized. Reasoning-style fields can be lifted per key through the `response.reasoning_field` override. |
| Usage accounting | Usage events vary by endpoint and upstream response. |

For response override details, see [Provider key schema](provider-key-schema.md#runtime-overrides). For usage behavior on `/v1/responses`, see [Responses](../integration/responses.md).

## Featured and Community Catalog Providers

In AISIX Cloud, the catalog distinguishes **featured** providers from community providers. Featured status affects discovery and AISIX Cloud web console presentation only.

Both featured and community providers resolve to one of the adapter families. The self-hosted gateway has no provider catalog and no featured concept; configure `provider`, `adapter`, and `api_base` on each provider key yourself. See [Adapter protocol families](adapters.md#cloud-catalog-and-self-hosted-providers).

## Related Reading

For adapter-family behavior, see [Adapter protocol families](adapters.md). For proxy routes and client-facing API behavior, see [Proxy API reference](proxy-api-reference.md) and [OpenAI-compatible API](../integration/openai-compatible-api.md).
