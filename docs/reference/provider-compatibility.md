---
title: Provider compatibility
description: Reference for current adapter-family coverage and compatibility boundaries in AISIX AI Gateway — which wire shape backs each provider and what each family supports.
sidebar_position: 64
---

This page is the lookup reference for which upstreams AISIX AI Gateway can reach and what each one currently supports. Compatibility is organized around the five [adapter protocol families](adapters.md), not a flat provider list: every upstream — catalog or bring-your-own — resolves to exactly one adapter, and the adapter determines the wire shape and the supported endpoints.

## Adapter families

The gateway encodes requests against a closed set of five adapter families. Vendor identity (`provider`) is a free-form string; the adapter is the closed enum that picks the bridge.

| Adapter | Upstream wire shape | Example upstreams |
|---|---|---|
| `openai` | OpenAI chat completions | OpenAI, plus every OpenAI-compatible vendor (DeepSeek, Groq, Mistral, Together.ai, Fireworks, Perplexity, …) and BYO endpoints (vLLM, SGLang, Ollama) |
| `anthropic` | Anthropic Messages | Anthropic (Claude) |
| `bedrock` | AWS Bedrock Runtime (Converse + Anthropic `/invoke`) | Claude, and other Bedrock publishers via Converse |
| `vertex` | Google Vertex AI Gemini | Gemini on Vertex |
| `azure-openai` | Azure OpenAI Service | Azure OpenAI deployments |

The `openai` family is the broadest: any vendor or self-hosted server that speaks the OpenAI chat-completions wire dispatches through it, differing only in `api_base` and credential. See [OpenAI-compatible vendor upstream](../integration/upstream-openai-compat.md) and [Bring your own endpoint](../configuration/byo-endpoint.md).

## Coverage matrix

Support depth varies by adapter family. The matrix below summarizes the current state; each integration guide documents the exact behavior.

| Capability | `openai` | `anthropic` | `bedrock` | `vertex` | `azure-openai` |
|---|---|---|---|---|---|
| Chat completions | Yes | Yes (Messages) | Yes | Yes (Gemini) | Yes |
| Streaming (SSE) | Yes | Yes | Yes | Yes (Gemini) | Yes |
| Embeddings | Yes (OpenAI / OpenAI-compatible) | No | No | No | No |
| Images, audio, responses | Yes (OpenAI / OpenAI-compatible) | No | No | No | No |
| Rerank | Yes (Cohere / Jina native surface) | No | No | No | No |

Notes on the matrix:

- The image, audio, `/v1/responses`, and embeddings endpoints are gated to OpenAI-shaped upstreams. A request that resolves to a non-OpenAI model on those endpoints is rejected rather than mis-dispatched.
- `/v1/rerank` is served by the Cohere and Jina native rerank surfaces, which bypass the chat bridge; it is keyed on the model's `provider`.
- `/v1/messages` accepts non-Anthropic models through a cross-provider translation path; see [Anthropic Messages](../integration/anthropic-messages.md).

## Per-family limitations

- **`openai`** — vendor-specific response extensions beyond the OpenAI envelope are not normalized. Reasoning-style fields can be lifted per key via the `response.reasoning_field` override (see [Provider key schema § response overrides](runtime-config-schema.md#response-overrides)).
- **`anthropic`** — the family speaks the Messages wire; it is not the OpenAI embeddings/images/audio surface.
- **`bedrock`** — Anthropic-on-Bedrock (Claude) models dispatch through the legacy `/invoke` route with an Anthropic Messages body; all other publishers use the unified Converse API. Cross-region inference profile prefixes (`us.`, `eu.`, `apac.`, `global.`, `us-gov.`) are supported. See [AWS Bedrock upstream](../integration/upstream-bedrock.md).
- **`vertex`** — Gemini chat and streaming are wired. **Anthropic-on-Vertex and Llama-on-Vertex are not yet implemented.** See [Google Vertex AI upstream § Limitations](../integration/upstream-vertex.md#limitations) and the [Roadmap](../roadmap.md).
- **`azure-openai`** — chat and streaming are wired for both the `api-key` and the Entra ID (AAD) `client_credentials` auth schemes. See [Azure OpenAI upstream](../integration/upstream-azure-openai.md).

## Featured versus non-featured catalog providers

In AISIX Cloud, the catalog distinguishes **featured** providers (the ranked set the dashboard surfaces first) from non-featured (Community) providers. Featured status affects discovery and presentation only — both featured and non-featured providers resolve to one of the five adapters through the same catalog mapping and run through the same bridges. The self-hosted gateway ships no catalog and has no featured concept; you set `provider`, `adapter`, and `api_base` on each provider key yourself. See [Adapter protocol families § Catalog versus bring-your-own](adapters.md#catalog-versus-bring-your-own).

## Compatibility boundary

Provider support is not identical across every endpoint and behavior surface.

Current reference point:

- the gateway exposes a mixed OpenAI-compatible and Anthropic-style surface
- support depth varies by adapter family and endpoint

This means provider compatibility is not a single yes/no question.

The real questions are:

- which endpoint family are you using
- which adapter backs the resolved model
- whether the path is provider-native or translated

## Practical reading guide

- start with integration docs for endpoint-family behavior
- use the feature matrix for current breadth versus limited support
- use roadmap items for parity you expect but do not yet see documented

Use the feature matrix and integration docs as the current contract, and treat broader provider parity as ongoing work.

## Related pages

- [Adapter protocol families](adapters.md) — the five families and how a model resolves to a bridge.
- [Feature Matrix](../overview/feature-matrix.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Roadmap](../roadmap.md)
