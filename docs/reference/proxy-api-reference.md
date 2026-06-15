---
title: Proxy API Reference
description: Reference for the current AISIX AI Gateway proxy surface and client-facing endpoints.
sidebar_position: 60
---

## Current Routes

The proxy router currently mounts:

- `GET /livez`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/messages`
- `POST /v1/rerank`
- `POST /v1/responses`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`
- `ANY /passthrough/:provider/*rest`

Use this page as the route inventory. Use the integration pages when you need behavior details and examples.

## Auth

Proxy requests use caller-facing API keys.

Current accepted forms:

- `Authorization: Bearer <plaintext>`
- `x-api-key: <plaintext>` fallback on proxy auth paths

The caller key is an AISIX gateway credential, not an upstream provider key.

## Route Families

You can think about the proxy surface in these groups:

- health and discovery: `/livez`, `/v1/models`
- modeled OpenAI-compatible endpoints: chat, completions, embeddings, images, audio, responses
- Anthropic-style endpoint: `/v1/messages`
- escape hatch: `/passthrough/:provider/*rest`

## Important Reference Boundaries

- `/v1/models` does not expose every callable alias in every case because routing aliases are hidden today
- `/v1/responses` is currently OpenAI-provider-only
- `/passthrough/:provider/*rest` is intentionally thinner than first-class modeled routes

## Related Pages

- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Headers And Error Codes](headers-and-error-codes.md)
- [Provider Compatibility](provider-compatibility.md)
