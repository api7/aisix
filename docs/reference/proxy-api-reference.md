---
title: Proxy API Reference
description: Reference for the current AISIX AI Gateway proxy surface and client-facing endpoints.
sidebar_position: 60
---

## Current Routes

The proxy router currently mounts:

- `GET /health`
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

## Auth

Proxy requests use caller-facing API keys.

Current accepted forms:

- `Authorization: Bearer <plaintext>`
- `x-api-key: <plaintext>` fallback on proxy auth paths

## Related Pages

- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Headers And Error Codes](headers-and-error-codes.md)
- [Provider Compatibility](provider-compatibility.md)
