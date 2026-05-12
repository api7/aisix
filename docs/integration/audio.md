---
title: Audio APIs
description: Learn how AISIX AI Gateway handles audio transcription, translation, and speech endpoints.
sidebar_position: 26
---

AISIX AI Gateway exposes three audio endpoints:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`

## Request Shapes

The current request contracts are:

- transcriptions: `multipart/form-data`
- translations: `multipart/form-data`
- speech: JSON

For multipart requests, the gateway resolves the AISIX model alias and rebuilds the multipart form with the upstream model id before forwarding.

## Response Behavior

The gateway returns the upstream response verbatim:

- JSON for transcription and translation results
- binary audio bytes for speech output

## Authentication And Authorization

These endpoints follow the same proxy rules as other client-facing routes:

- caller API key authentication
- model alias resolution
- `allowed_models` enforcement

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
- [Proxy API Reference](../reference/proxy-api-reference.md)
