---
title: Audio APIs
description: Learn how AISIX AI Gateway handles audio transcription, translation, and speech endpoints.
sidebar_position: 26
---

AISIX AI Gateway exposes three audio endpoints:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`

Use these endpoints when you want audio-related OpenAI-compatible request shapes at the gateway edge.

## Request Shapes

The current request contracts are:

- transcriptions: `multipart/form-data`
- translations: `multipart/form-data`
- speech: JSON

For multipart requests, the gateway resolves the AISIX model alias and rebuilds the multipart form with the upstream model id before forwarding.

That is the important gateway-specific behavior for transcription and translation: the client still sends the AISIX alias, but the upstream receives the provider model id.

## Response Behavior

The gateway returns the upstream response verbatim:

- JSON for transcription and translation results
- binary audio bytes for speech output

Your client should therefore handle the response based on the endpoint family, not just on the fact that everything goes through the gateway.

## Authentication And Authorization

These endpoints follow the same proxy rules as other client-facing routes:

- caller API key authentication
- model alias resolution
- `allowed_models` enforcement

## When To Use These Endpoints

- transcriptions for speech-to-text
- translations for speech-to-text with translation semantics
- speech for text-to-audio output

## Troubleshooting

### Multipart requests fail with `400`

Check form construction first, especially file upload fields and the presence of `model`.

### Speech output is not JSON

That is expected. `/v1/audio/speech` returns upstream audio bytes rather than a chat-style JSON body.

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
- [Proxy API Reference](../reference/proxy-api-reference.md)
