---
title: Audio APIs
description: Learn how AISIX AI Gateway handles OpenAI-style audio transcription, translation, and speech endpoints.
sidebar_position: 27
---

AISIX AI Gateway exposes three OpenAI-style audio endpoints:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`

Use these endpoints when you want audio-related OpenAI request shapes at the gateway edge.

## Request shapes

The current request contracts are:

- transcriptions: `multipart/form-data`
- translations: `multipart/form-data`
- speech: JSON

For multipart requests, the gateway resolves the AISIX model alias and rebuilds the multipart form with the upstream model id before forwarding. It preserves the other form fields, including file name and content type when present.

That is the important gateway-specific behavior for transcription and translation: the client still sends the AISIX alias, but the upstream receives the provider model id.

## Response behavior

The gateway relays the upstream response body and response content type:

- JSON for transcription and translation results
- binary audio bytes for speech output

Your client should therefore handle the response based on the endpoint family, not just on the fact that everything goes through the gateway. Do not rely on AISIX to normalize audio responses into a chat-style JSON body.

## Authentication and authorization

These endpoints follow the same proxy rules as other client-facing routes:

- caller API key authentication
- model alias resolution
- `allowed_models` enforcement

## Current provider boundary

Audio requests are forwarded to the resolved provider key's `api_base` with the AISIX model alias rewritten to the upstream model id.

The gateway does not translate audio request or response shapes across provider families. Use these endpoints with upstreams that expose the same OpenAI-style audio routes:

- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/v1/audio/speech`

If a provider does not expose the requested audio route, the failure is an upstream capability or base-URL issue, not a caller-auth issue.

Successful audio requests are attributed in gateway usage events. Token counts are populated only when the upstream response includes a recognized `usage` block; speech output and duration-based audio costs are not inferred from the binary response.

## When to use these endpoints

- transcriptions for speech-to-text
- translations for speech-to-text with translation semantics
- speech for text-to-audio output

## Troubleshooting

### Multipart requests fail with `400`

Check form construction first, especially file upload fields and the presence of `model`.

### Speech output is not JSON

That is expected. `/v1/audio/speech` returns upstream audio bytes rather than a chat-style JSON body.

### The request returns an upstream `404`

Check whether the resolved provider exposes the requested OpenAI-style audio route and whether `api_base` points to the route root the gateway should append `/v1/audio/...` to.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Provider keys](../configuration/provider-keys.md)
- [Provider compatibility](../reference/provider-compatibility.md)
- [Errors and retries](errors-and-retries.md)
- [Proxy API reference](../reference/proxy-api-reference.md)
