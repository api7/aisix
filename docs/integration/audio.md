---
title: Audio APIs
description: Learn how AISIX AI Gateway handles OpenAI-style audio transcription, translation, and speech endpoints.
sidebar_position: 27
toc_max_heading_level: 2
---

AISIX AI Gateway exposes OpenAI-style audio routes for transcription, translation, and speech generation.

Use these endpoints when audio clients should keep OpenAI request formats while AISIX manages caller authentication, model aliases, and upstream credentials.

## Audio Request Flow

Transcription and translation requests use `multipart/form-data`. Speech requests use JSON.

For multipart requests, the gateway resolves the AISIX model alias and rebuilds the multipart form with the upstream model id before forwarding. It preserves the other form fields, including file name and content type when present.

For transcription and translation, the client still sends the AISIX alias, while the upstream receives the provider model id.

The gateway relays the upstream response body and response content type. Transcription and translation responses are JSON results. Speech responses are binary audio bytes.

Your client should handle the response based on the endpoint family, not only on the fact that the request goes through the gateway. Do not rely on AISIX to normalize audio responses into a chat-style JSON body.

These endpoints follow the same proxy rules as other client-facing routes: caller API key authentication, model alias resolution, and `allowed_models` enforcement.

## Provider Support

Audio requests are forwarded to the resolved provider key's `api_base` with the AISIX model alias rewritten to the upstream model id.

The gateway does not translate audio request or response formats across provider families. Use these endpoints with upstreams that expose the same OpenAI-style audio routes: `/v1/audio/transcriptions`, `/v1/audio/translations`, and `/v1/audio/speech`.

If a provider does not expose the requested audio route, the failure is an upstream capability or base-URL issue, not a caller-auth issue.

Successful audio requests are attributed in gateway usage events. Token counts are populated only when the upstream response includes a recognized `usage` block; speech output and duration-based audio costs are not inferred from the binary response.

## Choose an Audio Endpoint

Use transcriptions for speech-to-text, translations for speech-to-text with translation semantics, and speech for text-to-audio output.

## Troubleshooting

Use these checks when audio requests fail before reaching the expected upstream path.

### Multipart Request Returns `400`

Check form construction first. Confirm the file upload fields and the presence of `model`.

### Speech Output Is Not JSON

`/v1/audio/speech` returns upstream audio bytes rather than a chat-style JSON body. Handle the response as binary output.

### The Request Returns an Upstream `404`

Check whether the resolved provider exposes the requested OpenAI-style audio route and whether `api_base` points to the route root the gateway should append `/v1/audio/...` to.

## Related Reading

[OpenAI-compatible API](openai-compatible-api.md) covers OpenAI-style gateway routes. Configure upstream credentials and base URLs with [Provider keys](../configuration/provider-keys.md), and check [Provider compatibility](../reference/provider-compatibility.md) before depending on an audio route. For proxy errors and gateway behavior, see [Errors and retries](errors-and-retries.md) and [Proxy API reference](../reference/proxy-api-reference.md).
