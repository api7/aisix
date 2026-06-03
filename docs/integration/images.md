---
title: Images
description: Learn how AISIX AI Gateway handles the OpenAI image generation endpoint and current support boundaries.
sidebar_position: 28
---

AISIX AI Gateway exposes `POST /v1/images/generations` as an OpenAI image-generation endpoint.

This guide explains the image-generation path through the same caller-auth and model-alias contract as the rest of the proxy surface.

## Current provider boundary

The gateway accepts this endpoint only when the resolved model's `provider` is
`openai`.

This is stricter than using the `openai` adapter. An OpenAI-compatible vendor
can work on `/v1/chat/completions` with `adapter: "openai"` and still be
rejected on `/v1/images/generations` if its provider label is not `openai`.

If the model is an OpenAI provider but the selected bridge does not implement
image generation, the gateway can return `501` with error type
`not_implemented`.

That is a provider or capability boundary, not an auth boundary.

## Gateway behavior

For image generation requests, the gateway:

1. authenticates the caller key
2. validates the request includes `model`
3. resolves the AISIX model alias
4. checks `allowed_models`
5. dispatches to the provider bridge

The caller continues to use the AISIX alias even when the upstream provider expects a different model identifier.

When the upstream image response includes token usage, the gateway records it. Some OpenAI image models do not return token usage; those successful requests are still visible, but per-image cost details such as image count, size, and quality are not inferred by the current proxy path.

## Example

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/images/generations \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image-prod",
    "prompt": "A minimal illustration of an AI gateway"
  }'
```

## When to use this endpoint

- image-generation APIs behind one gateway contract
- caller-side key management that should stay provider-agnostic

## Troubleshooting

### The request returns `501`

The resolved OpenAI-family bridge path does not implement image generation today.

### The request returns `400`

Check the model's `provider`. The current `/v1/images/generations` path requires `provider: "openai"`.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Provider compatibility](../reference/provider-compatibility.md)
- [Errors and retries](errors-and-retries.md)
