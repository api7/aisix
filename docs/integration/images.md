---
title: Image Generation
description: Learn how AISIX AI Gateway handles the OpenAI image generation endpoint and provider support.
sidebar_position: 28
toc_max_heading_level: 2
---

AISIX AI Gateway exposes `POST /v1/images/generations` as an OpenAI image-generation endpoint.

Image generation uses the same caller authentication and model-alias behavior as the rest of the proxy API.

## Send an Image Request

Send the image-generation request through the gateway proxy:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/images/generations" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image-prod",
    "prompt": "A minimal illustration of an AI gateway"
  }'
```

Use this endpoint when image-generation callers should share one gateway entry point and keep provider credentials out of application code.

## Provider and Gateway Behavior

The gateway accepts this endpoint only when the resolved model's `provider` is `openai`.

This is stricter than using the `openai` adapter. An OpenAI-compatible vendor can work on `/v1/chat/completions` with `adapter: "openai"` and still be rejected on `/v1/images/generations` if its provider label is not `openai`.

If the model is an OpenAI provider but the selected adapter does not implement image generation, the gateway can return `501` with error type `not_implemented`.

This is a provider or capability support issue, not a caller-authentication problem.

For image generation requests, AISIX authenticates the caller key, verifies that the request includes `model`, resolves the AISIX model alias, checks `allowed_models`, and sends the request through the provider adapter.

The caller continues to use the AISIX alias even when the upstream provider expects a different model identifier.

When the upstream image response includes token usage, the gateway records it. Some OpenAI image models do not return token usage; those successful requests are still visible, but per-image cost details such as image count, size, and quality are not inferred by this proxy path.

## Troubleshooting

Use these checks when image generation reaches a provider path that does not support the endpoint.

### The Request Returns `501`

The resolved OpenAI-family adapter does not implement image generation.

### The Request Returns `400`

Check the model's `provider`. The `/v1/images/generations` path requires `provider: "openai"`.

## Related Reading

For related endpoint and error behavior, see [OpenAI-compatible API](openai-compatible-api.md), [Provider compatibility](../reference/provider-compatibility.md), and [Errors and retries](errors-and-retries.md).
