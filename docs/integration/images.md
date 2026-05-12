---
title: Images
description: Learn how AISIX AI Gateway handles the OpenAI-compatible image generation endpoint and current support boundaries.
sidebar_position: 27
---

AISIX AI Gateway exposes `POST /v1/images/generations` as an OpenAI-compatible image-generation endpoint.

## Gateway Behavior

For image generation requests, the gateway:

1. authenticates the caller key
2. validates the request includes `model`
3. resolves the AISIX model alias
4. checks `allowed_models`
5. dispatches to the provider bridge

## Current Provider Boundary

Providers that do not implement image generation return:

- `501 Not Implemented`
- error type `not_implemented`

## Example

```bash title="Generate an image"
curl -sS -X POST http://127.0.0.1:3000/v1/images/generations \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image-prod",
    "prompt": "A minimal illustration of an AI gateway"
  }'
```

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Provider Compatibility](../reference/provider-compatibility.md)
- [Errors And Retries](errors-and-retries.md)
