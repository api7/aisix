---
title: Embeddings
description: Learn how AISIX AI Gateway handles the OpenAI-compatible embeddings endpoint, including request shape and current provider limits.
sidebar_position: 24
---

AISIX AI Gateway exposes `POST /v1/embeddings` as an OpenAI-compatible embeddings endpoint.

## Request Shape

The gateway accepts:

- `input` as a single string
- `input` as an array of strings

The gateway normalizes both forms before dispatching the request upstream.

## Example

```bash title="Create embeddings"
curl -sS -X POST http://127.0.0.1:3000/v1/embeddings \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-prod",
    "input": ["hello", "world"]
  }'
```

## Gateway Behavior

For this endpoint, the gateway:

1. authenticates the caller API key
2. resolves the model alias
3. checks `allowed_models`
4. dispatches through the configured provider bridge
5. returns an OpenAI-style embeddings response

## Current Provider Boundary

Providers that do not implement embeddings return:

- `501 Not Implemented`
- error type `not_implemented`

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
- [Provider Compatibility](../reference/provider-compatibility.md)
