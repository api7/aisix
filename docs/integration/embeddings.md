---
title: Embeddings
description: Learn how AISIX AI Gateway handles the OpenAI-compatible embeddings endpoint, including request shape and current provider limits.
sidebar_position: 24
---

AISIX AI Gateway exposes `POST /v1/embeddings` as an OpenAI-compatible embeddings endpoint.

Use this page when you need vector generation through the gateway while keeping OpenAI-compatible request shapes.

## Request Shape

The gateway accepts:

- `input` as a single string
- `input` as an array of strings

The gateway normalizes both forms before dispatching the request upstream.

That means callers do not need separate client-side logic just to switch between a single input and a batch input.

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

Typical successful responses follow the OpenAI embeddings shape with:

- `object: "list"`
- one `data[]` entry per normalized input item
- a `usage` block when the upstream/provider path returns token usage

## Gateway Behavior

For this endpoint, the gateway:

1. authenticates the caller API key
2. resolves the model alias
3. checks `allowed_models`
4. dispatches through the configured provider bridge
5. returns an OpenAI-style embeddings response

## When To Use This Endpoint

- semantic search indexing
- retrieval pipelines
- cache key or clustering workflows that depend on embeddings vectors

## Troubleshooting

### A provider returns `501`

The resolved provider does not implement embeddings on the current gateway path.

### A batch request returns fewer vectors than expected

Treat that as an upstream or product bug. The caller-visible contract should align with the normalized input count.

## Current Provider Boundary

Providers that do not implement embeddings return:

- `501 Not Implemented`
- error type `not_implemented`

## Related Pages

- [OpenAI-Compatible API](openai-compatible-api.md)
- [Errors And Retries](errors-and-retries.md)
- [Provider Compatibility](../reference/provider-compatibility.md)
