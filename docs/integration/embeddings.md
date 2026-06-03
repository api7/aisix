---
title: Embeddings
description: Learn how AISIX AI Gateway handles the OpenAI-compatible embeddings endpoint, including request shape and current provider limits.
sidebar_position: 25
---

AISIX AI Gateway exposes `POST /v1/embeddings` as an OpenAI-compatible embeddings endpoint.

This guide explains the embeddings path for applications that want vector generation through the gateway while keeping OpenAI-compatible request shapes.

## Current provider boundary

Embeddings use the resolved provider bridge's embeddings implementation. The
OpenAI bridge implements embeddings today. Providers or bridge paths that do not
implement embeddings return:

- `501 Not Implemented`
- error type `not_implemented`

This is different from `/v1/responses` and `/v1/images/generations`, which are
gated on `provider: "openai"`. For embeddings, the question is whether the
resolved bridge supports embeddings for the provider and model you configured.

## Request shape

The gateway accepts:

- `input` as a single string
- `input` as an array of strings

The gateway accepts both forms and preserves the caller's original wire shape when it dispatches the request upstream.

That means callers do not need separate client-side logic just to switch between a single input and a batch input.

## Example

```shell
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

## Gateway behavior

For this endpoint, the gateway:

1. authenticates the caller API key
2. resolves the model alias
3. checks `allowed_models`
4. rewrites `model` to the upstream provider model id
5. returns an OpenAI-style embeddings response

The gateway records usage when the upstream returns token usage. Embeddings do not use completion tokens, response caching, streaming, or guardrails on the current proxy path.

## When to use this endpoint

- semantic search indexing
- retrieval pipelines
- cache key or clustering workflows that depend on embeddings vectors

## Troubleshooting

### A provider returns `501`

The resolved provider does not implement embeddings on the current gateway path.

### A batch request returns fewer vectors than expected

Treat that as abnormal behavior. The caller-visible contract should return one embedding entry per input item.

## Next steps

- [OpenAI-compatible API](openai-compatible-api.md)
- [Errors and retries](errors-and-retries.md)
- [Provider compatibility](../reference/provider-compatibility.md)
