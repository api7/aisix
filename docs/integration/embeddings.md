---
title: Embeddings
description: Learn how AISIX AI Gateway handles the OpenAI-compatible embeddings endpoint, including request format and provider limits.
sidebar_position: 25
toc_max_heading_level: 2
---

AISIX AI Gateway exposes `POST /v1/embeddings` as an OpenAI-compatible embeddings endpoint.

Applications can generate vectors through the gateway while keeping OpenAI-compatible request formats and gateway-managed provider credentials.

## Send an Embeddings Request

Send the embeddings request through the gateway proxy:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/embeddings" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-prod",
    "input": ["hello", "world"]
  }'
```

Use embeddings for semantic search indexing, retrieval pipelines, and cache-key or clustering workflows that depend on vector representations.

## Provider and Request Behavior

Embeddings depend on the resolved provider adapter. OpenAI-compatible providers can use the OpenAI embeddings path. Providers that do not implement embeddings return `501 Not Implemented` with error type `not_implemented`.

This is different from `/v1/responses` and `/v1/images/generations`, which are gated on `provider: "openai"`. For embeddings, the question is whether the resolved provider supports embeddings for the model you configured.

The gateway accepts a single string or an array of strings and preserves the caller's original request format when it forwards the request upstream. Callers do not need separate client-side logic to switch between a single input and a batch input.

Successful responses follow the OpenAI embeddings format: `object: "list"`, one `data[]` entry per normalized input item, and a `usage` block when the upstream provider returns token usage.

For each request, AISIX authenticates the caller key, resolves the model alias, checks `allowed_models`, rewrites `model` to the upstream provider model id, and returns an OpenAI-style embeddings response.

The gateway records usage when the upstream returns token usage. Embeddings do not use completion tokens, response caching, streaming, or guardrails on this proxy path.

## Troubleshooting

Use these checks when an embeddings request reaches a provider path that does not support the endpoint.

### A Provider Returns `501`

The resolved provider does not implement embeddings on this gateway path.

### A Batch Request Returns Fewer Vectors Than Expected

The response should return one embedding entry per input item. If fewer vectors come back, inspect the upstream response and gateway logs for provider-specific batch handling.

## Related Reading

For related endpoint and error behavior, see [OpenAI-compatible API](openai-compatible-api.md), [Errors and retries](errors-and-retries.md), and [Provider compatibility](../reference/provider-compatibility.md).
