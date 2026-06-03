---
title: OpenAI-Compatible API
description: Learn how to call AISIX AI Gateway through its OpenAI-compatible proxy API, including authentication, model selection, error handling, and current endpoint coverage.
sidebar_position: 20
---

AISIX AI Gateway exposes an OpenAI-compatible proxy surface so existing SDKs and HTTP clients can talk to the gateway with minimal change.

This guide explains the OpenAI-compatible caller surface after the gateway can already serve a request through `/v1/chat/completions`.

This page explains the caller-facing contract: how clients authenticate, how model names are resolved, which endpoints are mounted, and which gateway errors callers should handle.

## Caller contract

OpenAI-compatible clients call AISIX instead of calling a provider directly:

- set the client base URL to the AISIX proxy listener
- send a caller API key, not the upstream provider key
- use an AISIX model alias in the request `model` field

AISIX resolves the caller key, model alias, provider key, upstream model name, routing, and policy at the gateway layer.

## What changes for clients

The application-level request stays familiar: clients send `Authorization`, `model`, `messages`, and other OpenAI-shaped fields to the AISIX proxy listener.

The important difference is where provider decisions live. The caller sends a gateway API key and a model alias. AISIX authenticates the caller, checks model access, applies policy, resolves the provider key and upstream model, and forwards the request.

The upstream provider credential is never sent by the caller. AISIX injects it from the configured provider key before forwarding the request upstream.

## Supported endpoints

The proxy router mounts several OpenAI-shaped routes, but they do not all have the same provider breadth. Treat `POST /v1/chat/completions` as the default route for OpenAI-compatible clients, then move to a specialized page when your application needs a narrower API family.

| Need | Start with | Notes |
| --- | --- | --- |
| Chat requests from OpenAI SDKs | `POST /v1/chat/completions` | Broadest provider path and the default OpenAI-compatible entry point. |
| Model discovery for a caller key | `GET /v1/models` | Returns non-routing aliases visible to the authenticated key. |
| Embeddings, images, audio, responses, or rerank | Endpoint-specific integration pages | Provider support differs by route. Check the page before using a non-OpenAI upstream. |
| Anthropic-style clients | [Anthropic-style Messages API](anthropic-messages.md) | Use `/v1/messages`, not the OpenAI-compatible chat path. |
| Provider-native routes | [Provider passthrough](passthrough.md) | Use when AISIX does not model the provider route directly. |

For the full mounted route list and current provider boundaries, see [Proxy API reference](../reference/proxy-api-reference.md) and [Provider compatibility](../reference/provider-compatibility.md).

## Authentication

Proxy requests use a caller-facing API key.

Use the standard bearer format:

```http
Authorization: Bearer YOUR_CALLER_API_KEY
```

The proxy also accepts `x-api-key: YOUR_CALLER_API_KEY` as a compatibility fallback, but `Authorization: Bearer ...` is the recommended form for OpenAI-compatible clients.

At runtime, the data plane hashes the bearer token and resolves it against the stored `key_hash` in the current snapshot.

## Model resolution

The model name seen by the caller is the configured `display_name`, not necessarily the upstream provider model identifier.

For a direct model, AISIX forwards to the configured provider key and upstream model name. For a routing model, AISIX chooses a target model according to the configured routing strategy before dispatching.

## `/v1/models`

`GET /v1/models` returns the subset of models the authenticated API key is allowed to access.

- wildcard keys see every non-routing model
- restricted keys see only explicitly allowed models
- routing aliases are not exposed through this list

Example:

```shell
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY"
```

## `/v1/chat/completions`

The chat-completions path is the main OpenAI-compatible entry point.

Example:

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "Hello from AISIX."}
    ]
  }'
```

## Error boundaries

Important proxy-side outcomes include:

- `400` if the request payload is malformed or invalid for the endpoint
- `401` if the caller key is missing or unknown
- `403` if the key is valid but not allowed to access the requested model
- `404` if the requested model alias is not found
- `413` if the request body exceeds the configured proxy body-size limit
- `422` if a guardrail blocks the content
- `429` if the request is blocked by limits or budget policy
- `503` if no bridge is registered for the resolved provider

For the current error taxonomy and header behavior, see [Headers and error codes](../reference/headers-and-error-codes.md).

## Next steps

- [Understand admin resources](../quickstart/first-model-first-key-first-request.md) — review the resources that make this proxy request work.
- [Models](../configuration/models.md) — configure caller-visible aliases and routing models.
- [API keys](../configuration/api-keys.md) — control caller authentication and model access.
- [Anthropic-style Messages API](anthropic-messages.md) — use the Anthropic-style proxy API.
- [Provider compatibility](../reference/provider-compatibility.md) — check provider and endpoint boundaries.
