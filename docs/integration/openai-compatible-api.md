---
title: OpenAI-Compatible API
description: Learn how to call AISIX AI Gateway through its OpenAI-compatible proxy API, including authentication, model selection, error handling, and endpoint coverage.
sidebar_position: 20
toc_max_heading_level: 2
---

AISIX AI Gateway exposes an OpenAI-compatible proxy API so existing SDKs and HTTP clients can talk to the gateway with minimal change. Use the OpenAI API reference for the base request and response schemas. The sections below cover gateway authentication, model aliases, provider routing, and error behavior.

## Client Configuration

OpenAI-compatible clients call AISIX instead of calling a provider directly. Set the client base URL to the AISIX proxy listener, use a gateway-issued caller API key, and send the AISIX model alias in `model`.

The application-level request stays familiar: clients send `Authorization`, `model`, `messages`, and other OpenAI-compatible fields to the AISIX proxy listener.

Provider selection and policy stay in the gateway configuration. The caller sends a gateway API key and a model alias. AISIX authenticates the caller, checks model access, applies policy, resolves the provider key and upstream model, and forwards the request.

The upstream provider credential is never sent by the caller. AISIX injects it from the configured provider key before forwarding the request upstream.

### Authentication

Proxy requests use a caller-facing API key.

Use the standard bearer format:

```http
Authorization: Bearer YOUR_CALLER_API_KEY
```

The proxy also accepts `x-api-key: YOUR_CALLER_API_KEY` for compatibility, but `Authorization: Bearer ...` is the recommended form for OpenAI-compatible clients.

## Choose an Endpoint

The proxy router mounts several OpenAI-compatible routes, but they do not all have the same provider breadth. Treat `POST /v1/chat/completions` as the default route for OpenAI-compatible clients, then use endpoint-specific guidance when your application needs a narrower API family.

Use `POST /v1/chat/completions` for chat requests from OpenAI SDKs and other OpenAI-compatible clients. Use `GET /v1/models` when a caller needs to discover the non-routing aliases visible to its API key.

For embeddings, images, audio, responses, or rerank, use the endpoint-specific integration page before routing a non-OpenAI upstream through that endpoint. Provider support differs by route.

If the client is Anthropic-style, use [Anthropic-style Messages API](anthropic-messages.md) and call `/v1/messages`. If the application needs a provider-native route that AISIX does not model directly, use [Provider passthrough](passthrough.md).

For the proxy route overview and provider support, see [Proxy API reference](../reference/proxy-api-reference.md) and [Provider compatibility](../reference/provider-compatibility.md).

## Model Resolution

The model name seen by the caller is the configured `display_name`, not necessarily the upstream provider model identifier.

For a direct model, AISIX forwards to the configured provider key and upstream model name. For a routing model, AISIX chooses a target model according to the configured routing strategy before sending the provider request.

### `GET /v1/models`

`GET /v1/models` returns the subset of models the authenticated API key is allowed to access.

Wildcard keys can see every non-routing model. Restricted keys see only the models explicitly allowed by the key. Routing aliases are not exposed through this list.

Example:

```shell
curl -sS "http://127.0.0.1:3000/v1/models" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY"
```

### `POST /v1/chat/completions`

The chat-completions path is the main OpenAI-compatible entry point.

Example:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "Hello from AISIX."}
    ]
  }'
```

## Error Handling

Gateway-generated errors use an OpenAI-compatible envelope on this API family. Request and configuration problems usually appear as `400`, `401`, `403`, `404`, or `413`. Guardrails, limits, budgets, and provider runtime state can produce `422`, `429`, or `503`.

For the error taxonomy and header behavior, see [Headers and error codes](../reference/headers-and-error-codes.md).

## Related Reading

Review the resources behind a working request in [Understand admin resources](../quickstart/first-model-first-key-first-request.md). Configure caller-visible aliases with [Models](../configuration/models.md), and control caller authentication with [API keys](../configuration/api-keys.md). For Anthropic-style clients and provider support, see [Anthropic-style Messages API](anthropic-messages.md) and [Provider compatibility](../reference/provider-compatibility.md).
