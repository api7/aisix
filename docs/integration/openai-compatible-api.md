---
title: OpenAI-Compatible API
description: Learn how to call AISIX AI Gateway through its OpenAI-compatible proxy API, including authentication, model selection, error handling, and current endpoint coverage.
sidebar_position: 20
---

AISIX AI Gateway exposes an OpenAI-compatible proxy surface so existing SDKs and HTTP clients can talk to the gateway with minimal change.

Use this page when you need to understand:

- how authentication works on the proxy surface
- which client-facing endpoints are currently exposed
- how model aliases are resolved
- what the current error and authorization boundaries look like

## Request Flow

A `POST /v1/chat/completions` (or any other proxy endpoint) takes this path through the gateway:

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Proxy as AISIX proxy (:3000)
    participant Hub as Provider bridge (hub)
    participant Upstream as Upstream LLM

    Client->>Proxy: POST /v1/chat/completions (Bearer sk-...)
    Note over Proxy: 1. hash bearer (SHA-256) and resolve ApiKey
    Note over Proxy: 2. resolve req.model to a Model in the snapshot
    Note over Proxy: 3. enforce allowed_models on the ApiKey
    Note over Proxy: 4. run input guardrails
    Note over Proxy: 5. budget pre-check (allow-all in standalone; cp-api /dp/budget_check in managed mode)
    Note over Proxy: 6. enforce per-key / per-model rate limits
    Note over Proxy: 7. cache lookup — on hit, return cached response and skip steps 8–10
    Proxy->>Hub: dispatch by Model.provider
    Hub->>Upstream: provider-native request shape
    Upstream-->>Hub: provider-native response (JSON or SSE)
    Hub-->>Proxy: normalised OpenAI-shaped response
    Note over Proxy: 8. run output guardrails
    Note over Proxy: 9. on cache miss, write the response into the CachePolicy backend
    Note over Proxy: 10. emit metrics + access log + UsageEvent
    Proxy-->>Client: 200 OK or proxy error envelope
```

Each step has its own failure path; see [Error boundaries](#error-boundaries) below for the matching status codes. The budget pre-check is a no-op in standalone mode (the DP runs an allow-all budget client) and a live call to cp-api over mTLS in managed mode — both modes traverse the same code path.

## Current Proxy Surface

The proxy router currently mounts:

- `GET /livez`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/messages`
- `POST /v1/rerank`
- `POST /v1/responses`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`
- `ANY /passthrough/:provider/*rest`

## Authentication

Proxy requests use a caller-facing API key.

The request format is:

```http
Authorization: Bearer YOUR_CALLER_API_KEY
```

At runtime, the data plane hashes the bearer token and resolves it against the stored `key_hash` in the current snapshot.

## Model Resolution

For a request like `/v1/chat/completions`, the gateway:

1. authenticates the caller API key
2. resolves `req.model` against the current model table
3. checks whether the API key can access that model
4. dispatches to the configured provider bridge

The model name seen by the caller is the configured `display_name`, not necessarily the upstream provider model identifier.

## Current Behavior Of `/v1/models`

`GET /v1/models` returns the subset of models the authenticated API key is allowed to access.

- wildcard keys see every non-routing model
- restricted keys see only explicitly allowed models
- routing aliases are not exposed through this list

Example:

```bash title="List models through the gateway"
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer YOUR_CALLER_API_KEY"
```

## Current Behavior Of `/v1/chat/completions`

The chat-completions path is the main OpenAI-compatible entry point.

At a high level, the request flow is:

1. authenticate the caller key
2. resolve the model
3. enforce allowlist authorization
4. run input guardrails
5. enforce rate limits and, in managed mode, budget checks
6. dispatch to the upstream bridge
7. render an OpenAI-shaped response
8. emit metrics, access logs, and usage events

Example:

```bash title="Send a chat completion request"
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

## Error Boundaries

Important proxy-side outcomes include:

- `400` if the request payload is malformed or invalid for the endpoint
- `401` if the caller key is missing or unknown
- `403` if the key is valid but not allowed to access the requested model
- `404` if the requested model alias is not found
- `422` if a guardrail blocks the content
- `429` if the request is blocked by limits or budget policy
- `503` if no bridge is registered for the resolved provider

## Related Pages

- [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md)
- [Admin API](../configuration/admin-api.md)
- [Feature Matrix](../overview/feature-matrix.md)
- [Roadmap](../roadmap.md)
