---
title: Admin API
description: Use the AISIX AI Gateway admin API to manage models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, and the in-process playground.
sidebar_position: 31
---

The AISIX AI Gateway admin API is the operator-facing surface for managing the gateway's dynamic configuration.

Use this API when you need to:

- create and update models
- create and rotate caller API keys
- manage upstream provider credentials
- manage guardrails, cache policies, and observability exporters
- inspect operator-facing health

## Listener And Auth Model

In standalone mode, the admin API runs on the admin listener configured in bootstrap config.

Admin authentication is static and bootstrap-based for the authenticated operator routes:

- admin keys come from `config.admin.admin_keys`
- `/admin/v1/*` routes expect `Authorization: Bearer <key>`
- `x-api-key` is also accepted as a fallback

The following routes are currently public on the admin listener:

- `GET /health`
- `GET /metrics`
- `GET /admin/openapi.json`
- `GET /admin/openapi-scalar`

Example:

```bash title="Authenticated admin request"
curl -sS http://127.0.0.1:3001/admin/v1/health \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

## Current Admin Surface

The current admin router exposes:

- `GET /health`
- `GET /metrics`
- `GET /admin/openapi.json`
- `GET /admin/openapi-scalar`
- `GET|POST /admin/v1/models`
- `GET|PUT|DELETE /admin/v1/models/:id`
- `GET|POST /admin/v1/apikeys`
- `GET|PUT|DELETE /admin/v1/apikeys/:id`
- `POST /admin/v1/apikeys/:id/rotate`
- `GET|POST /admin/v1/provider_keys`
- `GET|PUT|DELETE /admin/v1/provider_keys/:id`
- `GET|POST /admin/v1/guardrails`
- `GET|PUT|DELETE /admin/v1/guardrails/:id`
- `GET|POST /admin/v1/cache_policies`
- `GET|PUT|DELETE /admin/v1/cache_policies/:id`
- `GET|POST /admin/v1/observability_exporters`
- `GET|PUT|DELETE /admin/v1/observability_exporters/:id`
- `GET /admin/v1/health`
- `POST /playground/chat/completions`

## Error Envelope

The admin API does **not** use the OpenAI-style proxy error shape.

It uses a simpler envelope:

```json
{
  "error_msg": "missing or malformed admin authorization"
}
```

Current status behavior includes:

- `400` for bad request or schema validation failure
- `401` for missing or invalid admin auth
- `404` for missing resources
- `409` for conflicts such as duplicate names
- `500` for store failures

Public routes such as `/health`, `/metrics`, and the OpenAPI endpoints do not require admin auth.

## Models

`/admin/v1/models` manages model resources.

Current behavior:

- POST creates a UUID-backed resource entry
- PUT updates an existing model and bumps revision
- duplicate `display_name` values are rejected

Example:

```bash title="Create a model"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID"
  }'
```

## API Keys

`/admin/v1/apikeys` manages caller-facing API keys.

Important current behavior:

- the stored field is `key_hash`, not plaintext
- `allowed_models` controls model authorization
- `POST /admin/v1/apikeys/:id/rotate` returns a new plaintext key exactly once in the rotation response

## Provider Keys

`/admin/v1/provider_keys` manages upstream credentials reused by models.

Current fields include:

- `display_name`
- `secret`
- optional `api_base`

## Guardrails

`/admin/v1/guardrails` manages guardrail resources.

Current resource kinds are:

- `keyword`
- `bedrock`

Current operator guidance:

- use `keyword` for current in-process blocking behavior
- treat `bedrock` as a schema-backed but limited runtime path

See [Guardrails](guardrails.md).

## Cache Policies

`/admin/v1/cache_policies` manages cache-policy resources.

Current fields include:

- `name`
- `enabled`
- `backend`
- `ttl_seconds`
- `applies_to`

Current documented `applies_to` forms are:

- `all`
- `model:<display_name>`
- `api_key:<api_key_id>`

See [Caching](caching.md).

## Observability Exporters

`/admin/v1/observability_exporters` manages exporter resources.

Current behavior:

- `kind=otlp_http` is the supported resource type
- plain `http://` endpoints are rejected unless they are loopback-style development endpoints

See [Observability Exporters](observability-exporters.md).

## Health, Metrics, And Playground

### `GET /admin/v1/health`

This is the operator-facing health endpoint.

It reports top-level health plus current model health state.

### `GET /metrics`

This is the Prometheus scrape endpoint on the admin listener.

### `POST /playground/chat/completions`

The standalone admin playground is an in-process proxy to `/v1/chat/completions`.

Important current behavior:

- it expects a **proxy** API key, not an admin key
- it forwards into the proxy router inside the same process
- it runs the full proxy middleware path

## Verification

Verify that the admin surface is reachable:

```bash title="Check admin health"
curl -sS http://127.0.0.1:3001/admin/v1/health \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Then create a provider key, model, and API key as shown in [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md).

## Related Pages

- [Bootstrap Configuration](bootstrap-config.md)
- [Models](models.md)
- [Provider Keys](provider-keys.md)
- [API Keys](api-keys.md)
- [Guardrails](guardrails.md)
- [Caching](caching.md)
- [Observability Exporters](observability-exporters.md)
- [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Roadmap](../roadmap.md)
