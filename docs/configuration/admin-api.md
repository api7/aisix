---
title: Admin API
description: Use the AISIX AI Gateway admin API to manage models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, and the standalone playground.
sidebar_position: 31
---

The AISIX AI Gateway admin API is the operator-facing surface for managing the gateway's dynamic configuration.

:::note Standalone only
The `admin/v1` examples on this page apply to self-hosted standalone AISIX. A [Cloud managed data plane](../quickstart/aisix-cloud-managed-dp.md) only exposes proxy APIs locally and does **not** bind the standalone admin listener — provider keys, models, and caller API keys are managed through the AISIX Cloud control plane instead.
:::

Use this API when you need to:

- create and update models
- create and rotate caller API keys
- manage upstream provider credentials
- manage guardrails, cache policies, and observability exporters
- inspect operator-facing health

Use it as the write path for standalone deployments, not as a caller-facing integration surface.

## Listener and auth model

In standalone mode, the admin API runs on the admin listener configured in bootstrap config.

Admin authentication is static and bootstrap-based for the authenticated operator routes:

- admin keys come from `config.admin.admin_keys`
- `/admin/v1/*` routes expect `Authorization: Bearer <key>`
- `x-api-key` is also accepted as a fallback

The following routes are currently public on the admin listener:

- `GET /livez`
- `GET /metrics`
- `GET /admin/openapi.json`
- `GET /admin/openapi-scalar`

Example:

```shell
curl -sS http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Operationally, there are two very different key types in this product:

- admin keys for operator access to `/admin/v1/*`
- proxy caller API keys for `/v1/*`

Do not mix them.

## Admin surface

Think about the admin surface in four groups.

Public operator helpers cover liveness, metrics, and OpenAPI discovery.

CRUD resources cover models, API keys, provider keys, guardrails, cache
policies, and observability exporters.

Runtime status endpoints expose per-model runtime status and aggregated admin
health.

The operator playground forwards a local chat-completions request through the
proxy path for debugging.

Not every runtime resource has standalone admin CRUD today. `RateLimitPolicy` rows and `GuardrailAttachment` rows are loaded from the config store and can be projected by a control plane, but they are not exposed as `/admin/v1/rate_limit_policies` or `/admin/v1/guardrail_attachments` routes in the current standalone admin router.

For the exact route list, request schemas, and response schemas, use the generated [Admin API reference](/ai-gateway/reference/admin-api). To export the OpenAPI document or understand which standalone resources it covers, see [Admin API source notes](../reference/admin-api-reference.md).

## Error envelope

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

Public routes such as `/livez`, `/metrics`, and the OpenAPI endpoints do not require admin auth.

Use `GET /livez` for simple admin-listener reachability. Use `GET /admin/v1/health` when you need authenticated per-model operator health.

For automation, plan to branch on admin status codes and `error_msg`, not on the proxy-side OpenAI-compatible error envelope.

## Models

`/admin/v1/models` manages model resources.

Current behavior:

- POST creates a UUID-backed resource entry
- PUT updates an existing model and bumps revision
- duplicate `display_name` values are rejected

Use model CRUD when you need to change caller-visible routing behavior. A model row is the main bridge between your caller contract and the upstream provider configuration.

Example:

```shell
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

## API keys

`/admin/v1/apikeys` manages caller-facing API keys.

Important current behavior:

- the stored field is `key_hash`, not plaintext
- `allowed_models` controls model authorization
- `POST /admin/v1/apikeys/:id/rotate` returns a new plaintext key exactly once in the rotation response

This makes API-key creation and rotation an operator workflow with one-time secret reveal semantics. Treat the rotate response as the only chance to capture the new plaintext key.

## Provider keys

`/admin/v1/provider_keys` manages upstream credentials reused by models.

Provider keys should be reused across related models where that matches your
operational ownership boundary. That keeps upstream credential rotation
separate from model alias changes.

See [Provider keys](provider-keys.md).

## Guardrails

`/admin/v1/guardrails` manages guardrail resources.

Current resource kinds are:

- `keyword`
- `bedrock`
- `azure_content_safety`

Current operator guidance:

- use `keyword` for current in-process blocking behavior
- treat `bedrock` and `azure_content_safety` as remote guardrails that require provider credentials, network access, relevant build features, and an explicit `fail_open` decision

Create guardrails only when you are also clear about where they execute today. The current live guardrail path is narrower than the full schema surface.

Guardrail scoping is handled by `GuardrailAttachment` rows in the runtime snapshot. The standalone admin API does not expose CRUD routes for those attachment rows yet, so standalone-created guardrails with no attachment rows currently apply environment-wide through the compatibility fallback. See [Guardrails](guardrails.md#scope-guardrails).

See [Guardrails](guardrails.md).

## Cache policies

`/admin/v1/cache_policies` manages cache-policy resources.

Cache policies are a matching layer, not a guarantee that every request will be
cached. They must line up with the bootstrap cache backend and the current
request shape.

See [Caching](caching.md).

## Observability exporters

`/admin/v1/observability_exporters` manages exporter resources.

Current behavior:

- `kind=otlp_http` is the supported resource type
- plain `http://` endpoints are rejected unless they are loopback-style development endpoints

Use dynamic exporters when you want request telemetry fan-out to be configurable without restarting the gateway process.

See [Observability exporters](observability-exporters.md).

## Health, metrics, and playground

### `GET /admin/v1/health`

This is the operator-facing health endpoint.

It reports top-level health plus current model health state.

Use it to answer operator questions such as:

- is the admin surface alive
- does the process have a current snapshot
- are configured models currently healthy from the gateway's point of view

### `GET /metrics`

This is the Prometheus scrape endpoint on the admin listener.

### `POST /playground/chat/completions`

The standalone admin playground forwards requests to `/v1/chat/completions` through the local proxy router.

Important current behavior:

- it expects a **proxy** API key, not an admin key
- it forwards into the same proxy path used by normal caller traffic
- it runs the full proxy middleware path

This is useful for operator debugging because it exercises the normal proxy stack while avoiding a separate client setup step.

## Verify

Verify that the admin surface is reachable:

```shell
curl -sS http://127.0.0.1:3001/admin/v1/health \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Then create a provider key, model, and API key as shown in [Understand admin resources](../quickstart/first-model-first-key-first-request.md).

## Troubleshooting

### `401` on `/admin/v1/*`

Check the bootstrap admin key first. Do not test with a proxy caller key.

### A resource is created but proxy traffic still fails

Check configuration propagation before recreating the resource. Poll
`/v1/models` with the caller key, or retry the target proxy endpoint, until the
updated snapshot is visible.

### `409` on create

The most common cause is a duplicate logical name such as `display_name`.

## Next steps

- [Configuration overview](overview.md) — understand where the admin API fits
  in the configuration model.
- [Bootstrap configuration](bootstrap-config.md) — configure the admin listener and bootstrap admin keys.
- [Provider keys](provider-keys.md), [Models](models.md), and [API keys](api-keys.md) — create the minimum resources for proxy traffic.
- [Guardrails](guardrails.md), [Caching](caching.md), and [Observability exporters](observability-exporters.md) — add policy and telemetry resources.
- [Understand admin resources](../quickstart/first-model-first-key-first-request.md) — follow a resource-by-resource setup walkthrough.
- [OpenAI-compatible API](../integration/openai-compatible-api.md) — call the proxy after resources are configured.
