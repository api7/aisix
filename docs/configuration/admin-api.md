---
title: Admin API
description: Use the AISIX AI Gateway admin API to manage models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, and the standalone playground.
sidebar_position: 31
toc_max_heading_level: 2
---

The AISIX AI Gateway admin API manages dynamic configuration in standalone self-hosted deployments.

:::note Standalone Only
The `admin/v1` examples apply to self-hosted standalone AISIX. A [Cloud managed data plane](../quickstart/aisix-cloud-managed-dp.md) only exposes proxy APIs locally and does **not** bind the standalone admin listener. Provider keys, models, and caller API keys are managed through the AISIX Cloud control plane instead.
:::

The admin API creates and updates models, rotates caller API keys, manages upstream provider credentials, configures guardrails and cache policies, manages observability exporters, and reports management health. It is the write path for standalone deployments, not a caller-facing integration API.

## Admin API Access

In standalone mode, the admin API runs on the admin listener configured in bootstrap config.

Admin authentication is static and bootstrap-based for the authenticated administrative routes. Admin keys come from `config.admin.admin_keys`; `/admin/v1/*` routes expect `Authorization: Bearer <key>`, and `x-api-key` is accepted as a fallback.

The public admin-listener routes are `GET /livez`, `GET /metrics`, `GET /admin/openapi.json`, and `GET /admin/openapi-scalar`.

Example:

```shell
curl -sS "http://127.0.0.1:3001/admin/v1/models" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Operationally, admin keys and proxy caller API keys are different secrets. Admin keys authorize administrative access to `/admin/v1/*`; proxy caller API keys authorize client traffic to `/v1/*`. Do not mix them.

The admin API does **not** use the OpenAI-style proxy error format. It uses a simpler envelope:

```json
{
  "error_msg": "missing or malformed admin authorization"
}
```

| Status | Meaning |
| --- | --- |
| `400` | Bad request or schema validation failure. |
| `401` | Missing or invalid admin auth. |
| `404` | Missing resource. |
| `409` | Conflict such as a duplicate name. |
| `500` | Store failure. |

Public routes such as `/livez`, `/metrics`, and the OpenAPI endpoints do not require admin auth.

Use `GET /livez` for simple admin-listener reachability. Use `GET /admin/v1/health` when you need authenticated per-model health.

For automation, plan to branch on admin status codes and `error_msg`, not on the proxy-side OpenAI-compatible error envelope.

## Manage Standalone Resources

Admin routes fall into four groups.

Public admin-listener helpers cover liveness, metrics, and OpenAPI discovery.

CRUD resources cover models, API keys, provider keys, guardrails, cache policies, and observability exporters.

Runtime status endpoints expose per-model runtime status and aggregated admin health.

The standalone playground forwards a local chat-completions request through the proxy path for debugging.

Some runtime resources are loaded from the config store without standalone admin CRUD routes. `RateLimitPolicy` rows and `GuardrailAttachment` rows can be projected by a control plane, but standalone admin does not expose `/admin/v1/rate_limit_policies` or `/admin/v1/guardrail_attachments` routes.

Use the admin API to manage standalone dynamic resources. Start with the task guide for the resource you want to configure, then use the [Admin API reference](/ai-gateway/reference/admin-api) for exact routes, request schemas, and response schemas. To export the OpenAPI document or review which standalone resources it covers, see [Admin API Reference](../reference/admin-api-reference.md).

Models map caller-visible model aliases to upstream models or routing targets. API keys authenticate callers, control model access, and rotate caller keys. Provider keys store upstream credentials, base URLs, provider labels, and adapter families. Guardrails define request or response policy resources. Cache policies match requests that can use response caching, and observability exporters send dynamic request telemetry to OTLP or HTTP destinations.

Some resources have important workflow behavior. API-key rotation returns the new plaintext key exactly once, so capture the rotation response immediately. Standalone-created guardrails need attachment rows to scope them; without those rows, they can apply more broadly than intended. Plain `http://` exporter endpoints are rejected unless they target approved local development hosts.

## Health, Metrics, and Playground

**`GET /admin/v1/health`**

This is the authenticated management health endpoint.

It reports top-level health plus model health state.

Use it to confirm that the admin API is alive, that the process has a loaded proxy configuration, and that configured models are healthy from the gateway's point of view.

**`GET /metrics`**

This is the Prometheus scrape endpoint on the admin listener.

**`POST /playground/chat/completions`**

The standalone admin playground forwards requests to `/v1/chat/completions` through the local proxy router.

The playground expects a **proxy** API key, not an admin key. It forwards into the same proxy path used by normal caller traffic and runs the full proxy middleware path.

This is useful for local debugging because it exercises the normal proxy processing path while avoiding a separate client setup step.

## Verify

Verify that the admin API is reachable:

```shell
curl -sS "http://127.0.0.1:3001/admin/v1/health" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Then create a provider key, model, and API key as shown in [Understand admin resources](../quickstart/first-model-first-key-first-request.md).

## Troubleshooting

**`401` on `/admin/v1/*`**

Check the bootstrap admin key first. Do not test with a proxy caller key.

**A Resource Is Created but Proxy Traffic Still Fails**

Check configuration propagation before recreating the resource. Poll `/v1/models` with the caller key, or retry the target proxy endpoint, until the updated configuration is visible to the proxy.

**`409` on Create**

The most common cause is a duplicate logical name such as `display_name`.

## Related Reading

[Configuration overview](overview.md) places the admin API in the configuration model, and [Bootstrap configuration](bootstrap-config.md) covers the admin listener and bootstrap admin keys. For resource setup, see [Provider keys](provider-keys.md), [Models](models.md), [API keys](api-keys.md), [Guardrails](guardrails.md), [Caching](caching.md), and [Observability exporters](observability-exporters.md). For a full setup walkthrough, see [Understand admin resources](../quickstart/first-model-first-key-first-request.md), then call the proxy through the [OpenAI-compatible API](../integration/openai-compatible-api.md).
