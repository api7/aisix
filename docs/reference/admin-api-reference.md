---
title: Admin API Reference
description: Open the AISIX AI Gateway Admin API reference and understand its coverage.
sidebar_label: Admin API Reference
sidebar_position: 62
---

The standalone admin API publishes an OpenAPI 3.1 document from the gateway process. Use the [Admin API reference](/ai-gateway/reference/admin-api) for exact routes, request schemas, response schemas, and status-code details.

Open the OpenAPI reference from the hosted documentation or from a self-hosted gateway when you need the exact route contract. Some dynamic resources sit outside standalone admin CRUD and are noted below.

## Open the Admin API Reference

Open the Admin API reference at:

```text
/ai-gateway/reference/admin-api
```

When you run a self-hosted gateway, you can also open the live Scalar UI on the admin listener:

```text
http://127.0.0.1:3001/admin/openapi-scalar
```

The UI loads the machine-readable OpenAPI document from:

```text
http://127.0.0.1:3001/admin/openapi.json
```

You can also export the spec directly:

```shell
curl -sS "http://127.0.0.1:3001/admin/openapi.json" \
  -o aisix-admin-openapi.json
```

## Coverage

The OpenAPI document covers the routes mounted by the standalone admin router, so the route list, request schemas, and response schemas stay aligned with the running gateway.

It includes public admin-listener routes, authenticated admin CRUD routes, playground routes, and the resource schemas used by dynamic gateway resources.

For exact route, request, response, and status-code behavior, use the Admin API reference.

The Admin API reference does not describe the proxy API. For proxy endpoints such as `/v1/chat/completions`, see [Proxy API reference](proxy-api-reference.md).

## Authentication

The public admin-listener routes are liveness, metrics, and OpenAPI discovery. Authenticated admin routes use the configured admin key:

```http
Authorization: Bearer <admin-key>
```

`x-api-key: <admin-key>` is also accepted on admin auth paths.

This is separate from proxy caller API keys. `POST /playground/chat/completions` expects a proxy API key because it forwards through the proxy router.

## Managed Data Planes

The standalone admin API is not exposed on AISIX Cloud managed data planes.

In managed mode, use the AISIX Cloud control plane for provider keys, models, caller API keys, and related configuration. The local data plane exposes proxy APIs, not the standalone admin listener.

## Resources Outside Standalone Admin CRUD

Some dynamic resources do not have standalone admin CRUD routes.

`RateLimitPolicy` rows can be loaded from etcd or projected by a control plane. In self-hosted setups where you manage etcd directly, write them under the etcd `rate_limit_policies/<id>` prefix. See [Rate limits](../configuration/rate-limits.md#add-a-scoped-policy).

`GuardrailAttachment` rows bind guardrail definitions to `env`, `model`, `api_key`, or `team` scopes and are loaded from `guardrail_attachments/<id>`. See [Guardrails](../configuration/guardrails.md#scope-guardrails).

## Related Reading

[Admin API](../configuration/admin-api.md) covers standalone workflows and examples. For resource schemas and admin error behavior, see [Resource schemas](resource-schemas.md) and [Headers and error codes](headers-and-error-codes.md).
