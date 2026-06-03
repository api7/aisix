---
title: Admin API Source Notes
description: Understand where the generated AISIX AI Gateway Admin API reference comes from and what it covers.
sidebar_label: Admin API source notes
sidebar_position: 62
---

The standalone admin API publishes an OpenAPI 3.1 document from the gateway process.

Use the generated [Admin API reference](/ai-gateway/reference/admin-api) for the exact route list, request schemas, response schemas, and status-code details. This page explains where that generated reference comes from, how to export it, and which resource boundaries still live outside standalone admin CRUD.

## Open the generated reference

The docs site renders the generated OpenAPI document with Redoc:

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
curl -sS http://127.0.0.1:3001/admin/openapi.json \
  -o aisix-admin-openapi.json
```

## What the generated reference covers

The generated document covers the routes mounted by the standalone admin router. This avoids maintaining a second hand-written route inventory in the docs site.

It includes:

- public liveness, metrics, and OpenAPI discovery routes
- authenticated admin CRUD routes for resources such as provider keys, models, caller API keys, guardrails, cache policies, and observability exporters
- playground routes mounted on the admin listener
- request and response schemas merged from generated resource schemas

The generated reference does not describe the proxy API surface. For proxy endpoints such as `/v1/chat/completions`, see [Proxy API reference](proxy-api-reference.md).

## Source of truth

The admin API specification is generated from the AI Gateway repo and rendered by the docs site. Resource schemas are merged from:

```text
schemas/resources/
```

If the generated reference and a prose guide disagree, prefer the generated reference for the exact route, request, response, and status-code contract. Then update the prose guide so operators do not have to reconcile two different descriptions.

## Auth boundary

The public admin-listener routes are liveness, metrics, and OpenAPI discovery. Authenticated admin routes use the configured admin key:

```http
Authorization: Bearer <admin-key>
```

`x-api-key: <admin-key>` is also accepted on admin auth paths.

This is separate from proxy caller API keys. `POST /playground/chat/completions` expects a proxy API key because it forwards through the proxy router.

## Managed data-plane boundary

The standalone admin API is not exposed on AISIX Cloud managed data planes.

In managed mode, use the AISIX Cloud control plane for provider keys, models, caller API keys, and related configuration. The local data plane exposes proxy APIs, not the standalone admin listener.

## Resources outside standalone admin CRUD

Some runtime snapshot resources do not currently have standalone admin CRUD routes.

`RateLimitPolicy` rows are loaded from etcd directly and can be projected by a control plane. In self-hosted setups where you manage etcd directly, write them under the etcd `rate_limit_policies/<id>` prefix. See [Rate limits](../configuration/rate-limits.md#add-a-policy-limit).

`GuardrailAttachment` rows bind guardrail definitions to `env`, `model`, `api_key`, or `team` scopes and are loaded from `guardrail_attachments/<id>`. See [Guardrails](../configuration/guardrails.md#scope-guardrails).

## Related docs

- [Admin API](../configuration/admin-api.md) — operator workflow and examples.
- [Resource schemas](resource-schemas.md) — resource shape reference.
- [Headers and error codes](headers-and-error-codes.md) — admin error envelope and status boundaries.
