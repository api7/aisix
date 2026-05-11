---
title: Admin API Reference
description: Reference for the current standalone AISIX AI Gateway admin API surface.
sidebar_position: 61
---

## Public Admin-Listener Routes

- `GET /health`
- `GET /metrics`
- `GET /admin/openapi.json`
- `GET /admin/openapi-scalar`

## Authenticated Operator Routes

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

## Auth Model

Current authenticated operator routes use:

- `Authorization: Bearer <admin-key>`
- `x-api-key: <admin-key>` fallback

`POST /playground/chat/completions` expects a proxy API key, not an admin key.

## Related Pages

- [Admin API](../configuration/admin-api.md)
- [Resource Schemas](resource-schemas.md)
- [Headers And Error Codes](headers-and-error-codes.md)
