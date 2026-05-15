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

These are operator-facing helper and discovery routes, not dynamic-resource write paths.

## Authenticated Operator Routes

- `GET|POST /admin/v1/models`
- `GET|PUT|DELETE /admin/v1/models/:id`
- `GET /admin/v1/models/status`
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

Treat these as the standalone control surface for dynamic configuration.

## Auth Model

Current authenticated operator routes use:

- `Authorization: Bearer <admin-key>`
- `x-api-key: <admin-key>` fallback

This auth model is separate from proxy caller API keys.

`POST /playground/chat/completions` expects a proxy API key, not an admin key.

## Route Groups

- public admin-listener routes: process visibility and OpenAPI discovery
- CRUD resource routes: models, API keys, provider keys, guardrails, cache policies, exporters
- runtime model-state route: `/admin/v1/models/status`
- authenticated operator health route: `/admin/v1/health`
- operator convenience route: `/playground/chat/completions`

## Resources Without Admin CRUD

`RateLimitPolicy` rows are loaded from etcd directly and do not currently have admin API CRUD routes. Provision them through your control-plane projection or, in self-hosted setups, by writing under the etcd `rate_limit_policies/<id>` prefix. See [Rate Limits § Rate-Limit Policy Entities](../configuration/rate-limits.md#rate-limit-policy-entities).

## Runtime Model Status

`GET /admin/v1/models/status` returns one row per model in the current snapshot.

Current behavior is:

- direct models return runtime state keyed by their resolved model `id`
- routing models return `kind = routing` and `status = not_applicable`
- request-path retryable failures can mark a direct model `cooldown`
- background checks can mark a direct model `unhealthy`
- ignored background statuses like `408` or `429` remain visible through `last_check_status` and `status_reason = ignored_transient_error` without marking the model unhealthy

This route is the routing-exclusion signal.

`GET /admin/v1/health` remains the higher-level aggregated health view and does not define runtime routing exclusion.

## Related Pages

- [Admin API](../configuration/admin-api.md)
- [Resource Schemas](resource-schemas.md)
- [Headers And Error Codes](headers-and-error-codes.md)
