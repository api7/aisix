---
title: Resource Schemas
description: Reference for the current dynamic resource shapes used by AISIX AI Gateway.
sidebar_position: 62
---

## Current Dynamic Resource Types

- `Model`
- `ApiKey`
- `ProviderKey`
- `Guardrail`
- `CachePolicy`
- `ObservabilityExporter`
- `RateLimitPolicy`
- shared `RateLimit`
- shared `Routing`

Use this page as the schema map. Use the configuration pages when you need operator guidance and examples.

## Key Schema Notes

- `Model` is either direct upstream config or a routing model, never both.
- `Model.background_model_check` is direct-model-only.
- `ApiKey` requires `key_hash` and `allowed_models`.
- `ProviderKey` requires `display_name` and `secret`.
- `Guardrail` is discriminated by `kind` with current `keyword` and `bedrock` shapes.
- `CachePolicy` currently documents `name`, `enabled`, `backend`, `ttl_seconds`, and `applies_to`.
- `ObservabilityExporter` is currently `kind=otlp_http` only.
- `RateLimitPolicy` requires `name`, `scope` (`api_key` / `model` / `team` / `member`), `scope_ref`, and `window` (`second` / `minute` / `hour`); at least one of `max_requests` or `max_tokens` must be set. The standalone admin API does not currently expose CRUD routes for it — rows are written directly under the etcd `rate_limit_policies/<id>` prefix.

## How To Read These Schemas

- `Model` defines the caller-visible target contract
- `ApiKey` defines caller identity, authorization, and some policy
- `ProviderKey` defines upstream credential and base-url wiring
- `Guardrail`, `CachePolicy`, and `ObservabilityExporter` are dynamic policy or telemetry resources layered onto the serving path

## Runtime Versus Schema Boundary

Not every field or shape present in the schema should be interpreted as equally broad runtime support.

Examples:

- the typed `ApiKey` model and admin OpenAPI mention `max_budget_usd`, but the current standalone admin write validator rejects it and standalone hard-stop budget behavior is not the current documented default
- `Model.rate_limit` and `ApiKey.rate_limit` are both enforced today, alongside scope-matched `RateLimitPolicy` rows. See [Rate Limits](../configuration/rate-limits.md) for the layer order and the AND-combination semantics.
- `Model.background_model_check` exists in schema, but it only applies to direct models and its runtime effect is surfaced through `/admin/v1/models/status`
- `Guardrail.kind = bedrock` exists in schema, but current generally reliable runtime behavior is strongest on `keyword`

## Related Pages

- [Models](../configuration/models.md)
- [API Keys](../configuration/api-keys.md)
- [Provider Keys](../configuration/provider-keys.md)
- [Guardrails](../configuration/guardrails.md)
- [Caching](../configuration/caching.md)
- [Observability Exporters](../configuration/observability-exporters.md)
- [Rate Limits](../configuration/rate-limits.md)
