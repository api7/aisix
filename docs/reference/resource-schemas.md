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
- shared `RateLimit`
- shared `Routing`

## Key Schema Notes

- `Model` is either direct upstream config or a routing model, never both.
- `ApiKey` requires `key_hash` and `allowed_models`.
- `ProviderKey` requires `display_name` and `secret`.
- `Guardrail` is discriminated by `kind` with current `keyword` and `bedrock` shapes.
- `CachePolicy` currently documents `name`, `enabled`, `backend`, `ttl_seconds`, and `applies_to`.
- `ObservabilityExporter` is currently `kind=otlp_http` only.

## Related Pages

- [Models](../configuration/models.md)
- [API Keys](../configuration/api-keys.md)
- [Provider Keys](../configuration/provider-keys.md)
- [Guardrails](../configuration/guardrails.md)
- [Caching](../configuration/caching.md)
- [Observability Exporters](../configuration/observability-exporters.md)
