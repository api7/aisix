---
title: Resource Schemas
description: How to use JSON Schemas for AISIX AI Gateway dynamic resources.
sidebar_position: 62
---

AISIX dynamic resource schemas describe the configuration objects that the gateway accepts at runtime. Use schema files for exact field details, and use configuration guides for workflows, examples, and runtime behavior.

Each dynamic resource has a schema file and a matching Admin API reference.

## Choose the Right Reference

For day-to-day configuration, start with the task guides:

- [Provider keys](../configuration/provider-keys.md)
- [Models](../configuration/models.md)
- [API keys](../configuration/api-keys.md)
- [Guardrails](../configuration/guardrails.md)
- [Caching](../configuration/caching.md)
- [Observability exporters](../configuration/observability-exporters.md)
- [Rate limits](../configuration/rate-limits.md)

For exact admin request and response bodies, use the [Admin API reference](/ai-gateway/reference/admin-api).

## Resource Schema Files

Choose the schema file that sits closest to the configuration task.

| Resource | Schema file | What it describes |
| --- | --- | --- |
| `Model` | `model.schema.json` | Caller-visible model aliases, direct upstream targets, and routing models. |
| `ApiKey` | `api_key.schema.json` | Caller identity, model access, and key-level policy. |
| `ProviderKey` | `provider_key.schema.json` | Upstream credentials, adapter family, base URL, passthrough protections, and provider metadata. |
| `Guardrail` | `guardrail.schema.json` | Content-policy resources. |
| `CachePolicy` | `cache_policy.schema.json` | Response-cache matching and TTL. |
| `ObservabilityExporter` | `observability_exporter.schema.json` | Dynamic telemetry exporter configuration. |
| `RateLimitPolicy` | `rate_limit_policy.schema.json` | Scoped request or token quotas. |
| `RateLimit` | `rate_limit.schema.json` | Shared request, token, and concurrency limit schema embedded by other resources. |
| `Routing` | `routing.schema.json` | Shared routing-target and failover schema embedded by models. |

`GuardrailAttachment` rows bind guardrails to `env`, `model`, `api_key`, or `team` scopes in the loaded configuration. See [Guardrails](../configuration/guardrails.md#scope-guardrails).

## Admin API Reference

The standalone admin OpenAPI document includes these schemas in the Admin API reference:

```text
/ai-gateway/reference/admin-api
```

When you run a self-hosted gateway locally, you can also open the live Scalar reference from the admin listener at `http://127.0.0.1:3001/admin/openapi-scalar`.

## Runtime Behavior

Not every schema field implies broad runtime support on every path.

For example, `Model.rate_limit` and `ApiKey.rate_limit` are enforced alongside matching `RateLimitPolicy` rows. `Model.background_model_check` applies to direct models and appears through `/admin/v1/models/status`. Remote guardrail kinds such as `bedrock` and `azure_content_safety` depend on provider credentials, network access, build features, and `fail_open`.

If a schema accepts a field but a configuration guide does not describe runtime behavior for that field, validate the behavior in your deployment before using the field in production.
