---
title: Resource Schemas
description: How to use the generated JSON Schemas for AISIX AI Gateway dynamic resources.
sidebar_position: 62
---

AISIX dynamic resource schemas describe the configuration objects that
the gateway accepts at runtime. Use the generated schema files for the
exact field contract, and use the configuration guides for workflows,
examples, and runtime behavior.

This page explains which schema file describes each resource and how
those schemas connect to the generated Admin API reference.

## Start with the schema or guide you need

For day-to-day configuration, start with the task guides:

- [Provider keys](../configuration/provider-keys.md)
- [Models](../configuration/models.md)
- [API keys](../configuration/api-keys.md)
- [Guardrails](../configuration/guardrails.md)
- [Caching](../configuration/caching.md)
- [Observability exporters](../configuration/observability-exporters.md)
- [Rate limits](../configuration/rate-limits.md)

For the exact admin request and response bodies, use the generated [Admin API reference](/ai-gateway/reference/admin-api).

## Start with the schema file that matches your task

The generated files are stored in the AI Gateway repo under:

```text
schemas/resources/
```

Start with the resource that sits closest to the operator task.

| Resource | Schema file | What it describes |
| --- | --- | --- |
| `Model` | `model.schema.json` | Caller-visible model aliases, direct upstream targets, and routing models. |
| `ApiKey` | `api_key.schema.json` | Caller identity, model access, and key-level policy. |
| `ProviderKey` | `provider_key.schema.json` | Upstream credentials, adapter family, base URL, passthrough protections, and provider metadata. |
| `Guardrail` | `guardrail.schema.json` | Content-policy resources. |
| `CachePolicy` | `cache_policy.schema.json` | Response-cache matching and TTL. |
| `ObservabilityExporter` | `observability_exporter.schema.json` | Dynamic telemetry exporter configuration. |
| `RateLimitPolicy` | `rate_limit_policy.schema.json` | Scoped request or token quotas. |
| `RateLimit` | `rate_limit.schema.json` | Shared request, token, and concurrency limit shape embedded by other resources. |
| `Routing` | `routing.schema.json` | Shared routing-target and failover shape embedded by models. |

`GuardrailAttachment` rows bind guardrails to `env`, `model`, `api_key`, or
`team` scopes in the runtime snapshot, but standalone admin CRUD and a
standalone generated attachment schema file are not exposed today. See
[Guardrails](../configuration/guardrails.md#scope-guardrails).

## Unknown fields and strict validation

Most generated resource schemas reject unknown top-level fields.

Three schemas intentionally allow forward-compatible unknown fields:

- `guardrail.schema.json`
- `cache_policy.schema.json`
- `observability_exporter.schema.json`

If you validate resources outside AISIX, do not force these three schemas into strict unknown-field rejection. They can accept forward-compatible fields while the data plane, control plane, or dashboard rolls out new resource variants.

## How schemas feed the Admin API reference

The standalone admin OpenAPI document merges these schemas into the generated admin reference. Open the docs-site Redoc reference at:

```text
http://127.0.0.1:3000/ai-gateway/reference/admin-api
```

When you run a self-hosted gateway locally, you can also open the live Scalar reference from the admin listener at `http://127.0.0.1:3001/admin/openapi-scalar`.

## Runtime boundaries

Not every schema field implies broad runtime support on every path.

For example, `Model.rate_limit` and `ApiKey.rate_limit` are enforced today alongside matching `RateLimitPolicy` rows. `Model.background_model_check` applies to direct models and surfaces through `/admin/v1/models/status`. Remote guardrail kinds such as `bedrock` and `azure_content_safety` depend on provider credentials, network access, build features, and `fail_open`.

When a schema and a behavior guide appear to differ, treat the generated schema and admin reference as the accepted input contract, then verify the runtime behavior in the relevant configuration guide.

## If a schema and prose page differ

Treat the generated schema and generated admin reference as the exact accepted input contract. Treat the task guides as the explanation of how to use that contract safely.

If a prose page lists a field differently from the generated reference, prefer the generated reference and update the prose page. If the generated reference accepts or rejects a field incorrectly, fix the source that produces the generated schema rather than maintaining a separate hand-written field list in docs.
