---
title: Feature Status
description: Review the current AISIX AI Gateway and AISIX Cloud feature surface by status, including available, limited, preview, and planned capabilities.
sidebar_position: 4
---

This reference shows which capabilities are part of the current AISIX AI Gateway surface so you can plan around the current product boundary.

## Quick read

For self-hosted gateway evaluation, start with OpenAI-compatible proxying, caller API keys, provider keys, model aliases, routing models, rate limits, and observability exporters. These are documented as current gateway behavior.

Before depending on policy-heavy paths in production, read the limits for budgets, guardrails, and response caching. These features work in narrower runtime scopes or depend on Cloud connectivity, provider credentials, build features, or endpoint family.

For managed deployments, treat AISIX Cloud as the control plane for environments, certificates, configuration projection, usage events, and billing workflows. The Cloud playground is useful for control-plane feedback, but it is not the same as sending live traffic through a managed data plane.

## Status labels

Status labels mean:

- **Available**: documented as current customer-facing behavior.
- **Limited**: available with important runtime or scope limitations.
- **Preview**: customer-visible, but not production-equivalent or not broad enough to describe as generally available.
- **Planned**: not documented as current behavior.

If a capability is marked **Limited** or **Preview**, read the linked feature page before depending on it in production.

## AISIX AI Gateway

| Capability | Status | Current boundary |
| --- | --- | --- |
| OpenAI-compatible proxy API | Available | The proxy listener exposes OpenAI-shaped chat, completions, embeddings, image, audio, responses, rerank, and model-discovery paths. Provider and endpoint support still depends on the configured model adapter. |
| Anthropic-style Messages API | Available | `/v1/messages` and `/v1/messages/count_tokens` are first-class proxy routes. Message conversion and usage reporting vary by upstream provider and streaming mode. |
| Multi-provider model support | Available | Models can point at OpenAI-compatible providers and provider-specific adapters. Endpoint depth varies by provider and route. |
| Provider-specific passthrough | Available | `/passthrough/:provider/*rest` forwards provider-native routes that are not modeled by the gateway API. |
| Standalone admin API | Available | The self-hosted admin listener manages models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, OpenAPI, and playground resources. Managed data planes do not expose this listener. |
| Caller API key authentication | Available | Caller keys are stored as hashes, and each key carries an `allowed_models` list. Empty allowlists deny all models; `*` allows all models in scope. |
| Rate limits and concurrency limits | Available | The proxy evaluates inline key/model limits and matching `RateLimitPolicy` rows. Any configured layer can reject a request with `429`. See [Rate limits](../configuration/rate-limits.md). |
| Routing models and failover | Available | Routing models select among target models at request time. Current strategies include failover, round-robin, and weighted routing. See [Routing and failover](../configuration/routing-and-failover.md). |
| Observability exporters | Available | Observability exporter resources can forward per-request span telemetry over OTLP/HTTP to an external tracing backend. See [Observability exporters](../configuration/observability-exporters.md). |
| Budget checks | Limited | Budget checks are enforced when a managed data plane is connected to the Cloud budget-check endpoint. Standalone self-hosted deployments use the disabled budget client and allow requests through. See [Budgets](../configuration/budgets.md). |
| Keyword guardrails | Limited | Keyword guardrails run locally on `POST /v1/chat/completions` and `POST /v1/messages`. Other proxy endpoints do not run the same guardrail chain today. See [Guardrails](../configuration/guardrails.md). |
| Remote guardrails | Limited | Bedrock and Azure Content Safety guardrails are runtime-backed remote checks. They require provider credentials, network reachability, relevant build features, and a deliberate `fail_open` choice. See [Guardrails](../configuration/guardrails.md). |
| Response caching | Limited | Cache lookup and write are policy-gated. Current enforcement is on chat completions, with per-policy TTL applied to matching requests. Streaming responses are not cached at this layer. See [Caching](../configuration/caching.md). |
| Redis cache backend | Limited | The process-level cache backend can be switched from memory to Redis when `cache.backend` is `redis` and `cache.redis.url` is configured. A `CachePolicy.backend` field alone does not switch the runtime backend. See [Caching](../configuration/caching.md). |

Use the [Proxy API reference](../reference/proxy-api-reference.md) for the
gateway surface and [Provider compatibility](../reference/provider-compatibility.md)
for provider-specific boundaries.

## AISIX Cloud

| Capability | Status | Current boundary |
| --- | --- | --- |
| Environment-scoped control plane | Available | Cloud resources are organized around environments as first-class operational scopes. |
| Gateway certificate issuance | Available | The current managed-data-plane bootstrap flow is certificate-based. |
| Managed data-plane heartbeat and telemetry | Available | The current `/dp/*` surface is mTLS-authenticated in AISIX Cloud. |
| Resource projection into environment-scoped data planes | Available | Control-plane resources are projected into environment-scoped managed data planes. |
| Usage events and billing workflows | Available | Managed data planes emit usage-oriented telemetry for Cloud-side usage and billing workflows. |
| Cloud playground | Preview | The current Cloud playground goes directly from the control plane to the upstream provider and does not represent full data-plane behavior. |
| Advanced governance and team controls | Planned | Keep future governance detail out of current product docs until backed by current behavior. |

## How to use this page

Use the status to answer three questions:

1. Is this capability part of the current product surface?
2. Is it broadly documented as current behavior, or does it have important limits?
3. Which detailed page should you read before depending on it?

## Next steps

- [Provider compatibility](../reference/provider-compatibility.md) — check provider-specific behavior.
- [Provider keys](../configuration/provider-keys.md) — configure upstream credentials and adapter details.
- [Models](../configuration/models.md) — configure direct and routing model aliases.
- [Rate limits](../configuration/rate-limits.md) — configure request, token, and concurrency limits.
- [Routing and failover](../configuration/routing-and-failover.md) — configure routing models and fallback behavior.
