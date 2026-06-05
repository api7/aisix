---
title: Feature Availability
description: Review which AISIX AI Gateway and AISIX Cloud capabilities are available and which have runtime limits.
sidebar_position: 4
---

Review feature availability before planning production rollout. The tables summarize AISIX AI Gateway and AISIX Cloud capabilities and call out limits that affect deployment, provider choice, or traffic policy.

## Planning Summary

For self-hosted gateway evaluation, start with OpenAI-compatible proxying, caller API keys, provider keys, model aliases, routing models, rate limits, and observability exporters.

Before depending on policy-heavy paths in production, read the limits for budgets, guardrails, and response caching. These features work in narrower runtime scopes or depend on Cloud connectivity, provider credentials, build features, or endpoint family.

For managed deployments, treat AISIX Cloud as the control plane for environments, certificates, configuration projection, usage events, and billing workflows. The Cloud playground is useful for control-plane feedback, but it is not the same as sending live traffic through a managed data plane.

## Status Labels

**Available** means the capability is ready for normal use in the supported deployment modes. **Limited** means the capability is available with important runtime or scope limitations. **Preview** means the capability is available for evaluation, but is not production-equivalent or not broad enough to describe as generally available.

If a capability is marked **Limited** or **Preview**, review its limits before depending on it in production.

## AISIX AI Gateway

| Capability | Status | Support note |
| --- | --- | --- |
| OpenAI-compatible proxy API | Available | The proxy listener exposes OpenAI-compatible chat, completions, embeddings, image, audio, responses, rerank, and model-discovery paths. Provider and endpoint support still depends on the configured model adapter. |
| Anthropic-style Messages API | Available | `/v1/messages` and `/v1/messages/count_tokens` are first-class proxy routes. Message conversion and usage reporting vary by upstream provider and streaming mode. |
| Multi-provider model support | Available | Models can point at OpenAI-compatible providers and provider-specific adapters. Endpoint depth varies by provider and route. |
| Provider-specific passthrough | Available | `/passthrough/:provider/*rest` forwards provider-native routes that are not modeled by the gateway API. |
| Standalone admin API | Available | The self-hosted admin listener manages models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, OpenAPI, and playground resources. Managed data planes do not expose this listener. |
| Caller API key authentication | Available | Caller keys are stored as hashes, and each key carries an `allowed_models` list. Empty allowlists deny all models; `*` allows all models in scope. |
| Rate limits and concurrency limits | Available | The proxy evaluates inline key/model limits and matching `RateLimitPolicy` rows. Any configured layer can reject a request with `429`. See [Rate limits](../configuration/rate-limits.md). |
| Routing models and failover | Available | Routing models select among target models at request time. Strategies include failover, round-robin, and weighted routing. See [Routing and failover](../configuration/routing-and-failover.md). |
| Observability exporters | Available | Observability exporter resources can forward per-request span telemetry over OTLP/HTTP to an external tracing backend. See [Observability exporters](../configuration/observability-exporters.md). |
| Budget checks | Limited | Budget checks are enforced when a managed data plane is connected to the Cloud budget-check endpoint. Standalone self-hosted deployments use the disabled budget client and allow requests through. See [Budgets](../configuration/budgets.md). |
| Keyword guardrails | Limited | Keyword guardrails run locally on `POST /v1/chat/completions` and `POST /v1/messages`. Other proxy endpoints do not run the same guardrail chain. See [Guardrails](../configuration/guardrails.md). |
| Remote guardrails | Limited | Bedrock and Azure Content Safety guardrails are runtime-backed remote checks. They require provider credentials, network reachability, relevant build features, and a deliberate `fail_open` choice. See [Guardrails](../configuration/guardrails.md). |
| Response caching | Limited | Cache lookup and write are policy-gated. Enforcement applies to chat completions, with per-policy TTL applied to matching requests. Streaming responses are not cached at this layer. See [Caching](../configuration/caching.md). |
| Redis cache backend | Limited | The process-level cache backend can be switched from memory to Redis when `cache.backend` is `redis` and `cache.redis.url` is configured. A `CachePolicy.backend` field alone does not switch the runtime backend. See [Caching](../configuration/caching.md). |

The [Proxy API reference](../reference/proxy-api-reference.md) covers the gateway API, and [Provider compatibility](../reference/provider-compatibility.md) covers provider-specific support.

## AISIX Cloud

| Capability | Status | Support note |
| --- | --- | --- |
| Environment-scoped control plane | Available | Cloud resources are organized around environments as first-class operational scopes. |
| Gateway certificate issuance | Available | The managed-data-plane bootstrap flow is certificate-based. |
| Managed data-plane heartbeat and telemetry | Available | The `/dp/*` endpoints are mTLS-authenticated in AISIX Cloud. |
| Resource projection into environment-scoped data planes | Available | Control-plane resources are projected into environment-scoped managed data planes. |
| Usage events and billing workflows | Available | Managed data planes emit usage-oriented telemetry for Cloud-side usage and billing workflows. |
| Cloud playground | Preview | The Cloud playground goes directly from the control plane to the upstream provider and does not represent full data-plane behavior. |

## Related Reading

For provider-specific behavior, see [Provider compatibility](../reference/provider-compatibility.md). Configure upstream access and aliases with [Provider keys](../configuration/provider-keys.md) and [Models](../configuration/models.md). For traffic controls and routing models, see [Rate limits](../configuration/rate-limits.md) and [Routing and failover](../configuration/routing-and-failover.md).
