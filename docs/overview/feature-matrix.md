---
title: Feature Matrix
description: Review the current AISIX AI Gateway and AISIX Cloud feature matrix, including available, limited, and preview capabilities.
sidebar_position: 4
---

This matrix summarizes the current product surface for AISIX AI Gateway and AISIX Cloud.

Use it as a navigation aid, not as a replacement for detailed feature pages.

## Status Labels

- **Available**: documented as current customer-facing behavior
- **Limited**: available with important runtime or scope limitations
- **Preview**: customer-visible but not production-equivalent or not yet broad enough to describe as generally available
- **Planned**: not documented as current behavior; see the [Roadmap](../roadmap.md)

## AISIX AI Gateway

| Capability | Status | Notes |
| --- | --- | --- |
| OpenAI-compatible proxy API | Available | Includes chat, completions, embeddings, images, audio, responses, rerank, and passthrough routes currently wired by the proxy router. |
| Anthropic-style `/v1/messages` path | Available | Current behavior is implemented as a first-class route. Feature depth still varies by provider and message content shape. |
| Multi-provider model support | Available | The gateway supports multiple provider families and provider integrations, with endpoint depth varying by provider and route. |
| Provider-specific passthrough | Available | Use `/passthrough/:provider/*rest` for unsupported or provider-native routes. |
| Standalone admin API | Available | Current admin surface includes models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, OpenAPI, and playground. |
| API key allowlist authz | Available | Uses hashed caller keys and model allowlists. |
| Per-key budgets | Limited | Standalone deployments do not enforce per-key budgets; managed deployments enforce via the budget controller. |
| Rate limits and concurrency limits | Available | Three layers are AND-combined per request: `ApiKey.rate_limit`, `Model.rate_limit`, and scope-matched `RateLimitPolicy` rows (`api_key` / `model` / `team` / `member`). |
| Routing models and failover | Available | Current model schema supports routing strategies and retry budget behavior. |
| Keyword guardrails | Available | Current runtime enforcement is on `POST /v1/chat/completions`; non-chat endpoints do not run the guardrail chain today. |
| Bedrock guardrails | Limited | Current code includes feature-gated runtime wiring. Treat it as an advanced capability with deployment and support caveats rather than as a planned-only feature. |
| Memory-backed response caching | Available | Current cache policy behavior centers on memory-backed caching. |
| Redis-backed cache policy | Limited | Current code includes Redis backend selection and connection logic. Treat it as implemented with support caveats until the full cache docs land. |
| Observability exporters | Available | Current admin surface and resource model include observability exporters. |

## How To Use This Matrix

Use the matrix to answer three questions:

1. Is this capability part of the current product surface?
2. Is it broadly documented as current behavior, or does it have important limits?
3. Which detailed page should I read next before I depend on it?

If a capability is marked **Available**, use the main docs as the current contract.

If a capability is marked **Limited** or **Preview**, expect narrower scope, sharper edges, or workflow caveats and read the detailed page before you adopt it.

## AISIX Cloud

| Capability | Status | Notes |
| --- | --- | --- |
| Environment-scoped control plane | Available | Current Cloud code and e2e flows are built around environments as first-class resources. |
| Gateway certificate issuance | Available | Current managed-data-plane bootstrap flow is certificate-based. |
| Managed data-plane heartbeat and telemetry | Available | Current `/dp/*` surface is mTLS-authenticated in AISIX Cloud. |
| Resource projection into env-scoped data plane | Available | Current architecture and tests rely on control-plane projection into env-scoped configuration paths. |
| Usage events and billing workflows | Available | Current Cloud e2e coverage includes usage-event and billing-oriented flows. |
| Cloud playground | Preview | Current Cloud playground goes directly from the control plane to the upstream provider and does not represent full data-plane behavior. |
| Advanced governance and team controls | Planned | Keep future governance detail on the roadmap until backed by current product behavior. |

## Related Pages

- [What Is AISIX AI Gateway](what-is-aisix-ai-gateway.md)
- [Deployment Modes](deployment-modes.md)
- [Roadmap](../roadmap.md)
