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
| Multi-provider model support | Available | Current provider enum includes OpenAI, Anthropic, Google (Gemini), DeepSeek, Cohere, and Jina. |
| Provider-specific passthrough | Available | Use `/passthrough/:provider/*rest` for unsupported or provider-native routes. |
| Standalone admin API | Available | Current admin surface includes models, API keys, provider keys, guardrails, cache policies, observability exporters, health, metrics, OpenAPI, and playground. |
| API key allowlist authz | Available | Uses hashed caller keys and model allowlists. |
| Per-key budgets | Limited | Standalone deployments do not enforce per-key budgets; managed deployments enforce via the budget controller. |
| Rate limits and concurrency limits | Available | Three layers are AND-combined per request: `ApiKey.rate_limit`, `Model.rate_limit`, and scope-matched `RateLimitPolicy` rows (`api_key` / `model` / `team` / `member`). |
| Routing models and failover | Available | Current model schema supports routing strategies and retry budget behavior. |
| Guardrails | Available | Content-policy enforcement on the input and output hooks across all guarded proxy surfaces. Kinds: `keyword` (in-process), AWS Bedrock (`ApplyGuardrail`), Azure AI Content Safety (Prompt Shield and Text Moderation), and Aliyun content moderation. A block returns `422 content_filter`. See [Guardrails](../configuration/guardrails.md). |
| Memory-backed response caching | Available | Current cache policy behavior centers on memory-backed caching. |
| Redis-backed cache policy | Limited | Current code includes Redis backend selection and connection logic. Treat it as implemented with support caveats until the full cache docs land. |
| Observability exporters | Available | Current admin surface and resource model include observability exporters. |

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

## How To Read This Matrix

If a capability is marked **Available**, use the main docs.

If a capability is marked **Limited** or **Preview**, read the corresponding page carefully for current boundaries.

If a capability is marked **Planned**, use the [Roadmap](../roadmap.md) instead of expecting a full product page.

## Related Pages

- [What Is AISIX AI Gateway](what-is-aisix-ai-gateway.md)
- [Deployment Modes](deployment-modes.md)
- [Roadmap](../roadmap.md)
