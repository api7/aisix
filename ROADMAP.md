# Roadmap

This page lists capabilities that are planned or in progress but not yet generally available. It shows direction, not dates, and is not a delivery commitment.

For what the gateway does today — including [semantic routing](https://docs.api7.ai/ai-gateway/routing-and-resilience/semantic-routing), [ensemble models](https://docs.api7.ai/ai-gateway/routing-and-resilience/ensemble-models), [caching](https://docs.api7.ai/ai-gateway/add-traffic-controls/response-caching), and [guardrails](https://docs.api7.ai/ai-gateway/add-traffic-controls/guardrails) — see the [AISIX AI Gateway documentation](https://docs.api7.ai/ai-gateway/).

## How to read this page

- **Now** — in active design or development.
- **Next** — planned after the current focus.
- **Later** — on the longer-term horizon.

The **Surface** column shows where a capability lands: **Gateway** is the AISIX AI Gateway data plane; **Cloud** is the AISIX Cloud control plane and dashboard.

## Now

| Capability | What's planned | Surface |
| --- | --- | --- |
| MCP gateway | Register MCP servers as first-class resources and govern them with the same caller keys, teams, and policies as model traffic, including transport, authentication, per-tool access control, and per-server usage. | Gateway · Cloud |
| Enterprise SSO | Single sign-on through SAML and generic OIDC, beyond today's social logins. | Cloud |
| Directory sync (SCIM) | Provision and deprovision users and groups from your identity provider. | Cloud |
| Service accounts | Login-less, first-class principals for automated callers. | Cloud |
| Smart routing strategies | Cost-aware, latency-aware, and least-connections target selection, alongside today's ordered, weighted, failover, and semantic routing. | Gateway |
| Semantic caching | Serve responses for prompts close in meaning, on top of today's exact-match cache. | Gateway |
| PII guardrails | Detect and redact personally identifiable information in requests and responses. | Gateway |

## Next

| Capability | What's planned | Surface |
| --- | --- | --- |
| Fine-grained authorization | Custom roles with per-resource and per-action permissions, beyond today's fixed roles and read/write scopes. | Cloud |
| Conditional and wildcard routing | Route on request metadata, headers, and tags, and match upstreams by wildcard names such as `provider/*`. | Gateway |
| Prompt management | Store, version, and reuse prompt templates with variables, resolved at the gateway. | Gateway · Cloud |
| Caller key rotation experience | Self-service key rotation in the dashboard, plus scheduled auto-rotation with a grace overlap. | Cloud |
| Production-path playground | Run the Cloud playground through the managed data plane so it reflects real routing, caching, guardrails, and rate limiting. | Cloud |
| Cross-provider endpoint parity | Consistent embeddings, image generation, and Responses behavior across more providers. | Gateway |

## Later

| Capability | What's planned | Surface |
| --- | --- | --- |
| External secret management | Manage provider and API credentials through external KMS and secret stores such as Vault. | Gateway · Cloud |
| Expanded observability export | OTLP export for metrics and logs, alerting integrations such as Slack and PagerDuty, and first-party data-warehouse sinks. | Gateway · Cloud |
| More proxy endpoints | Passthrough for Batch, Files, and Fine-tuning, plus Realtime endpoints. | Gateway |
| Metered usage billing | Usage-based billing in addition to subscription plans. | Cloud |
| SDKs and agent-framework integrations | First-party SDKs and integrations with common agent frameworks. | Gateway · Cloud |

## Related pages

- [AISIX AI Gateway documentation](https://docs.api7.ai/ai-gateway/)
- [AISIX Cloud](https://api7.ai/ai-gateway)
- Tracked live in [issues](https://github.com/api7/aisix/issues)
