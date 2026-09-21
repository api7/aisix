# Roadmap

This page lists capabilities that are planned or in progress but not yet generally available. It shows direction, not dates, and is not a delivery commitment.

For what the gateway does today — including [semantic routing](https://docs.api7.ai/ai-gateway/routing/semantic-routing), [ensemble models](https://docs.api7.ai/ai-gateway/routing/ensemble-models), [caching](https://docs.api7.ai/ai-gateway/traffic-controls/caching), and [guardrails](https://docs.api7.ai/ai-gateway/traffic-controls/guardrails/overview) — see the [AISIX AI Gateway documentation](https://docs.api7.ai/ai-gateway/).

## How to read this page

- **Now** — in active design or development.
- **Next** — planned after the current focus.
- **Later** — on the longer-term horizon.

The **Surface** column shows where a capability lands: **Gateway** is the AISIX AI Gateway runtime; **Cloud** is the AISIX Cloud control plane and dashboard.

## Now

| Capability | What's planned | Surface |
| --- | --- | --- |
| Enterprise SSO | Single sign-on through SAML and generic OIDC, beyond today's social logins. | Cloud |
| Service accounts | Non-human, team-scoped principals that own caller keys without being modeled as organization members, on top of today's login-less members. | Cloud |

## Next

| Capability | What's planned | Surface |
| --- | --- | --- |
| Prompt management | Store, version, and reuse prompt templates with variables, resolved at the gateway. | Gateway · Cloud |
| Scheduled key rotation | Scheduled auto-rotation of caller keys with a grace overlap, on top of today's manual rotation. | Cloud |
| Production-path playground | Run the Cloud playground through a connected AISIX gateway so it reflects real routing, caching, guardrails, and rate limiting. | Cloud |
| Cross-provider endpoint parity | Consistent embeddings, image generation, and Responses behavior across more providers. | Gateway |

## Later

| Capability | What's planned | Surface |
| --- | --- | --- |
| External secret management | Manage provider and API credentials through external KMS and secret stores such as Vault. | Gateway · Cloud |
| Expanded observability export | OTLP export for metrics and logs, alerting integrations beyond today's webhook and Slack budget alerts (such as PagerDuty), and first-party data-warehouse sinks on top of today's object-storage staging. | Gateway · Cloud |
| Metered usage billing | Usage-based billing in addition to subscription plans. | Cloud |
| SDKs and agent-framework integrations | First-party SDKs and integrations with common agent frameworks. | Gateway · Cloud |

## Related pages

- [AISIX AI Gateway documentation](https://docs.api7.ai/ai-gateway/)
- [AISIX Cloud](https://api7.ai/ai-gateway)
- Tracked live in [issues](https://github.com/api7/aisix/issues)
