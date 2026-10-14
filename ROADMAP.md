# Roadmap

This page lists capabilities that are planned or in progress but not yet generally available. It shows direction, not dates, and is not a delivery commitment.

For what the gateway does today — including [semantic routing](https://docs.api7.ai/ai-gateway/routing/semantic-routing), [ensemble models](https://docs.api7.ai/ai-gateway/routing/ensemble-models), [caching](https://docs.api7.ai/ai-gateway/traffic-controls/caching), and [guardrails](https://docs.api7.ai/ai-gateway/traffic-controls/guardrails/overview) — see the [AISIX documentation](https://docs.api7.ai/ai-gateway/).

## How to read this page

- **Now** — in active design or development.
- **Next** — planned after the current focus.
- **Later** — on the longer-term horizon.

The **Products** column identifies the products expected to expose the capability. A shared gateway or control-plane implementation does not imply availability in every product.

## Now

| Capability | What's planned | Products |
| --- | --- | --- |
| Additional SSO protocols | Single sign-on through SAML and generic OIDC, beyond today's social logins. | AISIX Enterprise · AISIX Cloud |
| Service accounts | Login-less, first-class principals for automated callers. | AISIX Enterprise · AISIX Cloud |

## Next

| Capability | What's planned | Products |
| --- | --- | --- |
| Prompt management | Store, version, and reuse prompt templates with variables, resolved at the gateway. | AISIX OSS · AISIX Enterprise · AISIX Cloud |
| Scheduled key rotation | Scheduled auto-rotation of caller keys with a grace overlap, on top of today's manual rotation. | AISIX Enterprise · AISIX Cloud |
| Production-path playground | Run the dashboard playground through a connected AISIX gateway so it reflects real routing, caching, guardrails, and rate limiting. | AISIX Enterprise · AISIX Cloud |
| Cross-provider endpoint parity | Consistent embeddings, image generation, and Responses behavior across more providers. | AISIX OSS · AISIX Enterprise · AISIX Cloud |

## Later

| Capability | What's planned | Products |
| --- | --- | --- |
| External secret management | Manage provider and API credentials through external KMS and secret stores such as Vault. | AISIX OSS · AISIX Enterprise · AISIX Cloud |
| Expanded observability export | OTLP export for metrics and logs, and alerting integrations such as Slack and PagerDuty. | AISIX OSS · AISIX Enterprise · AISIX Cloud |
| Metered usage billing | Usage-based billing in addition to subscription plans. | AISIX Cloud |
| SDKs and agent-framework integrations | First-party SDKs and integrations with common agent frameworks. | AISIX OSS · AISIX Enterprise · AISIX Cloud |

## Related pages

- [AISIX documentation](https://docs.api7.ai/ai-gateway/)
- [AISIX Cloud](https://api7.ai/ai-gateway)
- Tracked live in [issues](https://github.com/api7/aisix/issues)
