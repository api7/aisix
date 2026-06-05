---
title: Upgrades and Compatibility
description: Upgrade AISIX AI Gateway conservatively and validate runtime compatibility across config, snapshot, and provider behavior.
toc_max_heading_level: 2
sidebar_position: 57
---

Upgrade AISIX AI Gateway conservatively when production traffic depends on dynamic configuration and provider behavior.

Treat an upgrade as a behavior change to verify, not only as a binary replacement.

## Compatibility Checks

Before widening traffic to a new version, verify the new binary parses the configured bootstrap file and starts with the expected listeners. Confirm etcd-backed resources are readable, model aliases resolve as expected, and caller-facing proxy behavior still matches the applications that call the gateway.

Also verify provider-specific behavior for each endpoint family you use. In managed deployments, confirm managed bootstrap and projection still work before widening traffic.

## Upgrade Flow

Review release notes or change notes for configuration, provider, and endpoint behavior before starting the rollout. Start the new version without sending full production traffic, then confirm proxy liveness and, in standalone mode, admin health.

Before widening traffic, confirm `GET /v1/models` with a representative caller API key, send one real request on each endpoint family your clients use, and check logs, metrics, headers, and usage events for the upgraded path.

## Compatibility Risks

Pay special attention to bootstrap config fields, etcd TLS and trust roots, dynamic resource schemas, cache backend selection, provider adapter behavior, managed certificate bootstrap, `/dp/*` connectivity, Cloud projection, and budget workflows.

If you use several endpoint families, test each one. A successful chat-completions request does not confirm that embeddings, streaming, Anthropic Messages, or passthrough behavior is compatible.

## Rollback Considerations

Before upgrading, decide which bootstrap config version will be used for rollback. Also verify whether dynamic resources written during the upgrade remain readable by the previous version, whether provider-key, model, or policy changes were made during the rollout, and whether managed projection state needs time to converge after rollback.

## Troubleshooting

Use these checks when an upgraded gateway starts successfully but endpoint behavior changes.

### The New Version Starts but One Endpoint Behaves Differently

Treat this as a compatibility issue even if health checks are green. Check the failing endpoint against a known-good request path, then inspect provider adapter behavior, request headers, response format, and policy resources.

## Related Reading

[Production deployment](/ai-gateway/operations/production-deployment) covers the production baseline, and [Testing and verification](/ai-gateway/operations/testing-and-verification) covers validation flow. For production readiness, see [Feature availability](/ai-gateway/overview/feature-matrix).
