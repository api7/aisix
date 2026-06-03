---
title: Upgrades and Compatibility
description: Upgrade AISIX AI Gateway conservatively and validate runtime compatibility across config, snapshot, and provider behavior.
sidebar_position: 57
---

Upgrade AISIX AI Gateway conservatively when production traffic depends
on dynamic configuration and provider behavior.

Treat an upgrade as a behavior change to verify, not only as a binary
replacement.

## What compatibility means

Before widening traffic to a new version, verify that:

- bootstrap config still parses
- etcd-backed dynamic resources are still readable
- model aliases resolve as expected
- caller-facing proxy behavior still matches your applications
- provider-specific upstream behavior still works for the endpoint
  families you use
- managed bootstrap and projection still work in managed deployments

## Suggested upgrade flow

1. Review release notes or change notes for config, provider, and
   endpoint behavior.
2. Start the new version without sending full production traffic.
3. Confirm proxy liveness and, in standalone mode, admin health.
4. Confirm `GET /v1/models` with a representative caller API key.
5. Send one real request on each endpoint family your clients use.
6. Check logs, metrics, headers, and usage events for the upgraded path.
7. Widen traffic only after the request path is verified.

## Areas to treat carefully

- bootstrap config fields
- etcd TLS and trust roots
- dynamic resource schemas
- cache backend selection
- provider adapter behavior
- managed certificate bootstrap and `/dp/*` connectivity
- Cloud projection and budget workflows

If you use several endpoint families, test each one. A successful
chat-completions request does not prove that embeddings, streaming,
Anthropic Messages, or passthrough behavior is compatible.

## Rollback considerations

Before upgrading, make sure you know:

- which bootstrap config version will be used for rollback
- whether dynamic resources written during the upgrade remain readable by
  the previous version
- whether provider-key, model, or policy changes were made during the
  rollout
- whether managed projection state needs time to converge after rollback

## Troubleshooting

### The new version starts but one endpoint behaves differently

Treat this as a compatibility issue even if health checks are green.
Compare the failing endpoint against a known-good request path, then
inspect provider adapter behavior, request headers, response shape, and
policy resources.

## Next steps

- [Production deployment](/ai-gateway/operations/production-deployment)
  gives the production baseline.
- [Testing and verification](/ai-gateway/operations/testing-and-verification)
  gives the validation flow.
- [Feature Status](/ai-gateway/overview/feature-matrix) shows the current
  product boundary.
