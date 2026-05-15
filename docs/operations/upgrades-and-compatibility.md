---
title: Upgrades And Compatibility
description: Upgrade AISIX AI Gateway conservatively and validate runtime compatibility across config, snapshot, and provider behavior.
sidebar_position: 56
---

Upgrade the gateway conservatively when dynamic configuration and provider behavior matter to production traffic.

Treat upgrades as behavior changes to be verified, not just binary replacements.

## Compatibility Principles

- bootstrap config must still parse on the new binary
- dynamic resources in etcd must remain readable by the new loader
- client-visible proxy behavior must be validated on real request paths

## Practical Upgrade Checks

Before and after an upgrade, verify:

1. `GET /livez`
2. admin-listener `GET /livez`
3. `GET /admin/v1/health`
4. `GET /v1/models`
5. one real request on each critical endpoint your clients use

If you use several endpoint families in production, test each one you depend on rather than assuming chat-completions success proves all compatibility.

## Areas To Treat Carefully

- managed-mode bootstrap path
- etcd TLS and trust roots
- cache backend selection
- dynamic resources written by a newer or older control plane

## Suggested Upgrade Flow

1. verify bootstrap config parses with the new binary
2. start the new instance without sending full traffic
3. confirm health, config freshness, and one real request per critical path
4. only then widen traffic exposure

## Troubleshooting

### The new binary starts but behaves differently on one endpoint family

Treat that as a compatibility issue even if health checks are green.

## Related Pages

- [Production Deployment](production-deployment.md)
- [Testing And Verification](testing-and-verification.md)
- [Roadmap](../roadmap.md)
