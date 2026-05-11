---
title: Upgrades And Compatibility
description: Upgrade AISIX AI Gateway conservatively and validate runtime compatibility across config, snapshot, and provider behavior.
sidebar_position: 56
---

Upgrade the gateway conservatively when dynamic configuration and provider behavior matter to production traffic.

## Compatibility Principles

- bootstrap config must still parse on the new binary
- dynamic resources in etcd must remain readable by the new loader
- client-visible proxy behavior must be validated on real request paths

## Practical Upgrade Checks

Before and after an upgrade, verify:

1. `GET /health`
2. `GET /admin/v1/health`
3. `GET /v1/models`
4. one real request on each critical endpoint your clients use

## Areas To Treat Carefully

- managed-mode bootstrap path
- etcd TLS and trust roots
- cache backend selection
- dynamic resources written by a newer or older control plane

## Related Pages

- [Production Deployment](production-deployment.md)
- [Testing And Verification](testing-and-verification.md)
- [Roadmap](../roadmap.md)
