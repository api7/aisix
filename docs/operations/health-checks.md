---
title: Health Checks
description: Use proxy and admin health endpoints to verify process availability, model health, and config freshness in AISIX AI Gateway.
sidebar_position: 53
---

AISIX AI Gateway currently exposes two different health surfaces in standalone mode.

## Proxy Health

`GET /health` on the proxy listener is the simple process-level check.

Use it to confirm:

- the proxy listener is up
- the process can return a basic JSON response

## Admin Health

`GET /admin/v1/health` is the operator-facing health endpoint.

It currently includes:

- top-level `status`
- per-model health entries
- optional config freshness data when watch status is wired

Example fields in the optional config block:

- `snapshot_revision`
- `snapshot_age_seconds`

## Why Config Freshness Matters

Per-model upstream health alone does not tell you whether the gateway is serving fresh config.

The watch-status block helps detect a frozen snapshot, stalled watch stream, or delayed config apply path.

## Operational Use

Use proxy health for liveness-style checks.

Use admin health for operator checks, rollout verification, and debugging propagation or watch issues.

## Related Pages

- [Configuration Propagation](../configuration/configuration-propagation.md)
- [Metrics And Logs](metrics-and-logs.md)
- [Troubleshooting](troubleshooting.md)
