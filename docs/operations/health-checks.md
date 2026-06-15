---
title: Health Checks
description: Use proxy and admin liveness endpoints plus the per-model health endpoint to verify process availability, model health, and config freshness in AISIX AI Gateway.
sidebar_position: 53
---

AISIX AI Gateway currently exposes three health surfaces. Use them for different jobs — they are not interchangeable.

## Proxy Liveness — `GET /livez`

`GET /livez` on the proxy listener is the public, unauthenticated liveness check.

Use it to confirm:

- the proxy listener is up
- the process is not shutting down

It returns `200 OK` with the plain-text body `ok` on a healthy process. During graceful shutdown it returns `500 Internal Server Error` with a body ending in `livez check failed`, which Kubernetes liveness probes and load balancers can match on.

Append `?verbose=1` for a multi-line plain-text body that ends with `livez check passed` (healthy) or includes the shutdown reason (failed). The verbose body is intended for human operators using `curl`, not for automated probes.

The body is intentionally minimal — snapshot counts, provider bridge counts, and configuration metadata are **not** exposed on this route. Operators looking for that information should use the authenticated admin endpoints below.

## Admin Liveness — `GET /livez`

The admin listener exposes the same `/livez` route with the same response contract. Use it to confirm the admin listener is reachable and the process is not shutting down. The admin-listener liveness is independent of the proxy listener so a probe failure points at the specific socket.

## Per-Model Health — `GET /admin/v1/health`

`GET /admin/v1/health` is the authenticated operator-facing endpoint. It requires an admin-key bearer token and returns one entry per model in the current snapshot plus an optional config-freshness block.

Response shape:

```json
{
  "status": "ok",
  "models": [
    {"id": "m-uuid-1", "name": "gpt-4o-prod", "health": 0},
    {"id": "m-uuid-2", "name": "claude-prod", "health": 1}
  ],
  "config": {
    "snapshot_revision": 1234567,
    "snapshot_age_seconds": 5
  }
}
```

Per-model `health` levels:

- `0` — Healthy (no recent failures)
- `1` — Degraded (4 to 7 consecutive upstream failures)
- `2` — Down (8 or more consecutive upstream failures)

The `config` block surfaces the etcd watch supervisor's freshness state. `snapshot_revision` is the highest etcd revision currently reflected in the snapshot. `snapshot_age_seconds` is `null` before the first apply and a number afterwards; a large value (for example, more than 300) suggests a stalled watch. The whole `config` block is omitted when the supervisor is not wired into admin state.

## Why Config Freshness Matters

Per-model upstream health alone does not tell you whether the gateway is serving fresh configuration. The watch-status block helps detect a frozen snapshot, a stalled watch stream, or a delayed config apply path. See [Configuration Propagation](../configuration/configuration-propagation.md) for how admin writes reach the proxy.

## Operational Use

Use `/livez` (proxy or admin) for liveness-style probes — Kubernetes `livenessProbe`, load balancer health checks, container orchestration restart triggers.

Use `/admin/v1/health` for operator diagnosis, rollout verification, and debugging propagation or watch issues.

## Minimal Runbook

1. If proxy `GET /livez` fails, treat it as a process or listener problem.
2. If admin `GET /livez` fails, inspect admin binding, mTLS settings (in managed mode), and deployment topology.
3. If both liveness routes succeed but traffic still fails, inspect `GET /admin/v1/health` for per-model degradation.
4. If admin health reports a stale `snapshot_age_seconds`, focus on configuration propagation and etcd watch freshness.
5. If model health is degraded but the snapshot is fresh, focus on upstream provider path issues (credentials, network, provider outages).

## Troubleshooting

### Liveness is green but requests still fail

That is expected in some failure modes. Liveness is intentionally narrow — process up, not shutting down — and is independent of snapshot, upstream health, and provider credentials. Use `GET /admin/v1/health` to see whether specific models are degraded.

### `snapshot_age_seconds` keeps growing

Indicates a stalled etcd watch. Check etcd connectivity and the supervisor logs.

## Related Pages

- [Configuration Propagation](../configuration/configuration-propagation.md)
- [Metrics And Logs](metrics-and-logs.md)
- [Troubleshooting](troubleshooting.md)
