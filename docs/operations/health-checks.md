---
title: Health Checks
description: Use proxy and admin liveness endpoints plus the per-model health endpoint to verify process availability, model health, and config freshness in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 53
---

AISIX AI Gateway exposes multiple health endpoints. Use each one for the job it is designed to answer.

Use `GET /livez` on the proxy listener as the caller-facing liveness probe. It is unauthenticated and only confirms that the proxy listener is up.

Use `GET /livez` on the admin listener to confirm that the private admin listener is reachable in standalone mode. It is also unauthenticated, so keep the admin listener private.

Use `GET /admin/v1/health` on the admin listener when you need authenticated management detail, including model health and configuration freshness.

## Proxy Liveness

`GET /livez` on the proxy listener confirms that the proxy listener is up and the process is not shutting down.

Healthy response:

```text
200 OK
ok
```

During graceful shutdown, the route returns `500 Internal Server Error` with a body ending in `livez check failed`. This lets Kubernetes probes and load balancers stop routing traffic during drain.

Append `?verbose=1` for a multi-line body suitable for manual `curl` checks. Do not depend on the verbose body for automated probes.

Proxy liveness is intentionally narrow. It does not expose snapshot details, provider adapter inventory, provider credentials, or model health.

## Admin Liveness

The admin listener exposes the same `/livez` route. Use it to confirm the admin listener is reachable in standalone mode.

Because proxy and admin listeners are separate sockets, a failure on one does not necessarily mean the other listener is unhealthy.

## Per-Model Health

`GET /admin/v1/health` is the authenticated management endpoint. It requires an admin-key bearer token and returns per-model health from the latest accepted snapshot.

Example response:

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

Model health levels:

- `0`: healthy, with no recent upstream failure streak
- `1`: degraded, after 4 to 7 consecutive upstream failures
- `2`: down, after 8 or more consecutive upstream failures

The optional `config` block reports snapshot freshness. A growing `snapshot_age_seconds` can indicate a stalled watch or delayed configuration propagation. The block is omitted when snapshot freshness is not available. When freshness tracking is available but has no age yet, `snapshot_age_seconds` can be `null`.

## Diagnosis Flow

When proxy `GET /livez` fails, inspect process state and proxy listener binding. When admin `GET /livez` fails in standalone mode, inspect admin binding, network placement, and listener TLS.

If liveness is green but traffic fails, inspect `GET /admin/v1/health` for model degradation in standalone mode. If `snapshot_age_seconds` keeps growing, check etcd connectivity and watch freshness. If model health is degraded while configuration is fresh, check upstream provider credentials, network, and provider availability.

## Troubleshooting

Use these checks when liveness endpoints pass but proxy requests still fail.

### Liveness Is Green but Requests Still Fail

Liveness only confirms that the process and listener are up. It does not confirm that a model alias exists, a provider key is valid, or an upstream provider is reachable.

### Snapshot Age Keeps Growing

Treat this as a configuration propagation issue. Check etcd connectivity, configuration watch logs, and whether the gateway can read the configured etcd TLS files.

## Related Reading

[Configuration propagation](/ai-gateway/configuration/configuration-propagation) explains how admin writes reach the proxy. For observability signals and broader diagnosis flow, see [Metrics and logs](/ai-gateway/operations/metrics-and-logs) and [Troubleshooting](/ai-gateway/operations/troubleshooting).
