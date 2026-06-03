---
title: Health Checks
description: Use proxy and admin liveness endpoints plus the per-model health endpoint to verify process availability, model health, and config freshness in AISIX AI Gateway.
sidebar_position: 53
---

AISIX AI Gateway exposes multiple health surfaces. Use each one for the
job it is designed to answer.

Use `GET /livez` on the proxy listener as the caller-facing liveness
probe. It is unauthenticated and only confirms that the proxy listener is
up.

Use `GET /livez` on the admin listener to confirm that the private admin
listener is reachable in standalone mode. It is also unauthenticated, so
keep the admin listener private.

Use `GET /admin/v1/health` on the admin listener when you need
authenticated operator detail, including model health and configuration
freshness.

## Proxy liveness

`GET /livez` on the proxy listener confirms that the proxy listener is up
and the process is not shutting down.

Healthy response:

```text
200 OK
ok
```

During graceful shutdown, the route returns `500 Internal Server Error`
with a body ending in `livez check failed`. This lets Kubernetes probes
and load balancers stop routing traffic during drain.

Append `?verbose=1` for a multi-line body intended for operators using
`curl`. Do not depend on the verbose body for automated probes.

Proxy liveness is intentionally narrow. It does not expose snapshot
counts, provider bridge counts, provider credentials, or model health.

## Admin liveness

The admin listener exposes the same `/livez` route. Use it to confirm the
admin listener is reachable in standalone mode.

Because proxy and admin listeners are separate sockets, a failure on one
does not necessarily mean the other listener is unhealthy.

## Per-model health

`GET /admin/v1/health` is the authenticated operator endpoint. It
requires an admin-key bearer token and returns per-model health from the
current snapshot.

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

The optional `config` block reports snapshot freshness. A growing
`snapshot_age_seconds` can indicate a stalled watch or delayed
configuration propagation. The block is omitted when the watch supervisor
is not wired into the admin state. When the supervisor is wired but has no
age yet, `snapshot_age_seconds` can be `null`.

## Minimal runbook

1. If proxy `GET /livez` fails, inspect process state and proxy listener
   binding.
2. If admin `GET /livez` fails in standalone mode, inspect admin binding,
   network placement, and listener TLS.
3. If liveness is green but traffic fails, inspect `GET /admin/v1/health`
   for model degradation in standalone mode.
4. If `snapshot_age_seconds` keeps growing, focus on etcd connectivity
   and watch freshness.
5. If model health is degraded but config is fresh, focus on upstream
   provider credentials, network, and provider availability.

## Troubleshooting

### Liveness is green but requests still fail

That can happen. Liveness only proves that the process and listener are
up. It does not prove that a model alias exists, a provider key is valid,
or an upstream provider is reachable.

### `snapshot_age_seconds` keeps growing

Treat this as a configuration propagation issue. Check etcd connectivity,
watch supervisor logs, and whether the gateway can read the configured
etcd TLS files.

## Next steps

- [Configuration propagation](/ai-gateway/configuration/configuration-propagation)
  explains how admin writes reach the proxy.
- [Metrics and logs](/ai-gateway/operations/metrics-and-logs) explains
  observability signals.
- [Troubleshooting](/ai-gateway/operations/troubleshooting) gives a
  broader diagnosis flow.
