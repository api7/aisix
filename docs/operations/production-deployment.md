---
title: Production Deployment
description: Deploy AISIX AI Gateway in production with correct bootstrap, listeners, etcd, cache, and graceful shutdown expectations.
sidebar_position: 50
---

Production deployment starts with a correct bootstrap config and a reachable etcd cluster.

Use this page as the minimum operator checklist before you call a gateway deployment production-ready.

## Core Runtime Shape

At boot, the gateway currently:

1. loads bootstrap config
2. connects to etcd
3. seeds the initial snapshot
4. starts the watch supervisor
5. builds shared proxy components
6. binds the proxy and, in standalone mode, admin listeners

This sequence matters because a process can be alive while still being unusable if the config-store path or initial snapshot is broken.

## Recommended Baseline

- run etcd separately from the gateway process
- bind the proxy listener to the network interface your clients use
- keep the admin listener private to operators
- enable TLS on proxy and admin listeners when exposing them outside local development

For most teams, a solid first production baseline is:

- one gateway process
- one separately managed etcd cluster
- loopback or internal-only admin listener
- TLS on exposed listeners
- memory cache unless you have a concrete reason to introduce Redis immediately

## Cache Backend Choice

The process always builds the in-process memory cache. Add a `cache.redis` block to also build the shared Redis cache. Which backend serves a request is selected by the matched cache policy's `backend` field — a policy that requests `redis` on a process without `cache.redis` gets no caching for its requests (no silent fallback to memory).

The legacy `cache.backend` knob no longer selects a single global cache; `backend: redis` without `cache.redis.url` still fails at startup so misconfigurations surface early.

`memory`-backed policies remain the simplest production baseline, making them the lowest-risk default for first rollout.

## Managed Versus Standalone

In standalone mode:

- the admin API binds
- the standalone playground binds

In managed mode:

- the admin API is not bound
- the standalone playground is not exposed
- the data plane reads config through the managed path

Your operational playbook should match the mode:

- standalone: operators write config locally through the admin API
- managed: operators expect control-plane projection and managed bootstrap flows

## Shutdown Behavior

The server currently handles graceful shutdown on `SIGINT` and `SIGTERM`.

On shutdown it stops accepting new work and coordinates listener shutdown with background tasks.

## Preflight Checklist

Before routing real traffic, verify:

1. bootstrap config is correct for the intended mode
2. etcd is reachable from the gateway host or container
3. proxy health works
4. admin health works in standalone mode
5. at least one provider key, model, and API key can propagate and serve a real request

## First Production Checks

After deployment, confirm:

1. `GET /livez` returns `200`
2. admin-listener `GET /livez` returns `200` in standalone mode
3. `GET /admin/v1/health` returns `200` in standalone mode
4. `GET /v1/models` returns the expected caller-visible aliases for a test key
5. one real request succeeds on each endpoint family you actually use

## Troubleshooting

### The process is up but real requests fail

Treat that as a configuration or propagation problem first, not as proof that the deployment succeeded.

### The admin API is missing in production

Check whether the deployment is intentionally running in managed mode.

## Related Pages

- [Bootstrap Configuration](../configuration/bootstrap-config.md)
- [Network And Security](network-and-security.md)
- [Upgrades And Compatibility](upgrades-and-compatibility.md)
