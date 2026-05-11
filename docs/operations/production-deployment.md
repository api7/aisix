---
title: Production Deployment
description: Deploy AISIX AI Gateway in production with correct bootstrap, listeners, etcd, cache, and graceful shutdown expectations.
sidebar_position: 50
---

Production deployment starts with a correct bootstrap config and a reachable etcd cluster.

## Core Runtime Shape

At boot, the gateway currently:

1. loads bootstrap config
2. connects to etcd
3. seeds the initial snapshot
4. starts the watch supervisor
5. builds shared proxy components
6. binds the proxy and, in standalone mode, admin listeners

## Recommended Baseline

- run etcd separately from the gateway process
- bind the proxy listener to the network interface your clients use
- keep the admin listener private to operators
- enable TLS on proxy and admin listeners when exposing them outside local development

## Cache Backend Choice

Current bootstrap cache backends are:

- `memory`
- `redis`

`memory` is the simplest production baseline. If you select `redis`, the bootstrap config must include `cache.redis.url` or startup will fail.

## Managed Versus Standalone

In standalone mode:

- the admin API binds
- the standalone playground binds

In managed mode:

- the admin API is not bound
- the standalone playground is not exposed
- the data plane reads config through the managed path

## Shutdown Behavior

The server currently handles graceful shutdown on `SIGINT` and `SIGTERM`.

On shutdown it stops accepting new work and coordinates listener shutdown with background tasks.

## Related Pages

- [Bootstrap Configuration](../configuration/bootstrap-config.md)
- [Network And Security](network-and-security.md)
- [Upgrades And Compatibility](upgrades-and-compatibility.md)
