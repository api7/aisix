---
title: Production Deployment
description: Deploy AISIX AI Gateway in production with correct bootstrap, listeners, etcd, cache, and graceful shutdown expectations.
sidebar_position: 50
---

Production deployment starts with a correct bootstrap config, a reachable
etcd cluster, and a clear decision about whether the gateway runs in
standalone or managed mode.

## Understand the runtime startup path

At startup, AISIX:

1. Loads bootstrap configuration.
2. Connects to etcd.
3. Builds the initial resource snapshot.
4. Starts the watch supervisor.
5. Builds shared proxy components.
6. Binds the proxy listener and, in standalone mode, the admin listener.

A process can be alive while still unable to serve useful traffic if the
configuration store, initial snapshot, or provider resources are not
ready. Always verify the request path, not only process startup.

## Start with this baseline checklist

For a first production rollout, use this baseline unless you have a
specific reason to diverge:

- run etcd separately from the gateway process
- expose the proxy listener only to intended callers
- keep the admin listener on loopback or a private network
- enable TLS on listeners that leave local development
- start with the memory cache backend unless Redis is required
- create at least one provider key, model, and caller API key before
  considering the deployment ready

If you choose Redis for cache, `cache.redis.url` must be present in the
bootstrap config or startup fails.

## Match the playbook to the deployment mode

Your production playbook should match the deployment mode.

In standalone mode, AISIX binds the local admin API when configured.
Operators manage resources through the admin API, and those writes are
stored as etcd-backed resources. The standalone playground is part of the
local admin surface. Verify the deployment with admin health checks and
real proxy requests.

In managed data-plane mode, AISIX does not expose the standalone admin
listener or standalone playground locally. Operators manage resources in
AISIX Cloud, and the Cloud control plane projects those resources into
the connected data plane. Bootstrap uses the Cloud certificate bundle and
managed path. Verify the deployment with managed heartbeat and real proxy
requests.

## Run these preflight checks

Before routing real traffic:

1. Confirm bootstrap config is correct for the intended mode.
2. Confirm etcd is reachable in standalone mode.
3. Confirm the proxy listener is reachable from intended callers.
4. Confirm the admin listener is private in standalone mode.
5. Confirm TLS and mTLS files are readable where configured.
6. Confirm at least one model alias can serve a real request.

## Run these first production checks

After deployment:

1. `GET /livez` returns `200` on the proxy listener.
2. In standalone mode, admin-listener `GET /livez` returns `200`.
3. In standalone mode, `GET /admin/v1/health` returns `200`.
4. `GET /v1/models` returns the expected caller-visible aliases for a
   test key.
5. One real request succeeds on each endpoint family you use.
6. Metrics, logs, or configured exporters show the request path from the
   smoke test.

## Shutdown behavior

AISIX handles graceful shutdown on `SIGINT` and `SIGTERM`. During
shutdown, the server stops accepting new work and coordinates listener
shutdown with background tasks.

Treat a failing `/livez` during shutdown as expected. Do not use it as
proof of an unexpected process failure unless the process was not meant
to be draining.

## Troubleshooting

### The process is up but real requests fail

Treat this as a configuration, propagation, credential, or upstream path
problem. Check `/v1/models`, admin health in standalone mode, and one
real request with the caller API key that should have access.

### The admin API is missing

Check whether the deployment is running as a managed data plane. Managed
mode does not bind the standalone admin listener.

## Next steps

- [Network and security](/ai-gateway/operations/network-and-security)
  explains listener and secret boundaries.
- [Health checks](/ai-gateway/operations/health-checks) explains the
  health surfaces.
- [Testing and verification](/ai-gateway/operations/testing-and-verification)
  gives the minimum validation flow.
