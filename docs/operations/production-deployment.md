---
title: Production Deployment
description: Deploy AISIX AI Gateway in production with correct bootstrap, listeners, etcd, cache, and graceful shutdown expectations.
toc_max_heading_level: 2
sidebar_position: 50
---

Production deployment starts with a correct bootstrap config, a reachable etcd cluster, and a clear decision about whether the gateway runs in standalone or managed mode.

## Production Runtime

At startup, AISIX loads bootstrap configuration, connects to etcd, builds the initial resource view, starts the configuration watcher, builds shared proxy components, and binds the proxy listener. In standalone mode, it also binds the admin listener.

A process can be alive while still unable to serve useful traffic if the configuration store, loaded resources, or provider resources are not ready. Always verify the request path, not only process startup.

Match the production playbook to the deployment mode. In standalone mode, AISIX binds the local admin API when configured. Gateway resources are managed through the admin API, and those writes are stored as etcd-backed resources. The standalone playground is part of the local admin API.

In managed data-plane mode, AISIX does not expose the standalone admin listener or standalone playground locally. AISIX Cloud manages resources and projects them into the connected data plane. Bootstrap uses the Cloud certificate bundle and managed path.

## Deployment Baseline

For a first production rollout, use this baseline unless you have a specific reason to diverge. Run etcd separately from the gateway process, expose the proxy listener only to intended callers, and keep the admin listener on loopback or a private network. Enable TLS on listeners that leave local development.

Start with memory cache unless Redis is required. Before considering the deployment ready, create at least one provider key, model, and caller API key and verify that the model can serve a real request.

If you choose Redis for cache, `cache.redis.url` must be present in the bootstrap config or startup fails.

## Verify Readiness

Before routing real traffic, confirm the bootstrap config matches the intended mode, the proxy listener is reachable from intended callers, and at least one model alias can serve a real request. In standalone mode, also confirm etcd is reachable and the admin listener is private. For deployments with TLS or mTLS, confirm the configured certificate and key files are readable.

After deployment, run production checks. `GET /livez` on the proxy listener should return `200`. In standalone mode, admin-listener `GET /livez` and `GET /admin/v1/health` should also return `200`. `GET /v1/models` with a test key should return the expected caller-visible aliases. One real request per endpoint family in use should succeed through the proxy, and metrics, logs, or configured exporters should show the smoke-test request path.

## Shutdown Behavior

AISIX handles graceful shutdown on `SIGINT` and `SIGTERM`. During shutdown, the server stops accepting new work and coordinates listener shutdown with background tasks.

Treat a failing `/livez` during shutdown as expected. Do not treat it as an unexpected process failure unless the process was not meant to be draining.

## Troubleshooting

Use these checks when the gateway process is healthy but real provider-backed requests fail.

### The Process Is Up but Real Requests Fail

Treat this as a configuration, propagation, credential, or upstream path problem. Check `/v1/models`, admin health in standalone mode, and one real request with the caller API key that should have access.

### The Admin API Is Missing

Check whether the deployment is running as a managed data plane. Managed mode does not bind the standalone admin listener.

## Related Reading

[Network and security](/ai-gateway/operations/network-and-security) covers listener and secret handling. For health endpoints and validation flow, see [Health checks](/ai-gateway/operations/health-checks) and [Testing and verification](/ai-gateway/operations/testing-and-verification).
