---
title: AISIX Cloud Managed Data Plane Quickstart
description: Understand the current AISIX Cloud managed data-plane bootstrap flow based on gateway certificates and mTLS.
sidebar_position: 15
---

This guide explains the current managed data-plane bootstrap path for AISIX Cloud.

Use it when you want to understand how a data plane connects to the AISIX Cloud control plane, receives configuration, and starts serving traffic.

Use it before:

- deploying a managed data plane into your own environment
- debugging why a data plane is not connecting to AISIX Cloud
- reading the deeper Cloud pages about projection, resilience, and certificate rotation

## Current Bootstrap Model

The current managed flow is certificate-based.

At a high level:

1. create an environment in AISIX Cloud
2. request a gateway certificate bundle for that environment
3. start the AISIX data plane with the issued certificate, key, and CA bundle
4. let the data plane connect to the control plane over mTLS
5. receive projected configuration and begin serving requests

The current `/dp/*` surface is mTLS-authenticated. The older `/dp/register` bearer-token pattern is not the current bootstrap path.

That distinction matters because older comments and older mental models can make the bootstrap path look token-based when current Cloud behavior is certificate-based.

## What The Control Plane Issues

The control plane issues a gateway certificate bundle for the target environment.

That bundle contains:

- a client certificate PEM
- a private key PEM
- a CA bundle PEM

The data plane uses those materials to authenticate to the control-plane data-plane manager.

The bundle is environment-scoped. The managed data plane uses it to connect to the control-plane surface that owns heartbeat, telemetry, certificate rotation, and budget checks.

## What The Data Plane Receives

In managed mode, the data plane is started with environment variables for:

- `AISIX_MANAGED__CP_BASE_URL`
- `AISIX_MANAGED__CP_CERT_PEM`
- `AISIX_MANAGED__CP_KEY_PEM`
- `AISIX_MANAGED__CP_CA_PEM`

`AISIX_MANAGED__CP_BASE_URL` is the data-plane manager `/dp/*` mTLS
surface, for example `https://dpm.example.com:7944` or
`https://dpm:7944` inside the AISIX Cloud Compose network. Do not point
it at the browser dashboard or cp-api origin such as
`http://api:8080`; heartbeat is always built as
`<CP_BASE_URL>/dp/heartbeat`.

`AISIX_MANAGED__CP_ETCD_ENDPOINT` is **optional** and most deployments
should leave it unset. When it is not provided, the data plane derives
the etcd endpoint from `CP_BASE_URL` (the control plane multiplexes the
REST and etcd gRPC surfaces on the same `host:port`). Set it only when
your control plane serves etcd on a different `host:port` than the
`/dp/*` REST surface. Managing or knowing the control plane's etcd is
not part of the normal Cloud user workflow.

### Selecting the config file with `AISIX_CONFIG_PATH`

The published container image's entrypoint selects which config file to
load via `AISIX_CONFIG_PATH` (default `/etc/aisix/config.yaml`). For a
managed data plane, point it at the baked managed config:

```bash
AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml
```

This variable is read by the image entrypoint, not the gateway binary
itself — the binary's own `--config` flag (env `AISIX_CONFIG`) is the
lower-level equivalent when you run the binary directly.

For deployments that should not inline PEM material into environment
variables, use the file-based equivalents instead:

- `AISIX_MANAGED__CP_CERT_FILE`
- `AISIX_MANAGED__CP_KEY_FILE`
- `AISIX_MANAGED__CP_CA_FILE`

The inline and file variants are mutually exclusive for each cert/key/CA
pair.

The data plane persists the issued mTLS bundle, `dp_id`, and runtime
state under `/var/lib/aisix` by default. If the container runs as a
non-default user and reads bind-mounted PEM files, make `/var/lib/aisix`
writable by that same user, for example by mounting a host-owned state
directory there. Mounting only the `mtls` subdirectory is not enough
because the process also writes sidecar files next to it.

The managed data plane then uses the same single binary, but follows the managed bootstrap path instead of binding the standalone admin surface.

In other words:

- standalone mode expects operator-driven admin writes on `:3001`
- managed mode expects control-plane projection and mTLS-authenticated control-plane coordination

## Operational Notes

- configuration is projected from the control plane into the managed environment
- the data plane persists a local config cache for offline resilience
- the control-plane data-plane manager exposes `/dp/heartbeat`, `/dp/telemetry`, `/dp/rotate-cert`, and `/dp/budget_check` behind mTLS

## What This Quickstart Does Not Cover

This page is a bootstrap overview, not a full deployment manual.

It does not try to document:

- every dashboard click path
- every environment creation workflow
- every certificate rotation sequence
- every billing or usage-event detail

## Product Boundary

This page is intentionally about the bootstrap flow, not a full step-by-step dashboard tutorial.

For current customer-facing Cloud behavior, continue with the Cloud docs section.

## Troubleshooting Pointers

### The data plane never becomes healthy

Start with:

- certificate bundle correctness
- `AISIX_MANAGED__CP_BASE_URL`
- `AISIX_MANAGED__CP_ETCD_ENDPOINT` (only if you set it explicitly; otherwise it is derived from `CP_BASE_URL`)
- trust chain in `AISIX_MANAGED__CP_CA_PEM`
- writable state directory for `/var/lib/aisix`

If logs show `/dp/heartbeat` returning 404, `CP_BASE_URL` usually points
at the wrong service. It should point at the DPM mTLS endpoint, not the
cp-api/dashboard origin.

### The data plane starts but does not receive configuration

Focus on control-plane projection and environment-scoped resource visibility rather than the standalone admin API, because managed mode does not use the standalone admin surface as its source of truth.

## Related Pages

- [AISIX Cloud Overview](../cloud/overview.md)
- [Gateway Certificates And Managed DP](../cloud/gateway-certificates-and-managed-dp.md)
- [Resource Projection](../cloud/resource-projection.md)
- [Offline Resilience](../cloud/offline-resilience.md)
