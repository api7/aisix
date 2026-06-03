---
title: Connect a Managed Data Plane
description: Bootstrap an AISIX AI Gateway data plane into AISIX Cloud with a gateway certificate bundle and mTLS.
sidebar_position: 15
---

This guide explains how an AISIX AI Gateway data plane connects to AISIX Cloud, authenticates with a gateway certificate bundle, and starts receiving projected configuration.

This guide shows how to bootstrap a managed data plane for an environment and what to check if the data plane does not connect to AISIX Cloud.

By the end of this guide, you should understand:

1. Which bootstrap inputs the data plane needs.
2. Which AISIX Cloud endpoint the data plane should connect to.
3. How managed mode differs from standalone self-hosted mode.
4. Which checks to run first when bootstrap fails.

:::note
This page describes the data-plane bootstrap shape. It does not cover every dashboard click path for creating environments or issuing certificates.
:::

## Prerequisites

Before you start, make sure you have:

- an AISIX Cloud environment for the managed data plane
- a gateway certificate bundle issued for that environment
- the AISIX AI Gateway container image or binary
- a writable runtime state directory, usually `/var/lib/aisix`
- network access from the data plane to the AISIX Cloud data-plane-manager endpoint

## Bootstrap model

Managed bootstrap is certificate-based.

At a high level:

1. Create or select an environment in AISIX Cloud.
2. Issue a gateway certificate bundle for that environment.
3. Start the AISIX data plane with the certificate, private key, CA bundle, and data-plane-manager URL.
4. Let the data plane connect to the `/dp/*` surface over mTLS.
5. Receive projected configuration and begin serving requests.

The current `/dp/*` surface is mTLS-authenticated. The older `/dp/register` bearer-token pattern is not the current bootstrap path.

That distinction matters when you are debugging bootstrap failures: a managed data plane should have a certificate bundle and mTLS connection settings, not a bearer registration token.

## Certificate bundle

The control plane issues a gateway certificate bundle for the target environment.

That bundle contains:

- a client certificate PEM
- a private key PEM
- a CA bundle PEM

The bundle is environment-scoped. The data plane uses it to authenticate to the AISIX Cloud data-plane manager and to derive the managed environment scope used for projected configuration.

## Runtime inputs

In managed mode, the data plane starts with these environment variables:

- `AISIX_MANAGED__CP_BASE_URL`
- `AISIX_MANAGED__CP_CERT_PEM`
- `AISIX_MANAGED__CP_KEY_PEM`
- `AISIX_MANAGED__CP_CA_PEM`

`AISIX_MANAGED__CP_BASE_URL` must point to the Cloud data-plane-manager `/dp/*` mTLS surface, for example `https://dpm.example.com:7944` or `https://dpm:7944` inside the AISIX Cloud Compose network.

Do not point `AISIX_MANAGED__CP_BASE_URL` at the browser dashboard or the control-plane API origin such as `http://api:8080`. The heartbeat URL is always built as `<CP_BASE_URL>/dp/heartbeat`, so a wrong base URL usually shows up as `/dp/heartbeat` returning `404`.

`AISIX_MANAGED__CP_ETCD_ENDPOINT` is **optional** and most deployments should leave it unset. When it is not provided, the data plane derives the etcd endpoint from `CP_BASE_URL` because the control plane multiplexes the REST and etcd gRPC surfaces on the same `host:port`.

Set `AISIX_MANAGED__CP_ETCD_ENDPOINT` only when your control plane serves etcd on a different `host:port` than the `/dp/*` REST surface. Managing or knowing the control plane's etcd endpoint is not part of the normal Cloud user workflow.

## Select the managed config

The published container image's entrypoint selects which config file to load with `AISIX_CONFIG_PATH`. The default is `/etc/aisix/config.yaml`.

For a managed data plane, point the entrypoint at the baked managed config:

```shell
AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml
```

`AISIX_CONFIG_PATH` is read by the container entrypoint and is unset before the binary starts. When you run the binary directly, use the equivalent `--config` flag or `AISIX_CONFIG` environment variable:

```shell
aisix --config /path/to/config.managed.yaml
```

or:

```shell
AISIX_CONFIG=/path/to/config.managed.yaml aisix
```

## Provide certificate material

For deployments that should not inline PEM material into environment variables, use the file-based equivalents instead:

- `AISIX_MANAGED__CP_CERT_FILE`
- `AISIX_MANAGED__CP_KEY_FILE`
- `AISIX_MANAGED__CP_CA_FILE`

The inline and file variants are mutually exclusive for each cert/key/CA pair.

## Start the data plane

A managed data plane usually starts from the published container image with the managed config selected and the certificate bundle supplied by environment variables or mounted files.

The shape is:

```shell
docker run --rm \
  -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
  -e AISIX_MANAGED__CP_BASE_URL="https://dpm.example.com:7944" \
  -e AISIX_MANAGED__CP_CERT_FILE="/etc/aisix/mtls/client.crt" \
  -e AISIX_MANAGED__CP_KEY_FILE="/etc/aisix/mtls/client.key" \
  -e AISIX_MANAGED__CP_CA_FILE="/etc/aisix/mtls/ca.crt" \
  -v "$PWD/mtls:/etc/aisix/mtls:ro" \
  -v aisix-state:/var/lib/aisix \
  ghcr.io/api7/ai-gateway:dev
```

Replace the example URL, certificate paths, and image tag with the values issued for your environment. If you use inline PEM values instead of files, provide `AISIX_MANAGED__CP_CERT_PEM`, `AISIX_MANAGED__CP_KEY_PEM`, and `AISIX_MANAGED__CP_CA_PEM` together.

The data plane persists the issued mTLS bundle, `dp_id`, and runtime state under `/var/lib/aisix` by default. If the container runs as a non-default user and reads bind-mounted PEM files, make `/var/lib/aisix` writable by that same user, for example by mounting a host-owned state directory there.

Mounting only the `mtls` subdirectory is not enough because the process also writes sidecar files next to it.

## Managed mode behavior

The managed data plane uses the same `aisix` binary as standalone mode, but it follows the managed bootstrap path instead of binding the standalone admin surface.

In other words:

- standalone mode expects operator-driven admin writes on `:3001`
- managed mode expects control-plane projection and mTLS-authenticated control-plane coordination

When `managed.enabled` is true, the standalone admin API and Playground are not bound. Cloud-owned resources are projected to the data plane instead of being created through the local admin API.

The Cloud data-plane-manager surface includes:

- `POST /dp/heartbeat` for liveness and data-plane status
- `POST /dp/telemetry` for usage-oriented telemetry
- `POST /dp/rotate-cert` for certificate rotation
- `GET /dp/budget_check` for managed budget checks

## What this page does not cover

This bootstrap guide does not cover:

- every dashboard click path
- every environment creation workflow
- every certificate rotation sequence
- every billing or usage-event detail

## Troubleshooting pointers

### The data plane never becomes healthy

Check:

- certificate bundle correctness
- `AISIX_MANAGED__CP_BASE_URL`
- `AISIX_MANAGED__CP_ETCD_ENDPOINT`, only if you set it explicitly
- trust chain in `AISIX_MANAGED__CP_CA_PEM` or `AISIX_MANAGED__CP_CA_FILE`
- writable state directory for `/var/lib/aisix`

If logs show `/dp/heartbeat` returning `404`, `CP_BASE_URL` usually points at the wrong service. It should point at the data-plane-manager mTLS endpoint, not the control-plane API or dashboard origin.

### The data plane starts but does not receive configuration

Focus on control-plane projection and environment-scoped resource visibility. Managed mode does not use the standalone admin API as its source of truth.

## Next steps

- [Gateway certificates and managed data plane](../cloud/gateway-certificates-and-managed-dp.md) — review certificate issuance and bootstrap in more detail.
- [AISIX Cloud overview](../cloud/overview.md) — understand the managed control-plane workflow.
- [Resource projection](../cloud/resource-projection.md) — learn how Cloud resources become gateway configuration.
- [Offline resilience](../cloud/offline-resilience.md) — understand how managed data planes continue serving with cached configuration.
