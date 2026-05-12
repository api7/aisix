---
title: AISIX Cloud Managed Data Plane Quickstart
description: Understand the current AISIX Cloud managed data-plane bootstrap flow based on gateway certificates and mTLS.
sidebar_position: 14
---

This guide explains the current managed data-plane bootstrap path for AISIX Cloud.

Use it when you want to understand how a data plane connects to the AISIX Cloud control plane, receives configuration, and starts serving traffic.

## Current Bootstrap Model

The current managed flow is certificate-based.

At a high level:

1. create an environment in AISIX Cloud
2. request a gateway certificate bundle for that environment
3. start the AISIX data plane with the issued certificate, key, and CA bundle
4. let the data plane connect to the control plane over mTLS
5. receive projected configuration and begin serving requests

The current `/dp/*` surface is mTLS-authenticated. The older `/dp/register` bearer-token pattern is not the current bootstrap path.

## What The Control Plane Issues

The control plane issues a gateway certificate bundle for the target environment.

That bundle contains:

- a client certificate PEM
- a private key PEM
- a CA bundle PEM

The data plane uses those materials to authenticate to the control-plane data-plane manager.

## What The Data Plane Receives

In managed mode, the data plane is started with environment variables for:

- `AISIX_MANAGED__CP_BASE_URL`
- `AISIX_MANAGED__CP_ETCD_ENDPOINT`
- `AISIX_MANAGED__CP_CERT_PEM`
- `AISIX_MANAGED__CP_KEY_PEM`
- `AISIX_MANAGED__CP_CA_PEM`

The managed data plane then uses the same single binary, but follows the managed bootstrap path instead of binding the standalone admin surface.

## Operational Notes

- configuration is projected from the control plane into the managed environment
- the data plane persists a local config cache for offline resilience
- the control-plane data-plane manager exposes `/dp/heartbeat`, `/dp/telemetry`, `/dp/rotate-cert`, and `/dp/budget_check` behind mTLS

## Product Boundary

This page is intentionally about the bootstrap flow, not a full step-by-step dashboard tutorial.

For current customer-facing Cloud behavior, continue with the Cloud docs section.

## Related Pages

- [AISIX Cloud Overview](../cloud/overview.md)
- [Gateway Certificates And Managed DP](../cloud/gateway-certificates-and-managed-dp.md)
- [Resource Projection](../cloud/resource-projection.md)
- [Offline Resilience](../cloud/offline-resilience.md)
