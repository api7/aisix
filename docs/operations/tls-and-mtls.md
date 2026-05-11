---
title: TLS And mTLS
description: Understand listener TLS, etcd TLS, and managed-mode mTLS bootstrap in AISIX AI Gateway.
sidebar_position: 52
---

AISIX AI Gateway uses TLS in three distinct places:

- listener TLS for proxy and admin endpoints
- etcd TLS or mTLS for config transport
- managed-mode mTLS for data-plane communication with the control plane

## Listener TLS

Bootstrap config supports optional TLS on:

- `proxy.tls`
- `admin.tls`

Use listener TLS whenever these surfaces are exposed beyond local development.

## etcd TLS

`etcd.tls` can provide:

- CA certificate
- client certificate
- client private key
- optional domain name override

This is the right path when your etcd deployment requires TLS or mTLS.

## Managed mTLS Bundle

Managed mode expects a bundle rooted in:

- `ca.crt`
- `client.crt`
- `client.key`

The runtime stores and reads this bundle from the managed `mtls_dir`.

Current managed bootstrap paths include:

- pre-provisioned certificate bundle
- registration-token path still present in runtime

For current AISIX Cloud behavior, treat the certificate-bundle flow as the primary path.

## Failure Signals

Common TLS or mTLS failures surface as:

- startup failures reading certificate files
- outbound client build failures for heartbeat or budget check
- etcd connection failures that can look like transport or DNS errors

## Related Pages

- [Production Deployment](production-deployment.md)
- [Network And Security](network-and-security.md)
- [Gateway Certificates And Managed DP](../cloud/gateway-certificates-and-managed-dp.md)
