---
title: TLS And mTLS
description: Understand listener TLS, etcd TLS, and managed-mode mTLS bootstrap in AISIX AI Gateway.
sidebar_position: 52
---

AISIX AI Gateway uses TLS in three distinct places:

- listener TLS for proxy and admin endpoints
- etcd TLS or mTLS for config transport
- managed-mode mTLS for data-plane communication with the control plane

Use this page to separate those three concerns clearly during deployment and debugging.

## Listener TLS

Bootstrap config supports optional TLS on:

- `proxy.tls`
- `admin.tls`

Use listener TLS whenever these surfaces are exposed beyond local development.

This is the correct place to secure inbound client and operator traffic.

## etcd TLS

`etcd.tls` can provide:

- CA certificate
- client certificate
- client private key
- optional domain name override

This is the right path when your etcd deployment requires TLS or mTLS.

It is independent from listener TLS. A working HTTPS proxy listener does not tell you anything about etcd trust configuration.

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

## How To Think About Failures

- listener TLS failures usually show up on inbound proxy or admin traffic
- etcd TLS failures usually show up as bootstrap or watch/connectivity problems
- managed mTLS failures usually show up on heartbeat, rotation, or budget-check paths

## Failure Signals

Common TLS or mTLS failures surface as:

- startup failures reading certificate files
- outbound client build failures for heartbeat or budget check
- etcd connection failures that can look like transport or DNS errors

## Troubleshooting

### The process fails at startup with certificate errors

Check file readability and whether the configured cert/key pair matches the intended TLS use.

### Managed mode starts but never heartbeats

Treat that as a managed mTLS bundle or trust-root problem first.

## Related Pages

- [Production Deployment](production-deployment.md)
- [Network And Security](network-and-security.md)
- [Gateway Certificates And Managed DP](../cloud/gateway-certificates-and-managed-dp.md)
