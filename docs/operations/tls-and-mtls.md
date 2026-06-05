---
title: TLS and mTLS
description: Understand listener TLS, etcd TLS, and managed-mode mTLS bootstrap in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 52
---

AISIX AI Gateway uses TLS in three different places. Configure and troubleshoot them separately.

Listener TLS protects inbound proxy and admin traffic. Configure it with `proxy.tls` and `admin.tls`.

etcd TLS or mTLS protects the configuration-store transport. Configure it with `etcd.tls`.

Managed mTLS authenticates data-plane communication with AISIX Cloud. It uses the managed certificate bundle instead of the standalone listener TLS settings.

## Listener TLS

Use listener TLS when the proxy or admin listener is exposed beyond local development.

Bootstrap config supports optional TLS on `proxy.tls` and `admin.tls`.

Listener TLS protects inbound caller and admin traffic. It does not confirm that the gateway can connect to etcd or to AISIX Cloud.

## etcd TLS

`etcd.tls` configures trust for the etcd client connection. It can include a CA certificate, client certificate, client private key, and optional domain name override.

Use etcd TLS or mTLS when your etcd deployment requires encrypted or mutually authenticated transport. Certificate files must be readable by the gateway process at startup.

## Managed mTLS Bundle

Managed data planes authenticate to AISIX Cloud with a certificate bundle. The bundle contains `ca.crt`, `client.crt`, and `client.key`.

The runtime stores and reads the bundle from the managed `mtls_dir`. For AISIX Cloud operation, use the certificate-bundle bootstrap flow described in [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp).

## Failure Signals

If HTTPS proxy requests fail while the process is running, start with listener TLS.

If startup fails while connecting to etcd, inspect the etcd TLS settings and the network path to etcd.

If watch freshness stalls after boot, focus on etcd TLS, network connectivity, and configuration watch health.

If managed heartbeat never succeeds, inspect the managed mTLS bundle and the data-plane-manager URL.

If budget checks or certificate rotation fail in managed mode, inspect managed mTLS and `/dp/*` connectivity.

## Troubleshooting

Use these checks when TLS configuration prevents the gateway from starting or accepting traffic.

### The Process Fails at Startup with Certificate Errors

Check file paths, file permissions, certificate/key pairing, and whether the certificate is configured for the correct TLS area.

### Managed Mode Starts but Never Heartbeats

Check the managed certificate bundle, trust root, runtime state directory, and `AISIX_MANAGED__CP_BASE_URL`.

## Related Reading

[Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp) covers managed bootstrap. For listener exposure, see [Network and security](/ai-gateway/operations/network-and-security). For production placement, see [Production deployment](/ai-gateway/operations/production-deployment).
