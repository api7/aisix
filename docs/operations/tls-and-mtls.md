---
title: TLS and mTLS
description: Understand listener TLS, etcd TLS, and managed-mode mTLS bootstrap in AISIX AI Gateway.
sidebar_position: 52
---

AISIX AI Gateway uses TLS in three different places. Configure and
troubleshoot them separately.

Listener TLS protects inbound proxy and admin traffic. Configure it with
`proxy.tls` and `admin.tls`.

etcd TLS or mTLS protects the configuration-store transport. Configure it
with `etcd.tls`.

Managed mTLS authenticates data-plane communication with AISIX Cloud. It
uses the managed certificate bundle instead of the standalone listener TLS
settings.

## Listener TLS

Use listener TLS when the proxy or admin listener is exposed beyond local
development.

Bootstrap config supports optional TLS on:

- `proxy.tls`
- `admin.tls`

Listener TLS protects inbound caller and operator traffic. It does not
prove that the gateway can connect to etcd or to AISIX Cloud.

## etcd TLS

`etcd.tls` configures trust for the etcd client connection. It can
include:

- CA certificate
- client certificate
- client private key
- optional domain name override

Use etcd TLS or mTLS when your etcd deployment requires encrypted or
mutually authenticated transport. Certificate files must be readable by
the gateway process at startup.

## Managed mTLS bundle

Managed data planes authenticate to AISIX Cloud with a certificate
bundle. The bundle contains:

- `ca.crt`
- `client.crt`
- `client.key`

The runtime stores and reads the bundle from the managed `mtls_dir`.
For current AISIX Cloud operation, use the certificate-bundle bootstrap
flow documented in [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp).

## Failure signals

If HTTPS proxy requests fail while the process is running, start with
listener TLS.

If startup fails while connecting to etcd, inspect the etcd TLS settings
and the network path to etcd.

If watch freshness stalls after boot, focus on etcd TLS, network
connectivity, and watch supervisor health.

If managed heartbeat never succeeds, inspect the managed mTLS bundle and
the data-plane-manager URL.

If budget checks or certificate rotation fail in managed mode, inspect
managed mTLS and `/dp/*` connectivity.

## Troubleshooting

### The process fails at startup with certificate errors

Check file paths, file permissions, certificate/key pairing, and whether
the certificate is configured for the correct TLS area.

### Managed mode starts but never heartbeats

Check the managed certificate bundle, trust root, runtime state
directory, and `AISIX_MANAGED__CP_BASE_URL`.

## Next steps

- [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp)
  explains managed bootstrap.
- [Network and security](/ai-gateway/operations/network-and-security)
  explains listener exposure.
- [Production deployment](/ai-gateway/operations/production-deployment)
  covers production placement and verification.
