---
title: Network and Security
description: Operate AISIX AI Gateway with correct listener exposure, admin isolation, and credential handling boundaries.
sidebar_position: 51
---

AISIX AI Gateway has different network surfaces for caller traffic,
operator traffic, configuration storage, and managed data-plane
communication. Treat those surfaces as separate trust zones.

## Separate the listener trust zones

In standalone mode, AISIX has two main listeners:

- **Proxy listener**: caller-facing API traffic, such as
  `/v1/chat/completions`.
- **Admin listener**: operator-facing admin API, health, metrics, and
  OpenAPI surfaces.

Expose the proxy listener only to intended callers or to the ingress tier
that fronts caller traffic.

Keep the admin listener on loopback, a private subnet, or an
operator-only network. Do not place it on the public network.

Keep etcd on a private network reachable only by the gateway and
operators who manage gateway configuration.

In managed deployments, treat the `/dp/*` path as a private
mTLS-authenticated connection between the data plane and AISIX Cloud.

Do not rely on admin authentication alone as the network boundary. Some
admin-listener routes, such as `/livez`, `/metrics`, and OpenAPI
endpoints, are intentionally available on the private admin listener.

## Protect secrets and credentials

Credential handling differs by resource type:

- caller API keys are stored as hashes
- provider keys store upstream credentials on the standalone path
- OTLP exporter headers are stored as plaintext in the resource model

Protect both the admin surface and etcd. Anyone who can read the
standalone etcd keyspace can read sensitive provider credentials and
exporter headers.

In AISIX Cloud managed operation, provider-key handling is controlled by
the Cloud control plane and projected into the managed data plane. The
operator-facing boundary is different, but credentials should still be
treated as sensitive operational data.

## Protect the etcd boundary

Dynamic resources live in etcd and are consumed through the watch
supervisor. Protect etcd as part of the gateway control-plane boundary.

Use network isolation and TLS or mTLS where appropriate. If etcd TLS is
enabled, bootstrap config must point to certificate files that the
gateway process can read.

## Understand the managed security boundary

Managed AISIX Cloud deployments use mTLS-authenticated `/dp/*` paths.
The data plane authenticates with its certificate bundle, not with a
caller bearer token.

When diagnosing managed connectivity, check certificate identity, trust
root, and data-plane-manager URL before investigating higher-level
resource projection.

## Start with this security posture

1. Expose only the proxy listener to callers.
2. Keep admin on loopback or a private operator network.
3. Protect etcd with network isolation and TLS where appropriate.
4. Treat provider-key secrets and exporter headers as sensitive data.
5. Validate managed data-plane identity through the certificate-based
   bootstrap path.

## Troubleshooting

### Admin or metrics routes are reachable from the public network

Fix listener placement first. Do not rely on application logic to
compensate for a public admin surface.

### Provider credentials appear in an unexpected place

Check etcd access, admin API access, and any backup or logging pipeline
that can read dynamic resource payloads.

## Next steps

- [TLS and mTLS](/ai-gateway/operations/tls-and-mtls) explains transport
  security boundaries.
- [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp)
  explains managed bootstrap.
- [Production deployment](/ai-gateway/operations/production-deployment)
  gives the production baseline checklist.
