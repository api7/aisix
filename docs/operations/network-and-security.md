---
title: Network and Security
description: Operate AISIX AI Gateway with correct listener exposure, admin isolation, and credential handling.
toc_max_heading_level: 2
sidebar_position: 51
---

AISIX AI Gateway has different network entry points for caller traffic, admin traffic, configuration storage, and managed data-plane communication. Treat those entry points as separate trust zones.

## Protect Network Entry Points

In standalone mode, AISIX has two main listeners. The proxy listener serves caller-facing API traffic, such as `/v1/chat/completions`. The admin listener serves the admin API, health, metrics, and OpenAPI endpoints.

Expose the proxy listener only to intended callers or to the ingress tier that fronts caller traffic.

Keep the admin listener on loopback, a private subnet, or an admin-only network. Do not place it on the public network.

Keep etcd on a private network reachable only by the gateway and the systems or teams that manage gateway configuration.

In managed deployments, treat the `/dp/*` path as a private mTLS-authenticated connection between the data plane and AISIX Cloud.

Do not rely on admin authentication alone for network protection. Some admin-listener routes, such as `/livez`, `/metrics`, and OpenAPI endpoints, are intentionally available on the private admin listener.

Dynamic resources live in etcd and are consumed through the watch path. Protect etcd as part of the gateway control plane.

Use network isolation and TLS or mTLS where appropriate. If etcd TLS is enabled, bootstrap config must point to certificate files that the gateway process can read.

Managed AISIX Cloud deployments use mTLS-authenticated `/dp/*` paths. The data plane authenticates with its certificate bundle, not with a caller bearer token.

When diagnosing managed connectivity, check certificate identity, trust root, and data-plane-manager URL before investigating higher-level resource projection.

## Protect Secrets and Credentials

Credential handling differs by resource type. Caller API keys are stored as hashes. Provider keys store upstream credentials on the standalone path. OTLP exporter headers are stored as plaintext in the resource model.

Protect both the admin API and etcd. Anyone who can read the standalone etcd keyspace can read sensitive provider credentials and exporter headers.

In AISIX Cloud managed operation, provider-key handling is controlled by the Cloud control plane and projected into the managed data plane. The management path is different, but credentials should still be treated as sensitive operational data.

## Security Baseline

A safe baseline is to expose only the proxy listener to callers, keep the admin listener on loopback or a private admin network, and protect etcd with network isolation and TLS where appropriate.

Treat provider-key secrets and exporter headers as sensitive data. In managed deployments, validate data-plane identity through the certificate-based bootstrap path before investigating higher-level projection issues.

## Troubleshooting

Use these checks when admin, metrics, or proxy routes are reachable from a wider network than intended.

### Admin or Metrics Routes Are Reachable from the Public Network

Fix listener placement first. Do not rely on application logic to compensate for a public admin listener.

### Provider Credentials Appear in an Unexpected Place

Check etcd access, admin API access, and any backup or logging pipeline that can read dynamic resource payloads.

## Related Reading

[TLS and mTLS](/ai-gateway/operations/tls-and-mtls) covers transport security. For managed bootstrap, see [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp). For production placement, see [Production deployment](/ai-gateway/operations/production-deployment).
