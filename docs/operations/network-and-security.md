---
title: Network And Security
description: Operate AISIX AI Gateway with correct listener exposure, admin isolation, and credential handling boundaries.
sidebar_position: 51
---

AISIX AI Gateway has two main listener surfaces in standalone mode:

- the public proxy listener
- the operator-facing admin listener

## Listener Exposure

Recommended boundary:

- expose the proxy listener to callers
- keep the admin listener private to operators and internal networks

Current admin design intentionally leaves `/health`, `/metrics`, and OpenAPI endpoints unauthenticated on that private listener, so network placement matters.

## Secrets And Credentials

Current credential handling differs by resource type:

- caller API keys are stored as `key_hash`, not plaintext
- provider keys store plaintext upstream secrets on the standalone path
- OTLP exporter headers are plaintext in the current resource model

## etcd Boundary

Dynamic resources live in etcd and are consumed through the watch supervisor. Protect etcd as part of the gateway trust boundary.

If etcd TLS or mTLS is enabled, bootstrap config must point to readable certificate files.

## Managed Security Boundary

Managed AISIX Cloud deployments use mTLS-authenticated `/dp/*` paths and managed etcd access. The data plane authenticates as a certificate-bearing component, not with a bearer token.

## Related Pages

- [TLS And mTLS](tls-and-mtls.md)
- [Health Checks](health-checks.md)
- [AISIX Cloud Overview](../cloud/overview.md)
