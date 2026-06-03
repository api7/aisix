---
title: Operations Overview
sidebar_label: Overview
description: Choose the right operations guide for deploying, securing, observing, verifying, and troubleshooting AISIX AI Gateway.
sidebar_position: 49
---

Use the operations section when AISIX AI Gateway is moving from a local
quickstart into a real environment. These pages focus on the running data
plane: how it starts, what surfaces to expose, how to verify traffic, and
how to diagnose production failures.

If you are still creating provider keys, models, caller keys, or runtime
policies, start with [Configuration overview](../configuration/overview.md)
first.

## Start with the right guide

| What you need to do | Start with | What it answers |
| --- | --- | --- |
| Roll out a gateway for real traffic | [Production deployment](production-deployment.md) | What must be true before the proxy handles production AI requests. |
| Set network and credential boundaries | [Network and security](network-and-security.md) | Which listeners, stores, and secrets belong on private networks. |
| Configure encrypted transport | [TLS and mTLS](tls-and-mtls.md) | Where listener TLS, etcd TLS, and managed data-plane mTLS apply. |
| Probe runtime health | [Health checks](health-checks.md) | Which liveness, admin-health, and model-health endpoints to use. |
| Observe traffic and usage | [Metrics and logs](metrics-and-logs.md) | Which metrics, logs, headers, usage events, and exporters explain runtime behavior. |
| Prove a deployment works | [Testing and verification](testing-and-verification.md) | How to verify the full caller-to-provider path, not only process startup. |
| Diagnose a failure | [Troubleshooting](troubleshooting.md) | How to narrow a symptom to startup, propagation, access, policy, upstream, or managed connectivity. |
| Upgrade safely | [Upgrades and compatibility](upgrades-and-compatibility.md) | What to verify before widening traffic to a new version. |

## Recommended reading order

For a first production-minded rollout, read in this order:

1. [Production deployment](production-deployment.md)
2. [Network and security](network-and-security.md)
3. [Health checks](health-checks.md)
4. [Metrics and logs](metrics-and-logs.md)
5. [Testing and verification](testing-and-verification.md)
6. [Troubleshooting](troubleshooting.md)

This path starts with deployment shape and boundaries, then adds the signals
operators need to prove and debug the live request path.

## Standalone and managed differences

Standalone gateways expose a local admin listener when configured. Use that
listener for admin health, metrics, OpenAPI, and dynamic-resource management,
and keep it on a private operator network.

Managed data planes receive projected resources from AISIX Cloud and use the
managed mTLS path instead of the standalone admin API. For managed operation,
pair this section with [AISIX Cloud overview](../cloud/overview.md) and
[Gateway certificates and managed data plane](../cloud/gateway-certificates-and-managed-dp.md).

## Next steps

- [Production deployment](production-deployment.md)
- [Network and security](network-and-security.md)
- [Testing and verification](testing-and-verification.md)
- [Troubleshooting](troubleshooting.md)
