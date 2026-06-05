---
title: Operations Overview
sidebar_label: Overview
description: Choose the right operations guide for deploying, securing, observing, verifying, and troubleshooting AISIX AI Gateway.
sidebar_position: 49
---

When AISIX AI Gateway moves from a local quickstart into a real environment, focus on the running data plane: how it starts, which listeners to expose, how to verify traffic, and how to diagnose failures in production-like deployments.

If you are still creating provider keys, models, caller keys, or runtime policies, start with [Configuration overview](../configuration/overview.md) first.

## Deployment Path

Start with [Production deployment](production-deployment.md) when the gateway is moving toward real traffic. It sets the baseline for bootstrap configuration, listeners, etcd, cache, and the first production checks.

Before exposing the proxy, use [Network and security](network-and-security.md) and [TLS and mTLS](tls-and-mtls.md) to set listener placement, credential handling, and encrypted transport.

After the runtime is deployed, verify the caller-to-provider path with [Health checks](health-checks.md), [Metrics and logs](metrics-and-logs.md), and [Testing and verification](testing-and-verification.md). If a running deployment does not behave as expected, continue with [Troubleshooting](troubleshooting.md). Before widening traffic to a new version, review [Upgrades and compatibility](upgrades-and-compatibility.md).

## Standalone and Managed Differences

Standalone gateways expose a local admin listener when configured. Use it for admin health, metrics, OpenAPI, and dynamic-resource management, and keep it on a private admin network.

Managed data planes receive projected resources from AISIX Cloud and use the managed mTLS path instead of the standalone admin API. For managed operation, pair these runtime checks with [AISIX Cloud overview](../cloud/overview.md) and [Gateway certificates and managed data plane](../cloud/gateway-certificates-and-managed-dp.md).

## Related Reading

For the first production rollout, see [Production deployment](production-deployment.md). For end-to-end validation, see [Testing and verification](testing-and-verification.md). For symptoms that appear after deployment, see [Troubleshooting](troubleshooting.md).
