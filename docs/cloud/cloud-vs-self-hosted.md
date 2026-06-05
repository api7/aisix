---
title: Cloud and Self-Hosted
description: Review AISIX Cloud managed workflows alongside standalone self-hosted AISIX AI Gateway operation.
sidebar_position: 78
---

AISIX AI Gateway can run as a standalone self-hosted gateway or as a managed data plane connected to AISIX Cloud. Both modes use the same gateway runtime, but they differ in how resources are managed, delivered, and observed.

## Operating Model Comparison

| Area | Standalone self-hosted | AISIX Cloud managed data plane |
| --- | --- | --- |
| Management API | Local `/admin/v1/*` API | Cloud control plane |
| Resource scope | Gateway and etcd prefix | Organization and environment |
| Configuration delivery | Local etcd watch and in-memory configuration state | Cloud projection into the connected data plane |
| Bootstrap identity | Local bootstrap config and admin keys | Gateway certificate bundle and mTLS to `/dp/*` |
| Administrative responsibility | Gateway runtime, etcd, admin exposure, telemetry, and upgrades | Connected data-plane runtime, networking, and live traffic path |
| Cloud-side workflows | Not available by default | Usage, billing, budget checks, heartbeat, and managed visibility |

In both modes, callers send traffic to the AISIX data plane. What changes is where resources are managed and how they reach the data plane.

## Self-Hosted Operation

Self-hosted operation fits teams that want direct control over the gateway runtime, etcd, bootstrap configuration, admin API exposure, telemetry, and upgrade process. It also keeps Cloud organization, environment, usage, and billing workflows out of the default operating path.

## AISIX Cloud Operation

AISIX Cloud fits teams that want environment-scoped resource management, gateway-certificate bootstrap, Cloud-side projection, heartbeat, usage, and budget workflows. Callers still send traffic to the gateway; Cloud changes how resources are managed and delivered to the data plane.

## Operational Differences

Cloud mode is not self-hosted mode with a different UI. Resources are scoped by environment, so a resource must belong to the environment served by the target data plane. Resource changes are projected asynchronously, so a saved change may not be active on every data-plane instance immediately.

The data plane authenticates to managed `/dp/*` routes. Certificate, trust, and data-plane-manager connectivity are therefore part of the live path. Cloud control-plane checks are useful, but production behavior should still be validated through the managed data-plane endpoint.

Troubleshooting also starts from a different question. In self-hosted mode, confirm that the admin write reached the proxy configuration. In Cloud mode, confirm that the resource is in the correct environment and that projection reached the target data plane.

## Related Reading

For deployment model comparison, see [Deployment modes](/ai-gateway/overview/deployment-modes). For managed operation, see [AISIX Cloud overview](/ai-gateway/cloud/overview). To run the standalone path, see [Self-hosted quickstart](/ai-gateway/quickstart/self-hosted).
