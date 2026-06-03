---
title: Cloud vs. Self-Hosted
description: Compare AISIX Cloud managed workflows with standalone self-hosted AISIX AI Gateway operation.
sidebar_position: 78
---

AISIX AI Gateway can run as a standalone self-hosted gateway or as a
managed data plane connected to AISIX Cloud. Both modes use the same
gateway runtime, but they differ in how resources are managed,
delivered, and observed.

## Compare the two operating models

| Area | Standalone self-hosted | AISIX Cloud managed data plane |
| --- | --- | --- |
| Management surface | Local `/admin/v1/*` API | Cloud control plane |
| Resource scope | Gateway and etcd prefix | Organization and environment |
| Configuration delivery | Local etcd watch and in-memory snapshot | Cloud projection into the connected data plane |
| Bootstrap identity | Local bootstrap config and admin keys | Gateway certificate bundle and mTLS to `/dp/*` |
| Operator responsibility | Gateway runtime, etcd, admin exposure, telemetry, and upgrades | Connected data-plane runtime, networking, and live traffic path |
| Cloud-side workflows | Not available by default | Usage, billing, budget checks, heartbeat, and managed visibility |

In both modes, callers send traffic to the AISIX data plane. What changes is
where operators manage resources and how those resources reach the data plane.

## Choose self-hosted when

- you want direct local control of the gateway runtime
- you want to manage etcd and bootstrap config yourself
- you want to expose and operate the admin API directly
- you do not need Cloud-side organization, environment, usage, or billing
  workflows

## Choose AISIX Cloud when

- you want environment-scoped resource management
- you want certificate-based managed data-plane bootstrap
- you want Cloud-side projection, heartbeat, usage, and budget workflows
- you want callers to use the gateway while operators manage resources
  from a centralized control plane

## Important boundary

Cloud mode is not just self-hosted mode with a different UI. It changes
the operational model:

- resources are scoped by environment
- resource changes are projected asynchronously
- the data plane authenticates to managed `/dp/*` routes
- Cloud preview surfaces do not necessarily exercise the full live
  data-plane path

The practical troubleshooting question also changes. In self-hosted mode,
start with "did the admin write reach the proxy snapshot?" In Cloud mode,
start with "is this resource in the correct environment, and has projection
reached this data plane?"

## Next steps

- [Deployment modes](/ai-gateway/overview/deployment-modes) explains the
  broader deployment model.
- [AISIX Cloud overview](/ai-gateway/cloud/overview) introduces managed
  operation.
- [Self-hosted quickstart](/ai-gateway/quickstart/self-hosted) shows the
  standalone path.
