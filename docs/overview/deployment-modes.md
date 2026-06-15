---
title: Deployment Modes
description: Compare self-hosted AISIX AI Gateway and AISIX Cloud managed data-plane deployments.
sidebar_position: 2
---

AISIX AI Gateway supports two main deployment modes: a self-hosted gateway and a managed data-plane model coordinated by AISIX Cloud.

Use this page to decide which operating model fits your environment.

## Self-Hosted Gateway

In self-hosted mode, you run the gateway directly and expose both:

- the proxy listener
- the admin listener

Bootstrap configuration comes from the local config file, and dynamic resources are managed through the admin API and stored in etcd.

This mode is a good fit when you want:

- full control over deployment topology
- direct access to the admin surface
- self-managed etcd and credentials
- a local or private operational model without a managed control plane

## AISIX Cloud Managed Data Plane

In managed mode, AISIX Cloud becomes the control plane and AISIX AI Gateway runs as the data plane.

At the gateway level, this changes several behaviors:

- the admin API listener is not bound
- the standalone playground endpoint is not exposed
- dynamic configuration is read from the managed etcd path over an mTLS channel

From the current AISIX Cloud implementation, managed data-plane bootstrap is centered on **gateway certificates** and mTLS-authenticated `/dp/*` endpoints. The current Cloud e2e flow creates an environment, issues a gateway certificate bundle, starts the data plane with that bundle, and then confirms DP heartbeats and config propagation.

## Comparison

| Dimension | Self-Hosted Gateway | AISIX Cloud Managed Data Plane |
| --- | --- | --- |
| Product boundary | Gateway only | Gateway plus managed control plane |
| Admin API | Exposed by the gateway | Not exposed on the data plane |
| Dynamic resource management | Gateway admin API + etcd | AISIX Cloud control plane |
| Data-plane auth to control plane | Operator-managed | mTLS certificate flow |
| Environment model | Operator-defined | First-class Cloud environment |
| Cloud billing and usage workflows | Not built in | Managed by AISIX Cloud |

## How To Choose

Choose **self-hosted** when you want a standalone gateway that you control end to end.

Choose **AISIX Cloud** when you want:

- centralized environment management
- managed certificate and control-plane workflows
- usage-event and billing integration at the Cloud layer

## Important Boundary

Do not assume that every Cloud feature is a gateway feature.

For example, the current AISIX Cloud playground is a control-plane preview path and does **not** send traffic through the managed data plane. That means it does not exercise data-plane cache, guardrails, rate limiting, or routing behavior.

See the dedicated Cloud section for current managed control-plane and managed data-plane documentation.

## Related Pages

- [What Is AISIX AI Gateway](what-is-aisix-ai-gateway.md)
- [Core Concepts](core-concepts.md)
- [Feature Matrix](feature-matrix.md)
- [AISIX Cloud Overview](../cloud/overview.md)
- [Roadmap](../roadmap.md)
