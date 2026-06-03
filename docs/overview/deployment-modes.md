---
title: Deployment Modes
description: Compare self-hosted AISIX AI Gateway and AISIX Cloud managed data-plane deployments.
sidebar_position: 3
---

AISIX AI Gateway supports two main deployment modes: a self-hosted gateway and a managed data-plane model coordinated by AISIX Cloud.

This overview compares the available operating models so you can choose the one that fits your environment.

## How to choose

Choose **self-hosted gateway** when you want a standalone runtime that you
operate end to end. You manage the process, admin API, configuration store,
provider credentials, network exposure, and upgrades.

Choose **AISIX Cloud managed data plane** when you want AISIX Cloud to manage
the control-plane workflow while the gateway data plane still handles traffic in
your network. You bootstrap the data plane with gateway certificates, and AISIX
Cloud projects environment-scoped configuration to it.

If you are evaluating AISIX for the first time, start with the self-hosted
quickstart. It exposes both listeners locally and makes the resource model
visible. Move to the managed data-plane path when you are ready to use AISIX
Cloud as the control plane.

## At a glance

| Question | Self-hosted gateway | AISIX Cloud managed data plane |
| --- | --- | --- |
| Who manages configuration? | You write resources through the standalone admin API. | AISIX Cloud manages and projects environment-scoped resources. |
| Does the gateway expose an admin listener? | Yes, when configured. | No. The data plane exposes proxy traffic, not the standalone admin write path. |
| Where do provider credentials live? | In your self-hosted configuration store. | In the Cloud control plane and projected to the data plane. |
| What is the bootstrap focus? | Local gateway config, admin keys, proxy/admin listeners, and etcd. | Gateway certificates, mTLS control-plane communication, and environment binding. |

## Self-hosted gateway

In self-hosted mode, you run the gateway directly and expose both:

- the proxy listener
- the admin listener

Bootstrap configuration comes from the local config file, and dynamic resources are managed through the admin API and stored in etcd.

This mode is a good fit when you want:

- full control over deployment topology
- direct access to the admin surface
- self-managed etcd and credentials
- a local or private operational model without a managed control plane

For the hands-on path, see [Run from source](../quickstart/self-hosted.md).

## AISIX Cloud managed data plane

In managed mode, AISIX Cloud becomes the control plane and AISIX AI Gateway runs as the data plane.

At the gateway level, this changes several behaviors:

- the admin API listener is not bound
- the standalone playground endpoint is not exposed
- dynamic configuration is read from the managed etcd path over an mTLS channel

Managed data-plane bootstrap is centered on **gateway certificates** and mTLS-authenticated `/dp/*` endpoints. The Cloud flow creates an environment, issues a gateway certificate bundle, starts the data plane with that bundle, and then confirms data-plane heartbeats and configuration propagation.

For the hands-on path, see [Connect a managed data plane](../quickstart/aisix-cloud-managed-dp.md).

## Important boundary

Do not assume that every Cloud feature is a gateway feature.

For example, the current AISIX Cloud playground is a control-plane preview path and does **not** send traffic through the managed data plane. That means it does not exercise data-plane cache, guardrails, rate limiting, or routing behavior.

See [Cloud vs. self-hosted](../cloud/cloud-vs-self-hosted.md) for the deeper
comparison, and use the dedicated AISIX Cloud section for current managed
control-plane and managed data-plane documentation.

## Next steps

- [Core concepts](core-concepts.md) — learn the resources operators configure in either deployment mode.
- [Cloud vs. self-hosted](../cloud/cloud-vs-self-hosted.md) — compare the operating models in more detail.
- [Connect a managed data plane](../quickstart/aisix-cloud-managed-dp.md) — understand the Cloud bootstrap path.
- [AISIX Cloud overview](../cloud/overview.md) — review the managed control-plane workflow.
- [Feature status](feature-matrix.md) — check current feature status and boundaries.
