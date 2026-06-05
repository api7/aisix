---
title: Deployment Modes
description: Review self-hosted AISIX AI Gateway and AISIX Cloud managed data-plane deployments.
sidebar_position: 3
---

AISIX AI Gateway supports two main deployment modes: a self-hosted gateway and a managed data-plane model coordinated by AISIX Cloud.

Choose the operating model based on who manages configuration, credentials, certificates, and the control-plane workflow.

## Deployment Choice

Choose **self-hosted gateway** when you want a standalone runtime that you operate end to end. You manage the process, admin API, configuration store, provider credentials, network exposure, and upgrades.

Choose **AISIX Cloud managed data plane** when you want AISIX Cloud to manage the control-plane workflow while the gateway data plane still handles traffic in your network. You bootstrap the data plane with gateway certificates, and AISIX Cloud projects environment-scoped configuration to it.

If you are evaluating AISIX for the first time, start with the self-hosted quickstart. It exposes both listeners locally and makes the resource model visible. Move to the managed data-plane path when you are ready to use AISIX Cloud as the control plane.

## Mode Comparison

Self-hosted deployments expose the standalone admin API when configured. You write resources through that API, store provider credentials in your self-hosted configuration store, and bootstrap local gateway config, admin keys, proxy and admin listeners, and etcd.

Managed data planes do not expose the standalone admin write path. AISIX Cloud manages environment-scoped resources, stores provider credentials in the Cloud control plane, and projects configuration to the data plane. Bootstrap focuses on gateway certificates, mTLS control-plane communication, and environment binding.

## Self-Hosted Gateway

In self-hosted mode, you run the gateway directly and expose both the proxy listener and the admin listener.

Bootstrap configuration comes from the local config file. Dynamic resources are managed through the admin API and stored in etcd.

This mode is a good fit when you want direct control over deployment topology, admin access, etcd, credentials, and upgrades without a managed control plane.

For the hands-on path, see [Run from source](../quickstart/self-hosted.md).

## AISIX Cloud Managed Data Plane

In managed mode, AISIX Cloud becomes the control plane and AISIX AI Gateway runs as the data plane.

At the gateway level, this means the admin API listener is not bound, the standalone playground endpoint is not exposed, and dynamic configuration is read from the managed etcd path over an mTLS channel.

Managed data-plane bootstrap is centered on **gateway certificates** and mTLS-authenticated `/dp/*` endpoints. The Cloud flow creates an environment, issues a gateway certificate bundle, starts the data plane with that bundle, and confirms data-plane heartbeats and configuration propagation.

For the hands-on path, see [Connect a managed data plane](../quickstart/aisix-cloud-managed-dp.md).

## Cloud Playground Traffic

The AISIX Cloud playground is a control-plane check and does **not** send traffic through the managed data plane. It does not exercise data-plane cache, guardrails, rate limiting, or routing behavior.

See [Cloud vs. self-hosted](../cloud/cloud-vs-self-hosted.md) for the deeper comparison, and use the dedicated AISIX Cloud section for managed control-plane and managed data-plane documentation.

## Related Reading

The [Core concepts](core-concepts.md) page explains the resources configured in either deployment mode. Compare operating models in [Cloud vs. self-hosted](../cloud/cloud-vs-self-hosted.md), and review managed setup in [Connect a managed data plane](../quickstart/aisix-cloud-managed-dp.md) and [AISIX Cloud overview](../cloud/overview.md). For production readiness, see [Feature availability](feature-matrix.md).
