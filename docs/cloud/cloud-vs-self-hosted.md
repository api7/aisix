---
title: Cloud Vs Self-Hosted
description: Compare AISIX Cloud managed workflows with standalone self-hosted AISIX AI Gateway operation.
sidebar_position: 78
---

## Self-Hosted

- you run the standalone gateway
- you manage the admin API directly
- you manage bootstrap config and etcd directly

This is the right fit when you want local operational control and are comfortable owning the gateway runtime and config plane directly.

## AISIX Cloud

- you manage resources through the control plane
- the managed data plane consumes projected config
- gateway certificate issuance and managed `/dp/*` workflows replace direct standalone admin exposure on the managed path

This is the right fit when you want environment-scoped control-plane workflows and managed bootstrap behavior.

## Decision Guide

- choose self-hosted when you want direct local admin control
- choose Cloud when you want control-plane-managed environments, projection, and managed data-plane lifecycle

## Boundary Reminder

The two modes share the gateway runtime, but they do not share the same operational model.

## Related Pages

- [Deployment Modes](../overview/deployment-modes.md)
- [AISIX Cloud Overview](overview.md)
- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
