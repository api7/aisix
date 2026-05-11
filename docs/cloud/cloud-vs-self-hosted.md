---
title: Cloud Vs Self-Hosted
description: Compare AISIX Cloud managed workflows with standalone self-hosted AISIX AI Gateway operation.
sidebar_position: 78
---

## Self-Hosted

- you run the standalone gateway
- you manage the admin API directly
- you manage bootstrap config and etcd directly

## AISIX Cloud

- you manage resources through the control plane
- the managed data plane consumes projected config
- gateway certificate issuance and managed `/dp/*` workflows replace direct standalone admin exposure on the managed path

## Related Pages

- [Deployment Modes](../overview/deployment-modes.md)
- [AISIX Cloud Overview](overview.md)
- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
