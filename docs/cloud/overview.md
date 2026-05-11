---
title: AISIX Cloud Overview
description: Understand what AISIX Cloud adds on top of AISIX AI Gateway for managed control-plane and data-plane operation.
sidebar_position: 70
---

AISIX Cloud adds a managed control plane on top of AISIX AI Gateway.

Current Cloud-specific value includes:

- organizations and environments
- managed gateway certificate issuance
- resource projection into the managed data plane
- usage-event ingestion and billing workflows
- control-plane-managed resilience paths

## Current Managed DP Boundary

Current AISIX Cloud managed data-plane behavior is centered on:

- gateway certificate issuance through the control plane
- mTLS-authenticated `/dp/*` routes
- config propagation from the control plane into the data plane

## Related Pages

- [Organizations And Environments](organizations-and-environments.md)
- [Gateway Certificates And Managed DP](gateway-certificates-and-managed-dp.md)
- [Cloud Vs Self-Hosted](cloud-vs-self-hosted.md)
