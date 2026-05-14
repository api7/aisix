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

Use AISIX Cloud when you want the gateway runtime but do not want every environment to be managed only through a local standalone admin API.

## Current Managed DP Boundary

Current AISIX Cloud managed data-plane behavior is centered on:

- gateway certificate issuance through the control plane
- mTLS-authenticated `/dp/*` routes
- config propagation from the control plane into the data plane

That means the center of operational control moves from local admin writes to control-plane-managed environment state.

## Typical Customer Journey

At a high level, the current Cloud path looks like this:

1. create or select an organization and environment
2. define environment-scoped resources in the control plane
3. issue a gateway certificate bundle for the managed data plane
4. start the data plane with the managed bootstrap inputs
5. wait for projection, heartbeat, and ready traffic flow

## When To Choose Cloud

- you want control-plane-managed environments
- you want managed certificate issuance and mTLS-managed data-plane flows
- you want Cloud-side usage-event, billing, and budget workflows

## What Cloud Does Not Mean

It does not mean every control-plane feature is identical to the standalone data-plane execution path.

The clearest example today is the Cloud playground, which is preview-oriented and does not represent the full managed data-plane path.

## Troubleshooting Pointers

### The managed data plane is running but does not reflect recent control-plane changes

Treat that first as a projection or propagation issue, not as a generic proxy failure.

### The Cloud playground behaves differently from live DP traffic

That is expected with the current product boundary.

## Related Pages

- [Organizations And Environments](organizations-and-environments.md)
- [Gateway Certificates And Managed DP](gateway-certificates-and-managed-dp.md)
- [Cloud Vs Self-Hosted](cloud-vs-self-hosted.md)
