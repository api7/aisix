---
title: Resource Projection
description: Understand how AISIX Cloud projects environment resources into the managed data plane.
sidebar_position: 73
---

AISIX Cloud manages resources at the control-plane layer and projects them into the managed data plane.

From the customer's point of view, the important behavior is:

- environment resources become available to the managed data plane after propagation
- the data plane serves traffic from its current projected snapshot
- propagation is asynchronous, not instantaneous

This is the Cloud equivalent of configuration propagation in standalone mode, but with an explicit control-plane to data-plane boundary.

## What Projection Means Operationally

From an operator point of view, projection is the mechanism that turns control-plane resource state into live data-plane behavior.

That means these are separate events:

1. saving the resource in Cloud
2. projecting it into the managed data plane
3. serving traffic from the new projected snapshot

## What To Expect

- configuration changes are fast, but not instantaneous
- the data plane can continue serving from an older projected snapshot during transient control-plane issues
- validation of live behavior should use the managed data plane, not just the control-plane UI or API response

## Troubleshooting

### The control plane shows the new resource, but live traffic does not use it yet

Treat that as a projection/readiness problem first.

## Related Pages

- [Organizations And Environments](organizations-and-environments.md)
- [Offline Resilience](offline-resilience.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
