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

## Related Pages

- [Organizations And Environments](organizations-and-environments.md)
- [Offline Resilience](offline-resilience.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
