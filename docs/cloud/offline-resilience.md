---
title: Offline Resilience
description: Understand the current AISIX Cloud and managed data-plane offline-resilience behavior.
sidebar_position: 77
---

AISIX Cloud and the managed data plane are designed so that transient control-plane loss does not immediately erase the data plane's ability to serve from its current config state.

Current resilience signals in code and e2e coverage include:

- on-disk snapshot cache behavior
- serving from previously projected config while control-plane paths are unavailable
- heartbeat and managed connectivity recovering when the control plane comes back

## Related Pages

- [Resource Projection](resource-projection.md)
- [Gateway Certificates And Managed DP](gateway-certificates-and-managed-dp.md)
- [Troubleshooting](../operations/troubleshooting.md)
