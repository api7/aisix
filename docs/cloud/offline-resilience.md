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

## What This Means Operationally

The managed data plane should not depend on a perfectly available control plane for every request once it has a valid projected snapshot.

That gives operators two important properties:

- transient control-plane issues do not necessarily mean immediate traffic outage
- recovery should preserve the ability to resume heartbeat and control-plane coordination when connectivity returns

## What Offline Resilience Does Not Mean

It does not mean the control plane is irrelevant.

New configuration changes, billing-oriented workflows, certificate rotation, and fresh control-plane decisions still depend on restoring the managed connection.

## Troubleshooting

### Traffic still flows during control-plane loss, but new changes do not apply

That is consistent with the current resilience model.

## Related Pages

- [Resource Projection](resource-projection.md)
- [Gateway Certificates And Managed DP](gateway-certificates-and-managed-dp.md)
- [Troubleshooting](../operations/troubleshooting.md)
