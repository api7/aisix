---
title: Offline Resilience
description: Understand AISIX Cloud and managed data-plane behavior during temporary control-plane connectivity loss.
sidebar_position: 77
---

AISIX Cloud and the managed data plane are designed so that temporary
control-plane connectivity loss does not immediately remove the data
plane's ability to serve from its current accepted configuration.

Once a managed data plane has a valid projected snapshot, it can continue
serving live traffic from that snapshot while Cloud connectivity is being
restored.

## What remains available

During a temporary control-plane outage, the managed data plane can keep
serving requests from its current projected configuration. That includes
the models, keys, policies, and routing state already accepted by the
data plane.

On restart, the data plane can use its persisted snapshot state while it
reconnects. This helps avoid turning every transient Cloud connectivity
issue into an immediate traffic outage.

## What still depends on Cloud

Offline resilience does not make the control plane optional. These
workflows still depend on restoring managed connectivity:

- new configuration changes
- projection of updated environment resources
- certificate rotation workflows
- usage telemetry delivery
- fresh budget decisions from Cloud
- heartbeat and managed health reporting

Use offline resilience as a continuity mechanism for already-projected
state, not as a long-term disconnected operating mode.

## Troubleshooting

### Traffic still flows, but new changes do not apply

That is consistent with the resilience model. The data plane can serve
from its current snapshot while new projected state waits for Cloud
connectivity to recover.

Check:

1. Connectivity from the data plane to the data-plane-manager endpoint.
2. Certificate validity and trust chain.
3. Heartbeat recovery.
4. Projection status after connectivity returns.

### Usage or budget state looks delayed

Check telemetry delivery and budget-check connectivity. Live request
success does not prove that every Cloud-side workflow is healthy.

## Next steps

- [Resource projection](/ai-gateway/cloud/resource-projection) explains
  how new Cloud state reaches live traffic.
- [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp)
  explains managed connectivity and mTLS bootstrap.
- [Troubleshooting](/ai-gateway/operations/troubleshooting) lists common
  diagnosis steps.
