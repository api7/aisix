---
title: Offline Resilience
description: Understand AISIX Cloud and managed data-plane behavior during temporary control-plane connectivity loss.
sidebar_position: 77
toc_max_heading_level: 2
---

AISIX Cloud and the managed data plane are designed so that temporary control-plane connectivity loss does not immediately remove the data plane's ability to serve from its latest accepted configuration.

Once a managed data plane has valid projected configuration, it can continue serving live traffic from that configuration while Cloud connectivity is being restored.

## Available During Connectivity Loss

During a temporary control-plane outage, the managed data plane can keep serving requests from its latest projected configuration. That includes the models, keys, policies, and routing state already accepted by the data plane.

On restart, the data plane can use its persisted configuration state while it reconnects. This helps avoid turning every transient Cloud connectivity issue into an immediate traffic outage.

## Cloud-Dependent Workflows

Offline resilience does not make the control plane optional. New configuration changes, updated resource projection, certificate rotation, usage telemetry delivery, fresh budget decisions, heartbeat, and managed health reporting all depend on restored managed connectivity.

Use offline resilience as a continuity mechanism for already-projected configuration, not as a long-term disconnected operating mode.

## Troubleshooting

Use these checks when existing traffic and new configuration changes behave differently.

### Traffic Still Flows, but New Changes Do Not Apply

This is consistent with the resilience model. The data plane can serve from its latest accepted configuration while new projected configuration waits for Cloud connectivity to recover.

Check that the data plane can reach the data-plane-manager endpoint, the certificate is valid and chains to the expected trust root, heartbeat recovers after connectivity returns, and new projected configuration reaches the data plane after recovery.

### Usage or Budget State Looks Delayed

Check telemetry delivery and budget-check connectivity. Live request success does not confirm that every Cloud-side workflow is healthy.

## Related Reading

For how new Cloud state reaches live traffic, see [Resource projection](/ai-gateway/cloud/resource-projection). For managed connectivity, mTLS bootstrap, and runtime diagnosis, see [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp) and [Troubleshooting](/ai-gateway/operations/troubleshooting).
