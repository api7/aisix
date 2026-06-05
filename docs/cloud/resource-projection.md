---
title: Resource Projection
description: Understand how AISIX Cloud projects environment resources into the managed data plane.
sidebar_position: 73
toc_max_heading_level: 2
---

AISIX Cloud stores environment-scoped resources in the control plane and projects them into the managed data plane. Projection is the Cloud equivalent of standalone configuration propagation, with an explicit control-plane to data-plane handoff.

Projection is the step that turns saved Cloud resource state into live data-plane behavior.

## Projection Path

The following diagram shows the path from saved Cloud resource state to live data-plane behavior:

```mermaid
flowchart LR
  save[Save resource in Cloud] --> project[Project environment config]
  project --> receive[Data plane receives config]
  receive --> serve[Live traffic uses new config]
```

Projection has four distinct events: the resource is saved in Cloud, Cloud projects the environment configuration, the managed data plane receives the projected configuration, and live traffic starts using the new configuration.

Projection is usually fast, but it is asynchronous. A successful save in Cloud does not confirm that every data-plane instance is already serving the new state.

## Projection Behavior

During temporary control-plane connectivity issues, a data plane can continue serving from its latest projected configuration. Validate live behavior through the managed data plane, not only through the Cloud UI or API response. If multiple data-plane instances are attached, they can briefly converge at different times.

## Troubleshooting

Use these checks when Cloud state and managed data-plane behavior do not line up.

### Cloud Shows the New Resource, but Live Traffic Does Not Use It

Confirm the resource belongs to the same environment as the target data plane, the data plane is healthy and connected, the projected configuration has reached the data plane, and the request is sent to the managed data plane rather than only to a Cloud UI check.

If the resource is a model, also confirm the caller is using the expected model alias. If the resource is a provider key or policy, confirm the model references the updated resource.

## Related Reading

For Cloud resource scope, see [Organizations and environments](/ai-gateway/cloud/organizations-and-environments). For temporary Cloud connectivity loss, see [Offline resilience](/ai-gateway/cloud/offline-resilience). To compare standalone propagation, see [Configuration propagation](/ai-gateway/configuration/configuration-propagation).
