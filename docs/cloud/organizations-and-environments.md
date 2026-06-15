---
title: Organizations And Environments
description: Understand how AISIX Cloud organizes tenant scope and environment-level gateway resources.
sidebar_position: 71
---

AISIX Cloud introduces organization and environment concepts that do not exist as first-class standalone gateway resources.

These are the main tenancy and scoping units for managed deployments.

Current customer-facing pattern:

- resources are managed under environments
- managed data planes operate within an environment scope
- environment scoping controls what resources project into a given data plane

## How To Think About Scope

- organization scope answers who owns the Cloud resources
- environment scope answers which managed data plane receives which projected configuration

For most operator reasoning, the environment is the important unit for deployment and traffic behavior.

## What Changes Relative To Self-Hosted

In standalone self-hosted mode, operators think directly in terms of one gateway instance plus its etcd state.

In Cloud mode, operators should think in terms of environment-scoped resources being projected into a managed data plane.

## Operational Implication

When a resource appears or does not appear in a managed data plane, the first question is often whether it belongs to the correct environment scope.

## Troubleshooting

### A resource exists in Cloud but not in the target data plane

Check environment scope and projection first.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Resource Projection](resource-projection.md)
- [Cloud Vs Self-Hosted](cloud-vs-self-hosted.md)
