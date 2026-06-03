---
title: Organizations and Environments
description: Understand how AISIX Cloud organizes tenant scope and environment-level gateway resources.
sidebar_position: 71
---

AISIX Cloud organizes managed gateway resources by organization and
environment. These concepts do not exist as first-class resources in a
standalone self-hosted gateway, but they are central to managed
operation.

An organization owns Cloud resources. An environment defines the managed
deployment scope that receives projected gateway configuration.

## Start by thinking about scope

An organization answers ownership: which tenant, account, or platform
team owns the Cloud resources.

An environment answers placement: which managed data plane should receive
this model, key, or policy.

For most traffic and troubleshooting work, the environment is the most
important unit. Models, provider keys, API keys, and policies must belong
to the environment that the target managed data plane serves.

## What changes from self-hosted mode

In self-hosted mode, operators usually reason about one gateway runtime
and its etcd-backed configuration. In Cloud mode, operators reason about
environment-scoped resources that are projected into one or more managed
data planes.

That changes the first troubleshooting question. Instead of only asking
"does this resource exist?", also ask "does this resource exist in the
environment served by this data plane?"

## Common checks

When a resource does not appear to affect live traffic:

1. Confirm the resource belongs to the expected environment.
2. Confirm the managed data plane is attached to that environment.
3. Check projection status and data-plane health.
4. Send a live request through the managed data plane, not only through a
   Cloud preview surface.

## Next steps

- [Resource projection](/ai-gateway/cloud/resource-projection) explains
  how environment resources reach the data plane.
- [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp)
  explains how a managed data plane joins Cloud.
- [Cloud vs. self-hosted](/ai-gateway/cloud/cloud-vs-self-hosted)
  compares the operating models.
