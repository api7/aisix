---
title: Organizations and Environments
description: Understand how AISIX Cloud organizes tenant scope and environment-level gateway resources.
sidebar_position: 71
---

AISIX Cloud organizes managed gateway resources by organization and environment. These concepts do not exist as first-class resources in a standalone self-hosted gateway, but they are central to managed operation.

An organization owns Cloud resources. An environment defines the managed deployment scope that receives projected gateway configuration.

## Environment Scope

An organization answers ownership: which tenant, account, or platform team owns the Cloud resources.

An environment answers placement: which managed data plane should receive this model, key, or policy.

For most traffic and troubleshooting work, the environment is the most important unit. Models, provider keys, API keys, and policies must belong to the environment that the target managed data plane serves.

## Managed Mode Differences

In self-hosted mode, teams usually reason about one gateway runtime and its etcd-backed configuration. In Cloud mode, teams reason about environment-scoped resources that are projected into one or more managed data planes.

That changes the first troubleshooting check. Confirm that the resource exists, then confirm that it belongs to the environment served by the target data plane.

## Common Checks

When a resource does not appear to affect live traffic, confirm the resource belongs to the expected environment, the managed data plane is attached to that environment, projection status and data-plane health are current, and the request goes through the managed data plane rather than only through a Cloud UI check.

## Related Reading

For how environment resources reach the data plane, see [Resource projection](/ai-gateway/cloud/resource-projection). For managed bootstrap and operating-model differences, see [Gateway certificates and managed data plane](/ai-gateway/cloud/gateway-certificates-and-managed-dp) and [Cloud vs. self-hosted](/ai-gateway/cloud/cloud-vs-self-hosted).
