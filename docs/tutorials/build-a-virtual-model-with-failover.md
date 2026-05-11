---
title: Build A Virtual Model With Failover
description: Create a routing model that fails over from one upstream-backed model to another.
sidebar_position: 81
---

This tutorial shows how to expose one stable alias while keeping a fallback target behind it.

## Goal

Create a routing model with `strategy: "failover"` and two direct target models.

## Flow

1. create the upstream provider key or keys
2. create the primary direct model
3. create the secondary direct model
4. create the routing model with `targets` and `retry_budget`
5. allow the routing alias on the caller API key

## Expected Behavior

The proxy starts at the first target and moves to the next target only on retryable failures.

## Related Pages

- [Models](../configuration/models.md)
- [Routing And Failover](../configuration/routing-and-failover.md)
- [Troubleshooting](../operations/troubleshooting.md)
