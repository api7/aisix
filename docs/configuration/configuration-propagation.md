---
title: Configuration Propagation
description: Understand how admin writes propagate through etcd and the in-memory gateway snapshot in AISIX AI Gateway.
sidebar_position: 41
---

AISIX AI Gateway does not apply admin writes directly on the proxy hot path.

Instead, the current runtime model is:

1. write a dynamic resource through the admin API
2. persist it to the config store and etcd
3. let the watch supervisor rebuild and publish a fresh in-memory snapshot
4. serve new proxy requests from that updated snapshot

## What To Expect

Propagation is asynchronous.

The code and test harness document a target propagation budget around one watch tick, with admin health comments and test helpers treating `<=500ms` as the normal expectation. In real CI and shared environments, end-to-end readiness can take longer, so positive polling is safer than a fixed sleep.

## Practical Guidance

After writing dependent resources such as:

- provider key
- model
- API key

do one of the following before sending a production-like proxy request:

- poll `GET /v1/models` until the model appears
- poll the target endpoint until a known propagation error disappears
- use a short delay only for simple local demos

## Health Visibility

`GET /admin/v1/health` exposes watch freshness information through the optional `config` block when the watch supervisor is wired.

That block includes:

- `snapshot_revision`
- `snapshot_age_seconds`

This helps detect a stale or wedged config stream.

## Related Pages

- [Admin API](admin-api.md)
- [Health Checks](../operations/health-checks.md)
- [Testing And Verification](../operations/testing-and-verification.md)
