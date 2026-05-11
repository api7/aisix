---
title: Budgets
description: Understand the current budget-enforcement boundary in AISIX AI Gateway and AISIX Cloud managed paths.
sidebar_position: 37
---

Budget enforcement in the current gateway runtime is driven by the managed budget-check path, not by a standalone in-process budget engine.

## Current Runtime Model

Before dispatch, the proxy can call:

- `GET {dpmgr_base}/dp/budget_check?api_key_id=<uuid>`

This path is authenticated with the same managed mTLS bundle used by heartbeat.

The budget client caches decisions briefly and can fall back according to the last known fail mode if the control plane becomes unreachable.

## Managed Versus Standalone

Current boundary:

- managed deployments can attach a live budget client through the managed data-plane path
- standalone self-hosted deployments default to `BudgetClient::disabled()`, which allows requests through

Because of that boundary, `ApiKey.max_budget_usd` should be treated as a schema field and forward-facing config surface, not as a fully documented standalone self-hosted feature.

## Proxy Outcomes

When the budget decision denies a request, the proxy returns:

- `429`
- OpenAI-style error envelope
- error code `budget_exceeded`

## Operational Notes

- live budget decisions are cached for 5 seconds
- stale cached decisions can be honored up to `AISIX_DP_BUDGET_STALE_MAX_SECONDS` with a default of `600`
- without any cached decision, an unreachable control plane causes a deny on the sticky default path

## Related Pages

- [API Keys](api-keys.md)
- [AISIX Cloud Overview](../cloud/overview.md)
- [Roadmap](../roadmap.md)
