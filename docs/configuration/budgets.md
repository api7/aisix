---
title: Budgets
description: Understand the current budget-enforcement boundary in AISIX AI Gateway and AISIX Cloud managed paths.
sidebar_position: 37
---

Budget enforcement in the current gateway runtime is driven by the managed budget-check path, not by a standalone in-process budget engine.

Use this page to avoid over-assuming what `max_budget_usd` means in standalone deployments.

## Current Runtime Model

Before dispatch, the proxy can call:

- `GET {dpmgr_base}/dp/budget_check?api_key_id=<uuid>`

This path is authenticated with the same managed mTLS bundle used by heartbeat.

The budget client caches decisions briefly and can fall back according to the last known fail mode if the control plane becomes unreachable.

That design keeps the budget decision on the managed control-plane path rather than making the standalone data plane the source of truth for budget enforcement.

## Managed Versus Standalone

Current boundary:

- managed deployments can attach a live budget client through the managed data-plane path
- standalone self-hosted deployments default to `BudgetClient::disabled()`, which allows requests through

Because of that boundary, `max_budget_usd` should not be treated as part of the current verified standalone admin write contract.

Current standalone caveat:

- the typed `ApiKey` model and admin OpenAPI mention `max_budget_usd`
- the active standalone admin JSON Schema validator rejects `max_budget_usd` on write today

## Operator Guidance

- treat managed mode as the real budget-enforcement path today
- do not promise standalone hard-stop budgets to internal or external users unless your deployment has explicitly wired a managed budget client path

## Proxy Outcomes

When the budget decision denies a request, the proxy returns:

- `429`
- OpenAI-style error envelope
- error code `budget_exceeded`

This is a caller-visible denial, not just an internal accounting event.

## Operational Notes

- live budget decisions are cached for 5 seconds
- stale cached decisions can be honored up to `AISIX_DP_BUDGET_STALE_MAX_SECONDS` with a default of `600`
- without any cached decision, an unreachable control plane causes a deny on the sticky default path

## Troubleshooting

### A managed deployment denies traffic after control-plane instability

Inspect budget-check freshness and the cached-decision behavior first.

### A standalone deployment ignores `max_budget_usd`

That is expected with the current standalone runtime boundary.

If you are using the standalone admin API directly, do not send `max_budget_usd` until the write validator and runtime contract are aligned.

## Related Pages

- [API Keys](api-keys.md)
- [AISIX Cloud Overview](../cloud/overview.md)
- [Roadmap](../roadmap.md)
