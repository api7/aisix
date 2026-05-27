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

That design keeps the budget decision on the managed control-plane path rather than making the standalone data plane the source of truth for budget enforcement.

## Budget Scopes

The control plane evaluates **up to six budget rows simultaneously** for a single request, and returns the most-restrictive deny. The gateway sees a single `{allow, fail_mode, reason}` reply per `/dp/budget_check` call; the `reason.scope` field tells you which budget was the proximate cause when a request was denied.

| `reason.scope` | What this row caps | Applies when |
|---|---|---|
| `org` | Every request in the whole org | always |
| `environment` | Every request in this env | always |
| `api_key` | Every request for this api_key | always |
| `provider_key` | Every request whose model dispatches to that upstream credential | always |
| `team` | Every request whose `api_key.team_id` equals this team | only if the api_key was created with a `team_id` |
| `member` | Every request whose `api_key.user_id` equals this member | only if the api_key was created with a `user_id` |

`org`, `environment`, `api_key`, and `provider_key` are env- or org-scoped; `team` and `member` are **org-scoped only** and aggregate spend across every environment the bound api_keys live in.

All six rows are peers — no row "wraps" or "overrides" another. Any applicable row with `hard_stop=true` and `spent_cents >= limit_cents` rejects the request with `429`. Warn-only rows never block but surface alerts on the dashboard.

For the full applicability matrix, re-binding semantics, orphan handling, and worked scenarios, see [PRD-09b §4 in the AISIX-Cloud repo](https://github.com/api7/AISIX-Cloud/blob/main/docs/prd/prd-09b-budget.md#4-scope-model--applicability).

## Managed Versus Standalone

Current boundary:

- managed deployments can attach a live budget client through the managed data-plane path
- standalone self-hosted deployments default to `BudgetClient::disabled()`, which allows requests through

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
- `fail_mode` (`sticky` / `open` / `closed`) is a single org-level setting in AISIX Cloud — the same outage policy applies to all six scopes. Operators change it in the dashboard's **Settings → Budget** card. There is no per-scope or per-budget outage policy.

## Troubleshooting

### A managed deployment denies traffic after control-plane instability

Inspect budget-check freshness and the cached-decision behavior first.

## Related Pages

- [API Keys](api-keys.md)
- [AISIX Cloud Overview](../cloud/overview.md)
- [Roadmap](../roadmap.md)
