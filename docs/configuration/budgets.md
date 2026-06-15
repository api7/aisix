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

### Worked examples — coverage and calculation

The amounts and periods below are illustrative.

**1. Every applicable cap must have headroom — the tightest one wins.**
A key `K` runs in environment `prod` and was created with `team_id = frontend` and `user_id = alice`. Five hard-stop budgets are configured:

| Scope | Cap | Spent this period | Remaining |
|---|---|---|---|
| `org` | $1,000 / month | $610 | $390 |
| `environment` `prod` | $400 / month | $300 | $100 |
| `api_key` `K` | $50 / day | $12 | $38 |
| `team` `frontend` | $200 / month | $185 | $15 |
| `member` `alice` | $30 / month | $30 | **$0** |

A request on `K` is checked against **all five** configured budgets — they all apply (this scenario has no `provider_key` budget; that's the sixth scope). `alice`'s member cap is exactly spent, so the request is **denied with `429`** even though the org, env, api_key, and team budgets still have room. `reason.scope` is `member`. A caller always feels the cap with the least remaining room.

**2. `team` and `member` totals span every environment; `environment` and `api_key` do not.**
`team` / `member` budgets are org-scoped — they sum spend across *every* environment their bound keys run in. Team `frontend` owns `K1` in `prod` and `K2` in `staging`:

- `K1` (prod) spent $120 this month; `K2` (staging) spent $90.
- The `team` `frontend` budget ($200 / month) sees **$210** — both envs combined — so it is over, and requests on **both** `K1` and `K2` are denied.
- A `prod` `environment` budget, by contrast, counts only `K1`'s $120; an `api_key` budget on `K1` counts only `K1`.

**3. Which budgets apply is decided by the key's binding — there is no implicit membership.**

| Key | `team_id` | `user_id` | Budgets that apply |
|---|---|---|---|
| `K-plain` | — | — | `org`, `environment`, `api_key`, `provider_key` |
| `K-team` | `frontend` | — | …plus `team` `frontend` |
| `K-member` | — | `alice` | …plus `member` `alice` |
| `K-both` | `frontend` | `alice` | …plus `team` `frontend` **and** `member` `alice` |

A budget you create for team `frontend` does **not** apply to a key that wasn't given `team_id = frontend`, even if that key's owner belongs to the team — only the explicit binding on the key counts. See [API Keys](api-keys.md#budget-boundary) for setting it.

**4. Warn-only watches; hard-stop blocks.**
That same `team` `frontend` budget with `hard_stop = false` never returns `429` — traffic keeps flowing past $200 and the over-budget state surfaces on the dashboard, so you are alerted without an outage. With `hard_stop = true` the identical cap denies at $200.

**5. Spend follows the key's current binding.**
Re-point `K` from team `frontend` to team `platform` (change its `team_id`). From then on `K`'s spend counts toward `platform`'s budget, not `frontend`'s — totals are computed from the key's *current* binding, so its existing spend moves with it. The same applies to `user_id` and `member` budgets.

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
