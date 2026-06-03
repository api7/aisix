---
title: Budgets
description: Understand where budget decisions come from and how AISIX AI Gateway enforces them.
sidebar_position: 37
---

Budgets protect an environment from unexpected AI spend. In AISIX AI Gateway,
the data plane enforces budget decisions, but it does not own the budget ledger.

In managed deployments, the gateway asks the AISIX Cloud control plane whether a
caller key may continue. In standalone self-hosted deployments, the budget
client is disabled by default and allows requests through.

## Budget enforcement at a glance

| Deployment mode | Budget authority | Data-plane behavior |
| --- | --- | --- |
| Managed data plane | AISIX Cloud control plane | Checks the control plane before upstream dispatch and enforces the returned decision. |
| Standalone self-hosted | No budget ledger by default | Allows requests unless a live budget-check client is explicitly wired. |

Use managed deployments when you need gateway-enforced spend controls. Treat
standalone budgets as a product boundary unless your deployment has its own live
budget-check path.

## Where decisions come from

Before the proxy dispatches a request upstream, a live managed data plane can
call the budget-check endpoint:

```text
GET {dpmgr_base}/dp/budget_check?api_key_id=<uuid>
```

The request uses the same managed mTLS bundle as data-plane heartbeat traffic.
The control plane evaluates the current spend state and returns a compact
decision:

- whether the request is allowed
- what fail mode the data plane should use if the control plane later becomes
  unreachable
- optional budget totals for metrics
- an optional reason when the request is denied

That keeps budget calculation on the managed control-plane path. The gateway
only needs the final decision.

## How the data plane enforces a decision

When the decision allows the request, the proxy continues with the normal
request path. Budget checks run before upstream dispatch and before the caller
receives any model output.

When the decision denies the request, the proxy returns a caller-visible
`429` response with an OpenAI-style error envelope:

```json
{
  "error": {
    "message": "team budget 'frontend' exceeded ($1.00/month). Resets soon.",
    "type": "billing_error",
    "code": "budget_exceeded"
  }
}
```

For OpenAI-compatible responses, the gateway can also include structured budget
fields that the managed control plane returned, such as `scope`, `scope_ref`,
`limit_usd`, `spent_usd`, `period`, `period_resets_at`, and
`retry_after_seconds`.

## Budget scopes

The gateway accepts the scope details returned by the managed budget-check
service. Common scopes include organization, environment, API key, provider key,
team, and member.

Those scopes are Cloud budget concepts, not standalone Admin API resources. If a
request is denied, inspect the returned `reason.scope` and `reason.scope_ref` to
see which budget caused the denial.

Team and member budgets depend on the API key identity projected to the data
plane. The runtime `ApiKey` row can carry `team_id` and `user_id`, and the proxy
uses those values for metrics and managed budget decisions. The standalone
`/admin/v1/apikeys` API does not currently set those fields, so team/member
budget matching is a managed projection path today.

## Managed versus standalone

Use managed deployments when you need live budget enforcement. A managed data
plane wires the budget client from the same control-plane configuration used for
heartbeat.

Standalone deployments use `BudgetClient::disabled()` unless a live client is
explicitly wired by the runtime. Disabled mode is allow-all. It is useful for
local development and self-hosted setups that do their own accounting, but it is
not a hard-stop budget engine.

Do not promise standalone budget blocking to application teams unless your
deployment has a live budget-check path configured.

## Control-plane outages

The budget client caches live decisions briefly so the proxy does not need a
round trip to the control plane for every repeated decision.

- Fresh cached decisions are reused for 5 seconds.
- Cached decisions can be treated as usable stale decisions for up to
  `AISIX_DP_BUDGET_STALE_MAX_SECONDS`; the default is `600`.
- If the stale ceiling expires, the gateway applies the last returned fail mode:
  `open`, `closed`, or `sticky`.
- If there is no cached decision and the control plane is unreachable, the live
  client denies by default on the sticky path.

This behavior is only for live managed budget clients. Disabled standalone mode
does not call the control plane and allows requests through.

## Metrics

When the managed budget response includes totals, the proxy records budget
gauges with the caller key identity. Labels include the API key ID and, when
available, the projected `team_id` and `user_id`.

If the decision does not include budget totals, the proxy clears the budget
gauges for that key identity.

## Troubleshooting

### A managed deployment returns `budget_exceeded`

Check the error code and structured budget fields first. The denial came from
the managed budget-check response, not from the standalone Admin API.

Then check the Cloud budget configuration for the returned scope. If the
returned scope looks wrong, investigate the control-plane budget calculation or
the API-key projection that reached the data plane.

### Traffic is denied after control-plane instability

Check whether the data plane had a fresh cached decision, whether the stale
ceiling elapsed, and which fail mode was last returned by the control plane.

If the data plane had no cached decision, an unreachable control plane denies on
the sticky default path.

### Standalone traffic is not blocked by budgets

That is expected for the default standalone runtime. The budget client is
disabled and allows requests through unless a live managed budget client is
wired.

## Next steps

- [API keys](api-keys.md) explains caller key identity and model access.
- [Configuration overview](overview.md) explains how managed and standalone
  configuration authority differs.
- [AISIX Cloud overview](../cloud/overview.md) explains managed control-plane
  operation.
- [Feature status](../overview/feature-matrix.md) summarizes managed versus
  standalone support.
