---
title: Budgets
description: Understand where budget decisions come from and how AISIX AI Gateway enforces them.
toc_max_heading_level: 2
sidebar_position: 37
---

Budgets protect an environment from unexpected AI spend. In AISIX AI Gateway, the data plane enforces budget decisions, but it does not own the budget ledger.

In managed deployments, the gateway asks the AISIX Cloud control plane whether a caller key may continue. In standalone self-hosted deployments, the budget client is disabled by default and allows requests through.

## Deployment Modes

| Deployment mode | Budget authority | Data-plane behavior |
| --- | --- | --- |
| Managed data plane | AISIX Cloud control plane | Checks the control plane before sending the provider request and enforces the returned decision. |
| Standalone self-hosted | No budget ledger by default | Allows requests unless the deployment provides a live budget-check client. |

Use managed deployments when you need gateway-enforced spend controls. Standalone deployments do not enforce budgets by default unless your deployment has its own live budget-check path.

## Configure Budget Enforcement

In a managed deployment, configure budget policy in AISIX Cloud and connect the data plane with the managed control-plane settings used for heartbeat traffic. Before the proxy sends a provider request, the data plane asks the control plane for a budget decision:

```text
GET {dpmgr_base}/dp/budget_check?api_key_id=<uuid>
```

The request uses the same managed mTLS bundle as data-plane heartbeat traffic. The control plane evaluates spend state and returns a compact decision with the allowed status, fail mode, optional budget totals, and an optional denial reason.

### Budget Scopes

The gateway accepts the scope details returned by the managed budget-check service. Common scopes include organization, environment, API key, provider key, team, and member.

Those scopes are Cloud budget concepts, not standalone Admin API resources. If a request is denied, inspect the returned `reason.scope` and `reason.scope_ref` to see which budget caused the denial.

Team and member budgets depend on the API key identity projected to the data plane. The runtime `ApiKey` row can carry `team_id` and `user_id`, and the proxy uses those values for metrics and managed budget decisions. The standalone `/admin/v1/apikeys` API does not set those fields, so team/member budget matching uses the managed projection path.

### Standalone Deployments

Standalone deployments use the disabled budget client unless the deployment provides a live budget-check path. Disabled mode is allow-all. It is useful for local development and self-hosted setups that do their own accounting, but it is not a hard-stop budget engine.

Do not rely on standalone deployments for budget blocking unless your deployment has a live budget-check path configured.

## Behavior Details

When a managed decision allows the request, the proxy continues with the normal request path. Budget checks run before the provider request and before the caller receives any model output.

When the decision denies the request, the proxy returns a caller-visible `429` response with an OpenAI-style error envelope:

```json
{
  "error": {
    "message": "team budget 'frontend' exceeded ($1.00/month). Resets soon.",
    "type": "billing_error",
    "code": "budget_exceeded"
  }
}
```

For OpenAI-compatible responses, the gateway can also include structured budget fields that the managed control plane returned, such as `scope`, `scope_ref`, `limit_usd`, `spent_usd`, `period`, `period_resets_at`, and `retry_after_seconds`.

### Control-Plane Outages

The budget client caches live decisions briefly so the proxy does not need a round trip to the control plane for every repeated decision.

Fresh cached decisions are reused for 5 seconds. Usable stale decisions are reused for up to `AISIX_DP_BUDGET_STALE_MAX_SECONDS`; the default is `600`. When the stale ceiling expires, the client applies the last returned fail mode: `open`, `closed`, or `sticky`. If there is no cached decision and the control plane is unreachable, the sticky default path denies the request.

This behavior is only for live managed budget clients. Disabled standalone mode does not call the control plane and allows requests through.

### Metrics

When the managed budget response includes totals, the proxy records budget gauges with the caller key identity. Labels include the API key ID and, when available, the projected `team_id` and `user_id`.

If the decision does not include budget totals, the proxy clears the budget gauges for that key identity.

## Troubleshooting

Use these checks when budget decisions or budget metrics do not match the deployment state.

### A Managed Deployment Returns Budget Exceeded

Check the error code and structured budget fields first. The denial came from the managed budget-check response, not from the standalone Admin API.

Then check the Cloud budget configuration for the returned scope. If the returned scope looks wrong, investigate the control-plane budget calculation or the API-key projection that reached the data plane.

### Traffic Is Denied After Control-Plane Instability

Check whether the data plane had a fresh cached decision, whether the stale ceiling elapsed, and which fail mode was last returned by the control plane.

If the data plane had no cached decision, an unreachable control plane denies on the sticky default path.

### Standalone Traffic Is Not Blocked by Budgets

In the default standalone runtime, the budget client is disabled and allows requests through unless a live managed budget client is configured.

## Related Reading

The [API keys](api-keys.md) page covers caller key identity and model access. For the managed and standalone configuration split, see [Configuration overview](overview.md) and [AISIX Cloud overview](../cloud/overview.md). Check [Feature availability](../overview/feature-matrix.md) before planning budget behavior for production.
