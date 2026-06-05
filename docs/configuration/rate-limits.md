---
title: Rate Limits
description: Configure request, token, and concurrency limits for callers and models in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 36
---

Rate limits protect upstream providers and keep one caller, model, team, or member from consuming the whole gateway.

AISIX evaluates every request against each matching limit layer. The request must pass all layers before it reaches the upstream provider. If any layer has no headroom, the proxy returns `429`.

## Choose Where the Limit Belongs

Start with the narrowest place that matches your operating goal.

Use `ApiKey.rate_limit` when you want a caller-specific safety limit. This is the most direct way to protect the gateway from one application or tenant.

Use `Model.rate_limit` when you want to protect one model alias. This is useful when several caller keys share the same expensive or fragile upstream model.

Use `RateLimitPolicy` when the subject is wider than one key or one model. A policy can match an API key entry, a model entry, a team bucket, or a member bucket.

All matching layers are enforced together. A permissive API-key limit does not override a tighter model or policy limit.

## Configure Rate Limits

Configure inline limits on API keys or models, or use a scoped policy when the limit should apply across a wider bucket.

### Inline Limits

API keys and models share the same inline rate-limit configuration:

```json
{
  "rate_limit": {
    "rpm": 60,
    "tpm": 100000,
    "concurrency": 5
  }
}
```

Each field is optional. Missing fields do not limit that dimension, and an empty `rate_limit` object behaves like no limit.

Common fields are:

| Field | Limit |
| --- | --- |
| `rps`, `rpm`, `rph`, `rpd` | Request count per second, minute, hour, or day. |
| `tpm`, `tpd` | Token count per minute or day. |
| `concurrency` | In-flight requests. |

Token limits are configured with `tpm` and `tpd`.

### Create a Caller Limit

Add an inline limit to an API key when the quota should follow one caller key:

```json
{
  "key_hash": "YOUR_CALLER_KEY_HASH",
  "allowed_models": ["gpt-4o-prod"],
  "rate_limit": {
    "rpm": 60,
    "tpm": 100000,
    "concurrency": 5
  }
}
```

This limit is checked after the gateway authenticates the caller and before the request reaches the upstream provider.

### Create a Model Limit

Add an inline limit to a model when the quota should follow a model alias:

```json
{
  "display_name": "gpt-4o-prod",
  "provider": "openai",
  "model_name": "gpt-4o",
  "provider_key_id": "YOUR_PROVIDER_KEY_ID",
  "rate_limit": {
    "rpm": 300,
    "concurrency": 20
  }
}
```

Every caller that targets `gpt-4o-prod` shares the model limit.

### Add a Scoped Policy

`RateLimitPolicy` is a standalone rule loaded from gateway configuration. Use it for shared subjects such as teams and members:

```json
{
  "name": "team-acme-tpm",
  "scope": "team",
  "scope_ref": "team-uuid-acme",
  "window": "minute",
  "max_tokens": 1000000
}
```

Use a member-scoped policy when the limit should apply to one member identity:

```json
{
  "name": "member-burst",
  "scope": "member",
  "scope_ref": "member-uuid-1234",
  "window": "minute",
  "max_requests": 60
}
```

The supported policy scopes are:

| Scope | Match Target |
| --- | --- |
| `api_key` | Authenticated API-key entry ID. |
| `model` | Resolved model entry ID. |
| `team` | `ApiKey.team_id`. |
| `member` | `ApiKey.user_id`. |

For team and member policies to match, the authenticated API key must carry the corresponding `team_id` or `user_id`. Managed deployments can project those bindings from the control plane. The standalone `/admin/v1/apikeys` API does not set them, so self-hosted team/member policies require a control-plane or direct config-store path that writes those fields.

## Behavior Details

Review policy windows, config-store loading, and caller-visible response behavior before applying shared limits broadly.

### Policy Windows

Policy windows map into the same limiter fields used by inline limits:

| Window | Mapping |
| --- | --- |
| `second` | `max_requests` maps to `rps`. |
| `minute` | `max_requests` maps to `rpm`; `max_tokens` maps to `tpm`. |
| `hour` | `max_requests` maps to `rph`. |

The gateway does not convert a second window into a minute bucket, or an hour window into a day bucket. Those conversions would let a caller spend the declared window too quickly.

Token policy limits are enforced for `minute` windows. `max_tokens` on `second` or `hour` policies is accepted by the configuration but ignored by the quota mapper. AISIX logs a warning when it sees that configuration.

At least one of `max_requests` or `max_tokens` must be set.

### Configure Policy Rows

`RateLimitPolicy` rows are loaded from etcd under `<prefix>/rate_limit_policies/<id>`.

The standalone admin API does not expose CRUD routes for rate-limit policies. In managed mode, use the control-plane projection path. In self-hosted setups, write rows directly through your config-store workflow.

Malformed rows are skipped and reported through configuration load diagnostics; other valid rows continue to load.

For the exact policy schema, use [Resource schemas](../reference/resource-schemas.md).

### Response Behavior

When any layer rejects a request, the proxy returns `429`. If the limiter can calculate a retry window, the response includes `Retry-After`.

Successful non-streaming chat responses include `x-ratelimit-*` headers based on the limiter state after the provider request. Use those headers for debugging and client-side adaptive throttling.

## Troubleshooting

Use these checks when rate-limit decisions do not match the configured limits.

### A Caller Sees `429` Unexpectedly

Check the matching layers in this order: `ApiKey.rate_limit` on the authenticated API key, `Model.rate_limit` on the resolved model, and any `RateLimitPolicy` rows that match the API key, model, team, or member.

Any one of those layers can be the gating layer.

### A Team or Member Policy Does Not Take Effect

Check whether the API key row includes `team_id` or `user_id`. Team and member policies match the API-key row, not an external membership database.

If the key was created through the standalone admin API, those fields are not present. Use API-key inline limits, model inline limits, or a projection path that writes the bucket fields.

### Limits Seem to Apply Only to Chat

The shared quota gate is used across the supported LLM endpoint set. If another endpoint does not appear to trip the limit, first confirm it resolves a model and is sending enough traffic to exhaust the configured bucket.

## Related Reading

For inline key and model limits, see [API keys](api-keys.md) and [Models](models.md). For managed spending controls, see [Budgets](budgets.md). For caller-visible `429` responses and retry hints, see [Headers and error codes](../reference/headers-and-error-codes.md).
