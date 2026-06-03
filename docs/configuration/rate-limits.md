---
title: Rate Limits
description: Configure request, token, and concurrency limits for callers and models in AISIX AI Gateway.
sidebar_position: 36
---

Rate limits protect upstream providers and keep one caller, model, team, or
member from consuming the whole gateway.

AISIX evaluates every request against each matching limit layer. The request
must pass all layers before it is dispatched upstream. If any layer has no
headroom, the proxy returns `429`.

## Choose where the limit belongs

Start with the narrowest place that matches your operating goal.

Use `ApiKey.rate_limit` when you want a caller-specific safety limit. This is
the most direct way to protect the gateway from one application or tenant.

Use `Model.rate_limit` when you want to protect one model alias. This is useful
when several caller keys share the same expensive or fragile upstream model.

Use `RateLimitPolicy` when the subject is wider than one key or one model. A
policy can match an API key entry, a model entry, a team bucket, or a member
bucket.

All matching layers are enforced together. A permissive API-key limit does not
override a tighter model or policy limit.

## Inline limits

API keys and models share the same inline rate-limit shape:

```json
{
  "rate_limit": {
    "rpm": 60,
    "tpm": 100000,
    "concurrency": 5
  }
}
```

Each field is optional. Missing fields do not limit that dimension, and an empty
`rate_limit` object behaves like no limit.

Common fields are:

- `rpm` for requests per minute.
- `tpm` for tokens per minute.
- `concurrency` for in-flight request protection.
- `rps`, `rph`, and `rpd` when the request window should be one second, one
  hour, or one day.
- `tpd` when you need a daily token cap.

There is no `tps` or `tph` token field today.

## Create a caller limit

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

This limit is checked after the gateway authenticates the caller and before the
request reaches the upstream provider.

## Create a model limit

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

## Add a policy limit

`RateLimitPolicy` is a standalone rule loaded from the gateway snapshot. Use it
for shared subjects such as teams and members:

```json
{
  "name": "team-acme-tpm",
  "scope": "team",
  "scope_ref": "team-uuid-acme",
  "window": "minute",
  "max_tokens": 1000000
}
```

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

- `api_key`, matched against the authenticated API-key entry ID.
- `model`, matched against the resolved model entry ID.
- `team`, matched against `ApiKey.team_id`.
- `member`, matched against `ApiKey.user_id`.

For team and member policies to match, the authenticated API key must carry the
corresponding `team_id` or `user_id`. Managed deployments can project those
bindings from the control plane. The standalone `/admin/v1/apikeys` API does not
currently set them, so self-hosted team/member policies require a control-plane
or direct config-store path that writes those fields.

## Policy windows

Policy windows map into the same limiter fields used by inline limits:

- `second` maps `max_requests` to `rps`.
- `minute` maps `max_requests` to `rpm` and `max_tokens` to `tpm`.
- `hour` maps `max_requests` to `rph`.

The data plane intentionally does not convert a second window into a minute
bucket, or an hour window into a day bucket. Those conversions would let a caller
spend the declared window too quickly.

Token policy limits are enforced for `minute` windows today. `max_tokens` on
`second` or `hour` policies is accepted by the policy shape but ignored by the
quota mapper because per-second and per-hour token counters are not implemented.
The data plane logs a warning when it sees that shape.

At least one of `max_requests` or `max_tokens` must be set.

## Provision policy rows

`RateLimitPolicy` rows are loaded from etcd under
`<prefix>/rate_limit_policies/<id>`.

The standalone admin API does not currently expose CRUD routes for rate-limit
policies. In managed mode, use the control-plane projection path. In self-hosted
setups, write rows directly through your config-store workflow.

Malformed rows are skipped and surfaced through the rejection signal; other
valid rows continue to load.

For the exact policy schema, use the generated [Resource schemas](../reference/resource-schemas.md).

## Response behavior

When any layer rejects a request, the proxy returns `429`. If the limiter can
calculate a retry window, the response includes `Retry-After`.

Successful non-streaming chat responses include `x-ratelimit-*` headers based on
the limiter state after dispatch. Use those headers for debugging and client-side
adaptive throttling.

## Troubleshooting

### A caller sees `429` unexpectedly

Check the matching layers in this order:

1. The authenticated `ApiKey.rate_limit`.
2. The resolved `Model.rate_limit`.
3. Any `RateLimitPolicy` rows that match the API key, model, team, or member.

Any one of those layers can be the gating layer.

### A team or member policy does not take effect

Check whether the API key row includes `team_id` or `user_id`. Team and member
policies match the API-key row, not an external membership database.

If the key was created through the standalone admin API, those fields are not
present today. Use API-key inline limits, model inline limits, or a projection
path that writes the bucket fields.

### Limits seem to apply only to chat

The shared quota gate is used across the current LLM endpoint set. If another
endpoint does not appear to trip the limit, first confirm it resolves a model and
is sending enough traffic to exhaust the configured bucket.

## Next steps

- [API keys](api-keys.md)
- [Models](models.md)
- [Budgets](budgets.md)
- [Headers and error codes](../reference/headers-and-error-codes.md)
