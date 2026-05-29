---
title: API Keys
description: Configure caller-facing API keys, model access, rate limits, and current budget boundaries in AISIX AI Gateway.
sidebar_position: 34
---

API keys are the caller-facing credentials used on the proxy surface.

The gateway does not store plaintext caller keys in the `ApiKey` resource. It stores `key_hash`, which is the SHA-256 hex digest of the plaintext bearer token.

This resource controls who can call the proxy and which model aliases they can use.

## Current Fields

- `key_hash`
- `allowed_models`
- optional `rate_limit`
- optional `team_id`
- optional `user_id`

Think of those fields as four distinct control layers:

- identity: `key_hash`
- authorization: `allowed_models`
- inline policy: `rate_limit`
- bucket identity: `team_id` and `user_id`

`team_id` and `user_id` are not access controls in themselves. They are the bucket keys that scoped policies on the control plane match against. Set them when you want a policy to span all keys belonging to the same team or to the same member.

Two policy families match against these buckets today:

- `team`-scoped and `member`-scoped [`RateLimitPolicy`](rate-limits.md#rate-limit-policy-entities) rows.
- `team`-scoped and `member`-scoped [budget](budgets.md#budget-scopes) rows.

Setting `team_id` opts the key into both the team rate-limit pool and the team budget (if either exists). Setting `user_id` does the same on the member side. Leaving them null means the key is unbound — the team and member rows of either policy family simply don't apply to it.

## Create A Caller Key

Hash the plaintext key first:

```bash title="Hash a caller API key"
printf 'sk-demo-caller' | sha256sum | cut -d' ' -f1
```

Then create the admin resource:

```bash title="Create an API key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["gpt-4o-prod"],
    "rate_limit": {
      "rpm": 60,
      "concurrency": 5
    }
  }'
```

The plaintext bearer token is the value your clients will send in `Authorization: Bearer ...`.

## Model Authorization

`allowed_models` controls which model aliases the caller may use.

Current behavior:

- `["*"]` allows access to every model alias visible to that key
- an explicit list allows only those model aliases
- an empty array is valid but denies every model

Choose explicit allowlists unless you intentionally want a wildcard operator or internal key.

## Rotation

`POST /admin/v1/apikeys/:id/rotate` generates a new plaintext bearer, stores only its hash, and returns the plaintext exactly once.

That means rotation is both a security action and a distribution event. You need a plan for getting the new plaintext into the caller before the old one is retired.

Example response shape:

```json
{
  "entry": {
    "id": "...",
    "revision": 2,
    "value": {
      "key_hash": "...",
      "allowed_models": ["gpt-4o-prod"]
    }
  },
  "plaintext": "sk-550e8400e29b41d4a716446655440000"
}
```

## Rate Limits

The inline rate-limit object on `ApiKey` supports:

- `tpm`
- `tpd`
- `rpm`
- `rpd`
- `concurrency`

`ApiKey.rate_limit` is one of three layers the proxy enforces. The other two are `Model.rate_limit` (inline on the resolved model) and standalone `RateLimitPolicy` rows. All applicable layers are AND-combined per request.

See [Rate Limits](rate-limits.md) for the full enforcement model and for `team`/`member`-scope policies that match against `team_id` / `user_id`.

## Budget Boundary

Managed budget enforcement exists on the managed `/dp/budget_check` path.

Up to six budget rows can apply to a single request — see [Budget Scopes](budgets.md#budget-scopes) for the full list and applicability rules. The `team_id` and `user_id` fields on this api_key control which of the `team` and `member` scope rows apply:

- `team_id = null` and `user_id = null` → only `org`, `environment`, `api_key`, `provider_key` scope rows apply
- `team_id = T` → `team` scope rows whose `scope_ref = T` also apply
- `user_id = M` → `member` scope rows whose `scope_ref = M` also apply

Adding a user to a team does **not** by itself make that user's keys count against the team budget — only the explicit `team_id` on the api_key does. If you want a key to contribute to a team's budget, set `team_id` at creation time.

Current standalone boundary:

- standalone self-hosted deployments default to a disabled budget client, which is allow-all

## Troubleshooting

### A valid key gets `403`

Check `allowed_models` first. That is an authorization failure, not an authentication failure.

### The caller lost access after rotation

Make sure the client is using the newly returned plaintext key, not the old one.

### Rate-limit behavior is not matching the layer you configured

The proxy combines `ApiKey.rate_limit`, `Model.rate_limit`, and matching `RateLimitPolicy` rows on every request. If a layer looks silent, check whether another layer is the actual gating one — a tighter `ApiKey.rate_limit` will trip before a looser `Model.rate_limit`. See [Rate Limits § Troubleshooting](rate-limits.md#troubleshooting).

## Related Pages

- [Models](models.md)
- [Rate Limits](rate-limits.md)
- [Budgets](budgets.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
