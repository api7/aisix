---
title: API Keys
description: Configure caller-facing API keys, model access, rate limits, and current budget boundaries in AISIX AI Gateway.
sidebar_position: 34
---

API keys authenticate callers on the proxy surface.

Clients send the plaintext key in `Authorization: Bearer <key>` or `x-api-key`.
The gateway stores only `key_hash`, the SHA-256 hex digest of that plaintext
key. On each request, the proxy hashes the presented key and looks up the
matching `ApiKey` resource.

## Prerequisites

This page assumes you have:

- a running self-hosted gateway with the admin and proxy listeners available
- an admin key for `Authorization: Bearer YOUR_ADMIN_KEY`
- at least one model alias the caller should be allowed to use

If you have not created a model yet, configure [Provider keys](provider-keys.md)
and [Models](models.md) first.

## Create a caller key

Choose the plaintext key you will give to the caller, then hash it before
writing the admin resource.

```shell
if command -v sha256sum >/dev/null 2>&1; then
  printf '%s' 'sk-demo-caller' | sha256sum | cut -d' ' -f1
else
  printf '%s' 'sk-demo-caller' | shasum -a 256 | awk '{print $1}'
fi
```

Create the API key resource with that hash:

```shell
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

Give the plaintext key to the caller. Do not give the caller `key_hash`.

## Verify access

First, check what the caller key can see:

```shell
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer sk-demo-caller"
```

Then send a request to an allowed model alias. The caller uses the plaintext
key, not the hash:

```shell
curl -sS http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

If the key was created successfully but the proxy still returns `401` or `403`,
wait briefly for configuration propagation and then use the troubleshooting
section below.

## Model access

`allowed_models` controls which model aliases the caller may use.

Use explicit allowlists for ordinary callers:

```json
{
  "allowed_models": ["gpt-4o-prod", "chat-prod"]
}
```

Use `["*"]` only when the key should access every model visible to it.

```json
{
  "allowed_models": ["*"]
}
```

An empty array is valid and denies every model.

```json
{
  "allowed_models": []
}
```

`GET /v1/models` applies the same access boundary. A wildcard key sees all
non-routing models. A restricted key sees only allowed non-routing models. An
empty allowlist returns an empty list.

## Rotate a key

`POST /admin/v1/apikeys/:id/rotate` generates a new plaintext bearer, stores
only its hash, and returns the plaintext once.

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys/API_KEY_ID/rotate \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Example response:

```json
{
  "entry": {
    "id": "API_KEY_ID",
    "revision": 2,
    "value": {
      "key_hash": "...",
      "allowed_models": ["gpt-4o-prod"]
    }
  },
  "plaintext": "sk-550e8400e29b41d4a716446655440000"
}
```

Capture `plaintext` immediately. Later reads return only the hash. The old key
stops working after the updated resource propagates to the proxy snapshot.

## Rate limits

`ApiKey.rate_limit` is an inline policy on the caller key.

It can limit:

- `rps`, `rpm`, `rph`, and `rpd` for request count
- `tpm` and `tpd` for token count
- `concurrency` for in-flight requests

The proxy combines all applicable rate-limit layers with AND semantics:

- `ApiKey.rate_limit`
- `Model.rate_limit`
- matching `RateLimitPolicy` rows

The tightest applicable layer wins in practice. See [Rate limits](rate-limits.md)
for the full enforcement model.

## Team and member bindings

The runtime `ApiKey` schema includes optional `team_id` and `user_id`.

Those fields are bucket identities, not access controls by themselves. The data
plane uses them to match `team`-scoped and `member`-scoped rate-limit policies
and managed budget rows.

The standalone admin API currently accepts and returns only `key_hash`,
`allowed_models`, and `rate_limit`. It does not set `team_id` or `user_id` on
`/admin/v1/apikeys` requests.

That means team and member bindings are currently a managed control-plane
projection concern, or a direct config-store concern for self-hosted operators
who intentionally write runtime rows outside the standalone admin API.

## Budget boundary

Managed budget enforcement runs on the managed `/dp/budget_check` path.

In standalone self-hosted deployments, the budget client defaults to disabled
and allows requests. The standalone admin API also does not set `team_id` or
`user_id`, so team and member budget scopes do not match keys created through
`/admin/v1/apikeys`.

For budget scope details, see [Budgets](budgets.md).

## Troubleshooting

### A valid key gets `403`

Check `allowed_models` first. `403` usually means the key authenticated but is
not allowed to use the requested model alias.

### A caller gets `401`

Check that the client is sending the plaintext key, not `key_hash`. Also check
that the updated API-key resource has propagated to the proxy snapshot.

### The caller lost access after rotation

Make sure the client is using the newly returned plaintext key. The old
plaintext no longer matches the stored hash after rotation propagates.

### Rate-limit behavior is not matching the layer you configured

Remember that key, model, and scoped policy layers are combined. If one layer
appears silent, another tighter layer may be the one rejecting requests.

## Next steps

- [Models](models.md) defines the aliases API keys can access.
- [Rate limits](rate-limits.md) explains inline and scoped request controls.
- [Budgets](budgets.md) explains managed budget boundaries.
- [OpenAI-compatible API](../integration/openai-compatible-api.md) shows proxy calls with caller keys.
