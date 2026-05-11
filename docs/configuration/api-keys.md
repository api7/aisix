---
title: API Keys
description: Configure caller-facing API keys, model access, rate limits, and current budget boundaries in AISIX AI Gateway.
sidebar_position: 34
---

API keys are the caller-facing credentials used on the proxy surface.

The gateway does not store plaintext caller keys in the `ApiKey` resource. It stores `key_hash`, which is the SHA-256 hex digest of the plaintext bearer token.

## Current Fields

- `key_hash`
- `allowed_models`
- optional `rate_limit`
- optional `max_budget_usd`

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

## Model Authorization

`allowed_models` controls which model aliases the caller may use.

Current behavior:

- `["*"]` allows access to every model alias visible to that key
- an explicit list allows only those model aliases
- an empty array is valid but denies every model

## Rotation

`POST /admin/v1/apikeys/:id/rotate` generates a new plaintext bearer, stores only its hash, and returns the plaintext exactly once.

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
  "plaintext": "sk-abcd1234ef567890"
}
```

## Rate Limits

The current rate-limit object supports:

- `tpm`
- `tpd`
- `rpm`
- `rpd`
- `concurrency`

Current enforcement uses the API key's `rate_limit` object. Model-level `rate_limit` exists in the schema, but current hot-path enforcement is keyed off the authenticated API key.

## Budget Boundary

`max_budget_usd` exists in the `ApiKey` schema and admin OpenAPI surface.

Current runtime boundary:

- managed deployments can run budget checks through the managed `/dp/budget_check` path
- standalone self-hosted deployments default to a disabled budget client, which is allow-all

Do not treat `max_budget_usd` as a fully documented standalone self-hosted budget control yet.

## Related Pages

- [Models](models.md)
- [Rate Limits](rate-limits.md)
- [Budgets](budgets.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
