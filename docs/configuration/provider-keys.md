---
title: Provider Keys
description: Configure upstream provider credentials and base URLs for AISIX AI Gateway models.
sidebar_position: 33
---

Provider keys store upstream credentials that one or more models can reuse.

Use a provider key when you want to:

- store one upstream API key once
- reuse it across multiple models
- rotate upstream credentials without recreating every model

## Current Fields

- `display_name`
- `secret`
- optional `api_base`

Example:

```bash title="Create a provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-prod",
    "secret": "YOUR_PROVIDER_API_KEY",
    "api_base": "https://api.openai.com/v1"
  }'
```

## `api_base` Behavior

`api_base` overrides the provider's default upstream base URL.

Current gateway behavior accepts both common OpenAI-style forms:

- `https://api.openai.com`
- `https://api.openai.com/v1`

The runtime normalizes these forms for the endpoints that build `/v1/...` upstream URLs.

## Reuse Model References

Models reference provider keys by `provider_key_id`, not by `display_name`.

Typical flow:

1. create one `ProviderKey`
2. create one or more `Model` rows that point at its returned `id`
3. rotate the provider key later with `PUT /admin/v1/provider_keys/:id`

## Operational Notes

- `secret` is stored as plaintext in the standalone gateway path.
- Duplicate `display_name` values are rejected with `409`.
- A model that points at a provider key not yet visible in the proxy snapshot can temporarily fail dispatch until propagation completes.

## Related Pages

- [Models](models.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Configuration Propagation](configuration-propagation.md)
