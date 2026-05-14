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

Think of a provider key as the upstream credential container, not the client-facing contract.

## Current Fields

- `display_name`
- `secret`
- optional `api_base`

In practice:

- `display_name` is for operator readability
- `secret` is the actual upstream credential used at dispatch time
- `api_base` is how you override the provider's default endpoint root

Example:

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` configured in [`config.yaml`](bootstrap-config.md). Anyone with read access to the etcd keyspace can read the credential. For production, front etcd with encryption-at-rest, restrict etcd network access to the gateway, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md), where the secret stays in the control plane and only the projected `provider_key_id` reference reaches the data plane.
:::

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

`api_base` overrides the provider's default upstream base URL. The gateway uses the value almost as-is — it only strips a trailing `/`. There is **no `/v1` normalization**. Each provider bridge appends a different suffix at request time, so the exact form `api_base` must take depends on which `provider` your model selects.

Each provider has its own convention — the four current bridges do **not** share one. Use the table below; do not generalize from one row to another.

| `provider` | What `api_base` should be | Bridge appends | Default if `api_base` is omitted |
|---|---|---|---|
| `openai` | include `/v1` | `/chat/completions`, `/embeddings`, `/completions`, `/images/generations`, `/audio/*` | `https://api.openai.com/v1` |
| `deepseek` | bare host (DeepSeek serves OpenAI-compatible paths at the host root) | `/chat/completions` | `https://api.deepseek.com` |
| `gemini` | host plus the OpenAI-compat prefix `/v1beta/openai` | `/chat/completions` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `anthropic` | bare host | `/v1/messages` | `https://api.anthropic.com` |

The OpenAI and Anthropic conventions match each upstream's official SDK — `openai-python` initialises `base_url = "https://api.openai.com/v1"`, while `anthropic-sdk-python` initialises `base_url = "https://api.anthropic.com"` and appends `/v1/messages` itself. DeepSeek is OpenAI-compatible but exposes `/chat/completions` directly at the host root, and Gemini's OpenAI-compatible surface lives under a fixed `/v1beta/openai` prefix that the bridge does not synthesize.

Wrong forms fail at request time with an upstream `404`, not at admin-write time. For example, `api_base: "https://api.openai.com"` for an `openai` provider produces an upstream request to `https://api.openai.com/chat/completions` — the upstream returns `404` because OpenAI's API lives under `/v1`. There is no admin-side validation today that catches this. Tracking issue: [api7/ai-gateway#270](https://github.com/api7/ai-gateway/issues/270).

## Reuse Model References

Models reference provider keys by `provider_key_id`, not by `display_name`.

Typical flow:

1. create one `ProviderKey`
2. create one or more `Model` rows that point at its returned `id`
3. rotate the provider key later with `PUT /admin/v1/provider_keys/:id`

This is the main reason to avoid embedding provider credentials conceptually into each model row.

## Operational Notes

- `secret` is stored as plaintext in the standalone gateway path.
- Duplicate `display_name` values are rejected with `409`.
- A model that points at a provider key not yet visible in the proxy snapshot can temporarily fail dispatch until propagation completes.

## Troubleshooting

### Requests fail after changing `api_base`

Treat that first as an upstream endpoint construction issue, not as a caller-key or model-auth issue.

### Several models fail at once after provider-key rotation

That is expected if they all share the same provider key. The shared key is the common dependency.

## Related Pages

- [Models](models.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Configuration Propagation](configuration-propagation.md)
