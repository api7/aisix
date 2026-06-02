---
title: First Model, First Key, First Request
description: Create a provider key, model, and API key through the AISIX AI Gateway admin API, then send your first successful proxy request.
sidebar_position: 11
---

This guide shows how to move from a running self-hosted [gateway](../overview/glossary.md#gateway) to a working end-to-end request. You will create:

- one `ProviderKey`
- one `Model`
- one caller-facing `ApiKey`

Then you will verify that the new configuration is visible on the proxy surface.

:::note Standalone only
This guide uses the standalone `admin/v1` API on `127.0.0.1:3001`. A [Cloud managed data plane](aisix-cloud-managed-dp.md) only exposes proxy APIs locally and does **not** bind the standalone admin listener — create provider keys, models, and caller API keys through the AISIX Cloud control plane instead.
:::

## Prerequisites

- A running gateway from the [Self-Hosted Quickstart](self-hosted.md)
- An API key from an upstream provider (OpenAI, Anthropic, Google, DeepSeek, or another supported provider). The key looks like `sk-...` and is what you will paste into the `YOUR_PROVIDER_API_KEY` placeholder in Step 1. If you do not have one, sign up at the provider (for example, <https://platform.openai.com/api-keys>).
- Your admin key from the bootstrap config

## What This Quickstart Configures

The standalone gateway uses:

- **[provider keys](../overview/glossary.md#provider-key)** to store upstream credentials and optional base URLs
- **models** to expose operator-defined model aliases on the proxy surface
- **[API keys](../overview/glossary.md#api-key)** to control which callers can access which models

## Step 1: Create a Provider Key

Create a provider key that points at your upstream provider.

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the [etcd](../overview/glossary.md#etcd) `prefix` you configured in [`config.yaml`](self-hosted.md#step-2-create-a-bootstrap-config). For production, front etcd with encryption-at-rest, or use [AISIX Cloud](../overview/glossary.md#aisix-cloud)'s managed [Provider Key Rotation](../cloud/provider-key-rotation.md), which holds the secret in the [control plane](../overview/glossary.md#control-plane) and projects only what each environment needs.
:::

```bash title="Create a provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-upstream",
    "secret": "YOUR_PROVIDER_API_KEY",
    "api_base": "https://api.openai.com/v1"
  }'
```

:::caution `api_base` convention differs per provider
Each provider has its own canonical form — do not generalize from this OpenAI example. `openai` expects `api_base` to include `/v1`; `deepseek` wants the bare host (`https://api.deepseek.com`); `google` wants the OpenAI-compat prefix (`https://generativelanguage.googleapis.com/v1beta/openai`); `anthropic` wants the bare host (the bridge appends `/v1/messages` itself). The gateway tolerates common paste-mistakes such as trailing slashes, full endpoint URLs, and (for the canonical OpenAI/DeepSeek hosts) the missing or extra `/v1` segment. See [Provider Keys § `api_base` Behavior](../configuration/provider-keys.md#api_base-behavior) for the full truth table and the tolerated forms.
:::

The admin envelope returns a `ResourceEntry` shape:

```json
{
  "id": "...",
  "revision": 1,
  "value": {
    "display_name": "openai-upstream",
    "secret": "YOUR_PROVIDER_API_KEY",
    "api_base": "https://api.openai.com/v1"
  }
}
```

Capture the returned `id` for use as `provider_key_id` in the next step — copy it by eye from the response above, or use this jq-capturing form **instead of** the curl above (running both creates a duplicate `display_name` and the second POST returns 409):

```bash title="Create and capture the id in one shot"
PROVIDER_KEY_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"display_name":"openai-upstream","secret":"YOUR_PROVIDER_API_KEY","api_base":"https://api.openai.com/v1"}' \
  | jq -r .id)
```

The same pattern applies to the model `id` and API-key `id` captures in the next two steps.

:::warning
The `secret` field is returned as plaintext in this response. Treat the command output as sensitive — avoid pasting it into shared documents, issue trackers, or chat. The admin API also returns the plaintext on subsequent `GET /admin/v1/provider_keys/:id` calls, so the same handling applies any time you read this resource.
:::

## Step 2: Create a Model

Create a model alias that the proxy will expose to callers.

:::tip
This guide uses `gpt-4o-mini` to minimize evaluation costs. Swap in any chat-completions-compatible model supported by your upstream provider.
:::

```bash title="Create a model"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID"
  }'
```

The `display_name` is the model name your clients will send in proxy requests.

## Step 3: Create a Caller API Key

The data plane stores `key_hash`, not plaintext API keys. Hash your chosen plaintext key first, then create the API key resource.

```bash title="Hash a plaintext caller key"
printf 'sk-demo-caller' | sha256sum | cut -d' ' -f1
```

Use the resulting hash in the admin API request:

```bash title="Create a caller API key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["gpt-4o-prod"]
  }'
```

## Step 4: Wait For Configuration Propagation

Admin writes do not become visible to the proxy instantly. The gateway publishes dynamic resources through the watch-driven snapshot path. On a healthy local etcd this is typically under 500 ms; on a slow CI runner or a cold etcd it can be several seconds.

A fixed sleep is fine for local evaluation:

```bash title="Wait briefly for propagation"
sleep 1
```

For automation or slow environments, poll the proxy until the model becomes visible — this is what the e2e harness does with its `waitConfigPropagation` helper:

```bash title="Poll until the model is visible"
until curl -sf http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer sk-demo-caller" \
  | grep -q '"gpt-4o-prod"'; do
  sleep 1
done
```

If a subsequent proxy call returns `404 model_not_found`, propagation is still in flight — wait longer or switch to the polling form.

## Step 5: Verify `/v1/models`

Call the proxy with the plaintext caller key you chose before hashing it.

```bash title="List visible models"
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer sk-demo-caller"
```

Expected result:

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o-prod",
      "object": "model",
      "created": 1715000000,
      "owned_by": "openai"
    }
  ]
}
```

`created` is the gateway-side unix timestamp at response time, not when the model resource was provisioned. The OpenAI SDK accepts it as-is.

## Step 6: Send The First Chat Request

```bash title="Send a chat completion request"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "Say hello from AISIX."}
    ]
  }'
```

If the upstream provider is reachable and the model is configured correctly, the response follows the OpenAI chat-completions shape.

## Step 7: Verify The Auth And Allowlist Contract

Two negative-path checks prove the proxy is doing the work the configuration claims it is doing.

### Missing bearer returns `401`

```bash title="Verify missing-bearer rejection"
curl -sS -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-prod","messages":[{"role":"user","content":"hi"}]}'
```

Expected: `401`. The response body uses the proxy error envelope:

```json
{
  "error": {
    "message": "missing or malformed Authorization header",
    "type": "invalid_api_key"
  }
}
```

### Unauthorized model returns `403`

Ask for a model alias the caller key is **not** in `allowed_models` for:

```bash title="Verify unauthorized-model rejection"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"some-model-not-in-allowed-models","messages":[{"role":"user","content":"hi"}]}'
```

Expected: `403` with `"type": "permission_denied"`. Or `404` with `"type": "model_not_found"` if the alias does not exist in the snapshot at all.

These two paths exercise the same authentication and authorization code paths that gate every production request — passing them proves the gateway is correctly enforcing the caller key and the `allowed_models` list, not just returning `200` because the upstream happens to be reachable.

## Verification Notes

- `401` (`invalid_api_key`) — the caller key is missing, malformed, or unknown to the snapshot.
- `403` (`permission_denied`) — the key exists, but the resolved model is not in its `allowed_models`.
- `404` (`model_not_found`) — the model alias does not resolve in the current snapshot.
- `503` (`provider_unavailable`) — no provider bridge is registered for the resolved provider.
- Admin errors (`/admin/v1/*`) use a different envelope: `{"error_msg": "..."}`. See [Admin API](../configuration/admin-api.md).

## Cleanup

Delete the resources you created so they don't leak into other work. Delete in reverse dependency order — caller key first (so the model can no longer be reached), then the model, then the provider key.

```bash title="Delete the caller API key"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/apikeys/YOUR_APIKEY_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

```bash title="Delete the model"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/YOUR_MODEL_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

```bash title="Delete the provider key"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/provider_keys/YOUR_PROVIDER_KEY_ID \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Use the `id` values captured from each `POST` response. To remove the gateway itself, see the [self-hosted cleanup](self-hosted.md#cleanup).

## Related Pages

- [Self-Hosted Quickstart](self-hosted.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Bootstrap Configuration](../configuration/bootstrap-config.md)
- [Admin API](../configuration/admin-api.md)
