---
title: First Model, First Key, First Request
description: Create a provider key, model, and API key through the AISIX AI Gateway admin API, then send your first successful proxy request.
sidebar_position: 11
---

This guide shows how to move from a running self-hosted gateway to a working end-to-end request. You will create:

- one `ProviderKey`
- one `Model`
- one caller-facing `ApiKey`

Then you will verify that the new configuration is visible on the proxy surface.

## Prerequisites

- A running gateway from the [Self-Hosted Quickstart](self-hosted.md)
- A reachable upstream OpenAI-compatible endpoint
- Your admin key from the bootstrap config

## What This Quickstart Configures

The standalone gateway uses:

- **provider keys** to store upstream credentials and optional base URLs
- **models** to expose operator-defined model aliases on the proxy surface
- **API keys** to control which callers can access which models

## Step 1: Create a Provider Key

Create a provider key that points at your upstream provider.

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

Capture the returned `id`. You will use it as `provider_key_id` when creating the model.

## Step 2: Create a Model

Create a model alias that the proxy will expose to callers.

```bash title="Create a model"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o",
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

Admin writes do not become visible to the proxy instantly. The gateway publishes dynamic resources through the watch-driven snapshot path.

In practice, allow a short delay before the proxy request:

```bash title="Wait briefly for propagation"
sleep 1
```

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
      "owned_by": "openai"
    }
  ]
}
```

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

If the upstream provider is reachable and the model is configured correctly, the response will follow the OpenAI chat-completions shape.

## Verification Notes

- `401` usually means the caller API key is missing or incorrect.
- `403` means the key exists, but the model is not in `allowed_models`.
- `404` means the model alias does not resolve in the current snapshot.
- `500` on the admin API usually means the store or etcd path failed.

## Related Pages

- [Self-Hosted Quickstart](self-hosted.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Bootstrap Configuration](../configuration/bootstrap-config.md)
- [Admin API](../configuration/admin-api.md)
