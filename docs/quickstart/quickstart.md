---
title: Quickstart
description: Run AISIX AI Gateway locally, create your first model and API key, and send your first proxy request.
sidebar_position: 10
---

This quickstart walks you through running AISIX AI Gateway locally, configuring the minimum resources required for traffic, and sending your first request through the proxy.

By the end of this guide, you will have:

1. Started a local gateway instance.
2. Created a provider key, model alias, and caller API key.
3. Verified that the proxy can see the configured model.
4. Sent a request through the OpenAI-compatible API.

**Time to complete**: about 15 minutes if Rust is already installed. Allow longer for the first Rust build.

## Before You Start

This page is the main first-user path for the self-hosted gateway.

If you only want to verify local bootstrap, use [Boot A Self-Hosted Gateway](self-hosted.md) instead.

If you want the same first-request flow with more explanation about each admin resource and the failure modes to verify, continue to [First Model, First Key, First Request](first-model-first-key-first-request.md) after you finish this page.

## Prerequisites

| Requirement | Notes |
|---|---|
| Git | Needed to clone the repository. |
| Rust | Install through [rustup](https://rustup.rs). The repo pins its toolchain in `rust-toolchain.toml`. |
| Docker | Used to run a local etcd instance. |
| curl | Used to verify the admin and proxy APIs. |
| Upstream provider API key | Required for the final successful LLM request. OpenAI is used in the example below. |

## Step 1: Clone the repository

```bash title="Clone the repository"
git clone https://github.com/api7/ai-gateway.git
cd ai-gateway
```

## Step 2: Start etcd

Start a local etcd 3.5 instance in Docker. AISIX AI Gateway uses etcd as its configuration store in self-hosted mode.

```bash title="Start etcd"
docker run -d \
  --name aisix-etcd \
  -p 2379:2379 \
  -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --advertise-client-urls=http://0.0.0.0:2379 \
  --listen-client-urls=http://0.0.0.0:2379
```

Verify that the container is running:

```bash title="Check the etcd container"
docker ps --filter name=aisix-etcd
```

## Step 3: Create a local config

Copy the example configuration:

```bash title="Create a local config"
cp config.example.yaml config.yaml
```

The example configuration already points at local etcd and binds:

- proxy listener on `127.0.0.1:3000`
- admin listener on `127.0.0.1:3001`

If either port is already in use on your machine, update `proxy.addr` or `admin.addr` in `config.yaml` before starting the gateway.

## Step 4: Start the gateway

```bash title="Run the gateway"
cargo run -p aisix-server --bin aisix -- --config config.yaml
```

The first run downloads dependencies and compiles the workspace, so it can take several minutes. Keep this terminal open while the gateway is running.

## Step 5: Verify the listeners

In a new terminal, verify that both listeners are healthy:

```bash title="Check the proxy listener"
curl -sS http://127.0.0.1:3000/livez
```

```bash title="Check the admin listener"
curl -sS http://127.0.0.1:3001/livez
```

Expected response:

```text
ok
```

If you only want to verify local bootstrap, you can stop here. The remaining steps configure a real provider-backed request path.

## Step 6: Export your local variables

Export the values used by the remaining commands:

```bash title="Set local variables"
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export CALLER_KEY="sk-demo-caller"
```

Replace `YOUR_OPENAI_API_KEY` with a real upstream key. Without a valid provider credential, the admin resources below can still be created, but the final proxy request will fail at the upstream call.

The remaining steps create three distinct resources:

- a **provider key** for the upstream credential
- a **model alias** that callers use on the proxy API
- a **caller API key** that authenticates client traffic to AISIX

## Step 7: Create a provider key

```bash title="Create a provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-upstream",
    "secret": "'"${OPENAI_API_KEY}"'",
    "api_base": "https://api.openai.com/v1"
  }'
```

This creates the upstream credential the gateway will use when it forwards requests to OpenAI.

Copy the returned `id` field and export it:

```bash title="Capture the provider key id"
export PROVIDER_KEY_ID="YOUR_PROVIDER_KEY_ID"
```

## Step 8: Create a model alias

```bash title="Create a model"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "provider_key_id": "'"${PROVIDER_KEY_ID}"'"
  }'
```

Copy the returned `id` field and export it:

```bash title="Capture the model id"
export MODEL_ID="YOUR_MODEL_ID"
```

Clients will use `gpt-4o-prod` as the `model` value on the proxy API.

## Step 9: Create a caller API key

AISIX stores a hash of the caller key rather than the plaintext value. Hash the caller key first:

```bash title="Hash the caller key"
CALLER_KEY_HASH=$(printf '%s' "${CALLER_KEY}" | shasum -a 256 | awk '{print $1}')
```

Then create the API key resource:

```bash title="Create the caller API key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "'"${CALLER_KEY_HASH}"'",
    "allowed_models": ["gpt-4o-prod"]
  }'
```

Copy the returned `id` field and export it:

```bash title="Capture the API key id"
export APIKEY_ID="YOUR_APIKEY_ID"
```

## Step 10: Wait for propagation

The admin API writes to etcd first. The proxy picks up those changes through the watch-driven snapshot path.

For local setup, a short wait is usually enough:

```bash title="Wait for propagation"
sleep 1
```

## Step 11: Verify model visibility

```bash title="Check /v1/models"
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer ${CALLER_KEY}"
```

Expected response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o-prod",
      "object": "model"
    }
  ]
}
```

At this point, you have verified that:

- the gateway booted successfully
- the admin API accepted your resources
- the proxy can see the resolved model alias

## Step 12: Send your first proxy request

```bash title="Send a chat completion request"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${CALLER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-prod",
    "messages": [
      {"role": "user", "content": "Say hello from AISIX AI Gateway."}
    ]
  }'
```

With a valid upstream provider key, the response follows the OpenAI chat-completions shape.

The gateway authenticates to the upstream provider with the provider key you created earlier, while the caller authenticates to AISIX with the caller API key. This separation is one of the core operating patterns in AISIX AI Gateway.

## Clean up

Delete the created resources in reverse dependency order:

```bash title="Delete the caller API key"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/apikeys/${APIKEY_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```bash title="Delete the model"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/${MODEL_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```bash title="Delete the provider key"
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/provider_keys/${PROVIDER_KEY_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

Then stop the local etcd container:

```bash title="Stop etcd"
docker rm -f aisix-etcd
```

## Next steps

Follow the docs in this order if you want to keep going:

1. [First Model, First Key, First Request](first-model-first-key-first-request.md) for the same flow with more detail about the resource contract, propagation behavior, and negative-path checks.
2. [OpenAI SDK Quickstart](openai-sdk.md) if your application already uses the OpenAI SDK.
3. [Anthropic SDK Quickstart](anthropic-sdk.md) if your application expects the Anthropic-style `messages` API.
4. [OpenAI-Compatible API](../integration/openai-compatible-api.md) for the broader proxy contract.
5. [Bootstrap Configuration](../configuration/bootstrap-config.md) before adapting the local config for a shared environment.

- [Boot A Self-Hosted Gateway](self-hosted.md) for the narrower bootstrap-only flow.
- [First Model, First Key, First Request](first-model-first-key-first-request.md) for a deeper walkthrough of the admin resources.
- [OpenAI SDK Quickstart](openai-sdk.md) if you want to switch from `curl` to an SDK client.
