---
title: Quickstart
description: Run AISIX AI Gateway locally, create your first model and API key, and send your first proxy request.
sidebar_position: 10
slug: /quickstart
---

This quickstart shows you how to run AISIX AI Gateway locally, configure the minimum resources required for traffic, and send one request through the proxy.

By the end of this guide, you will have:

1. Started a local gateway instance.
2. Created a provider key, model alias, and caller API key.
3. Verified that the proxy can see the configured model.
4. Sent a request through the OpenAI-compatible API.

**Time to complete**: about 10 minutes.

## Request path

The first working request path needs three dynamic resources:

```text
caller key -> model alias -> provider key -> upstream model
```

In this quickstart:

- `sk-demo-caller` is the caller key your application sends to AISIX.
- `gpt-4o-prod` is the model alias your application uses on the proxy API.
- `openai-upstream` is the provider key AISIX uses to call OpenAI.
- `gpt-4o-mini` is the upstream model AISIX sends to OpenAI.

The caller never sends the upstream provider key. AISIX keeps that credential on the provider side of the gateway.

## Prerequisites

Before you start, make sure you have:

- Docker with Docker Compose, used to run AISIX AI Gateway and etcd locally.
- `curl`, used to verify the admin and proxy APIs.
- `jq`, used to capture IDs from admin API responses.
- An upstream provider API key. This guide uses OpenAI for the final LLM request.

## Step 1: Create a working directory

```shell
mkdir aisix-quickstart
cd aisix-quickstart
```

## Step 2: Create a local config

Create `config.yaml` for the local gateway container:

```yaml title="config.yaml"
etcd:
  endpoints:
    - "http://etcd:2379"
  prefix: "/aisix"
  dial_timeout_ms: 5000
  request_timeout_ms: 5000

proxy:
  addr: "0.0.0.0:3000"
  request_body_limit_bytes: 10485760

admin:
  addr: "0.0.0.0:3001"
  admin_keys:
    - "admin-local-only-change-me"

observability:
  service_name: "aisix"
  log_level: "info"
  access_log: true

cache:
  backend: "memory"
```

AISIX uses etcd as its configuration store in self-hosted mode. In this Compose stack, the gateway reaches etcd through the service name `etcd`.

## Step 3: Create the Compose stack

Create `docker-compose.yml`:

```yaml title="docker-compose.yml"
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    command:
      - /usr/local/bin/etcd
      - --advertise-client-urls=http://0.0.0.0:2379
      - --listen-client-urls=http://0.0.0.0:2379
    ports:
      - "2379:2379"

  aisix:
    image: ghcr.io/api7/ai-gateway:dev
    volumes:
      - ./config.yaml:/etc/aisix/config.yaml:ro
    ports:
      - "3000:3000"
      - "3001:3001"
    depends_on:
      - etcd
```

:::note
`ghcr.io/api7/ai-gateway:dev` tracks the `main` branch. For a reproducible deployment, pin a released version tag once one is available.
:::

## Step 4: Start the gateway

```shell
docker compose up -d
```

Verify that both containers are running:

```shell
docker compose ps
```

The gateway exposes:

- proxy listener on `http://127.0.0.1:3000`
- admin listener on `http://127.0.0.1:3001`

## Step 5: Verify the listeners

In a new terminal, verify that both listeners are healthy:

```shell
curl -sS http://127.0.0.1:3000/livez
```

```shell
curl -sS http://127.0.0.1:3001/livez
```

Expected response:

```text
ok
```

The remaining steps configure a real provider-backed request path.

## Step 6: Export your local variables

Export the values used by the remaining commands:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export CALLER_KEY="sk-demo-caller"
```

Replace `YOUR_OPENAI_API_KEY` with a real upstream key. Without a valid provider credential, the admin resources can still be created, but the final proxy request will fail when AISIX calls the upstream provider.

The remaining steps create three resources:

- a **provider key** for the upstream credential
- a **model alias** that callers use on the proxy API
- a **caller API key** that authenticates client traffic to AISIX

## Step 7: Create a provider key

```shell
PROVIDER_KEY_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-upstream",
    "provider": "openai",
    "adapter": "openai",
    "secret": "'"${OPENAI_API_KEY}"'",
    "api_base": "https://api.openai.com/v1"
  }' | jq -r .id)
```

This creates the upstream credential AISIX uses when it forwards requests to OpenAI and stores the returned resource ID in `PROVIDER_KEY_ID`.

:::warning Production credentials
In standalone self-hosted mode, AISIX stores provider-key `secret` values as plaintext under the configured etcd `prefix`. For production, protect etcd with the same care as any secret store, including encryption at rest and restricted access. In AISIX Cloud managed deployments, provider credentials are managed by the control plane and projected to data planes.
:::

:::note Provider base URL
This quickstart uses OpenAI, so `api_base` is `https://api.openai.com/v1`. Do not reuse that value for every provider. Provider base URL requirements differ; see [Provider keys](../configuration/provider-keys.md#base-url) and [URL normalization](../configuration/provider-keys.md#url-normalization).
:::

## Step 8: Create a model alias

In this resource, `display_name` is the model alias callers send to AISIX. `model_name` is the upstream model ID that AISIX sends to the provider.

```shell
MODEL_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "provider_key_id": "'"${PROVIDER_KEY_ID}"'"
  }' | jq -r .id)
```

Clients will use `gpt-4o-prod` as the `model` value on the proxy API. The upstream provider receives `gpt-4o-mini`.

## Step 9: Create a caller API key

AISIX stores a hash of the caller key rather than the plaintext value. Hash the caller key first:

```shell
if command -v sha256sum >/dev/null 2>&1; then
  CALLER_KEY_HASH=$(printf '%s' "${CALLER_KEY}" | sha256sum | cut -d' ' -f1)
else
  CALLER_KEY_HASH=$(printf '%s' "${CALLER_KEY}" | shasum -a 256 | awk '{print $1}')
fi
```

Then create the API key resource:

```shell
APIKEY_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "'"${CALLER_KEY_HASH}"'",
    "allowed_models": ["gpt-4o-prod"]
  }' | jq -r .id)
```

Verify that all three resource IDs were captured:

```shell
printf 'provider key: %s\nmodel: %s\napi key: %s\n' \
  "${PROVIDER_KEY_ID}" "${MODEL_ID}" "${APIKEY_ID}"
```

If any value is empty or `null`, check the previous command output for an `error_msg` before continuing.

## Step 10: Verify model visibility

The admin API writes to etcd first. The proxy picks up those changes through the watch-driven snapshot path.

Poll `/v1/models` until the model alias is visible to the caller key:

```shell
MODEL_VISIBLE=false
for i in $(seq 1 20); do
  if curl -sS http://127.0.0.1:3000/v1/models \
    -H "Authorization: Bearer ${CALLER_KEY}" \
    | jq -e '.data[]? | select(.id == "gpt-4o-prod")' >/dev/null; then
    MODEL_VISIBLE=true
    echo "model alias is visible"
    break
  fi
  sleep 0.5
done

if [ "${MODEL_VISIBLE}" != "true" ]; then
  echo "model alias is not visible yet; check the admin resources and proxy logs" >&2
fi
```

Optionally inspect the caller-visible model list:

```shell
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer ${CALLER_KEY}"
```

The response should include `gpt-4o-prod`.

At this point, you have verified that:

- the gateway booted successfully
- the admin API accepted your resources
- the proxy can see the resolved model alias

You have not called the upstream provider yet. `/v1/models` proves that caller authentication, model allowlisting, and configuration propagation are working. The next step sends real AI traffic through the provider key.

## Step 11: Send your first proxy request

```shell
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

With a valid upstream provider key, the response follows the OpenAI chat-completions shape and includes the model alias `gpt-4o-prod`.

The gateway authenticates to the upstream provider with the provider key you created earlier, while the caller authenticates to AISIX with the caller API key. This separation is one of the core operating patterns in AISIX AI Gateway.

Keep this local stack and these resources running if you want to continue through the next docs. The follow-up pages reuse the same caller key and model alias.

## Next steps

Follow the docs in this order:

1. [Understand admin resources](first-model-first-key-first-request.md) to inspect the resource chain, propagation behavior, negative-path checks, and cleanup.
2. [What is AISIX AI Gateway](../overview/what-is-aisix-ai-gateway.md) to understand where the gateway fits and when to use it instead of direct provider integrations or AI plugins on normal API gateway routes.
3. [Client APIs overview](../integration/overview.md) to choose the caller-facing API surface for your application.
4. [Configuration overview](../configuration/overview.md) before adapting the local config for a shared environment.
5. [Operations overview](../operations/overview.md) when you are preparing a runtime for real traffic.

If you want to build and run the gateway from source, use [Boot a self-hosted gateway](self-hosted.md).
