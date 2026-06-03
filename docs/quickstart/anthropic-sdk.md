---
title: Anthropic SDK Quickstart
description: Configure an Anthropic-compatible client to call AISIX AI Gateway through the /v1/messages proxy surface.
sidebar_position: 13
toc_max_heading_level: 2
---

This quickstart shows how to point an Anthropic-style SDK client at AISIX AI Gateway.

This guide shows how to call AISIX from the Anthropic SDK when your application already uses Claude-style request and response shapes. If your application is already built around OpenAI SDKs, use [OpenAI SDK quickstart](openai-sdk.md) instead.

By the end of this guide, you will have:

1. Added an Anthropic-backed model alias to the local gateway.
2. Allowed the quickstart caller key to use that alias.
3. Called `POST /v1/messages` through the Anthropic Python SDK.

## Prerequisites

- A running gateway from the [Quickstart](../quickstart).
- The quickstart admin key and caller key. This guide uses `admin-local-only-change-me` and `sk-demo-caller`.
- An Anthropic API key.
- Python 3.8 or later.
- `curl` and `jq`, used to create and verify the example alias.

The main quickstart creates an OpenAI-backed alias named `gpt-4o-prod`. This page adds a second alias, `claude-prod`, backed by Anthropic.

## Request path

The Anthropic SDK will call the gateway, not Anthropic directly:

```text
Anthropic SDK -> AISIX /v1/messages -> claude-prod -> Anthropic provider key -> upstream model
```

The application sends:

- AISIX caller key: `sk-demo-caller`
- AISIX model alias: `claude-prod`
- Gateway base URL: `http://127.0.0.1:3000`

AISIX resolves `claude-prod` to the upstream Anthropic model and injects the stored provider credential on the upstream side.

## Step 1: Set Anthropic setup variables

Export the values used by the admin commands:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export CALLER_KEY="sk-demo-caller"
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
export ANTHROPIC_MODEL="claude-sonnet-4-6"
export AISIX_ANTHROPIC_ALIAS="claude-prod"
```

Replace `YOUR_ANTHROPIC_API_KEY` with a real Anthropic API key.

`ANTHROPIC_MODEL` is the upstream model ID AISIX sends to Anthropic. `AISIX_ANTHROPIC_ALIAS` is the caller-facing model name your application sends to AISIX.

## Step 2: Find the quickstart API key resource

If you are reusing the quickstart caller key, recover the matching API key resource from the admin API:

```shell
if command -v sha256sum >/dev/null 2>&1; then
  CALLER_KEY_HASH=$(printf '%s' "${CALLER_KEY}" | sha256sum | cut -d' ' -f1)
else
  CALLER_KEY_HASH=$(printf '%s' "${CALLER_KEY}" | shasum -a 256 | awk '{print $1}')
fi

APIKEY_ID=$(curl -sS http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  | jq -r --arg hash "${CALLER_KEY_HASH}" \
    '.[] | select(.value.key_hash == $hash) | .id' \
  | head -n 1)

if [ -z "${APIKEY_ID}" ]; then
  echo "No API key resource found for ${CALLER_KEY}; rerun the main quickstart API-key step." >&2
  exit 1
fi
```

## Step 3: Create an Anthropic provider key

Create a provider key that stores the upstream Anthropic credential:

```shell
ANTHROPIC_PROVIDER_KEY_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "anthropic-upstream",
    "provider": "anthropic",
    "adapter": "anthropic",
    "secret": "'"${ANTHROPIC_API_KEY}"'",
    "api_base": "https://api.anthropic.com"
  }' | jq -r .id)
```

The Anthropic bridge appends `/v1/messages` to `api_base`, so use the bare host `https://api.anthropic.com`.

## Step 4: Create an Anthropic-backed model alias

Create the caller-facing model alias:

```shell
ANTHROPIC_MODEL_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "'"${AISIX_ANTHROPIC_ALIAS}"'",
    "provider": "anthropic",
    "model_name": "'"${ANTHROPIC_MODEL}"'",
    "provider_key_id": "'"${ANTHROPIC_PROVIDER_KEY_ID}"'"
  }' | jq -r .id)
```

The caller sends `claude-prod` to AISIX. The upstream provider receives `claude-sonnet-4-6`.

## Step 5: Allow the caller key to use the alias

Update the quickstart API key so the same caller key can access both `gpt-4o-prod` and `claude-prod`:

```shell
curl -sS -X PUT http://127.0.0.1:3001/admin/v1/apikeys/${APIKEY_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "'"${CALLER_KEY_HASH}"'",
    "allowed_models": ["gpt-4o-prod", "'"${AISIX_ANTHROPIC_ALIAS}"'"]
  }'
```

## Step 6: Verify the alias is visible

Poll `/v1/models` until the new alias appears for the caller key:

```shell
ANTHROPIC_ALIAS_VISIBLE=false
for i in $(seq 1 20); do
  if curl -sS http://127.0.0.1:3000/v1/models \
    -H "Authorization: Bearer ${CALLER_KEY}" \
    | jq -e --arg model "${AISIX_ANTHROPIC_ALIAS}" \
      '.data[]? | select(.id == $model)' >/dev/null; then
    ANTHROPIC_ALIAS_VISIBLE=true
    echo "Anthropic alias is visible"
    break
  fi

  sleep 0.5
done

if [ "${ANTHROPIC_ALIAS_VISIBLE}" != "true" ]; then
  echo "Anthropic alias is not visible yet; check the admin resources and proxy logs" >&2
  exit 1
fi
```

## Step 7: Install the Anthropic SDK

Create a small Python environment and install the SDK:

```shell
python -m venv .venv
. .venv/bin/activate
```

```shell
python -m pip install anthropic
```

## Step 8: Create the SDK example

Use your Anthropic-compatible client with the gateway base URL and your AISIX caller key:

```python title="anthropic-sdk-example.py"
import os

from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ["AISIX_API_KEY"],
    base_url=os.environ["AISIX_BASE_URL"],
)

message = client.messages.create(
    model=os.environ.get("AISIX_MODEL", "claude-prod"),
    max_tokens=128,
    messages=[
        {"role": "user", "content": "Say hello from AISIX."}
    ],
)

print(message.content[0].text)
```

Set the gateway-facing SDK values:

```shell
export AISIX_API_KEY="sk-demo-caller"
export AISIX_MODEL="claude-prod"
export AISIX_BASE_URL="http://127.0.0.1:3000"
```

Run the example:

```shell
python anthropic-sdk-example.py
```

You should see a short assistant response. The exact text depends on the upstream model.

## Expected result

If the gateway can resolve `claude-prod` and the upstream is reachable, the client receives an Anthropic-style message response from AISIX.

At the client edge:

- the SDK sends the request to `POST /v1/messages`
- `model` is the AISIX model alias
- `messages` and `max_tokens` follow the Anthropic Messages shape

At the gateway layer, AISIX authenticates the caller key, checks `allowed_models`, resolves the alias, and injects the upstream Anthropic provider key.

## Compatibility notes

`POST /v1/messages` can resolve both Anthropic-backed and non-Anthropic-backed model aliases. Anthropic-backed aliases preserve Anthropic-specific request and response behavior most directly.

Non-Anthropic translation is useful when you need a stable Anthropic-style client edge, but it is not feature-identical to native Anthropic behavior. If your application depends on tool-result round trips, thinking blocks, image blocks, or other Anthropic-specific content blocks, prefer an Anthropic-backed alias and validate the exact flow.

For the full endpoint behavior, see [Anthropic Messages](../integration/anthropic-messages.md).

## Troubleshooting

### The client gets `404`

Check that `AISIX_MODEL` is the AISIX alias, such as `claude-prod`, not only the raw upstream Anthropic model ID.

### The client gets `403`

The caller key is valid, but its `allowed_models` list does not include the alias you requested.

### The client works with `curl` but not the SDK

Compare these three values first:

- `AISIX_API_KEY`
- `AISIX_BASE_URL`
- `AISIX_MODEL`

Then compare the SDK request body with the request pattern in [Anthropic Messages](../integration/anthropic-messages.md#request-pattern).

## Clean up the extra alias

If you created the Anthropic-backed alias only for this guide, delete it before deleting the provider key:

```shell
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/models/${ANTHROPIC_MODEL_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

```shell
curl -sS -X DELETE http://127.0.0.1:3001/admin/v1/provider_keys/${ANTHROPIC_PROVIDER_KEY_ID} \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

If you want to remove every local quickstart resource, run the cleanup section in [Understand admin resources](first-model-first-key-first-request.md#clean-up-when-done), then stop the local stack.

## Next steps

- [Anthropic Messages](../integration/anthropic-messages.md) — review the full `/v1/messages` gateway contract.
- [Streaming](../integration/streaming.md) — understand SSE behavior across endpoint families.
- [OpenAI client to Anthropic upstream](../tutorials/openai-client-to-anthropic-upstream.md) — route OpenAI-shaped clients to an Anthropic upstream.
- [Provider compatibility](../reference/provider-compatibility.md) — check current provider and endpoint boundaries.
