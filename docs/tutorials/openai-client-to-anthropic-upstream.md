---
title: Use an OpenAI Client with an Anthropic Upstream
description: Route an OpenAI-style client through AISIX AI Gateway to an Anthropic upstream model, with the gateway translating the wire shape in both directions.
sidebar_position: 80
---

This tutorial wires an OpenAI-compatible client to an Anthropic upstream model. The caller speaks OpenAI Chat Completions; the gateway speaks Anthropic Messages to the upstream; the gateway returns an OpenAI-shaped response back to the caller.

In this tutorial, you will:

1. Create an Anthropic provider key.
2. Create a model named `claude-prod`.
3. Call `claude-prod` with the OpenAI SDK.
4. Verify that AISIX returns an OpenAI-shaped response.

## Prerequisites

- A running gateway from the [Quickstart](../quickstart)
- `jq`, used to capture resource IDs from admin API responses
- An Anthropic API key
- A caller API key from [Understand admin resources](../quickstart/first-model-first-key-first-request.md), with `claude-prod` in `allowed_models` (or `["*"]`)

## Set variables

Export the values used by the tutorial:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
export AISIX_API_KEY="sk-demo-caller"
```

## Create an Anthropic provider key

:::note Anthropic api_base
The Anthropic bridge appends `/v1/messages` to the resolved base URL. The canonical value is the bare host, `https://api.anthropic.com`. If you paste `https://api.anthropic.com/v1` or `https://api.anthropic.com/v1/messages`, the bridge normalizes it back to the bare host.
:::

```shell
ANTHROPIC_PK_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "anthropic-prod",
    "provider": "anthropic",
    "adapter": "anthropic",
    "secret": "'"${ANTHROPIC_API_KEY}"'",
    "api_base": "https://api.anthropic.com"
  }' | jq -r .id)
```

## Create a model

```shell
CLAUDE_PROD_ID=$(curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "claude-prod",
    "provider": "anthropic",
    "model_name": "claude-3-5-haiku-20241022",
    "provider_key_id": "'"${ANTHROPIC_PK_ID}"'"
  }' | jq -r .id)
```

- `provider: "anthropic"` selects the Anthropic bridge at dispatch time.
- `model_name` is the upstream model identifier — what the gateway sends to Anthropic. Verify the exact value in the [Anthropic Messages API reference](https://docs.anthropic.com/en/api/messages).

Wait for the model alias to become visible to the caller key:

```shell
MODEL_VISIBLE=false
for i in $(seq 1 20); do
  MODELS_RESPONSE=$(curl -sS http://127.0.0.1:3000/v1/models \
    -H "Authorization: Bearer ${AISIX_API_KEY}")

  if echo "${MODELS_RESPONSE}" | jq -e '.data[]? | select(.id == "claude-prod")' >/dev/null; then
    MODEL_VISIBLE=true
    echo "claude-prod is visible"
    break
  fi
  sleep 0.5
done

if [ "${MODEL_VISIBLE}" != "true" ]; then
  echo "claude-prod is not visible yet; check the admin resources and proxy logs" >&2
fi
```

If the loop does not report `claude-prod is visible`, the admin write may not have reached the proxy snapshot yet. See [Verify propagation to the proxy](../quickstart/first-model-first-key-first-request.md#verify-propagation-to-the-proxy) for the full propagation check.

## Call with the OpenAI SDK

The caller does not change provider, base URL, or request shape relative to a normal OpenAI gateway call. Only `model` changes — it is now the gateway alias `claude-prod`.

```js title="anthropic-via-openai-sdk.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,        // sk-demo-caller
  baseURL: "http://127.0.0.1:3000/v1",
});

const completion = await client.chat.completions.create({
  model: "claude-prod",
  messages: [{ role: "user", content: "Say hello." }],
});

console.log(completion.choices[0]?.message.content);
console.log("usage:", completion.usage);
```

Run with:

```shell
node anthropic-via-openai-sdk.mjs
```

## Verify

The response object is OpenAI-shaped. Check the published wire properties so you have proof the translation worked, not just that the call returned `200`:

- `completion.object === "chat.completion"`
- `completion.choices[0].message.role === "assistant"`
- `completion.choices[0].message.content` is the text content from Anthropic's `content[0].text`
- `completion.usage.prompt_tokens` is Anthropic's `input_tokens`
- `completion.usage.completion_tokens` is Anthropic's `output_tokens`
- `completion.usage.total_tokens` is the sum

If you prefer raw HTTP and want to inspect the response body directly:

```shell
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-prod",
    "messages": [{"role":"user","content":"Say hello."}]
  }'
```

You should see a single OpenAI-shaped chat-completions object — no Anthropic-shaped fields leak through.

## Delete resources

```shell
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/models/${CLAUDE_PROD_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/provider_keys/${ANTHROPIC_PK_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

## Next steps

- [Models](../configuration/models.md) — direct model field reference, including the difference between `display_name` and `model_name`
- [Provider keys](../configuration/provider-keys.md) — `api_base` conventions per provider
- [Anthropic messages](../integration/anthropic-messages.md) — the Anthropic-shaped endpoint surface and current translation boundaries
- [OpenAI-compatible API](../integration/openai-compatible-api.md) — what gets normalized and what is forwarded as-is
