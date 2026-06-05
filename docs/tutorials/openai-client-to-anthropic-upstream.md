---
title: Use an OpenAI Client with an Anthropic Upstream
description: Route an OpenAI-style client through AISIX AI Gateway to an Anthropic upstream model, with the gateway translating requests and responses in both directions.
sidebar_position: 80
toc_max_heading_level: 2
---

Route an OpenAI-compatible client to an Anthropic upstream model. The caller sends OpenAI Chat Completions requests, AISIX sends Anthropic Messages requests upstream, and the caller receives an OpenAI-compatible response. The path uses an Anthropic provider key, a model alias named `claude-prod`, and an OpenAI SDK request through AISIX.

## Prerequisites

Before you start, run the gateway from the [Quickstart](../quickstart), install `jq`, and prepare an Anthropic API key. You also need a caller API key from [Understand Admin Resources](../quickstart/first-model-first-key-first-request.md) that can use `claude-prod`, or a wildcard `allowed_models` value of `["*"]`.

## Configure the Anthropic Upstream

In this section, you will create the provider key, model alias, and caller access needed for the Anthropic upstream.

### Set Variables

Export the values used in the commands:

```shell
export AISIX_ADMIN_KEY="admin-local-only-change-me"
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
export AISIX_API_KEY="sk-demo-caller"
```

### Create an Anthropic Provider Key

:::note Anthropic `api_base`
AISIX appends `/v1/messages` to the resolved base URL. Use the bare host, `https://api.anthropic.com`. If you paste `https://api.anthropic.com/v1` or `https://api.anthropic.com/v1/messages`, AISIX normalizes it back to the bare host.
:::

Create the Anthropic provider key and capture its returned ID:

```shell
ANTHROPIC_PK_ID=$(curl -sS -X POST "http://127.0.0.1:3001/admin/v1/provider_keys" \
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

### Create a Model

Create the caller-facing model alias backed by Anthropic:

```shell
CLAUDE_PROD_ID=$(curl -sS -X POST "http://127.0.0.1:3001/admin/v1/models" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "claude-prod",
    "provider": "anthropic",
    "model_name": "claude-3-5-haiku-20241022",
    "provider_key_id": "'"${ANTHROPIC_PK_ID}"'"
  }' | jq -r .id)
```

In this model, `provider: "anthropic"` identifies the upstream provider, and `model_name` is the upstream model identifier that the gateway sends to Anthropic. Verify the exact model value in the [Anthropic Messages API reference](https://docs.anthropic.com/en/api/messages).

Wait for the model alias to become visible to the caller key:

```shell
MODEL_VISIBLE=false
for i in $(seq 1 20); do
  MODELS_RESPONSE=$(curl -sS "http://127.0.0.1:3000/v1/models" \
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

If the loop does not report `claude-prod is visible`, the admin write may not have reached the loaded proxy configuration yet. See [Verify propagation to the proxy](../quickstart/first-model-first-key-first-request.md#verify-propagation-to-the-proxy) for the full propagation check.

## Call Through AISIX

After the upstream resources are ready, call AISIX from OpenAI-compatible clients.

### Use the OpenAI SDK

The caller does not change provider, base URL, or request format relative to a normal OpenAI gateway call. Only `model` changes: it is now the gateway alias `claude-prod`.

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

## Verify Translation

The response object is OpenAI-compatible. Check the response fields instead of relying only on the `200` status code. `completion.object` should be `chat.completion`, `completion.choices[0].message.role` should be `assistant`, and `completion.choices[0].message.content` should contain text from the Anthropic response. Usage fields are also normalized: `prompt_tokens` maps from Anthropic `input_tokens`, `completion_tokens` maps from Anthropic `output_tokens`, and `total_tokens` is their sum.

To inspect the HTTP response body directly:

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer ${AISIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-prod",
    "messages": [{"role":"user","content":"Say hello."}]
  }'
```

The caller receives a single OpenAI-compatible chat-completions object. Anthropic-specific fields should not appear in the caller response.

## Clean Up

Delete the model and provider key when you finish the tutorial:

```shell
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/models/${CLAUDE_PROD_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
curl -sS -X DELETE "http://127.0.0.1:3001/admin/v1/provider_keys/${ANTHROPIC_PK_ID}" \
  -H "Authorization: Bearer ${AISIX_ADMIN_KEY}"
```

## Related Reading

For direct model fields, including the difference between `display_name` and `model_name`, see [Models](../configuration/models.md). For provider-specific `api_base` behavior, see [Provider Keys](../configuration/provider-keys.md). For the Anthropic-style proxy API and OpenAI-compatible client behavior, see [Anthropic-style Messages API](../integration/anthropic-messages.md) and [OpenAI-compatible API](../integration/openai-compatible-api.md).
