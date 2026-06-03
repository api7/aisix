---
title: OpenAI SDK Quickstart
description: Configure the official OpenAI SDK to call AISIX AI Gateway through the OpenAI-compatible proxy surface.
sidebar_position: 12
---

This quickstart shows how to point the official OpenAI SDK at AISIX AI Gateway instead of sending requests directly to an upstream provider.

This guide continues from the [Quickstart](../quickstart) and shows how to call the gateway with the OpenAI SDK. If you cleaned up the quickstart resources, run the quickstart again first.

By the end of this guide, your OpenAI SDK client will:

- authenticate to AISIX with a caller API key
- send requests to the gateway's `/v1` proxy surface
- use an AISIX model alias instead of a raw upstream model ID
- receive OpenAI-compatible chat-completions responses

## Prerequisites

- A running gateway with one provider key, model alias, and caller-facing API key. If you have not created them yet, start with the [Quickstart](../quickstart).
- The quickstart caller key and model alias. This guide uses `sk-demo-caller` and `gpt-4o-prod`.
- Node.js 20 LTS or newer with `npm`. Verify with `node --version && npm --version`.

## What changes in your application

Keep the OpenAI SDK surface, but change the gateway-facing inputs:

| SDK setting | Use this value |
|---|---|
| `apiKey` | AISIX caller API key, such as `sk-demo-caller` |
| `baseURL` | Gateway `/v1` proxy URL, such as `http://127.0.0.1:3000/v1` |
| `model` | AISIX model alias, such as `gpt-4o-prod` |

Your code still calls `client.chat.completions.create(...)`, sends OpenAI-style `messages`, and receives OpenAI-compatible JSON or SSE chunks.

## Install the SDK

Create a small demo project:

```shell
mkdir aisix-openai-demo && cd aisix-openai-demo
npm init -y
```

```shell
npm install openai
```

Set the gateway values that the examples use:

```shell
export AISIX_API_KEY="sk-demo-caller"
export AISIX_MODEL="gpt-4o-prod"
export AISIX_BASE_URL="http://127.0.0.1:3000/v1"
```

## Minimal example

Use the `.mjs` extension so Node treats top-level `await` and `import` as ES modules without extra configuration.

```js title="openai-sdk-example.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,
  baseURL: process.env.AISIX_BASE_URL,
});

const response = await client.chat.completions.create({
  model: process.env.AISIX_MODEL ?? "gpt-4o-prod",
  messages: [{ role: "user", content: "Say hello from AISIX." }],
});

console.log(response.choices[0]?.message.content);
```

## Run it

```shell
node openai-sdk-example.mjs
```

You should see a short assistant response. The exact text depends on the upstream model.

If the gateway can resolve `gpt-4o-prod` and the upstream provider is reachable, the SDK returns a standard OpenAI chat-completions object.

The important caller-visible properties are:

- `response.object` is `chat.completion`
- `response.choices[0].message.role` is `assistant`
- `response.choices[0].message.content` contains the model output

At the gateway layer, AISIX resolves `gpt-4o-prod` to the configured upstream model and injects the provider credential from the stored `ProviderKey`.

:::note
If you prefer TypeScript, save the file as `openai-sdk-example.ts` and run it with `npx tsx openai-sdk-example.ts`. Plain `node openai-sdk-example.ts` does not work because Node cannot execute TypeScript without a loader such as `tsx` or `ts-node`.
:::

## Streaming example

The same `baseURL` works for streaming.

```js title="openai-sdk-streaming.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,
  baseURL: process.env.AISIX_BASE_URL,
  maxRetries: 0,
});

const stream = await client.chat.completions.create({
  model: process.env.AISIX_MODEL ?? "gpt-4o-prod",
  messages: [{ role: "user", content: "Stream a short greeting." }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

```shell
node openai-sdk-streaming.mjs
```

You should see streamed text printed to the terminal.

## Production setup pattern

In most deployments, application code needs only three gateway-facing inputs:

- gateway base URL
- AISIX caller API key
- AISIX model alias

The upstream details stay behind the gateway:

- upstream provider API key
- upstream base URL
- upstream model identifier
- routing or failover policy
- rate limits, guardrails, and observability hooks

This separation lets operators rotate provider credentials, change upstream model IDs, or add gateway policy without changing the SDK call site.

## Troubleshooting

### The SDK still talks to OpenAI directly

Check `baseURL`. It must point to the gateway `/v1` proxy prefix, not to `https://api.openai.com/v1`.

### The request fails with `404`

The `model` value must be the AISIX model alias, not the raw upstream model name unless they are intentionally the same.

### The request fails with `403`

The caller key exists, but its `allowed_models` list does not include the alias you requested.

### The request works in curl but not in the SDK

Compare these three values first:

- `AISIX_API_KEY`
- `AISIX_BASE_URL`
- `AISIX_MODEL`

If `curl` and the SDK use the same values, compare the SDK request body with the request that passed in the [Quickstart](../quickstart#step-11-send-your-first-proxy-request).

## Next steps

- [OpenAI-compatible API](../integration/openai-compatible-api.md) — review the full OpenAI-compatible proxy contract.
- [Streaming](../integration/streaming.md) — use SSE responses through AISIX.
- [Tool calling](../integration/tool-calling.md) — pass tool definitions through supported OpenAI-compatible paths.
- [Anthropic SDK quickstart](anthropic-sdk.md) — use the Anthropic Messages surface when your application expects Claude-style requests.
