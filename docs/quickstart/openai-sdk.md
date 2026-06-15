---
title: OpenAI SDK Quickstart
description: Configure the official OpenAI SDK to call AISIX AI Gateway through the OpenAI-compatible proxy surface.
sidebar_position: 12
---

This quickstart shows the smallest working setup for the official OpenAI SDK against AISIX AI Gateway.

Use it when:

- your application already uses the official OpenAI SDK
- you want to keep OpenAI request and response shapes unchanged
- you want AISIX to own provider selection, upstream credentials, and policy enforcement

Use this page after you have already created:

- a [provider key](../overview/glossary.md#provider-key)
- a model alias
- a caller-facing [API key](../overview/glossary.md#api-key)

If you have not done that yet, start with [First Model, First Key, First Request](first-model-first-key-first-request.md).

## Prerequisites

- A running gateway with at least one configured provider key, model alias, and caller-facing API key (see the prior quickstart pages).
- **Node.js 18 or newer with `npm`.** Install from <https://nodejs.org> or via [nvm](https://github.com/nvm-sh/nvm) and verify with `node --version && npm --version`. The official OpenAI SDK used below requires Node.js 18+.

## What Changes In The SDK

Point the SDK at the gateway instead of the upstream provider:

- keep your caller-facing AISIX API key as `apiKey`
- set `baseURL` to the gateway's `/v1` prefix
- use the gateway model alias in `model`

What does **not** change:

- you still call `client.chat.completions.create(...)`
- you still send OpenAI-style `messages`
- you still receive OpenAI-style JSON or SSE chunks

## Install The SDK

All commands below run from the same project directory. Create one first:

```bash title="Create the demo project directory"
mkdir aisix-openai-demo && cd aisix-openai-demo
```

```bash title="Install openai"
npm install openai
```

## Minimal Example

Use the `.mjs` extension so Node treats top-level `await` and `import` as ES modules without extra configuration.

```js title="openai-sdk-example.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,
  baseURL: "http://127.0.0.1:3000/v1",
});

const response = await client.chat.completions.create({
  model: "gpt-4o-prod",
  messages: [{ role: "user", content: "Say hello from AISIX." }],
});

console.log(response.choices[0]?.message.content);
```

## Run It

```bash title="Run the OpenAI SDK example"
AISIX_API_KEY=sk-demo-caller node openai-sdk-example.mjs
```

:::note
If you prefer TypeScript, save the file as `openai-sdk-example.ts` and run it with `npx tsx openai-sdk-example.ts`. Plain `node openai-sdk-example.ts` does not work because Node cannot execute TypeScript without a loader such as `tsx` or `ts-node`.
:::

## Expected Result

If the gateway can resolve `gpt-4o-prod` and the upstream provider is reachable, the SDK returns a standard OpenAI chat-completions object.

The important caller-visible properties are:

- `response.object` is `chat.completion`
- `response.choices[0].message.role` is `assistant`
- `response.choices[0].message.content` contains the model output

At the gateway layer, AISIX resolves `gpt-4o-prod` to the configured upstream model and injects the provider credential from the stored `ProviderKey`.

## Streaming Example

The same `baseURL` works for streaming.

```js title="openai-sdk-streaming.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,
  baseURL: "http://127.0.0.1:3000/v1",
  maxRetries: 0,
});

const stream = await client.chat.completions.create({
  model: "gpt-4o-prod",
  messages: [{ role: "user", content: "Stream a short greeting." }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

```bash title="Run the streaming example"
AISIX_API_KEY=sk-demo-caller node openai-sdk-streaming.mjs
```

## When To Use This Quickstart

Choose this path when your client code is already built around:

- OpenAI SDKs
- OpenAI-compatible chat-completions requests
- OpenAI-style streaming consumers

If you instead want Claude-style request and response shapes, use [Anthropic SDK Quickstart](anthropic-sdk.md).

## Common Setup Pattern

In most deployments, your application should know only three gateway-specific inputs:

- gateway URL such as `http://127.0.0.1:3000`
- AISIX caller API key such as `sk-demo-caller`
- AISIX model alias such as `gpt-4o-prod`

Everything else stays behind the gateway:

- upstream provider API key
- upstream base URL
- upstream model identifier
- routing and failover policy
- rate limits, guardrails, and observability hooks

## What Stays The Same

- request and response shapes follow the OpenAI chat-completions API
- the SDK still sends requests to `/chat/completions` under the configured `baseURL`
- streaming remains SSE-based

## What Changes At The Gateway Layer

- authentication uses the AISIX caller API key, not the upstream provider key
- `model` is the AISIX model alias such as `gpt-4o-prod`
- the gateway resolves the alias to the configured upstream model and provider key

## Verification Notes

- `401` means the AISIX caller API key is missing or invalid
- `403` means the key cannot access the requested model alias
- `404` means the model alias is not present in the current gateway snapshot
- upstream `4xx` errors are returned in the proxy error envelope
- upstream `5xx` errors collapse to `502`

## Troubleshooting

### The SDK still talks to OpenAI directly

Check `baseURL`. It must point to the gateway, not to `api.openai.com`.

### The request fails with `404`

The `model` value must be the AISIX model alias, not the raw upstream model name unless they are intentionally the same.

### The request fails with `403`

The caller key exists, but its `allowed_models` list does not include the alias you requested.

### The request works in curl but not in the SDK

Compare these three values first:

- `apiKey`
- `baseURL`
- `model`

## Related Pages

- [First Model, First Key, First Request](first-model-first-key-first-request.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Streaming](../integration/streaming.md)
- [Tool Calling](../integration/tool-calling.md)
