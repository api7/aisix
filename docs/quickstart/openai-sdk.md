---
title: OpenAI SDK Quickstart
description: Configure the official OpenAI SDK to call AISIX AI Gateway through the OpenAI-compatible proxy API.
toc_max_heading_level: 2
sidebar_position: 12
---

Point the official OpenAI SDK at AISIX AI Gateway instead of sending requests
directly to an upstream provider.

Continue from the [Quickstart](../quickstart). If you cleaned up the quickstart
resources, run the quickstart again first.

The example authenticates to AISIX with a caller API key, sends requests to the
gateway's `/v1` proxy API, uses an AISIX model alias instead of the upstream
model ID, and receives OpenAI-compatible chat-completions responses.

## Prerequisites

Before you start, run the gateway with one provider key, model alias, and
caller-facing API key. If you have not created them yet, start with the
[Quickstart](../quickstart). The examples use the quickstart caller key
`sk-demo-caller` and model alias `gpt-4o-mini`. You also need Node.js 20 LTS or
newer with `npm`; verify with `node --version && npm --version`.

## What Changes in Your Application

Keep the OpenAI SDK client, but change the gateway-facing inputs:

| SDK Setting | Use This Value |
| --- | --- |
| `apiKey` | AISIX caller API key, such as `sk-demo-caller` |
| `baseURL` | Gateway `/v1` proxy URL, such as `http://127.0.0.1:3000/v1` |
| `model` | AISIX model alias, such as `gpt-4o-mini` |

Your code still calls `client.chat.completions.create(...)`, sends OpenAI-style
`messages`, and receives OpenAI-compatible JSON or SSE chunks.

## Configure the SDK

Create a small Node.js project, install the SDK, and point it at the gateway
proxy URL.

### Install the SDK

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
export AISIX_MODEL="gpt-4o-mini"
export AISIX_BASE_URL="http://127.0.0.1:3000/v1"
```

### Create the Chat Example

Use the `.mjs` extension so Node treats top-level `await` and `import` as ES
modules without extra configuration.

```js title="openai-sdk-example.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,
  baseURL: process.env.AISIX_BASE_URL,
});

const response = await client.chat.completions.create({
  model: process.env.AISIX_MODEL ?? "gpt-4o-mini",
  messages: [{ role: "user", content: "Say hello from AISIX." }],
});

console.log(response.choices[0]?.message.content);
```

### Run the Example

```shell
node openai-sdk-example.mjs
```

You should see a short assistant response. The exact text depends on the
upstream model.

If the gateway can resolve `gpt-4o-mini` and the upstream provider is
reachable, the SDK returns a standard OpenAI chat-completions object.

The response should have `response.object` set to `chat.completion`,
`response.choices[0].message.role` set to `assistant`, and
`response.choices[0].message.content` populated with model output.

At the gateway layer, AISIX resolves `gpt-4o-mini` to the configured upstream
model and injects the provider credential from the stored `ProviderKey`.

:::note
If you prefer TypeScript, save the file as `openai-sdk-example.ts` and run it
with `npx tsx openai-sdk-example.ts`. Plain `node openai-sdk-example.ts` does
not work because Node cannot execute TypeScript without a loader such as `tsx`
or `ts-node`.
:::

## Streaming Responses

The same `baseURL` works for streaming.

```js title="openai-sdk-streaming.mjs"
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.AISIX_API_KEY,
  baseURL: process.env.AISIX_BASE_URL,
  maxRetries: 0,
});

const stream = await client.chat.completions.create({
  model: process.env.AISIX_MODEL ?? "gpt-4o-mini",
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

## Production Setup Pattern

In most deployments, application code needs only the gateway base URL, AISIX
caller API key, and AISIX model alias. Upstream credentials, base URLs, model
identifiers, routing policies, rate limits, guardrails, and observability hooks
stay behind the gateway.

This separation lets you rotate provider credentials, change upstream model
IDs, or add gateway policy without changing the SDK call site.

## Troubleshooting

### The SDK Still Talks to OpenAI Directly

Check `baseURL`. It must point to the gateway `/v1` proxy prefix, not to
`https://api.openai.com/v1`.

### The Request Fails With `404`

The `model` value must be the AISIX model alias, not the upstream model name
unless they are intentionally the same.

### The Request Fails With `403`

The caller key exists, but its `allowed_models` list does not include the alias
you requested.

### The Request Works in `cURL` but Not in the SDK

Check `AISIX_API_KEY`, `AISIX_BASE_URL`, and `AISIX_MODEL` first. If `curl` and
the SDK use the same values, compare the SDK request body with the request that
passed in [Send a Proxy Request](../quickstart#send-a-proxy-request).

## Related Reading

For OpenAI-compatible proxy behavior, see
[OpenAI-compatible API](../integration/openai-compatible-api.md). For SSE
responses and tool definitions, see [Streaming](../integration/streaming.md)
and [Tool calling](../integration/tool-calling.md). If your application expects
Claude-style requests, continue with
[Anthropic SDK Quickstart](anthropic-sdk.md).
