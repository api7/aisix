---
title: OpenAI SDK Quickstart
description: Configure the official OpenAI SDK to call AISIX AI Gateway through the OpenAI-compatible proxy surface.
sidebar_position: 12
---

This quickstart shows the smallest working setup for the official OpenAI SDK against AISIX AI Gateway.

Use this page after you have already created:

- a provider key
- a model alias
- a caller-facing API key

If you have not done that yet, start with [First Model, First Key, First Request](first-model-first-key-first-request.md).

## What Changes In The SDK

Point the SDK at the gateway instead of the upstream provider:

- keep your caller-facing AISIX API key as `apiKey`
- set `baseURL` to the gateway's `/v1` prefix
- use the gateway model alias in `model`

## Install The SDK

```bash title="Install openai"
npm install openai
```

## Minimal Example

```ts title="openai-sdk-example.ts"
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
AISIX_API_KEY=sk-demo-caller node openai-sdk-example.ts
```

## Streaming Example

The same `baseURL` works for streaming.

```ts title="openai-sdk-streaming.ts"
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

## Related Pages

- [First Model, First Key, First Request](first-model-first-key-first-request.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Streaming](../integration/streaming.md)
- [Tool Calling](../integration/tool-calling.md)
