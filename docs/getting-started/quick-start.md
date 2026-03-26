---
title: Quick Start
slug: /aisix/getting-started/quick-start
description: Get AISIX AI Gateway up and running in under 5 minutes with Docker.
---

This guide walks you through starting AISIX with a single command and making your first AI request in under 5 minutes.

## Prerequisites

- **Docker** with the Compose plugin. Install from [docs.docker.com/get-docker](https://docs.docker.com/get-docker/).

## Step 1: Start AISIX

Run the following command to download and start AISIX:

```bash
curl -fsSL https://run.api7.ai/aisix/quickstart | sh
```

This script:
- Downloads `docker-compose.yaml` and `config.yaml` to `~/.aisix/`
- Generates a random Admin Key and writes it into the config
- Pulls and starts AISIX and etcd via Docker Compose

When the script completes, you will see output like:

```text
[aisix] AISIX is running!

  Proxy API:   http://127.0.0.1:3000
  Admin API:   http://127.0.0.1:3001/aisix/admin
  Admin UI:    http://127.0.0.1:3001/ui
  API Docs:    http://127.0.0.1:3001/openapi
  Admin Key:   <generated-admin-key>

  Export it:    export ADMIN_KEY=<generated-admin-key>
```

Copy the `export` line and run it in your terminal — all examples in this guide use the `$ADMIN_KEY` variable:

```bash
export ADMIN_KEY=<your-admin-key>
```

## Step 2: Configure a Model

Tell AISIX which upstream LLM to use. This example uses OpenAI's `gpt-4`:

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "openai-gpt4",
    "model": "openai/gpt-4",
    "provider_config": {
      "api_key": "<YOUR_OPENAI_API_KEY>"
    }
  }'
```

- Replace `<YOUR_OPENAI_API_KEY>` with your OpenAI API key.
- `name` is the identifier used in requests to AISIX.
- `model` specifies the provider (`openai`) and the upstream model ID.

A `201` response confirms the model was created.

## Step 3: Configure an API Key

Create an API key to authenticate requests to the proxy:

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/apikeys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "my-secret-key",
    "allowed_models": ["openai-gpt4"]
  }'
```

- `key` is the secret used in the `Authorization` header when calling the proxy.
- `allowed_models` controls which models this key can access.

## Step 4: Make Your First Request

Send a chat completion request through AISIX to the upstream provider:

```bash
curl -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{
    "model": "openai-gpt4",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a fun fact about AI"
      }
    ]
  }'
```

You will receive a standard OpenAI-compatible chat completion response from the upstream provider, proxied through AISIX.

## Example: Other Providers

Adding other providers follows the same pattern — use the correct `model` prefix and provide the right credentials in `provider_config`.

### Google Gemini

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gemini",
    "model": "gemini/gemini-2.5-flash",
    "provider_config": {
      "api_key": "<YOUR_GEMINI_API_KEY>"
    }
  }'
```

### Anthropic

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-anthropic",
    "model": "anthropic/claude-3-5-sonnet-20241022",
    "provider_config": {
      "api_key": "<YOUR_ANTHROPIC_API_KEY>"
    }
  }'
```

### DeepSeek

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-deepseek",
    "model": "deepseek/deepseek-chat",
    "provider_config": {
      "api_key": "<YOUR_DEEPSEEK_API_KEY>"
    }
  }'
```

After creating a new model, update your API key's `allowed_models` list to grant access to it.

## Next Steps

- **Explore More Providers**: Add other supported providers like **DeepSeek**. The process is the same: create a new model with the `deepseek/deepseek-chat` prefix and your DeepSeek API key, then update your API key to allow access.
- **Configure Rate Limiting**: Add rate limits to your models and API keys to control costs and prevent abuse.
- **Admin UI**: AISIX includes a built-in Admin UI and Chat Playground. See [Admin UI](./admin-ui.md).
