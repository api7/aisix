---
title: Quick Start
slug: /ai-gateway/getting-started/quick-start
description: Get AISIX AI Gateway up and running in 5 minutes.
---

This guide walks you through installing and configuring AISIX, and making your first request to an LLM in under 5 minutes. You will run the gateway from source using `cargo` and configure it entirely via the Admin API.

## Prerequisites

Ensure you have the following installed:

- **Rust**: Latest stable version (see [rustup.rs](https://rustup.rs/)).
- **etcd**: A running instance. For local testing, use Docker:

  ```bash
  docker run -d -p 2379:2379 --name etcd gcr.io/etcd-development/etcd:v3.6.8 etcd --advertise-client-urls http://0.0.0.0:2379 --listen-client-urls http://0.0.0.0:2379
  ```

- **etcdctl**: The command-line client for etcd.
- **Git**: To clone the AISIX repository.

## Step 1: Clone the Repository

Clone the AISIX repository:

```bash
git clone https://github.com/api7/ai-gateway-stash.git
cd ai-gateway-stash
```

## Step 2: Modify the Admin Key (Recommended)

A `config.yaml` file already exists in the project root with the following defaults:

```yaml
# config.yaml (excerpt)
deployment:
  etcd:
    host:
      - "http://127.0.0.1:2379"
    prefix: /aisix
    timeout: 30
  admin:
    admin_key:
      - key: admin            # <-- default value
```

The default `admin_key` is `admin`. **We strongly recommend replacing it** with a randomly generated string before proceeding:

```bash
export ADMIN_KEY=$(openssl rand -base64 32)
sed -i.bak "s/key: admin/key: $ADMIN_KEY/" config.yaml
echo "Your admin key: $ADMIN_KEY"
```

All examples in this guide use the `$ADMIN_KEY` environment variable.

## Step 3: Run AISIX

Run the gateway using `cargo`:

```bash
RUST_LOG=info cargo run
```

AISIX starts two services by default:
- **Data Plane**: Listens on `0.0.0.0:3000` for AI requests.
- **Admin API**: Listens on `127.0.0.1:3001` for configuration management.

The log output will indicate that both servers are running.

## Step 4: Configure a Model

Configure a model using the Admin API. This tells AISIX how to connect to an upstream LLM provider. This example uses OpenAI's `gpt-4.1-mini`.

Execute the following `curl` command to create a model. Note the `Authorization` header, which uses the `admin_key` you just configured.

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
-d '{
  "name": "openai-gpt4-mini",
  "model": "openai/gpt-4.1-mini",
  "provider_config": {
    "api_key": "<YOUR_OPENAI_API_KEY>"
  }
}'
```

- Replace `<YOUR_OPENAI_API_KEY>` with your OpenAI API key.
- The `name` field (`openai-gpt4-mini`) is the identifier used in requests to AISIX.
- The `model` field (`openai/gpt-4.1-mini`) specifies the provider (`openai`) and the upstream model ID.

A successful request returns a JSON object confirming the creation, including a unique `key` (e.g., `/models/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).

## Step 5: Configure an API Key

Now, create an API key to authenticate your AI requests. This is also done via the Admin API.

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/apikeys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "my-secret-key",
    "allowed_models": ["openai-gpt4-mini"]
  }'
```

- `key` is the secret key used in the `Authorization` header.
- `allowed_models` is a list of model names this key is permitted to use.

## Step 6: Make Your First Request

You are now ready to make your first request through AISIX.

Use any OpenAI-compatible SDK or a `curl` command. Point your client to the AISIX Data Plane endpoint (`http://localhost:3000/v1`) and use your API key.

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer my-secret-key" \
-d '{
  "model": "openai-gpt4-mini",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a fun fact about AI"
    }
  ]
}'
```

If configured correctly, you will receive a standard OpenAI-compatible chat completion response from the upstream provider, proxied through AISIX.

You have successfully installed, configured, and used AISIX to proxy your first AI request.

## Example: Using Other Providers

Adding other providers like Google Gemini or Anthropic follows the same pattern. The key is to use the correct `model` prefix and provide the right credentials in the `provider_config`.

### Google Gemini

- **Model Prefix**: `gemini/`
- **Example**: `gemini/gemini-2.5-flash`

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gemini",
    "model": "gemini/gemini-2.5-flash",
    "provider_config": {
      "api_key": "YOUR_GEMINI_API_KEY"
    }
  }'
```

### Anthropic

- **Model Prefix**: `anthropic/`
- **Example**: `anthropic/claude-3-5-sonnet-20241022`

```bash
# Create an Anthropic Model
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-anthropic",
    "model": "anthropic/claude-3-5-sonnet-20241022",
    "provider_config": {
      "api_key": "YOUR_ANTHROPIC_API_KEY"
    }
  }'
```

After creating the model, update your API key to grant access to it, and you can start sending requests.



## Next Steps

- **Explore More Providers**: Add other supported providers like **DeepSeek**. The process is the same: create a new model with the `deepseek/deepseek-chat` prefix and your DeepSeek API key, then update your API key to allow access.
- **Configure Rate Limiting**: Add rate limits to your models and API keys to control costs and prevent abuse.
- **Admin UI**: AISIX includes a built-in Admin UI and Chat Playground. See [Admin UI](./admin-ui.md) for build and setup instructions.
