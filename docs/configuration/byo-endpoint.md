---
title: Bring Your Own Endpoint
description: Route AISIX AI Gateway to a private or self-hosted OpenAI-compatible endpoint such as vLLM, SGLang, or Ollama, including custom per-token pricing for budget tracking.
toc_max_heading_level: 2
sidebar_position: 42
keywords:
  - AISIX AI Gateway
  - bring your own endpoint
  - vLLM
  - SGLang
  - Ollama
  - OpenAI-compatible API
---

A bring-your-own (BYO) endpoint is any OpenAI-compatible HTTP server you operate yourself: a [vLLM](https://docs.vllm.ai/) or [SGLang](https://docs.sglang.ai/) inference server, an [Ollama](https://ollama.com/) host, or a self-hosted proxy in front of your own models. Register it with AISIX AI Gateway so callers reach it through the same OpenAI-compatible proxy API and the same caller API keys as any catalog provider.

Set `adapter` to `openai` for BYO endpoints. AISIX sends a standard `POST /chat/completions` request to your endpoint and returns an OpenAI-compatible chat-completions response to the caller.

## When to Use

Use a BYO endpoint when you run an inference server that exposes the OpenAI chat-completions API, such as vLLM, SGLang, Ollama, another OpenAI-compatible proxy, or your own service. This lets a private or air-gapped model share the gateway's auth, allowlist, rate limiting, and usage accounting.

Do not use this path for AWS Bedrock, Google Vertex AI, or Azure OpenAI. Those providers use native APIs and dedicated guides: [AWS Bedrock](../integration/upstream-bedrock.md), [Google Vertex AI](../integration/upstream-vertex.md), and [Azure OpenAI](../integration/upstream-azure-openai.md).

## Required Resources

A BYO endpoint is configured through a provider key and a model. The [provider key](provider-keys.md) holds the endpoint credential (`secret`) and base URL (`api_base`). The direct [model](models.md) maps a caller-facing alias (`display_name`) to the upstream model ID (`model_name`) and references the provider key.

Set `api_base` explicitly for every BYO endpoint. AISIX only has a built-in default for the OpenAI provider itself; it does not guess the base URL for a private vLLM, SGLang, Ollama, or private proxy endpoint. A BYO provider key without `api_base` fails before it can reach the upstream.

For private or self-hosted endpoints, set `api_base` to the root where your server serves the OpenAI-compatible API. If your server serves the OpenAI API at `http://10.0.0.5:8000/v1`, set exactly that. See [Provider Keys](provider-keys.md#configure-the-base-url) for base URL guidance and [Base URL Normalization](provider-keys.md#base-url-normalization) for the normalization rules.

## Prerequisites

Before you start, run the gateway with admin on `:3001` and proxy on `:3000`, prepare your admin key from the bootstrap config, and make sure the OpenAI-compatible endpoint is reachable. The examples below assume vLLM at `http://10.0.0.5:8000/v1` serving the model ID `meta-llama/Llama-3.1-8B-Instruct`.

For common OpenAI-compatible inference servers, use the `/v1` form: `http://host:8000/v1` for vLLM, `http://host:30000/v1` for SGLang, and `http://host:11434/v1` for Ollama. AISIX appends `/chat/completions` to the configured base URL.

## Configure the BYO Endpoint

Create a provider key, model alias, and caller API key. Together, these resources let callers send an AISIX model alias while the gateway sends the served model ID and endpoint credential upstream.

### Create a Provider Key

Many self-hosted inference servers do not require an API key. The `secret` field is required by the schema, so use a non-empty placeholder when your endpoint is unauthenticated. AISIX sends it as the bearer token, and your server can ignore it.

:::warning Production Credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` from [`config.yaml`](bootstrap-config.md). For production, protect etcd with encryption at rest and restricted network access, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md).
:::

Create a provider key for the private OpenAI-compatible endpoint:

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/provider_keys" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "vllm-private",
    "provider": "vllm",
    "adapter": "openai",
    "secret": "not-used-by-vllm",
    "api_base": "http://10.0.0.5:8000/v1"
  }'
```

Use any short `provider` label that makes sense for your environment, such as `vllm`, `sglang`, `ollama`, or `private-proxy`. Do not set `provider` to `openai` unless the upstream is actually OpenAI; that label can allow fallback to the public OpenAI host if `api_base` is removed.

Set `adapter` to `openai` for a BYO OpenAI-compatible endpoint. Set `api_base` to the endpoint root your server expects, including `/v1` when that is part of the server's OpenAI-compatible route.

Save the returned provider key `id` for the model resource.

### Create a Model

Map a caller-facing alias to the upstream model ID your endpoint serves.

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/models" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "llama-3-private",
    "provider": "vllm",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID",
    "cost": {
      "input_per_1k": 0.0,
      "output_per_1k": 0.0
    }
  }'
```

`display_name` is the alias callers send in `model` and the value `response.model` echoes back. `model_name` is the upstream id your endpoint expects; for vLLM and SGLang this is the served model name, and for Ollama it is the local model tag, such as `llama3.1:8b`. `cost` is optional; see [Pricing Metadata](#pricing-metadata).

### Create a Caller API Key

AISIX stores `key_hash`, not the plaintext caller key. Hash a plaintext caller key, then create the key resource scoped to the new alias.

```shell
if command -v sha256sum >/dev/null 2>&1; then
  printf '%s' 'sk-demo-caller' | sha256sum | cut -d' ' -f1
else
  printf '%s' 'sk-demo-caller' | shasum -a 256 | awk '{print $1}'
fi
```

Create an API key resource with the hash from the previous command:

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/apikeys" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["llama-3-private"]
  }'
```

### Pricing Metadata

Catalog providers carry pricing from the models.dev catalog. A BYO endpoint is not in that catalog, so the gateway has no price for it unless you set one.

Attach a `cost` block to the model to enable per-token budget accounting:

```json
{
  "cost": {
    "input_per_1k": 0.10,
    "output_per_1k": 0.30
  }
}
```

Both values are in USD per 1,000 tokens. `input_per_1k` applies to prompt tokens and `output_per_1k` to completion tokens. The gateway multiplies each token count by its rate and sums them. Both fields are required when the `cost` block is present.

Standalone self-hosted deployments store `cost` metadata but do not enforce budget checks from that value at request time. Set `cost` on a BYO model so a managed deployment, or your own usage-event consumer, has the per-token rate available. See [Models](models.md#cost-metadata) and [Budgets](budgets.md).

## Send a Test Request

Admin API writes propagate to the proxy asynchronously. If the alias is not visible immediately, check configuration propagation and retry after the proxy has loaded the model alias.

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-private",
    "messages": [
      {"role": "user", "content": "Say hello from the private model."}
    ]
  }'
```

## Verify the Endpoint

After the test request succeeds, confirm the caller-facing alias and upstream endpoint.

```shell
curl -sS -X POST "http://127.0.0.1:3000/v1/chat/completions" \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-private","messages":[{"role":"user","content":"ping"}]}' \
  | grep -o '"model":"[^"]*"'
```

The output should be `"model":"llama-3-private"`, your caller-facing alias, not the upstream `meta-llama/Llama-3.1-8B-Instruct`. If the upstream ID appears instead, check that the request is using the AISIX proxy URL and that the caller key is allowed to use the `llama-3-private` alias.

Check the endpoint access log for a `POST /v1/chat/completions` entry from AISIX. If AISIX returns an upstream route or connection error, check `api_base`, the served model name, and endpoint reachability.

## Limitations

BYO is for OpenAI-compatible chat-completions endpoints. Endpoints with a different API format need a native adapter. See [Adapter Protocol Families](../reference/adapters.md).

Always set `api_base` for BYO. A non-`openai` provider key without `api_base` cannot route requests because AISIX has no endpoint host to call.

Standalone deployments record `cost` metadata but do not enforce budget checks from that value at request time. See [Pricing Metadata](#pricing-metadata).

## Related Reading

[Choose a Provider Upstream](../integration/provider-upstreams.md) compares setup paths. For the resources used here, see [Provider Keys](provider-keys.md), [Models](models.md), and [Adapter Protocol Families](../reference/adapters.md). For the caller-facing proxy API, see [OpenAI-compatible API](../integration/openai-compatible-api.md).
