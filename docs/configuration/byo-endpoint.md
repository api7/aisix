---
title: Bring your own endpoint
description: Route AISIX AI Gateway to a private or self-hosted OpenAI-compatible endpoint such as vLLM, SGLang, or Ollama, including custom per-token pricing for budget tracking.
sidebar_position: 42
keywords:
  - AISIX AI Gateway
  - bring your own endpoint
  - vLLM
  - SGLang
  - Ollama
  - OpenAI-compatible API
---

A bring-your-own (BYO) endpoint is any OpenAI-compatible HTTP server you operate yourself — a [vLLM](https://docs.vllm.ai/) or [SGLang](https://docs.sglang.ai/) inference server, an [Ollama](https://ollama.com/) host, or a self-hosted proxy in front of your own models. This page shows how to register one against AISIX AI Gateway so callers reach it through the same OpenAI-compatible proxy surface and the same caller API keys as any catalog provider.

A BYO endpoint uses the `openai` adapter family. The gateway sends a standard `POST /chat/completions` to your endpoint and renders the response back to the caller as an OpenAI chat-completions envelope.

## When to use this

- Use this when you run an inference server that exposes the OpenAI chat-completions API (vLLM, SGLang, Ollama, other OpenAI-compatible proxies, or your own service).
- Use this when you want a private or air-gapped model to share the gateway's auth, allowlist, rate limiting, and usage accounting.
- Do not use this for AWS Bedrock, Google Vertex AI, or Azure OpenAI — those have native wire shapes and dedicated guides: [AWS Bedrock](../integration/upstream-bedrock.md), [Google Vertex AI](../integration/upstream-vertex.md), [Azure OpenAI](../integration/upstream-azure-openai.md).

## How it works

A BYO endpoint is configured through two resources, exactly like a catalog upstream:

1. A [provider key](provider-keys.md) holding the endpoint's credential (`secret`) and its base URL (`api_base`).
2. A direct [model](models.md) that maps a caller-facing alias (`display_name`) to the upstream model id (`model_name`) and references the provider key.

The difference from a catalog provider is that **you set `api_base` explicitly**. The OpenAI-family bridge only falls back to `https://api.openai.com` when the provider key's vendor identity is `openai` (or empty). For any other vendor it refuses to guess a base URL and returns an error, so a BYO key without `api_base` will fail dispatch. Set `api_base` to your endpoint's root.

Outside the canonical OpenAI host, the gateway trusts your `api_base` verbatim — it appends the endpoint path (`/chat/completions`) but does not synthesize a `/v1` segment. If your server serves the OpenAI API at `http://10.0.0.5:8000/v1`, set exactly that. See [Provider keys § `api_base` behavior](provider-keys.md#api_base-behavior) for the full normalization rules.

## Prerequisites

- A running self-hosted gateway (admin on `:3001`, proxy on `:3000`). See the [Self-Hosted Quickstart](../quickstart/self-hosted.md).
- Your admin key from the bootstrap config.
- A reachable OpenAI-compatible endpoint. The examples below assume vLLM at `http://10.0.0.5:8000/v1` serving the model id `meta-llama/Llama-3.1-8B-Instruct`.

:::note vLLM, SGLang, and Ollama base URLs
- **vLLM** serves the OpenAI API under `/v1` (for example `http://host:8000/v1`).
- **SGLang** also serves under `/v1` (for example `http://host:30000/v1`).
- **Ollama** serves its OpenAI-compatible surface under `/v1` (for example `http://host:11434/v1`).

Use the `/v1` form for all three. The gateway appends `/chat/completions` to whatever you supply.
:::

## Configuration

### Step 1: Create the provider key

Many self-hosted inference servers do not require an API key. The `secret` field is required by the schema, so use a non-empty placeholder when your endpoint is unauthenticated — the bridge sends it as the bearer token and your server ignores it.

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` from [`config.yaml`](bootstrap-config.md). For production, front etcd with encryption-at-rest, restrict etcd network access to the gateway, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md), where the secret stays in the control plane and only the projected reference reaches the data plane.
:::

```bash title="Create a BYO provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "vllm-internal",
    "provider": "vllm",
    "adapter": "openai",
    "secret": "not-used-by-vllm",
    "api_base": "http://10.0.0.5:8000/v1"
  }'
```

Field notes:

- `provider` is a free-form vendor identity. Use any short label (`vllm`, `sglang`, `ollama`, `internal-proxy`); it is for your own readability and metrics. It must not be `openai` unless you genuinely point at OpenAI, because that would let the bridge fall back to the public OpenAI host if `api_base` were ever cleared.
- `adapter` pins the wire shape to `openai`. For a BYO OpenAI-compatible endpoint this is the only valid value.
- `api_base` is required and trusted verbatim.

Capture the returned `id` for the next step. The admin API returns a `ResourceEntry` with an `id` field; the [first-request quickstart](../quickstart/first-model-first-key-first-request.md#step-1-create-a-provider-key) shows a `jq`-capturing one-liner if you want to script it.

### Step 2: Create the model

Map a caller-facing alias to the upstream model id your endpoint serves.

```bash title="Create a model for the BYO endpoint"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "llama-3-internal",
    "provider": "vllm",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID",
    "cost": {
      "input_per_1k": 0.0,
      "output_per_1k": 0.0
    }
  }'
```

- `display_name` is the alias callers send in `model` and the value `response.model` echoes back.
- `model_name` is the upstream id your endpoint expects — for vLLM and SGLang this is the served model name; for Ollama it is the local model tag (for example `llama3.1:8b`).
- `cost` is optional; see [BYO pricing](#byo-pricing) below.

### Step 3: Create a caller API key

The data plane stores `key_hash`, not plaintext. Hash a plaintext caller key, then create the key resource scoped to your new alias.

```bash title="Hash a plaintext caller key"
printf 'sk-demo-caller' | sha256sum | cut -d' ' -f1
```

```bash title="Create a caller API key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/apikeys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_hash": "YOUR_CALLER_KEY_HASH",
    "allowed_models": ["llama-3-internal"]
  }'
```

### Step 4: Send a request

Admin writes propagate to the proxy asynchronously; allow about a second, or poll `/v1/models` until the alias appears.

```bash title="Send a chat completion to the BYO endpoint"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-internal",
    "messages": [
      {"role": "user", "content": "Say hello from the internal model."}
    ]
  }'
```

## BYO pricing

Catalog providers carry pricing from the models.dev catalog. A BYO endpoint is not in that catalog, so the gateway has no price for it unless you set one. Attach a `cost` block to the model to enable per-token budget accounting:

```json title="Model cost block"
{
  "cost": {
    "input_per_1k": 0.10,
    "output_per_1k": 0.30
  }
}
```

Both values are in USD per 1,000 tokens. `input_per_1k` applies to prompt tokens and `output_per_1k` to completion tokens; the gateway multiplies each token count by its rate and sums them. Both fields are required when the `cost` block is present.

:::note Pricing is enforced in AISIX Cloud, not standalone
The standalone OSS proxy stores `cost` but does not consult it at request time — it always emits `cost_usd=0.0`. Pricing-aware budget enforcement runs through the AISIX Cloud control plane, which reads the `cost` block when emitting usage events. Set `cost` on a BYO model so that a future managed deployment, or your own usage-event consumer, has the per-token rate available. See [Models § field notes](models.md#field-notes) and [Budgets](budgets.md).
:::

## Verification

A `200` alone does not prove the gateway reached your endpoint and applied the alias contract. Verify the two observable facts that do.

### The alias is restored on `response.model`

```bash title="Confirm response.model echoes your alias"
curl -sS -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-internal","messages":[{"role":"user","content":"ping"}]}' \
  | grep -o '"model":"[^"]*"'
```

Expected: `"model":"llama-3-internal"` — your caller-facing alias, **not** the upstream `meta-llama/Llama-3.1-8B-Instruct`. This proves the request resolved through your model and the gateway restored the alias on the way out. If you see the upstream id instead, the request did not flow through the gateway's render path.

### The endpoint actually received the request

Confirm the request reached your server, not the public OpenAI host. Check your endpoint's access log for a `POST /v1/chat/completions` entry, or temporarily point `api_base` at an unreachable host and confirm the gateway returns an upstream error rather than a `200`:

```bash title="Negative check — unreachable endpoint surfaces an upstream error"
curl -sS -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-caller" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-internal","messages":[{"role":"user","content":"ping"}]}'
```

With a healthy endpoint, expect `200`. With `api_base` pointing at a dead host, expect a `5xx` upstream error — confirming dispatch targets your `api_base` and not a default.

## Limitations

- BYO is for OpenAI-compatible chat-completions endpoints. Endpoints with a non-OpenAI wire shape need a native adapter — see [Adapter protocol families](../reference/adapters.md).
- A missing `api_base` on a non-`openai` vendor fails dispatch with a configuration error. Always set `api_base` for BYO.
- Standalone deployments record but do not enforce `cost`; see [BYO pricing](#byo-pricing).

## Related pages

- [Provider keys](provider-keys.md) — the credential resource and the full `api_base` normalization rules.
- [Models](models.md) — direct and routing model configuration, including the `cost` block.
- [Adapter protocol families](../reference/adapters.md) — why a BYO OpenAI-compatible endpoint uses the `openai` adapter.
- [OpenAI-compatible API](../integration/openai-compatible-api.md) — the proxy surface callers use to reach the endpoint.
