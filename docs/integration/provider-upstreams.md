---
title: Choose a Provider Upstream
description: Choose the right upstream provider setup path for AISIX AI Gateway.
sidebar_position: 30
keywords:
  - AISIX AI Gateway
  - provider upstream
  - adapter
  - OpenAI-compatible provider
  - AWS Bedrock
  - Vertex AI
  - Azure OpenAI
---

A provider upstream is the model service AISIX AI Gateway calls after it authenticates a caller, resolves the requested model alias, and selects a provider key.

This guide helps you choose the right provider setup path and identify which values you need from the upstream platform before you add a new provider.

## Choose the setup path

Start from the upstream you operate or buy from.

| If your upstream is | Use this guide | Adapter |
| --- | --- | --- |
| A public OpenAI-compatible vendor such as DeepSeek, Groq, Mistral, Together.ai, Fireworks, or Perplexity | [OpenAI-compatible vendor upstream](upstream-openai-compat.md) | `openai` |
| A private OpenAI-compatible server such as vLLM, SGLang, Ollama, or an internal model proxy | [Bring your own endpoint](../configuration/byo-endpoint.md) | `openai` |
| AWS Bedrock Runtime | [AWS Bedrock upstream](upstream-bedrock.md) | `bedrock` |
| Google Vertex AI | [Google Vertex AI upstream](upstream-vertex.md) | `vertex` |
| Azure OpenAI Service | [Azure OpenAI upstream](upstream-azure-openai.md) | `azure-openai` |

Choose the adapter that matches the upstream wire shape, not the provider's marketing category. For example, DeepSeek and a vLLM server both use `adapter: "openai"` because AISIX sends them OpenAI-compatible chat-completions requests.

## What you configure

Every upstream setup creates the same three AISIX resources:

1. A provider key that stores the upstream credential, provider label, adapter, and optional base URL.
2. A model that maps the caller-facing alias to the upstream model or deployment id.
3. A caller API key that allows clients to use the alias.

The details differ by upstream family:

- **OpenAI-compatible vendor** — `secret` is the provider API key, `model_name` is the vendor model id, and `api_base` is required for non-OpenAI vendors unless a built-in default applies.
- **BYO OpenAI-compatible endpoint** — `secret` is the provider API key or a placeholder for an unauthenticated endpoint, `model_name` is the served model name or local model tag, and `api_base` is the endpoint root, including `/v1` when the server serves there.
- **Bedrock** — `secret` is a JSON AWS credential with region, `model_name` is the Bedrock model id or inference profile id, and `api_base` is only needed for a private endpoint or VPC endpoint override.
- **Vertex AI** — `secret` is a JSON GCP credential with project and region, `model_name` is the Vertex publisher model id, and `api_base` is only needed when routing through a proxy or private endpoint.
- **Azure OpenAI** — `secret` is either the resource api-key string or a JSON Entra ID credential, `model_name` is the Azure deployment name, and `api_base` is the resource host, bare resource name, or override URL.

## How a request uses the upstream

```text
client request -> caller API key -> model alias -> provider key -> adapter bridge -> upstream provider
```

The client sends the caller-facing alias in `model`. AISIX rewrites that value to the upstream `model_name` before dispatch and restores the alias in normalized chat responses.

## Before exposing an alias

Check these items before giving the caller key to an application team:

- The caller key's `allowed_models` includes the alias.
- `/v1/models` shows the alias for that caller key after configuration propagation.
- A test request returns `response.model` as the alias, not the upstream model id.
- The upstream platform's logs, metrics, or error responses show that the request reached the expected provider.
- The endpoint you plan to use is supported by that provider family. See [Provider compatibility](../reference/provider-compatibility.md).

## Next steps

- [OpenAI-compatible vendor upstream](upstream-openai-compat.md)
- [Bring your own endpoint](../configuration/byo-endpoint.md)
- [AWS Bedrock upstream](upstream-bedrock.md)
- [Google Vertex AI upstream](upstream-vertex.md)
- [Azure OpenAI upstream](upstream-azure-openai.md)
- [Adapter protocol families](../reference/adapters.md)
