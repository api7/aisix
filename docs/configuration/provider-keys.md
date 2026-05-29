---
title: Provider Keys
description: Configure upstream provider credentials and base URLs for AISIX AI Gateway models.
sidebar_position: 33
---

Provider keys store upstream credentials that one or more models can reuse.

Use a provider key when you want to:

- store one upstream API key once
- reuse it across multiple models
- rotate upstream credentials without recreating every model

Think of a provider key as the upstream credential container, not the client-facing contract.

## Current Fields

- `display_name`
- `secret`
- optional `api_base`
- optional `provider`
- optional `adapter`
- optional `telemetry_tags`

In practice:

- `display_name` is for operator readability
- `secret` is the actual upstream credential used at dispatch time
- `api_base` is how you override the provider's default endpoint root
- `provider` is the upstream vendor identity (a free-form lowercase label such as `openai`, `anthropic`, `deepseek`, `vllm`); it is the first-tier [dispatch key](../reference/adapters.md#how-a-model-resolves-to-a-bridge) and a metrics label
- `adapter` pins the upstream wire shape to one of `openai`, `anthropic`, `bedrock`, `vertex`, `azure-openai`; it is the second-tier dispatch key that routes Bedrock, Vertex, Azure OpenAI, and long-tail OpenAI-compatible vendors to the right bridge
- `telemetry_tags` carries attribution metadata (`kind` of `catalog` or `byo`, plus optional labels); it is populated by AISIX Cloud and is not required for self-hosted use

For the complete field reference — every type, validation rule, and the `request`/`response` runtime-config overrides — see the [Provider key schema](../reference/runtime-config-schema.md).

Example:

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` configured in [`config.yaml`](bootstrap-config.md). Anyone with read access to the etcd keyspace can read the credential. For production, front etcd with encryption-at-rest, restrict etcd network access to the gateway, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md), where the secret stays in the control plane and only the projected `provider_key_id` reference reaches the data plane.
:::

```bash title="Create a provider key"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "openai-prod",
    "provider": "openai",
    "adapter": "openai",
    "secret": "YOUR_PROVIDER_API_KEY",
    "api_base": "https://api.openai.com/v1"
  }'
```

## `api_base` Behavior

`api_base` overrides the provider's default upstream base URL. Each provider bridge appends a different path at request time, so the canonical form `api_base` should take depends on which `provider` your model selects.

Each provider has its own convention — the bridges do **not** share one. Use the table below; do not generalize from one row to another. The first rows are keyed on the vendor `provider`; the last two rows are keyed on the `adapter` family (Bedrock and Azure OpenAI dispatch by adapter, not by a fixed vendor string). For how `provider` and `adapter` select a bridge, see [Adapter protocol families](../reference/adapters.md).

| `provider` / `adapter` | Canonical `api_base` form | Bridge appends | Default if `api_base` is omitted |
|---|---|---|---|
| `openai` | include `/v1` | `/chat/completions`, `/embeddings`, `/completions`, `/images/generations`, `/audio/*` | `https://api.openai.com/v1` |
| `deepseek` | bare host (DeepSeek serves OpenAI-compatible paths at the host root) | `/chat/completions` | `https://api.deepseek.com` |
| `google` | host plus the OpenAI-compat prefix `/v1beta/openai` | `/chat/completions` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `anthropic` | bare host | `/v1/messages` | `https://api.anthropic.com` |
| `google-vertex` | bare host, no path | `/v1/projects/<project>/locations/<region>/publishers/google/models/<model>:generateContent` (non-streaming) or `:streamGenerateContent?alt=sse` (streaming). `<project>` and `<region>` come from the SA JSON inside `secret`. | `https://<region>-aiplatform.googleapis.com` |
| `bedrock` (adapter) | `api_base` usually **unset** | `/model/<model>/converse` or the Anthropic `/invoke` route, SigV4-signed | Region-keyed `bedrock-runtime.<region>.amazonaws.com`; the region comes from the `region` field in the credential JSON inside `secret`, not from `api_base`. Set `api_base` only for a private (VPC) Bedrock endpoint. |
| `azure-openai` (adapter) | the resource host `https://<resource>.openai.azure.com` (a bare resource name is also accepted) | `/openai/deployments/<deployment>/chat/completions?api-version=<version>` | No default — `api_base` is required and supplies the resource host. A verbatim override host that does not end in `.openai.azure.com` is trusted as-is for a corporate proxy or mock. |

The OpenAI and Anthropic conventions match each upstream's official SDK — `openai-python` initialises `base_url = "https://api.openai.com/v1"`, while `anthropic-sdk-python` initialises `base_url = "https://api.anthropic.com"` and appends `/v1/messages` itself. DeepSeek is OpenAI-compatible but exposes `/chat/completions` directly at the host root, and Google's Gemini OpenAI-compatible surface lives under a fixed `/v1beta/openai` prefix that the bridge does not synthesize. The Vertex bridge appends a parameterized URL of the form `/v1/projects/<project>/locations/<region>/publishers/google/models/<model>:generateContent`, so the canonical `api_base` form for `google-vertex` is the bare host root — operators behind a corporate proxy or air-gapped network point `api_base` at their proxy host, and the bridge tacks on the rest. Note that OAuth token minting still hits `secret.token_uri` (controlled by the SA JSON, not `api_base`); operators behind a fully air-gapped network must additionally point `token_uri` at their internal token endpoint.

### Forms the gateway tolerates

The gateway accepts several common operator paste-mistakes and normalizes them to the canonical form. Tolerance is intentionally conservative — `/v1` synthesis and stripping happens **only for the canonical upstream host of each provider**. Corporate proxies and any non-default path the operator chose on purpose pass through unchanged.

All providers (always tolerated):

- leading and trailing whitespace
- trailing slash
- accidental endpoint suffix appended to the URL — e.g. pasting `https://api.openai.com/v1/chat/completions` works the same as `https://api.openai.com/v1`. The same applies to `/embeddings`, `/completions`, `/images/generations`, and `/audio/*` for OpenAI-compat bridges, and to `/v1/messages` and `/v1` for the Anthropic bridge.

For `openai` on the canonical OpenAI host only:

- pasting the bare host without `/v1` — `api_base: "https://api.openai.com"` is accepted and the bridge adds `/v1` at dispatch time.

For `deepseek` on the canonical DeepSeek host only:

- pasting an extra `/v1` segment — `api_base: "https://api.deepseek.com/v1"` is accepted and the bridge strips `/v1` at dispatch time. Common copy-paste habit from OpenAI conventions.

For `anthropic` (always tolerated):

- the full upstream URL `…/v1/messages` or the `/v1` prefix is stripped — `api_base: "https://api.anthropic.com/v1/messages"`, `…/v1`, and bare host all converge to the bare host at dispatch time.

For `google` and any other variant: only suffix stripping; the bridge does not synthesize the `/v1beta/openai` prefix. Operators should paste the full canonical form.

### Outside the canonical hosts

For corporate proxies or alternative deployments, the operator's `api_base` is trusted verbatim (after suffix stripping and trailing-slash trim). If your proxy serves the OpenAI API at `https://my-proxy.example.com/openai-shim`, set `api_base` to exactly that — the bridge will not unilaterally add `/v1`.

## Reuse Model References

Models reference provider keys by `provider_key_id`, not by `display_name`.

Typical flow:

1. create one `ProviderKey`
2. create one or more `Model` rows that point at its returned `id`
3. rotate the provider key later with `PUT /admin/v1/provider_keys/:id`

This is the main reason to avoid embedding provider credentials conceptually into each model row.

## Operational Notes

- `secret` is stored as plaintext in the standalone gateway path.
- Duplicate `display_name` values are rejected with `409`.
- A model that points at a provider key not yet visible in the proxy snapshot can temporarily fail dispatch until propagation completes.

## Troubleshooting

### Requests fail after changing `api_base`

Treat that first as an upstream endpoint construction issue, not as a caller-key or model-auth issue.

### Several models fail at once after provider-key rotation

That is expected if they all share the same provider key. The shared key is the common dependency.

## Related Pages

- [Models](models.md)
- [Provider key schema](../reference/runtime-config-schema.md) — the complete field reference, including `request`/`response` overrides.
- [Adapter protocol families](../reference/adapters.md) — how `provider` and `adapter` select a bridge.
- [Bring your own endpoint](byo-endpoint.md) — point the `openai` adapter at a private or self-hosted endpoint.
- [OpenAI-compatible vendor upstream](../integration/upstream-openai-compat.md) — onboard a public OpenAI-compatible vendor (DeepSeek, Groq, Mistral).
- [AWS Bedrock upstream](../integration/upstream-bedrock.md), [Google Vertex AI upstream](../integration/upstream-vertex.md), [Azure OpenAI upstream](../integration/upstream-azure-openai.md) — the specialized-family guides.
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Configuration Propagation](configuration-propagation.md)
