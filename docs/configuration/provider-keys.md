---
title: Provider Keys
description: Configure upstream provider credentials and base URLs for AISIX AI Gateway models.
sidebar_position: 32
---

Use provider keys to store upstream credentials and endpoint settings outside
of caller-facing model aliases.

A direct [model](models.md) references a provider key by `provider_key_id`.
This lets you reuse one upstream credential across multiple models and rotate
that credential without recreating every alias.

## Prerequisites

This page assumes you have:

- a running self-hosted gateway with the admin listener available
- an admin key for `Authorization: Bearer YOUR_ADMIN_KEY`
- an upstream provider credential or private endpoint credential

Provider keys are operator-managed secrets. Decide who owns the upstream
credential before sharing the key across many models.

## Create a provider key

```shell
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

Use the returned `id` as the model's `provider_key_id`.

```json
{
  "display_name": "gpt-4o-prod",
  "provider": "openai",
  "model_name": "gpt-4o",
  "provider_key_id": "PROVIDER_KEY_ID_FROM_ADMIN_API"
}
```

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix`
configured in [`config.yaml`](bootstrap-config.md). Anyone with read access to
that etcd keyspace can read the credential. In production, restrict etcd
network access, use encryption at rest where available, and keep the
gateway-to-etcd channel inside the trusted infrastructure boundary.
:::

## Verify

Confirm the admin API can read the provider key:

```shell
curl -sS http://127.0.0.1:3001/admin/v1/provider_keys \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

The provider key is not useful by itself on the proxy surface. To verify
end-to-end traffic, attach it to a [model](models.md), allow that model on a
caller [API key](api-keys.md), and send a proxy request.

## Understand provider and adapter

`provider` and `adapter` are separate fields:

| Field | Purpose | Example values |
| --- | --- | --- |
| `provider` | Identifies the upstream vendor or endpoint. It is an open string. | `openai`, `anthropic`, `deepseek`, `openrouter`, `internal-vllm` |
| `adapter` | Identifies the upstream wire shape. It is a closed enum because AISIX can only encode implemented protocol families. | `openai`, `anthropic`, `bedrock`, `vertex`, `azure-openai` |

At dispatch time, the gateway tries a provider-specific bridge first. If no
provider-specific bridge is registered, it falls back to the adapter-family
bridge. This is why a long-tail OpenAI-compatible provider can use
`adapter: "openai"` with its own `provider` and `api_base`.

For the dispatch model, see [Adapter protocol families](../reference/adapters.md).

## Choose an `api_base`

`api_base` controls where the gateway sends upstream requests.

The safest rule is: configure the base URL exactly as the selected adapter
expects. The gateway tolerates common copy-paste forms, but it does not try to
guess arbitrary provider URL layouts.

Use these common patterns:

| Upstream | `adapter` | `api_base` pattern |
| --- | --- | --- |
| OpenAI | `openai` | `https://api.openai.com/v1` |
| DeepSeek | `openai` | `https://api.deepseek.com` |
| Gemini OpenAI-compatible API | `openai` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Anthropic | `anthropic` | `https://api.anthropic.com` |
| Azure OpenAI | `azure-openai` | `https://<resource>.openai.azure.com` |
| AWS Bedrock | `bedrock` | usually unset |
| Google Vertex AI | `vertex` | `https://<region>-aiplatform.googleapis.com` |

For OpenAI itself:

```json
{
  "provider": "openai",
  "adapter": "openai",
  "api_base": "https://api.openai.com/v1"
}
```

For DeepSeek's OpenAI-compatible API:

```json
{
  "provider": "deepseek",
  "adapter": "openai",
  "api_base": "https://api.deepseek.com"
}
```

For Gemini's OpenAI-compatible API:

```json
{
  "provider": "google",
  "adapter": "openai",
  "api_base": "https://generativelanguage.googleapis.com/v1beta/openai"
}
```

For Anthropic, the bridge appends `/v1/messages`.

```json
{
  "provider": "anthropic",
  "adapter": "anthropic",
  "api_base": "https://api.anthropic.com"
}
```

For Azure OpenAI, a bare resource name is also accepted. The bridge builds the deployment URL from the selected model name and API version.

```json
{
  "provider": "azure",
  "adapter": "azure-openai",
  "api_base": "https://example-resource.openai.azure.com"
}
```

For Bedrock, `api_base` is usually unset. The region comes from the Bedrock credential JSON in `secret`, and the gateway builds the Bedrock Runtime endpoint. Set `api_base` only when routing through a private or custom Bedrock endpoint.

For Vertex AI, project, region, and token endpoint details come from the service-account JSON in `secret`.

```json
{
  "provider": "google-vertex",
  "adapter": "vertex",
  "api_base": "https://us-central1-aiplatform.googleapis.com"
}
```

## URL normalization

The gateway normalizes common `api_base` paste mistakes:

- leading and trailing whitespace is trimmed
- trailing slashes are removed
- full endpoint URLs such as `/chat/completions`, `/embeddings`, `/v1/messages`,
  and `/audio/transcriptions` are reduced to the expected base URL
- OpenAI's canonical host can be supplied with or without `/v1`
- DeepSeek's canonical host tolerates an accidental `/v1`
- Anthropic tolerates bare host, `/v1`, and `/v1/messages`

This tolerance is conservative. For corporate proxies, private gateways, or
non-canonical paths, the gateway preserves the operator's chosen base after the
basic trimming and endpoint-suffix cleanup. It does not invent `/v1` for an
unknown host.

## Passthrough header stripping

The passthrough endpoint uses the selected provider key to add the upstream
credential. To avoid forwarding caller credentials to the upstream provider,
the gateway strips these headers by default:

```text
authorization
cookie
set-cookie
x-api-key
```

`strip_headers` lets you customize the list. Entries are trimmed, lowercased,
deduplicated, and empty entries are dropped on load. Hop-by-hop headers such as
`host` and `content-length` are stripped separately and cannot be re-enabled
through `strip_headers`.

Only change this field when you have a concrete forwarding requirement and
accept the credential-leak risk. For endpoint behavior, see
[Passthrough](../integration/passthrough.md).

## Runtime compatibility overrides

Provider keys can include optional `request` and `response` override blocks for
provider compatibility. Examples include parameter renames, temperature clamps,
default outbound headers, default body fields, content-list flattening, stream
done-marker policy, and reasoning-field extraction.

These blocks are advanced compatibility knobs. The generated schema is the
contract for the accepted shape, but each provider bridge decides which
overrides apply to its wire path. Verify bridge behavior before relying on an
override for a specific provider family.

For exact fields and source-of-truth guidance, see
[Provider key schema](../reference/provider-key-schema).

## Operational guidance

Provider keys are shared dependencies. Rotating one provider key affects every
model that references it.

Duplicate `display_name` values are rejected with `409`.

A model that points at a newly written provider key can fail temporarily until
the proxy snapshot sees both resources. See
[Configuration propagation](configuration-propagation.md).

## Troubleshooting

### Requests fail after changing `api_base`

Treat this first as an upstream URL construction issue. Confirm the provider
key's `provider`, `adapter`, and `api_base` match the upstream protocol family.

### Several models fail after provider-key rotation

Check whether they share the same provider key. If they do, the rotated key is
the common dependency.

### A bring-your-own OpenAI-compatible endpoint fails without `api_base`

Configure `api_base` explicitly. The OpenAI-family bridge refuses to fall back
to `api.openai.com` for non-OpenAI provider identities.

## Next steps

- [Models](models.md) attaches provider keys to caller-facing aliases.
- [Bring your own endpoint](byo-endpoint.md) points the `openai` adapter at a private or self-hosted endpoint.
- [OpenAI-compatible vendor upstream](../integration/upstream-openai-compat.md) onboards public OpenAI-compatible providers.
- [AWS Bedrock upstream](../integration/upstream-bedrock.md), [Google Vertex AI upstream](../integration/upstream-vertex.md), and [Azure OpenAI upstream](../integration/upstream-azure-openai.md) cover specialized provider families.
- [Provider key schema](../reference/provider-key-schema) explains the generated schema source of truth.
