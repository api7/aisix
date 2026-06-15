---
title: Provider key schema
description: Complete JSON schema reference for the AISIX AI Gateway ProviderKey resource — every field, type, validation rule, the adapter enum, telemetry tags, and the request/response runtime-config overrides.
sidebar_position: 67
keywords:
  - AISIX AI Gateway
  - ProviderKey
  - schema
  - adapter
  - runtime config
  - AI gateway
---

This page is the complete schema reference for the `ProviderKey` resource — the upstream credential container every direct [model](../configuration/models.md) references by `provider_key_id`. It documents every field, its type, whether it is required, the validation rule the admin API enforces, and the closed `adapter` enum. Use it as the lookup companion to the task-oriented [Provider keys](../configuration/provider-keys.md) guide.

The admin API validates every write against a JSON Schema (Draft 2020-12) before persisting. The top-level object is closed: unknown fields are rejected.

## Top-level fields

| Field | Type | Required | Validation | Description |
|---|---|---|---|---|
| `display_name` | string | yes | `minLength: 1`; unique within the gateway | Operator-facing label. Surfaces in the admin list view and dashboard. Duplicate values are rejected with `409`. |
| `secret` | string | yes | `minLength: 1` | Upstream provider credential. Stored as plaintext on the standalone path — see the production warning below. |
| `api_base` | string | no | any string | Override for the upstream base URL. The canonical form depends on the `adapter` — see [Provider keys § `api_base` behavior](../configuration/provider-keys.md#api_base-behavior). |
| `provider` | string | no | free-form string | Vendor identity (for example `openai`, `anthropic`, `deepseek`, `vllm`). First-tier dispatch key. Defaults to an empty string when omitted. |
| `adapter` | string (enum) | no | one of `openai`, `anthropic`, `bedrock`, `vertex`, `azure-openai` | Wire-shape protocol family. Second-tier dispatch key. See [the adapter enum](#the-adapter-enum). |
| `telemetry_tags` | object | no | closed object | Attribution metadata. See [telemetry_tags](#telemetry_tags). |
| `request` | object | no | closed object | Per-key request-shape overrides. See [request overrides](#request-overrides). |
| `response` | object | no | closed object | Per-key response-shape overrides. See [response overrides](#response-overrides). |
| `strip_headers` | array of string | no | array of strings | Inbound headers stripped before forwarding on the `passthrough` endpoint. Defaults to `["authorization", "cookie", "set-cookie", "x-api-key"]` when the field is absent. See [Passthrough](../integration/passthrough.md). |

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` from [`config.yaml`](../configuration/bootstrap-config.md). Anyone with read access to the etcd keyspace can read the credential. For production, front etcd with encryption-at-rest, restrict etcd network access to the gateway, or use AISIX Cloud's managed [Provider Key Rotation](../cloud/provider-key-rotation.md), where the secret stays in the control plane and only the projected `provider_key_id` reference reaches the data plane.
:::

:::note `provider` validation differs between resources
On a `ProviderKey`, `provider` is an unconstrained string. On a [`Model`](../configuration/models.md), `provider` is also free-form but additionally enforces `minLength: 1`, `maxLength: 64`, and the pattern `^[a-z0-9][a-z0-9._-]*$`. Neither field is a closed enum — only `adapter` is.
:::

## The adapter enum

`adapter` is the one closed enum on the resource. It pins the upstream wire shape the gateway encodes against. It serializes in `kebab-case`, so the Azure family is the wire string `azure-openai`.

| Value | Upstream wire shape | Integration guide |
|---|---|---|
| `openai` | OpenAI chat completions | [OpenAI-compatible vendor upstream](../integration/upstream-openai-compat.md), [BYO endpoint](../configuration/byo-endpoint.md) |
| `anthropic` | Anthropic Messages | [Anthropic Messages](../integration/anthropic-messages.md) |
| `bedrock` | AWS Bedrock Runtime (Converse + Anthropic `/invoke`) | [AWS Bedrock upstream](../integration/upstream-bedrock.md) |
| `vertex` | Google Vertex AI Gemini | [Google Vertex AI upstream](../integration/upstream-vertex.md) |
| `azure-openai` | Azure OpenAI Service | [Azure OpenAI upstream](../integration/upstream-azure-openai.md) |

Any string outside this set is rejected at write time. For how `provider` and `adapter` combine at dispatch, see [Adapter protocol families § How a model resolves to a bridge](adapters.md#how-a-model-resolves-to-a-bridge).

## telemetry_tags

Attribution metadata carried alongside the key. The object is closed (unknown keys rejected). All fields are optional.

| Field | Type | Validation | Description |
|---|---|---|---|
| `kind` | string (enum) | one of `catalog`, `byo` | Whether the key is a curated catalog provider or a bring-your-own endpoint. |
| `featured` | boolean | — | Whether the provider is surfaced in the dashboard's featured (ranked) list. Defaults to `false`. |
| `branded_provider` | string or null | — | Branded provider slug for catalog entries (for example `openai`, `deepseek`). Null for BYO. |
| `pk_label` | string or null | — | Operator-defined label for the key (for example `production`, `shared-test`). |
| `byo_label` | string or null | — | Operator-defined label for bring-your-own entries (for example an internal team name). |

```json title="Catalog telemetry tags"
{
  "telemetry_tags": {
    "kind": "catalog",
    "featured": true,
    "branded_provider": "deepseek",
    "pk_label": "production"
  }
}
```

```json title="Bring-your-own telemetry tags"
{
  "telemetry_tags": {
    "kind": "byo",
    "branded_provider": null,
    "byo_label": "platform-team"
  }
}
```

## request overrides

Per-key overrides applied to the outbound request body and headers. The object is closed. Each field maps to a primitive transform the gateway applies before dispatch.

| Field | Type | Validation | Description |
|---|---|---|---|
| `param_renames` | object (string → string) | values are strings | Top-level body keys named on the left are renamed to the right (for example `max_completion_tokens` → `max_tokens`). |
| `param_constraints` | object | closed; see below | Numeric clamps applied to the request body. |
| `default_headers` | object (string → string) | values are strings | Headers added to the outbound request when the caller did not set them. Reserved auth headers are dropped as defense-in-depth. |
| `default_body_fields` | object | free-form | Top-level body fields added when the caller did not set them (for example `safe_prompt`). |

`param_constraints` is a closed object with two fields:

| Field | Type | Description |
|---|---|---|
| `temperature_max` | number | Upper clamp for `temperature`. Values above are clamped down. |
| `temperature_min` | number | Lower clamp for `temperature`. Values below are clamped up. |

```json title="request overrides"
{
  "request": {
    "param_renames":      { "max_completion_tokens": "max_tokens" },
    "param_constraints":  { "temperature_max": 1.0 },
    "default_headers":    { "X-Foo": "bar" },
    "default_body_fields": { "safe_prompt": true }
  }
}
```

:::note Where overrides are applied
The `request` and `response` blocks are applied at dispatch by the **`openai` and `azure-openai`** bridges (the OpenAI-wire families): request `param_renames` / `param_constraints` / `default_headers` / `default_body_fields` are folded into the outbound call, and response `stream_done_marker` / `content_list_to_string` / `reasoning_field` shape how the upstream reply is interpreted. The `bedrock` and `vertex` bridges build their providers' native request shapes and do **not** apply these blocks today.

What is not yet shipped is the AISIX Cloud control-plane wiring that auto-populates these blocks from the dashboard, so in AISIX Cloud they are currently empty. A self-hosted operator who sets `request` / `response` directly through the admin API gets them applied on the `openai` / `azure-openai` paths.
:::

## response overrides

Per-key overrides describing how the gateway interprets the upstream response. The object is closed.

| Field | Type | Validation | Description |
|---|---|---|---|
| `stream_done_marker` | string (enum) | one of `required`, `optional`, `none` | The SSE `[DONE]` terminator expectation. `required` — upstream must emit `data: [DONE]`. `optional` — either is acceptable. `none` — upstream is expected to omit it. |
| `content_list_to_string` | boolean | — | When `true`, a `messages[*].content` array of text blocks is flattened to a single string before dispatch (for upstreams that only accept string content). Defaults to `false`. |
| `error_envelope` | string | open string | Error-translation strategy. The control-plane spec uses `openai` (project upstream errors into the OpenAI envelope) or `passthrough` (return the upstream body as-is). Validated as an open string today. |
| `reasoning_field` | string | — | Dotted path to lift an upstream reasoning field (for example `delta.reasoning_content`). |

```json title="response overrides"
{
  "response": {
    "stream_done_marker":     "required",
    "content_list_to_string": false,
    "error_envelope":         "openai",
    "reasoning_field":        "delta.reasoning_content"
  }
}
```

## Minimal and full examples

A `ProviderKey` requires only `display_name` and `secret`:

```json title="Minimal provider key"
{
  "display_name": "openai-prod",
  "secret": "YOUR_PROVIDER_API_KEY"
}
```

A fully populated catalog provider key:

```json title="Full provider key"
{
  "display_name": "deepseek-prod",
  "secret": "YOUR_PROVIDER_API_KEY",
  "api_base": "https://api.deepseek.com/v1",
  "provider": "deepseek",
  "adapter": "openai",
  "telemetry_tags": {
    "kind": "catalog",
    "featured": true,
    "branded_provider": "deepseek",
    "pk_label": "production"
  },
  "request": {
    "param_constraints": { "temperature_max": 1.0 }
  },
  "response": {
    "stream_done_marker": "required",
    "reasoning_field": "delta.reasoning_content"
  }
}
```

## Backward compatibility

Every field except `display_name` and `secret` is optional. Provider key payloads written before `provider`, `adapter`, `telemetry_tags`, `request`, `response`, or `strip_headers` existed continue to validate and load: the missing fields fall back to their defaults (`provider` to an empty string, `adapter` to absent, `telemetry_tags` to the all-default object, `strip_headers` to the four-header credential list).

## Related pages

- [Provider keys](../configuration/provider-keys.md) — the task-oriented configuration guide, including the `api_base` behavior table and tolerance rules.
- [Adapter protocol families](adapters.md) — how `provider` and `adapter` select a bridge.
- [Models](../configuration/models.md) — the resource that references a provider key by `provider_key_id`.
- [Resource schemas](resource-schemas.md) — schemas for the other admin-managed resources.
