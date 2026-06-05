---
title: Provider Key Schema
description: Understand ProviderKey fields and runtime behavior around provider keys.
sidebar_position: 67
keywords:
  - AISIX AI Gateway
  - ProviderKey
  - schema
  - adapter
  - runtime config
  - AI gateway
---

A `ProviderKey` stores the upstream credential and connection details that a [model](../configuration/models.md) uses when AISIX sends traffic to an AI provider.

The [Admin API reference](/ai-gateway/reference/admin-api) contains the exact request body, response body, defaults, enum values, and validation errors. Provider-key fields are grouped by credential storage, upstream routing, compatibility overrides, passthrough protection, and metadata.

For the configuration workflow, see [Provider keys](../configuration/provider-keys.md).

## Schema Reference

The standalone JSON Schema for provider keys is available at:

```text
schemas/resources/provider_key.schema.json
```

The admin OpenAPI document includes the same provider-key schema:

```text
/ai-gateway/reference/admin-api
```

When you run a self-hosted gateway locally, you can also open the live Scalar reference from the admin listener:

```text
http://127.0.0.1:3001/admin/openapi-scalar
```

For accepted request and response schemas, use the Admin API reference.

Provider keys are closed resources. Unknown top-level fields are rejected.

The required fields are:

| Field | Description |
| --- | --- |
| `display_name` | Display name for the provider key. |
| `secret` | Credential AISIX uses when it calls the upstream provider. |

The provider-key schema also defines optional fields for:

| Field group | Fields |
| --- | --- |
| Upstream routing identity | `provider`, `adapter`, and `api_base` |
| Request and response compatibility overrides | `request` and `response` |
| Passthrough credential protection | `strip_headers` |
| Attribution metadata | `telemetry_tags` |

Use the Admin API reference for complete field schemas.

:::warning Production Credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix` configured in [`config.yaml`](../configuration/bootstrap-config.md). Anyone with read access to that etcd keyspace can read the credential. In production, restrict etcd network access, use encryption at rest where available, and keep the gateway-to-etcd channel inside trusted infrastructure.
:::

## Provider and Adapter Fields

`provider` and `adapter` are related, but they do different jobs.

`provider` is the vendor or endpoint identity, such as `openai`, `anthropic`, `deepseek`, or a bring-your-own provider label. It is an open string so AISIX can route additional provider identities without adding every vendor name to the gateway runtime.

`adapter` is the upstream protocol family AISIX knows how to encode. Supported values include `openai`, `anthropic`, `bedrock`, `vertex`, and `azure-openai`.

At request time, AISIX first tries provider-specific request handling keyed by `provider`. If that lookup does not match, AISIX uses the adapter family selected by `adapter`. This is why an OpenAI-compatible vendor can use its own `provider` value while still using `adapter: "openai"`.

For the request model, see [Adapter protocol families](adapters.md).

`api_base` controls the upstream endpoint root.

Some built-in provider identities can infer a default base URL. For example, an OpenAI provider key can fall back to the OpenAI base URL, and an Anthropic provider key can fall back to the Anthropic base URL.

For other providers, bring-your-own endpoints, private gateways, and most catalog projections, set `api_base` explicitly. AISIX does not guess a default base URL for a different provider identity because that could send a credential to the wrong upstream.

The configuration guide has the practical examples: [Provider Keys](../configuration/provider-keys.md#configure-the-base-url).

## Runtime Overrides

Provider keys accept optional `request` and `response` override blocks. These blocks describe compatibility settings such as parameter renames, temperature clamps, default outbound headers, default outbound body fields, content-list flattening, stream `[DONE]` marker policy, and reasoning-field extraction.

The accepted configuration schema is separate from endpoint behavior. Not every adapter or proxy endpoint applies every override in the same way.

Adapter and endpoint behavior is runtime-specific:

| Runtime path | Override behavior |
| --- | --- |
| OpenAI-family and Azure OpenAI chat paths | Apply request-body, header, and selected response override behavior. |
| Vertex publisher rails | Apply the shared request-body override pipeline before Vertex-specific shaping. |
| Anthropic `/v1/messages` and `/v1/messages/count_tokens` paths | Apply request-side overrides to their outbound provider request. |
| Passthrough and provider-native forwarding paths | May bypass normalized adapter behavior. |

When an override matters for a provider family, confirm the behavior in the relevant integration guide and validate it in your deployment before relying on it in production.

## Passthrough and Telemetry Fields

`strip_headers` controls which inbound headers the passthrough endpoint removes before forwarding a request to the upstream provider.

When the field is absent, AISIX strips these credential headers:

```text
authorization
cookie
set-cookie
x-api-key
```

Entries are normalized when the provider key is loaded: whitespace is trimmed, names are lowercased, empty entries are dropped, and duplicates are removed. Hop-by-hop protocol headers and other non-configurable headers are stripped separately by the passthrough handler and cannot be re-enabled through this field.

For endpoint behavior, see [Passthrough](../integration/passthrough.md).

`telemetry_tags` carries attribution metadata alongside the provider key. In managed deployments, the control plane can use this block to distinguish catalog and bring-your-own provider keys and to carry display metadata.

Treat these tags as metadata, not routing controls. Provider request selection depends on the resolved `provider`, `adapter`, model, and provider-key connection settings.

## Upgrade Compatibility

Older provider-key payloads that omit newer optional fields can still deserialize. Missing optional fields fall back to their defaults.

That compatibility is useful during upgrades, but it should not become a reason to hand-edit stale schemas. If an existing payload fails validation, compare it with `schemas/resources/provider_key.schema.json` and the [Admin API reference](/ai-gateway/reference/admin-api).

## Related Reading

For related configuration and reference details, see [Provider keys](../configuration/provider-keys.md), [Adapter protocol families](adapters.md), [Resource schemas](resource-schemas.md), and [Admin API reference](/ai-gateway/reference/admin-api).
