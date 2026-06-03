---
title: Provider Key Schema
description: How to read the generated ProviderKey contract and the runtime boundaries around provider keys.
sidebar_position: 67
keywords:
  - AISIX AI Gateway
  - ProviderKey
  - schema
  - adapter
  - runtime config
  - AI gateway
---

A `ProviderKey` stores the upstream credential and connection details that a
[model](../configuration/models.md) uses when AISIX sends traffic to an AI
provider.

This reference explains where the generated provider-key contract comes from
and how to interpret the runtime caveats around provider keys. For the exact request
body, response body, defaults, enum values, and validation errors, use the
generated [Admin API reference](/ai-gateway/reference/admin-api).

For the operator workflow, see [Provider keys](../configuration/provider-keys.md).

## Source of truth

The standalone JSON Schema for provider keys is generated into:

```text
schemas/resources/provider_key.schema.json
```

That schema is merged into the generated admin OpenAPI document rendered by the
docs site:

```text
http://127.0.0.1:3000/ai-gateway/reference/admin-api
```

When you run a self-hosted gateway locally, you can also open the live Scalar
reference from the admin listener:

```text
http://127.0.0.1:3001/admin/openapi-scalar
```

If this page, a configuration guide, and the generated reference disagree,
prefer the generated reference. If the generated reference is wrong, fix the
source that generates the schema rather than maintaining a parallel field list
in this page.

## How to read the contract

Provider keys are closed resources. Unknown top-level fields are rejected.

The required fields are:

- `display_name`: the operator-facing name for the provider key
- `secret`: the credential AISIX uses when it calls the upstream provider

The generated schema also defines optional fields for:

- upstream routing identity: `provider`, `adapter`, and `api_base`
- request and response compatibility overrides: `request` and `response`
- passthrough credential protection: `strip_headers`
- attribution metadata: `telemetry_tags`

Do not use this page as the complete field reference. Use it to understand the
field groups and then check the generated admin reference for the exact shape.

:::warning Production credentials
The standalone gateway stores `secret` as plaintext under the etcd `prefix`
configured in [`config.yaml`](../configuration/bootstrap-config.md). Anyone
with read access to that etcd keyspace can read the credential. In production,
restrict etcd network access, use encryption at rest where available, and keep
the gateway-to-etcd channel inside the trusted infrastructure boundary.
:::

## Keep `provider` and `adapter` separate

`provider` and `adapter` are related, but they do different jobs.

`provider` is the vendor or endpoint identity, such as `openai`, `anthropic`,
`deepseek`, or a bring-your-own provider label. It is an open string because
AISIX can route catalog and long-tail provider identities without adding every
vendor name to the data plane.

`adapter` is the upstream protocol family AISIX knows how to encode. The
generated schema exposes the current closed set, including `openai`,
`anthropic`, `bedrock`, `vertex`, and `azure-openai`.

At dispatch time, AISIX first tries a provider-specific bridge keyed by
`provider`. If that lookup does not match, AISIX falls back to the adapter
family selected by `adapter`. This is why an OpenAI-compatible vendor can use
its own `provider` value while still dispatching through `adapter: "openai"`.

For the dispatch model, see [Adapter protocol families](adapters.md).

## Understand `api_base`

`api_base` controls the upstream endpoint root.

Some canonical bridges can infer a default base URL for their own canonical
provider. For example, an OpenAI provider key can fall back to the OpenAI base
URL, and an Anthropic provider key can fall back to the Anthropic base URL.

For non-canonical providers, bring-your-own endpoints, private gateways, and
most catalog projections, set `api_base` explicitly. AISIX refuses to guess a
canonical provider base URL for a different provider identity because that
could send a credential to the wrong upstream.

The configuration guide has the practical examples:
[Provider keys](../configuration/provider-keys.md#base-url).

## Understand runtime overrides

The generated schema accepts optional `request` and `response` override blocks.
These blocks describe compatibility knobs such as parameter renames,
temperature clamps, default outbound headers, default outbound body fields,
content-list flattening, stream `[DONE]` marker policy, and reasoning-field
extraction.

The schema tells you which shapes AISIX accepts. It does not mean every bridge
or every proxy endpoint applies every override in the same way.

Bridge and endpoint behavior is runtime-specific:

- OpenAI-family and Azure OpenAI chat paths apply request-body, header, and
  selected response override behavior.
- Vertex publisher rails apply the shared request-body override pipeline before
  Vertex-specific shaping.
- Anthropic `/v1/messages` and `/v1/messages/count_tokens` paths apply
  request-side overrides to their outbound provider request.
- Passthrough and provider-native forwarding paths may bypass normalized
  bridge behavior.

When an override matters for a provider family, confirm the behavior in the
relevant integration guide or source-backed runtime tests before relying on it
in production.

## Passthrough header stripping

`strip_headers` controls which inbound headers the passthrough endpoint removes
before forwarding a request to the upstream provider.

When the field is absent, AISIX strips these credential headers:

```text
authorization
cookie
set-cookie
x-api-key
```

Entries are normalized when the provider key is loaded: whitespace is trimmed,
names are lowercased, empty entries are dropped, and duplicates are removed.
Hop-by-hop protocol headers and other non-configurable headers are stripped
separately by the passthrough handler and cannot be re-enabled through this
field.

For endpoint behavior, see [Passthrough](../integration/passthrough.md).

## Telemetry tags

`telemetry_tags` carries attribution metadata alongside the provider key. In
managed deployments, the control plane can use this block to distinguish
catalog and bring-your-own provider keys and to carry display metadata.

Treat these tags as metadata, not dispatch controls. Dispatch depends on the
resolved `provider`, `adapter`, model, and provider-key connection settings.

## Compatibility boundary

Older provider-key payloads that omit newer optional fields can still
deserialize. Missing optional fields fall back to their defaults.

That compatibility is useful during upgrades, but it should not become a reason
to hand-edit stale shapes. If an existing payload fails validation, compare it
with `schemas/resources/provider_key.schema.json` and the generated
[Admin API reference](/ai-gateway/reference/admin-api).

## Next steps

- [Provider keys](../configuration/provider-keys.md) explains how to configure
  provider credentials in an operator workflow.
- [Adapter protocol families](adapters.md) explains how provider keys select
  upstream bridges.
- [Resource schemas](resource-schemas.md) explains how generated resource
  schemas are produced and verified.
- [Admin API reference](/ai-gateway/reference/admin-api) renders the generated
  OpenAPI reference.
