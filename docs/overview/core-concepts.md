---
title: Core Concepts
description: Understand the resource model behind AISIX AI Gateway.
sidebar_position: 2
---

AISIX AI Gateway is configured through a small set of resources. The most
important relationship is the path from caller credential to upstream provider.

## Request path

```text
caller token
  -> API key
  -> allowed model alias
  -> model
  -> provider key
  -> upstream provider
```

Read this page after the [Quickstart](../quickstart) when you
want the mental model behind the resources you created. Use the configuration
guides and generated schemas for exact field contracts.

If you just completed the quickstart, the abstract chain above maps to the
resources you created like this:

| Concept | Quickstart value | What it does |
| --- | --- | --- |
| Caller token | `sk-demo-caller` | The bearer token the application sends to AISIX. |
| API key | `ApiKey.key_hash` for `sk-demo-caller` | Authenticates the caller and limits which aliases it can use. |
| Model alias | `gpt-4o-prod` | The stable model name callers put in the proxy request. |
| Model | Direct `Model` row | Maps the caller-facing alias to the upstream model. |
| Provider key | `openai-upstream` | Stores the upstream credential and adapter settings. |
| Upstream provider | OpenAI with `gpt-4o-mini` | Receives the provider-authenticated request from AISIX. |

## Traffic resources

Most gateway setup starts with three resources.

### Model

A `Model` is the name callers send in the request's `model` field.

For a direct model, `display_name` is the caller-facing alias and `model_name` is
the upstream provider's model ID. This is the most important distinction in the
resource model: callers should not need to know whether `prod-chat` maps to
`gpt-4o`, a DeepSeek model, an internal vLLM model, or a routing group.

A direct model also references a `ProviderKey` through `provider_key_id`. Timeout
settings, inline rate limits, health-check behavior, cooldown behavior, and cost
metadata can be attached to the model when you need them.

See [Models](../configuration/models.md).

### Provider key

A `ProviderKey` stores the upstream credential and connection settings.

It keeps secrets out of model definitions and lets multiple models reuse the
same upstream credential. The provider key also tells the gateway which wire
shape to use through `adapter`, such as `openai`, `anthropic`, `bedrock`,
`vertex`, or `azure-openai`.

Provider identity and adapter family are not the same thing. For example, a
DeepSeek or internal vLLM endpoint can use the OpenAI-compatible wire shape
without pretending to be OpenAI.

See [Provider keys](../configuration/provider-keys.md).

### API key

An `ApiKey` is the caller credential.

The proxy never stores the plaintext caller token in the API key resource. It
hashes the incoming bearer token and compares it with `key_hash`. The rotate
endpoint returns a generated plaintext key exactly once; later reads only expose
the hash.

`allowed_models` is the access boundary. An empty list denies access to every
model. A wildcard entry, `"*"`, allows access to every model in scope.

The runtime API key row can also carry `team_id` and `user_id`. These are bucket
identifiers for team-scoped and member-scoped policy and metrics. They are not
access controls by themselves.

See [API keys](../configuration/api-keys.md).

## Routing models

A routing model is a model alias backed by a `routing` block instead of a single
upstream provider model.

Callers still send one stable alias. At request time, AISIX chooses a target
model using the configured strategy:

- `failover` tries targets in priority order.
- `round_robin` rotates traffic across targets.
- `weighted` selects targets according to configured weights.

Routing models are useful when you want to change upstream selection without
changing application code.

See [Routing and failover](../configuration/routing-and-failover.md).

## Policy resources

Policy resources add gateway behavior around the key-model-provider path.

### Rate-limit policy

A `RateLimitPolicy` is a standalone rate-limit rule. It can match an API key,
model, team, or member identity, and it is enforced alongside inline API-key and
model limits.

Any matching layer can reject a request with `429`.

See [Rate limits](../configuration/rate-limits.md).

### Guardrail

A `Guardrail` checks request or response content.

Keyword guardrails run locally in the data plane. Bedrock and Azure Content
Safety guardrails use remote provider services, so they require credentials,
network reachability, and an explicit outage posture.

See [Guardrails](../configuration/guardrails.md).

### Cache policy

A `CachePolicy` controls prompt-response cache lookup and storage.

Cache policy matching can apply globally, to a caller-facing model alias, or to
an API key entry. The process-level cache backend is selected from bootstrap
configuration; the policy shape includes a `backend` field, but that field does
not switch the runtime backend per policy today.

See [Caching](../configuration/caching.md).

### Observability exporter

An `ObservabilityExporter` sends gateway trace data to an OTLP/HTTP-compatible
backend.

Exporter traffic is sent by the data plane. It is metadata-oriented gateway
telemetry, not prompt or response body export.

See [Observability exporters](../configuration/observability-exporters.md).

## Managed concepts

AISIX Cloud adds managed control-plane concepts around the gateway runtime.

### Environment

An environment scopes the resources projected to a managed data plane. Projection
rules ensure a data plane only receives the resources intended for that
environment.

### Managed data plane

A managed data plane still runs AISIX AI Gateway, but it is operated through the
AISIX Cloud control plane.

In managed mode, the standalone admin listener is not exposed as the operator
write path. Dynamic resources come from the Cloud-managed configuration path,
and control-plane communication uses mTLS-authenticated `/dp/*` endpoints.

### Playground

The standalone gateway playground is mounted on the admin listener and forwards
requests through the local proxy router. It uses a proxy API key, not the admin
key, and the proxy middleware stack still runs.

The AISIX Cloud playground is a control-plane feature. Do not assume Cloud
playground behavior is identical to managed data-plane traffic unless the
specific feature says so.

## Source of truth

Concept pages explain how the pieces fit together. They are not the exact API
contract.

Use these sources when you need the precise accepted shape:

- [Admin API reference](/ai-gateway/reference/admin-api) for the generated
  OpenAPI reference.
- [Resource schemas](../reference/resource-schemas.md) for generated resource
  schemas.
- The configuration guide for the workflow you are implementing.

## Next steps

- [Provider keys](../configuration/provider-keys.md) — store upstream credentials and connection details.
- [Models](../configuration/models.md) — configure direct and routing model aliases.
- [API keys](../configuration/api-keys.md) — authenticate callers and control model access.
- [Rate limits](../configuration/rate-limits.md) — apply gateway-level traffic controls.
- [Routing and failover](../configuration/routing-and-failover.md) — configure virtual models and routing strategies.
