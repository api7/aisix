---
title: What Is AISIX AI Gateway
description: Learn what AISIX AI Gateway is, what problems it solves, and how it fits between applications and upstream model providers.
sidebar_position: 1
---

AISIX AI Gateway is a dedicated AI traffic gateway that sits between applications and upstream model providers.

Applications call AISIX with a gateway-issued API key and a model alias. Operators manage the upstream provider credentials, model mapping, routing, rate limits, guardrails, cache, and observability at the gateway layer.

This overview explains the operating model behind the first request you send through AISIX. If you have not run the gateway yet, start with the [Quickstart](../quickstart) first.

## What problem it solves

AI traffic often starts as direct provider integration: an application stores a provider key, chooses a provider model ID, and calls that provider's API.

That can work for one application. It becomes harder to operate when many applications need shared credentials, shared policy, provider failover, observability, or model changes that should not require application redeploys.

AISIX gives applications one stable gateway contract:

```text
caller key -> model alias -> provider key -> upstream model
```

Callers send a model alias such as `prod-chat`. Operators decide which upstream provider, credential, model ID, and policy that alias uses.

Use AISIX when AI traffic needs its own operator-managed contract. If one application calls one provider directly and does not need shared keys, shared policy, or provider abstraction, a direct provider integration may be enough.

## How requests move through AISIX

At request time, AISIX:

1. Authenticates the caller key.
2. Checks whether the caller can use the requested model alias.
3. Resolves the alias to provider-side configuration.
4. Applies gateway policy such as routing, rate limits, guardrails, cache, and observability.
5. Forwards the request to the upstream provider.

The main operating pattern is separation of concerns: applications use stable model aliases and gateway API keys, while operators manage provider choice and policy centrally.

For the resource-by-resource model, see [Core concepts](core-concepts.md).

## What operators manage

Most AISIX setup starts with three resources.

| Resource | Purpose |
| --- | --- |
| Provider key | Stores the upstream credential, provider identity, adapter family, and connection details. |
| Model | Defines the caller-facing model alias and how that alias resolves to an upstream model or routing group. |
| API key | Authenticates callers and controls which model aliases they can use. |

After the first working request path, operators can add routing, failover, rate limits, budgets, guardrails, response caching, and observability without changing application code.

## Runtime surfaces

A self-hosted AISIX gateway exposes two primary surfaces.

**Proxy API**

Applications and agents use the proxy API. It accepts OpenAI-compatible requests, Anthropic-style requests, and provider passthrough requests. The gateway authenticates the caller, resolves the model alias, applies policy, and forwards the request upstream.

**Admin API**

Operators use the admin API in self-hosted deployments. It manages gateway resources such as models, API keys, provider keys, guardrails, cache policies, observability exporters, health checks, and OpenAPI discovery.

In managed deployments, AISIX Cloud acts as the control plane. The data plane still exposes the proxy API, but the standalone admin listener is not exposed as the operator write path. Operators manage environments, certificates, and configuration projection through AISIX Cloud instead.

For exact route coverage, see [Proxy API reference](../reference/proxy-api-reference.md) and [Admin API reference](/ai-gateway/reference/admin-api).

## How AISIX relates to APISIX AI plugins

Apache APISIX and API7 Gateway can proxy AI traffic on normal gateway routes through AI plugins. That path is useful when an AI call is one route in a broader API gateway deployment.

AISIX is different because the AI gateway itself is the product boundary. It models provider keys, model aliases, caller API keys, routing, rate limits, cache, guardrails, and observability as AI gateway resources.

Choose the APISIX or API7 Gateway AI plugin path when AI behavior belongs to an existing API gateway route.

Choose AISIX when the main thing you need to operate is AI traffic itself: provider credentials, model aliases, model access, routing, failover, policy, and AI request telemetry.

## Deployment modes

AISIX can run in two operating modes.

| Mode | How it works |
| --- | --- |
| Self-hosted gateway | You run the gateway and manage bootstrap configuration, etcd, dynamic resources, provider credentials, and upgrades. |
| AISIX Cloud managed data plane | AISIX Cloud manages environments, certificates, and configuration projection while the gateway data plane still handles traffic. |

See [Deployment modes](deployment-modes.md) for the comparison.

## Provider and endpoint support

AISIX supports multiple upstream protocol families and provider integrations. Provider support is not identical across every endpoint or provider family.

For example, a model can work on the broad OpenAI-compatible chat route and still be rejected on an endpoint that requires a narrower provider-native shape.

Use [Feature status](feature-matrix.md) for the high-level product surface and [Provider compatibility](../reference/provider-compatibility.md) for provider and endpoint boundaries.

## Product boundary

`AISIX AI Gateway` is the gateway runtime documented in this repo. `AISIX Cloud` is the managed extension that adds environment management, certificate issuance, projection, usage-event collection, and Cloud-specific workflows.

:::note
Main docs describe current, verified behavior. Use [Feature Status](feature-matrix.md) to check whether a capability is available, limited, preview, or planned.
:::

## Next steps

- [Core concepts](core-concepts.md) — learn the resources behind models, provider keys, caller keys, routing, and policy.
- [Deployment modes](deployment-modes.md) — compare self-hosted and AISIX Cloud managed data-plane operation.
- [Feature status](feature-matrix.md) — check current feature status before planning production use.
- [OpenAI-compatible API](../integration/openai-compatible-api.md) — review the default caller-facing API surface.
