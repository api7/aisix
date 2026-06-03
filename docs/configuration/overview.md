---
title: Configuration Overview
description: Understand the AISIX AI Gateway configuration model before creating provider keys, models, caller keys, and runtime policies.
sidebar_position: 29
---

AISIX AI Gateway configuration has two layers:

- **Bootstrap configuration** starts the process and decides which listeners,
  config store, cache backend, and managed-mode settings are available.
- **Dynamic resources** define the gateway behavior that operators change over
  time, such as provider keys, models, caller API keys, guardrails, cache
  policies, and observability exporters.

Use bootstrap configuration to bring the gateway online. Use dynamic resources
to decide what caller traffic can do after the gateway is running.

## Recommended setup order

For a standalone self-hosted gateway, the shortest useful path is:

1. Start with [Bootstrap configuration](bootstrap-config.md) so the proxy,
   admin listener, and config store are available.
2. Create [Provider keys](provider-keys.md) for upstream credentials and base
   URLs.
3. Create [Models](models.md) for the caller-visible names that map to upstream
   model IDs.
4. Create [API keys](api-keys.md) so callers can authenticate and access those
   models.
5. Add policies such as [Rate limits](rate-limits.md),
   [Guardrails](guardrails.md), [Caching](caching.md), and
   [Observability exporters](observability-exporters.md) when you know what
   boundary each policy should protect.

Managed data planes follow the same resource model, but configuration authority
lives in the AISIX Cloud control plane. A managed gateway does not bind the
standalone admin listener locally.

## Separate bootstrap config from runtime resources

| Configuration area | Where it lives | Change requires restart? | Typical owner |
| --- | --- | --- | --- |
| Proxy and admin listener addresses | Bootstrap config file or environment | Yes | Platform operator |
| etcd endpoints, prefix, and TLS | Bootstrap config file or environment | Yes | Platform operator |
| Cache backend selection | Bootstrap config file or environment | Yes | Platform operator |
| Provider keys | Dynamic resource store | No | Platform or AI platform operator |
| Models and routing aliases | Dynamic resource store | No | Platform or AI platform operator |
| Caller API keys | Dynamic resource store | No | Platform or application owner |
| Guardrails, cache policies, and exporters | Dynamic resource store | No | Platform or security/observability owner |

This split is important when troubleshooting. A successful process start proves
that bootstrap configuration was accepted, but it does not prove that dynamic
resources are present, valid, or visible to the proxy snapshot yet.

## Create these three resources before sending traffic

A normal proxy request needs three dynamic resources to line up:

```text
caller API key -> allowed model alias -> provider key -> upstream provider
```

- The caller API key authenticates the client and controls which model aliases
  it may use.
- The model alias is the stable name callers send in the request body, such as
  `gpt-4o-prod`.
- The provider key supplies the upstream credential, base URL, provider label,
  and adapter family used to send the request to the provider.

If proxy traffic fails after an admin write, first check that all three
resources exist and have propagated to the current proxy snapshot.

## Standalone and managed control

In standalone mode, operators write dynamic resources through the local
`/admin/v1/*` API. The admin API uses bootstrap admin keys and is separate from
caller-facing proxy API keys.

In managed mode, the local data plane receives projected resources from the
control plane. Provider keys, models, caller keys, and managed policies are
owned by the control plane, not by a local standalone admin API.

Do not mix the two operating models in one deployment. Decide whether the local
gateway is self-managed or control-plane-managed before you design the
configuration workflow.

## Source of truth for exact API shapes

The configuration pages explain operator intent and safe usage patterns. For
exact request schemas, response schemas, and status codes, use the generated
[Admin API reference](/ai-gateway/reference/admin-api) and generated
[Resource schemas](../reference/resource-schemas.md).

If a schema or route is wrong, fix the generated source of truth rather than
duplicating a conflicting shape in prose.

## Next steps

- [Bootstrap configuration](bootstrap-config.md) starts the gateway process.
- [Provider keys](provider-keys.md), [Models](models.md), and [API keys](api-keys.md)
  create the minimum dynamic resources for proxy traffic.
- [Configuration propagation](configuration-propagation.md) explains when
  dynamic writes become visible to the proxy.
