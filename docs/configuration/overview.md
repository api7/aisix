---
title: Configuration Overview
description: Understand the AISIX AI Gateway configuration model before creating provider keys, models, caller keys, and runtime policies.
sidebar_position: 29
---

AISIX AI Gateway configuration has two layers:

- **Bootstrap configuration** starts the process and decides which listeners, config store, cache backend, and managed-mode settings are available.
- **Dynamic resources** define gateway behavior that can change over time, such as provider keys, models, caller API keys, guardrails, cache policies, and observability exporters.

Use bootstrap configuration to bring the gateway online. Use dynamic resources to decide what caller traffic can do after the gateway is running.

## Recommended Setup Order

For a standalone self-hosted gateway, start with [Bootstrap configuration](bootstrap-config.md) so the proxy, admin listener, and config store are available. Then create [Provider keys](provider-keys.md), [Models](models.md), and [API keys](api-keys.md) for the first proxy request.

After traffic is working, add policies such as [Rate limits](rate-limits.md), [Guardrails](guardrails.md), [Caching](caching.md), and [Observability exporters](observability-exporters.md) when you know what each policy should protect.

Managed data planes follow the same resource model, but configuration authority lives in the AISIX Cloud control plane. A managed gateway does not bind the standalone admin listener locally.

## Configuration Layers

| Configuration area | Where it lives | Change requires restart? | Typical owner |
| --- | --- | --- | --- |
| Proxy and admin listener addresses | Bootstrap config file or environment | Yes | Platform team |
| etcd endpoints, prefix, and TLS | Bootstrap config file or environment | Yes | Platform team |
| Cache backend selection | Bootstrap config file or environment | Yes | Platform team |
| Provider keys | Dynamic resource store | No | Platform or AI platform team |
| Models and routing aliases | Dynamic resource store | No | Platform or AI platform team |
| Caller API keys | Dynamic resource store | No | Platform or application owner |
| Guardrails, cache policies, and exporters | Dynamic resource store | No | Platform or security/observability owner |

This split is important when troubleshooting. A successful process start shows that bootstrap configuration was accepted, but it does not confirm that dynamic resources are present, valid, or visible to proxy traffic yet.

A normal proxy request needs three dynamic resources to line up:

```mermaid
flowchart LR
  CallerKey["Caller API key"] --> ModelAlias["Allowed model alias"]
  ModelAlias --> ProviderKey["Provider key"]
  ProviderKey --> Upstream["Upstream provider"]
```

The caller API key authenticates the client and controls which model aliases it may use. The model alias is the stable name callers send in the request body, such as `gpt-4o-prod`. The provider key supplies the upstream credential, base URL, provider label, and adapter family used to send the request to the provider.

If proxy traffic fails after an admin write, first check that all three resources exist and have propagated to the proxy.

## Standalone and Managed Control

In standalone mode, dynamic resources are written through the local `/admin/v1/*` API. The admin API uses bootstrap admin keys and is separate from caller-facing proxy API keys.

In managed mode, the local data plane receives projected resources from the control plane. Provider keys, models, caller keys, and managed policies are owned by the control plane, not by a local standalone admin API.

Do not mix the two operating models in one deployment. Decide whether the local gateway is self-managed or control-plane-managed before you design the configuration workflow.

Use the configuration guides for setup intent and safe usage patterns. For exact request schemas, response schemas, and status codes, use the [Admin API reference](/ai-gateway/reference/admin-api) and [Resource schemas](../reference/resource-schemas.md).

## Related Reading

Start with [Bootstrap configuration](bootstrap-config.md), then configure [Provider keys](provider-keys.md), [Models](models.md), and [API keys](api-keys.md) for proxy traffic. When changes are not visible at the proxy immediately, see [Configuration propagation](configuration-propagation.md).
