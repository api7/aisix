---
title: Port Reference
description: Reference for common AISIX AI Gateway local and self-hosted network ports.
sidebar_position: 65
toc_max_heading_level: 2
---

AISIX AI Gateway listener ports are configured in bootstrap config. The Quickstart and self-hosted examples use the ports below so the proxy, admin API, and etcd endpoints do not overlap.

| Port | Used By | Purpose | Exposure |
| --- | --- | --- | --- |
| `3000` | Proxy listener | Receives client-facing AI API requests, such as `/v1/chat/completions` and `/v1/models`. | Expose only to intended callers or the ingress tier in front of the gateway. |
| `3001` | Standalone admin listener | Receives admin API requests that create and update dynamic resources. It can also serve health, metrics, and OpenAPI routes. | Keep on loopback, a private subnet, or an admin-only network. |
| `2379` | etcd client listener | Stores dynamic gateway configuration for standalone deployments. | Keep private to AISIX and the systems that manage gateway configuration. |

## Configure Listener Ports

Set the proxy listener with `proxy.addr`:

```yaml
proxy:
  addr: "0.0.0.0:3000"
```

Set the standalone admin listener with `admin.addr`:

```yaml
admin:
  addr: "127.0.0.1:3001"
```

The proxy listener address is required in bootstrap config. The standalone admin listener defaults to `127.0.0.1:0`, so self-hosted deployments must set `admin.addr` when they need the Admin API.

## Related Reading

For the full bootstrap schema, see [Bootstrap Configuration](../configuration/bootstrap-config.md). For exposure guidance, see [Network and Security](../operations/network-and-security.md).
