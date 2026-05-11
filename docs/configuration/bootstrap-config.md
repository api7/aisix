---
title: Bootstrap Configuration
description: Configure AISIX AI Gateway bootstrap settings, including etcd, proxy and admin listeners, observability, cache backends, and managed-mode options.
sidebar_position: 30
---

Bootstrap configuration defines the static settings the gateway needs at startup. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are loaded later from etcd.

Use this page to understand the config file that starts the gateway process.

## Loading Model

Bootstrap configuration is loaded in this order:

1. defaults
2. file contents
3. environment-variable overrides using the `AISIX_` prefix and `__` as the nested separator

Example:

```bash title="Override the proxy listener address"
export AISIX_PROXY__ADDR="0.0.0.0:3000"
```

## Root Sections

The current root config includes:

- `etcd`
- `proxy`
- `admin`
- `observability`
- `cache`
- `managed`
- optional top-level `bedrock_endpoint_url`

## Minimal Self-Hosted Example

```yaml title="config.yaml" {1-22}
etcd:
  endpoints:
    - "http://127.0.0.1:2379"
  prefix: "/aisix"
  dial_timeout_ms: 5000
  request_timeout_ms: 5000

proxy:
  addr: "0.0.0.0:3000"
  request_body_limit_bytes: 10485760

admin:
  addr: "127.0.0.1:3001"
  admin_keys:
    - "YOUR_ADMIN_KEY"

observability:
  service_name: "aisix"
  log_level: "info"
  access_log: true

cache:
  backend: "memory"
```

## `etcd`

Use `etcd` to define:

- endpoints
- key prefix
- env scope
- optional auth
- optional TLS or mTLS bundle

Important fields:

| Field | Description |
| --- | --- |
| `endpoints` | etcd endpoints the gateway should connect to |
| `prefix` | base resource namespace, usually `/aisix` |
| `env_id` | optional environment scope for env-scoped keys |
| `dial_timeout_ms` | connection timeout |
| `request_timeout_ms` | request timeout |
| `tls` | optional etcd TLS or mTLS configuration |

## `proxy`

Use `proxy` to configure the public client-facing listener.

Important fields:

| Field | Description |
| --- | --- |
| `addr` | proxy listener address |
| `request_body_limit_bytes` | request-body limit enforced by the proxy listener |
| `tls` | optional TLS certificate and key for the proxy listener |

## `admin`

Use `admin` to configure the operator-facing listener.

Important fields:

| Field | Description |
| --- | --- |
| `addr` | admin listener address |
| `admin_keys` | static admin keys accepted by the admin auth layer |
| `tls` | optional TLS certificate and key for the admin listener |

Admin keys are static bootstrap configuration. They are not stored in the dynamic `ApiKey` table.

## `observability`

Use `observability` to configure:

- service name
- log level
- access logs
- Prometheus metrics
- OTLP metrics and tracing exporters

## `cache`

Use `cache` to choose the bootstrap cache backend.

Current backend selection supports:

- `memory`
- `redis`

`memory` is the default path. `redis` has runtime backend selection and connection logic, but the broader cache docs and support boundaries are still being expanded.

## `managed`

Use `managed` when the gateway runs under AISIX Cloud control-plane workflows.

Important current behaviors when `managed.enabled = true`:

- the admin API is not bound
- the standalone playground endpoint is not exposed
- dynamic resources are read through the managed etcd path

The current config schema supports both:

- registration-token-driven bootstrap
- pre-provisioned certificate-bundle bootstrap using inline PEM or file paths

`AISIX Cloud` currently uses the certificate-based managed bootstrap flow. The registration-token path remains in the gateway runtime, but should be treated as a legacy or self-managed bootstrap path unless your deployment explicitly uses it.

## `bedrock_endpoint_url`

Use `bedrock_endpoint_url` only when you need a deployment-wide override for Bedrock guardrail traffic.

This is a deployment concern, not a per-guardrail-row field.

## Verification

After updating the bootstrap config, start the gateway and verify:

```bash title="Verify proxy bootstrap"
curl -s http://127.0.0.1:3000/health
```

For standalone mode, also verify:

```bash title="Verify admin bootstrap"
curl -s \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  http://127.0.0.1:3001/admin/v1/health
```

## Related Pages

- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md)
- [Admin API](admin-api.md)
- [Roadmap](../roadmap.md)
