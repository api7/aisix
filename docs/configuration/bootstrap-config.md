---
title: Bootstrap Configuration
description: Configure AISIX AI Gateway bootstrap settings, including etcd, proxy and admin listeners, observability, cache backends, and managed-mode options.
sidebar_position: 30
toc_max_heading_level: 2
---

Bootstrap configuration defines the static settings the gateway needs at startup. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are loaded later from etcd.

Bootstrap config is for values that must exist before the process accepts traffic, not for day-to-day model and credential management.

## Loading Order

Bootstrap configuration is loaded in this order:

1. defaults
2. file contents
3. environment-variable overrides using the `AISIX_` prefix and `__` as the nested separator

This makes bootstrap config suitable for local file-based deployments and for containerized deployments where listener addresses and secret references are injected through environment variables.

Example:

```shell
export AISIX_PROXY__ADDR="0.0.0.0:3000"
```

## Root Sections

The root config is split by startup responsibility:

| Section | Purpose |
| --- | --- |
| `etcd` | Dynamic configuration store connection. |
| `proxy` | Public client-facing listener. |
| `admin` | Standalone admin listener. |
| `observability` | Process-wide logging, tracing, access-log, and metrics settings. |
| `cache` | Process-wide cache backend selection. |
| `managed` | Control-plane-managed bootstrap mode. |
| `bedrock_endpoint_url` | Optional deployment-wide Bedrock guardrail endpoint override. |

## Minimal Self-Hosted Example

Use this example as a starting point for a local self-hosted gateway:

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
  metrics:
    prometheus:
      enabled: true
      path: "/metrics"

cache:
  backend: "memory"
```

## Configuration Sections

Modify the minimal example for your deployment with the configuration sections that follow.

### etcd Configuration

This configuration controls where the gateway reads dynamic configuration after boot.

| Field | Purpose |
| --- | --- |
| `endpoints` | Required etcd endpoints the gateway connects to. |
| `prefix` | Base resource namespace. The default is `"/aisix"`. |
| `env_id` | Optional environment scope for env-scoped keys. The default is `""`, which means unscoped operation. |
| `dial_timeout_ms` | Connection timeout. The default is `5000`. |
| `request_timeout_ms` | Request timeout. The default is `5000`. |
| `tls` | Optional etcd TLS or mTLS configuration. It is absent by default. |

Use a stable `prefix` such as `/aisix` for standalone deployments. Set `env_id` only when your deployment model expects environment-scoped keys. Choose timeouts that fail fast on broken config-store connectivity without treating normal network variance as failure.

### Proxy Listener

Use `proxy` to configure the public client-facing listener.

This is the only listener your callers need for model traffic.

| Field | Purpose |
| --- | --- |
| `addr` | Required proxy listener address. |
| `request_body_limit_bytes` | Request-body limit enforced by the proxy listener. The default is `10485760` bytes, or 10 MiB. |
| `tls` | Optional TLS certificate and key for the proxy listener. It is absent by default. |

Bind `0.0.0.0` only when the process should be network-reachable. Keep `request_body_limit_bytes` large enough for expected request families without setting it arbitrarily high.

### Admin Listener

Use `admin` to configure the standalone admin listener.

In standalone mode, this listener owns the write path for dynamic resources.

| Field | Purpose |
| --- | --- |
| `addr` | Admin listener address. The default is `"127.0.0.1:0"`, which is intentionally non-routable; standalone deployments must override it. |
| `admin_keys` | Static admin keys accepted by the admin auth layer. The default is `[]`, and it must be non-empty for standalone mode. |
| `tls` | Optional TLS certificate and key for the admin listener. It is absent by default. |

Admin keys are static bootstrap configuration. They are not stored in the dynamic `ApiKey` table.

Bind the admin listener to loopback or a private interface when possible. Do not reuse proxy caller API keys as admin keys. Rotate bootstrap admin keys through deployment or config management, not through the proxy-facing key lifecycle.

### Observability Settings

Use `observability` to set process-wide telemetry knobs.

`service_name` sets the service-name attribute on tracing initialized at boot. The default is `"aisix"`.

`log_level` sets the fallback logging directive when `RUST_LOG` is not set. The default is `"info"`.

`metrics.prometheus.enabled` controls whether the admin listener mounts the Prometheus scrape endpoint. When it is `false`, no `/metrics` route is registered. The default is `true`.

`metrics.prometheus.path` sets the Prometheus scrape path. The default is `"/metrics"`.

Bootstrap observability settings are process-wide. They are different from dynamic `ObservabilityExporter` rows, which control per-request span fan-out via OTLP/HTTP at runtime. For dynamic exporters added through the admin API, see [Observability exporters](observability-exporters.md).

### Cache Backend

Use `cache` to choose the bootstrap cache backend.

| Field | Purpose |
| --- | --- |
| `backend` | Cache backend for the process. Supported values are `memory` and `redis`; the default is `memory`. |
| `redis` | Redis connection block, including `url` and optional `mode`. It is only consulted when `backend: redis` and is absent by default. |

`memory` is the default path. Use `redis` when several data-plane instances should share cached responses. The Redis bootstrap path connects to a single Redis URL.

Use bootstrap cache settings to decide whether the process has a cache backend. Use dynamic cache policies to decide which requests participate in caching.

### Managed Mode

Use `managed` when the gateway runs under AISIX Cloud control-plane workflows.

When `managed.enabled = true`, the admin API is not bound, the standalone playground endpoint is not exposed, and dynamic resources are read through the managed etcd path.

This is the most important mode switch in the bootstrap config. It changes where configuration authority lives.

The config schema supports registration-token-driven bootstrap and pre-provisioned certificate-bundle bootstrap using inline PEM or file paths.

AISIX Cloud uses the certificate-based managed bootstrap flow. The registration-token path remains available, but treat it as a legacy or self-managed bootstrap path unless your deployment explicitly uses it.

Use standalone bootstrap when local control through `:3001` is required. Use managed bootstrap when AISIX Cloud is the control plane and the gateway should not expose a standalone admin write API.

Do not mix standalone and managed operating modes in one deployment.

### Bedrock Guardrail Endpoint

Use `bedrock_endpoint_url` only when you need a deployment-wide override for Bedrock guardrail traffic. Skip this field unless you use the AWS Bedrock guardrail integration (`kind: bedrock` on a [Guardrail](../overview/glossary.md#guardrail) row). The value overrides the default Bedrock endpoint for all Bedrock guardrail traffic in this deployment.

This is a deployment concern, not a per-guardrail-row field.

## Verify the Bootstrap Configuration

After updating the bootstrap config, start the gateway and verify:

```shell
curl -s "http://127.0.0.1:3000/livez"
```

For standalone mode, also verify:

```shell
curl -s "http://127.0.0.1:3001/livez"
```

## Troubleshooting

Use these checks when the process starts but the runtime resources are not available on the request path.

### Process Starts but Models Do Not Appear

Focus on etcd connectivity and prefix alignment first. Bootstrap success alone does not confirm that dynamic config reads are healthy.

### Proxy Is Reachable but Admin Listener Is Not

Check whether `managed.enabled = true`. In managed mode, the standalone admin API is intentionally not bound.

### Environment Variables Do Not Override File

Confirm the `AISIX_` prefix and nested `__` separator are correct.

## Related Reading

[Configuration overview](overview.md) explains the split between bootstrap settings and dynamic resources. To run a local gateway, see the [Quickstart](../quickstart). After bootstrap, [Admin API](admin-api.md) and [Understand admin resources](../quickstart/first-model-first-key-first-request.md) show how to create provider keys, models, and caller keys. For propagation timing, see [Configuration propagation](configuration-propagation.md).
