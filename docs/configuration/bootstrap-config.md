---
title: Bootstrap Configuration
description: Configure AISIX AI Gateway bootstrap settings, including etcd, proxy and admin listeners, observability, cache backends, and managed-mode options.
sidebar_position: 30
---

Bootstrap configuration defines the static settings the gateway needs at startup. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are loaded later from etcd.

Use this page to understand the config file that starts the gateway process.

Use bootstrap config for values that should exist before the process accepts traffic, not for day-to-day model and credential management.

## Loading Model

Bootstrap configuration is loaded in this order:

1. defaults
2. file contents
3. environment-variable overrides using the `AISIX_` prefix and `__` as the nested separator

This makes bootstrap config suitable for both:

- local file-based development
- containerized deployment where secrets and listener addresses are injected through environment variables

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

As a practical split:

- `etcd`, `proxy`, and `admin` define how the process starts
- `observability` and `cache` define process-wide runtime helpers
- `managed` switches the bootstrap mode from standalone to control-plane-managed

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
  metrics:
    prometheus:
      enabled: true
      path: "/metrics"

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

This section is the source of truth for where the gateway reads dynamic configuration after boot.

Important fields:

| Field | Description |
| --- | --- |
| `endpoints` | etcd endpoints the gateway should connect to |
| `prefix` | base resource namespace, usually `/aisix` |
| `env_id` | optional environment scope for env-scoped keys |
| `dial_timeout_ms` | connection timeout |
| `request_timeout_ms` | request timeout |
| `tls` | optional etcd TLS or mTLS configuration |

Operator guidance:

- use a stable `prefix` such as `/aisix` for standalone deployments
- use `env_id` only when your deployment model actually expects environment-scoped keys
- set timeouts aggressively enough to fail fast on broken config-store connectivity, but not so low that normal network variance looks like failure

## `proxy`

Use `proxy` to configure the public client-facing listener.

This is the only listener your callers need for model traffic.

Important fields:

| Field | Description |
| --- | --- |
| `addr` | proxy listener address |
| `request_body_limit_bytes` | request-body limit enforced by the proxy listener |
| `tls` | optional TLS certificate and key for the proxy listener |

Recommended pattern:

- bind `0.0.0.0` only when the process is intentionally network-reachable
- keep `request_body_limit_bytes` large enough for your expected request families, but avoid setting it arbitrarily high without a reason

## `admin`

Use `admin` to configure the operator-facing listener.

In standalone mode, this listener owns the write path for dynamic resources.

Important fields:

| Field | Description |
| --- | --- |
| `addr` | admin listener address |
| `admin_keys` | static admin keys accepted by the admin auth layer |
| `tls` | optional TLS certificate and key for the admin listener |

Admin keys are static bootstrap configuration. They are not stored in the dynamic `ApiKey` table.

Recommended pattern:

- bind the admin listener to loopback or an internal interface when possible
- do not reuse proxy caller API keys as admin keys
- rotate bootstrap admin keys through deployment/config management, not through the proxy-facing key lifecycle

## `observability`

Use `observability` to configure:

- service name
- log level
- access logs
- Prometheus metrics
- OTLP metrics and tracing exporters

Bootstrap observability settings are process-wide. They are different from dynamic `ObservabilityExporter` rows, which control data-plane telemetry fan-out for request events.

`observability.metrics.prometheus.enabled` controls whether the admin listener mounts the Prometheus scrape endpoint. `observability.metrics.prometheus.path` controls the mounted path and defaults to `/metrics`.

## `cache`

Use `cache` to choose the bootstrap cache backend.

Current backend selection supports:

- `memory`
- `redis`

`memory` is the default path. `redis` has runtime backend selection and connection logic, but the broader cache docs and support boundaries are still being expanded.

Use bootstrap cache settings to decide whether the process has a cache backend available at all. Use dynamic cache policies to decide which requests actually participate in caching.

## `managed`

Use `managed` when the gateway runs under AISIX Cloud control-plane workflows.

Important current behaviors when `managed.enabled = true`:

- the admin API is not bound
- the standalone playground endpoint is not exposed
- dynamic resources are read through the managed etcd path

This is the most important mode switch in the bootstrap config. It changes where operators should expect configuration authority to live.

The current config schema supports both:

- registration-token-driven bootstrap
- pre-provisioned certificate-bundle bootstrap using inline PEM or file paths

`AISIX Cloud` currently uses the certificate-based managed bootstrap flow. The registration-token path remains in the gateway runtime, but should be treated as a legacy or self-managed bootstrap path unless your deployment explicitly uses it.

## Choosing Between Standalone And Managed Bootstrap

- use standalone when you want local operator control through `:3001`
- use managed when AISIX Cloud is the control plane and the gateway should not expose a standalone admin write surface

Do not try to mix the two mental models in one deployment.

## `bedrock_endpoint_url`

Use `bedrock_endpoint_url` only when you need a deployment-wide override for Bedrock guardrail traffic.

This is a deployment concern, not a per-guardrail-row field.

## Verification

After updating the bootstrap config, start the gateway and verify:

```bash title="Verify proxy bootstrap"
curl -s http://127.0.0.1:3000/livez
```

For standalone mode, also verify:

```bash title="Verify admin bootstrap"
curl -s http://127.0.0.1:3001/livez
```

## Troubleshooting

### The process starts but no models ever appear

Focus on etcd connectivity and prefix alignment first. Bootstrap success alone does not prove dynamic config reads are healthy.

### The proxy is reachable but the admin listener is not

Check whether `managed.enabled = true`. In managed mode, the standalone admin API is intentionally not bound.

### Environment variables do not seem to override the file

Confirm the `AISIX_` prefix and nested `__` separator are correct.

## Related Pages

- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md)
- [Admin API](admin-api.md)
- [Roadmap](../roadmap.md)
