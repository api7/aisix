---
title: Bootstrap Configuration
description: Configure AISIX AI Gateway bootstrap settings, including etcd, proxy and admin listeners, observability, cache backends, and managed-mode options.
sidebar_position: 30
---

Bootstrap configuration defines the static settings the gateway needs at startup. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are loaded later from etcd.

This guide explains the config file that starts the gateway process. Bootstrap config is for values that must exist before the process accepts traffic, not for day-to-day model and credential management.

## Loading order

Bootstrap configuration is loaded in this order:

1. defaults
2. file contents
3. environment-variable overrides using the `AISIX_` prefix and `__` as the nested separator

This makes bootstrap config suitable for both:

- local file-based development
- containerized deployment where listener addresses and secret references are injected through environment variables

Example:

```shell
export AISIX_PROXY__ADDR="0.0.0.0:3000"
```

## Root sections

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

## Minimal self-hosted example

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

- `endpoints` is required and lists the etcd endpoints the gateway
  should connect to.
- `prefix` is the base resource namespace. The default is `"/aisix"`.
- `env_id` is the optional environment scope for env-scoped keys. The
  default is `""`, which means legacy or unscoped operation.
- `dial_timeout_ms` controls connection timeout. The default is `5000`.
- `request_timeout_ms` controls request timeout. The default is `5000`.
- `tls` configures optional etcd TLS or mTLS. It is absent by default.

Operator guidance:

- use a stable `prefix` such as `/aisix` for standalone deployments
- use `env_id` only when your deployment model actually expects environment-scoped keys
- set timeouts aggressively enough to fail fast on broken config-store connectivity, but not so low that normal network variance looks like failure

## `proxy`

Use `proxy` to configure the public client-facing listener.

This is the only listener your callers need for model traffic.

Important fields:

- `addr` is required and sets the proxy listener address.
- `request_body_limit_bytes` sets the request-body limit enforced by the
  proxy listener. The default is `10485760` bytes, or 10 MiB.
- `tls` configures an optional TLS certificate and key for the proxy
  listener. It is absent by default.

Recommended pattern:

- bind `0.0.0.0` only when the process is intentionally network-reachable
- keep `request_body_limit_bytes` large enough for your expected request families, but avoid setting it arbitrarily high without a reason

## `admin`

Use `admin` to configure the operator-facing listener.

In standalone mode, this listener owns the write path for dynamic resources.

Important fields:

- `addr` sets the admin listener address. The default is
  `"127.0.0.1:0"`, which is intentionally non-routable; standalone
  deployments must override it.
- `admin_keys` lists static admin keys accepted by the admin auth layer.
  The default is `[]`, and it must be non-empty for standalone mode.
- `tls` configures an optional TLS certificate and key for the admin
  listener. It is absent by default.

Admin keys are static bootstrap configuration. They are not stored in the dynamic `ApiKey` table.

Recommended pattern:

- bind the admin listener to loopback or an internal interface when possible
- do not reuse proxy caller API keys as admin keys
- rotate bootstrap admin keys through deployment/config management, not through the proxy-facing key lifecycle

## `observability`

Use `observability` to set process-wide telemetry knobs.

`service_name` is wired and sets the service-name attribute on the
tracing subscriber initialized at boot. The default is `"aisix"`.

`log_level` is wired and sets the fallback `EnvFilter` directive when
`RUST_LOG` is not set. The default is `"info"`.

`access_log` is currently reserved. Access logs are emitted by every
proxy handler regardless of this setting. The default is `true`.

`metrics.prometheus.enabled` is wired and controls whether the admin
listener mounts the Prometheus scrape endpoint. When it is `false`, no
`/metrics` route is registered. The default is `true`.

`metrics.prometheus.path` is wired and sets the Prometheus scrape path.
The default is `"/metrics"`.

`metrics.otlp.enabled` and `metrics.otlp.endpoint` are reserved. No OTLP
metrics export pipeline is installed in the current release.
`metrics.otlp.enabled` defaults to `false`.

`tracing.otlp.enabled`, `tracing.otlp.endpoint`, and
`tracing.otlp.sample_ratio` are partially wired for boot-time endpoint
validation, but the OTLP traces pipeline is deferred.
`tracing.otlp.enabled` defaults to `false`, and
`tracing.otlp.sample_ratio` defaults to `1.0`.

Bootstrap observability settings are process-wide. They are different from dynamic `ObservabilityExporter` rows, which control per-request span fan-out via OTLP/HTTP at runtime. For per-row dynamic exporters added at runtime via the admin API, see [Observability exporters](observability-exporters.md).

## `cache`

Use `cache` to choose the bootstrap cache backend.

Important fields:

- `backend` selects which cache backend the process uses. The current
  options are `memory` and `redis`; the default is `memory`.
- `redis` configures the Redis connection block, including `url` and
  optional `mode`. It is only consulted when `backend: redis` and is
  absent by default.

`memory` is the default path. Use `redis` when several data-plane instances should share cached responses. The current Redis bootstrap path connects to a single Redis URL; cluster and sentinel modes are not exposed through bootstrap config.

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

## Choosing between standalone and managed bootstrap

- use standalone when you want local operator control through `:3001`
- use managed when AISIX Cloud is the control plane and the gateway should not expose a standalone admin write surface

Do not try to mix the two mental models in one deployment.

## `bedrock_endpoint_url`

Use `bedrock_endpoint_url` only when you need a deployment-wide override for Bedrock guardrail traffic. Skip this field unless you actively use the AWS Bedrock guardrail integration (`kind: bedrock` on a [Guardrail](../overview/glossary.md#guardrail) row); it overrides the default Bedrock endpoint for all such traffic in this deployment.

This is a deployment concern, not a per-guardrail-row field.

## Verify

After updating the bootstrap config, start the gateway and verify:

```shell
curl -s http://127.0.0.1:3000/livez
```

For standalone mode, also verify:

```shell
curl -s http://127.0.0.1:3001/livez
```

## Troubleshooting

### The process starts but no models ever appear

Focus on etcd connectivity and prefix alignment first. Bootstrap success alone does not prove dynamic config reads are healthy.

### The proxy is reachable but the admin listener is not

Check whether `managed.enabled = true`. In managed mode, the standalone admin API is intentionally not bound.

### Environment variables do not seem to override the file

Confirm the `AISIX_` prefix and nested `__` separator are correct.

## Next steps

- [Configuration overview](overview.md) — understand the split between
  bootstrap settings and dynamic resources.
- [Quickstart](../quickstart) — run a local gateway with a working config file.
- [Admin API](admin-api.md) — manage dynamic resources after bootstrap.
- [Understand admin resources](../quickstart/first-model-first-key-first-request.md) — create provider keys, models, and caller keys.
- [Configuration propagation](configuration-propagation.md) — understand how dynamic resources reach the proxy.
