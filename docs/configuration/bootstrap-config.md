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
  real_ip:
    trusted_proxies: ["10.0.0.0/8", "127.0.0.1/32"]
    recursive: true
    header: x-forwarded-for

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

| Field | Description | Default |
| --- | --- | --- |
| `endpoints` | etcd endpoints the gateway should connect to | required |
| `prefix` | base resource namespace, usually `/aisix` | `"/aisix"` |
| `env_id` | optional environment scope for env-scoped keys | `""` (legacy / unscoped) |
| `dial_timeout_ms` | connection timeout | `5000` |
| `request_timeout_ms` | request timeout | `5000` |
| `tls` | optional etcd TLS or mTLS configuration | none |

Operator guidance:

- use a stable `prefix` such as `/aisix` for standalone deployments
- use `env_id` only when your deployment model actually expects environment-scoped keys
- set timeouts aggressively enough to fail fast on broken config-store connectivity, but not so low that normal network variance looks like failure

## `proxy`

Use `proxy` to configure the public client-facing listener.

This is the only listener your callers need for model traffic.

Important fields:

| Field | Description | Default |
| --- | --- | --- |
| `addr` | proxy listener address | required |
| `request_body_limit_bytes` | request-body limit enforced by the proxy listener | `10485760` (10 MiB) |
| `tls` | optional TLS certificate and key for the proxy listener | none |
| `real_ip` | downstream client-IP resolution from a forwarded header (see below) | trust nothing (log the TCP peer) |

Recommended pattern:

- bind `0.0.0.0` only when the process is intentionally network-reachable
- keep `request_body_limit_bytes` large enough for your expected request families, but avoid setting it arbitrarily high without a reason

### `proxy.real_ip`

Use `real_ip` to recover the original client IP for usage logs when the gateway sits behind an L7 load balancer or K8s ingress that sets `x-forwarded-for`. The resolution mirrors nginx's `set_real_ip_from` + `real_ip_recursive`: the immediate TCP peer is the client unless it matches a configured trusted proxy, in which case the forwarded header is walked to find the originating address. The resolved value is recorded as `client_source_ip` on each usage event.

| Field | Description | Default |
| --- | --- | --- |
| `trusted_proxies` | list of trusted upstream proxy CIDRs (a bare IP is treated as a `/32`/`/128` host route); malformed entries are rejected at startup | `[]` (trust nothing → always log the TCP peer) |
| `recursive` | when true, walk the forwarded header right-to-left skipping trusted addresses and take the first untrusted one; when false, take the rightmost header entry once the peer is trusted | `false` |
| `header` | forwarded header to consult | `"x-forwarded-for"` |

Operator guidance:

- leave `real_ip` unset for direct-to-client deployments — the TCP peer is already the real client
- set `trusted_proxies` to the CIDRs of your ingress / load balancer only; trusting too broad a range lets clients spoof `x-forwarded-for`

## `admin`

Use `admin` to configure the operator-facing listener.

In standalone mode, this listener owns the write path for dynamic resources.

Important fields:

| Field | Description | Default |
| --- | --- | --- |
| `addr` | admin listener address | `"127.0.0.1:0"` (intentionally non-routable; standalone deployments must override) |
| `admin_keys` | static admin keys accepted by the admin auth layer | `[]` (must be non-empty for standalone) |
| `tls` | optional TLS certificate and key for the admin listener | none |

Admin keys are static bootstrap configuration. They are not stored in the dynamic `ApiKey` table.

Recommended pattern:

- bind the admin listener to loopback or an internal interface when possible
- do not reuse proxy caller API keys as admin keys
- rotate bootstrap admin keys through deployment/config management, not through the proxy-facing key lifecycle

## `observability`

Use `observability` to set process-wide telemetry knobs. Today `service_name`, `log_level`, and the `metrics.prometheus.*` block are consulted at runtime; the other fields have varying current behavior — see the `Status` column below.

Important fields:

| Field | Description | Default | Status |
| --- | --- | --- | --- |
| `service_name` | service-name attribute on the tracing subscriber initialised at boot | `"aisix"` | wired |
| `log_level` | fallback `EnvFilter` directive when `RUST_LOG` is not set in the environment | `"info"` | wired |
| `access_log` | reserved field; access logs are currently emitted by every proxy handler regardless of this setting | `true` | reserved (not yet consulted) |
| `metrics.prometheus.enabled` | controls whether the admin listener mounts the Prometheus scrape endpoint; when `false`, no `/metrics` route is registered | `true` | wired |
| `metrics.prometheus.path` | mount path for the Prometheus scrape endpoint | `"/metrics"` | wired |
| `metrics.otlp.enabled` | reserved field; no OTLP metrics export pipeline is installed in the current release | `false` | reserved (not yet wired) |
| `metrics.otlp.endpoint` | OTLP/gRPC metrics endpoint | none | reserved (not yet wired) |
| `tracing.otlp.enabled` | boot-time endpoint validation; OTLP traces pipeline deferred | `false` | partial (validation only) |
| `tracing.otlp.endpoint` | OTLP/gRPC collector endpoint for traces | none | partial (validation only) |
| `tracing.otlp.sample_ratio` | head-based sampling ratio reserved for the future OTLP traces pipeline | `1.0` | reserved (not yet wired) |

Bootstrap observability settings are process-wide. They are different from dynamic `ObservabilityExporter` rows, which control per-request span fan-out via OTLP/HTTP at runtime. For per-row dynamic exporters added at runtime via the admin API, see [Observability Exporters](observability-exporters.md).

## `cache`

Use `cache` to declare which cache backends the process builds at startup.

The in-process memory cache is always built. The shared Redis cache is built only when the `redis` block is present. Which backend serves a given request is selected by the matched cache policy's `backend` field (a dynamic resource) — not by this section.

Important fields:

| Field | Description | Default |
| --- | --- | --- |
| `backend` | legacy knob — no longer selects a single global cache; `redis` without a `redis` block is rejected at boot | `memory` |
| `redis` | Redis connection block (`url`, optional `mode`); when present, the process builds the shared Redis cache | none |

A cache policy that requests `backend: redis` on a process without `cache.redis` gets no caching for its requests (`cache_status = disabled`, one warning per policy in the gateway log) — never a silent fallback to memory.

Use bootstrap cache settings to decide which cache backends the process has available. Use dynamic cache policies to decide which requests actually participate in caching, and on which backend.

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

Use `bedrock_endpoint_url` only when you need a deployment-wide override for Bedrock guardrail traffic. Skip this field unless you actively use the AWS Bedrock guardrail integration (`kind: bedrock` on a [Guardrail](../overview/glossary.md#guardrail) row); it overrides the default Bedrock endpoint for all such traffic in this deployment.

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
