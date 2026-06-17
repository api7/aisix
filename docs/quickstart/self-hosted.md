---
title: Self-Hosted Quickstart
description: Deploy a self-hosted AISIX AI Gateway instance and verify that the proxy and admin listeners are reachable.
sidebar_position: 10
---

This guide shows how to start a self-hosted AISIX AI Gateway instance with the local example configuration and verify that both the proxy and admin surfaces are reachable.

## Prerequisites

- **Rust 1.93 or newer with `cargo`.** Install via [rustup](https://rustup.rs) and verify with `cargo --version`. The repo pins this version through `rust-toolchain.toml`, so `rustup` selects the right channel automatically.
- Docker
- A reachable [etcd](../overview/glossary.md#etcd) instance

## Step 1: Start etcd

For local development, start etcd in Docker:

```bash title="Start etcd"
docker run -d \
  --name aisix-etcd \
  -p 2379:2379 \
  -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --advertise-client-urls=http://0.0.0.0:2379 \
  --listen-client-urls=http://0.0.0.0:2379
```

## Step 2: Create a bootstrap config

Create a local `config.yaml` based on the example config. Place this file in the repo root — Step 3's `cargo run` looks for `config.yaml` relative to your current working directory.

```yaml title="config.yaml" {2-7,9-14}
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

## Step 3: Start the gateway

```bash title="Build and run locally"
cargo run -p aisix-server --bin aisix -- --config config.yaml
```

The first time you run this, `cargo` will compile several hundred dependencies before the gateway starts, which typically takes 3–5 minutes on common hardware. Subsequent runs are incremental and much faster.

Keep this terminal running. In a new terminal, you should now have:

- proxy listener on `http://127.0.0.1:3000`
- admin listener on `http://127.0.0.1:3001`

## Alternative: Run with Docker Compose

If you only want to run a standalone gateway locally and don't need a Rust toolchain, use the published gateway image with Docker Compose instead of Steps 1–3. This brings up etcd and AISIX together.

Create a `config.yaml` next to the compose file. It is the same as the Step 2 config, except the etcd endpoint points at the `etcd` service name instead of `127.0.0.1`:

```yaml title="config.yaml"
etcd:
  endpoints:
    - "http://etcd:2379"
  prefix: "/aisix"
  dial_timeout_ms: 5000
  request_timeout_ms: 5000

proxy:
  addr: "0.0.0.0:3000"
  request_body_limit_bytes: 10485760

admin:
  addr: "0.0.0.0:3001"
  admin_keys:
    - "YOUR_ADMIN_KEY"

observability:
  service_name: "aisix"
  log_level: "info"
  access_log: true

cache:
  backend: "memory"
```

```yaml title="docker-compose.yml"
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    command:
      - /usr/local/bin/etcd
      - --advertise-client-urls=http://0.0.0.0:2379
      - --listen-client-urls=http://0.0.0.0:2379
    ports:
      - "2379:2379"

  aisix:
    image: ghcr.io/api7/aisix:dev
    command: ["--config", "/etc/aisix/config.yaml"]
    volumes:
      - ./config.yaml:/etc/aisix/config.yaml:ro
    ports:
      - "3000:3000"
      - "3001:3001"
    depends_on:
      - etcd
```

```bash title="Start the stack"
docker compose up -d
```

:::note
`ghcr.io/api7/aisix:dev` tracks the `main` branch. For a reproducible deployment, pin a released version tag (for example `ghcr.io/api7/aisix:v1.2.3`) once one is available.
:::

The proxy listener is now on `http://127.0.0.1:3000` and the admin listener on `http://127.0.0.1:3001`, the same as the local build, so the verification below applies unchanged. To stop the stack, run `docker compose down`.

## Step 4: Verify the listeners

Both listeners expose an unauthenticated liveness route at `/livez`. The proxy and admin handlers share the same response shape, so you can probe either with the same expectation.

Verify the proxy listener:

```bash title="Check proxy liveness"
curl -sS http://127.0.0.1:3000/livez
```

Verify the admin listener:

```bash title="Check admin liveness"
curl -sS http://127.0.0.1:3001/livez
```

## Expected Result

A healthy gateway returns `200 OK` with the plain-text body `ok` on both listeners:

```text
ok
```

The body is intentionally minimal — the unauthenticated liveness route does not expose snapshot counts or registered providers. During shutdown the same routes return `500 Internal Server Error` with a body ending in `livez check failed`, which Kubernetes-style probes can match on.

For more detail, append `?verbose=1`. The verbose body is human-readable plain text suitable for `curl`, ending with `livez check passed` when the process is healthy.

For authenticated per-model operator health after boot, use `GET /admin/v1/health`. That endpoint returns the per-model `health` level (`0` healthy, `1` degraded, `2` down) for every model in the current snapshot. See [Health Checks](../operations/health-checks.md) for the operator-facing routes.

:::note
This quickstart only verifies gateway bootstrap. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are managed after boot through the admin API.
:::

## Cleanup

Stop the gateway process (Ctrl-C in its terminal) and remove the etcd container so you don't leak local state:

```bash title="Stop the etcd container"
docker rm -f aisix-etcd
```

If you created admin resources later (models, API keys, provider keys), delete them through the admin API before stopping etcd, or remove the etcd `--prefix` keyspace if you want a clean slate.

## Next Steps

- Review [What Is AISIX AI Gateway](../overview/what-is-aisix-ai-gateway.md).
- Compare [Deployment Modes](../overview/deployment-modes.md).
- Continue to [First Model, First Key, First Request](first-model-first-key-first-request.md).
- Learn the current [OpenAI-Compatible API](../integration/openai-compatible-api.md).
