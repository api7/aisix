---
title: Run from Source
description: Build AISIX AI Gateway from the repository checkout, start it locally, and verify that the proxy and admin listeners are reachable.
sidebar_position: 16
---

This guide shows how to build AISIX AI Gateway from the repository checkout, start it with the local example configuration, and verify that the proxy and admin listeners are reachable.

This guide shows how to run AISIX directly from the repository so you can inspect or modify the gateway while it is running. For the fastest container-based path, start with the [Quickstart](../quickstart).

By the end of this guide, you will have:

1. Started local etcd.
2. Built and started the `aisix` binary from source.
3. Verified the proxy and admin listeners.

This page stops at gateway bootstrap. A source-built gateway still needs the same provider key, model alias, and caller API key as the container quickstart before it can proxy model traffic.

## Prerequisites

- Git
- **Rust 1.93 or newer with `cargo`.** Install via [rustup](https://rustup.rs) and verify with `cargo --version`. The repo pins this version through `rust-toolchain.toml`, so `rustup` selects the right channel automatically.
- Docker
- curl

## Step 1: Clone the repository

```shell
git clone https://github.com/api7/ai-gateway.git
cd ai-gateway
```

## Step 2: Start etcd

For local development, start etcd in Docker:

```shell
docker run -d \
  --name aisix-etcd \
  -p 2379:2379 \
  -p 2380:2380 \
  quay.io/coreos/etcd:v3.5.18 \
  /usr/local/bin/etcd \
  --advertise-client-urls=http://0.0.0.0:2379 \
  --listen-client-urls=http://0.0.0.0:2379
```

## Step 3: Create the bootstrap config

Create a local `config.yaml` based on the example config:

```shell
cp config.example.yaml config.yaml
```

The example configuration points at local etcd and binds:

- proxy listener on `0.0.0.0:3000`
- admin listener on `127.0.0.1:3001`
- admin key `admin-local-only-change-me`

If either port is already in use on your machine, update `proxy.addr` or `admin.addr` in `config.yaml` before starting the gateway.

## Step 4: Build and start the gateway

```shell
cargo run -p aisix-server -- --config config.yaml
```

The package defines `aisix` as its binary, so `cargo run -p aisix-server` starts the gateway. The first run compiles the Rust workspace and can take several minutes; later runs are incremental and much faster.

Keep this terminal running. In a new terminal, you should now have:

- proxy listener on `http://127.0.0.1:3000`
- admin listener on `http://127.0.0.1:3001`

## Step 5: Verify the listeners

Both listeners expose an unauthenticated liveness route at `/livez`. The proxy and admin handlers share the same response shape, so you can probe either with the same expectation.

Verify the proxy listener:

```shell
curl -sS http://127.0.0.1:3000/livez
```

Verify the admin listener:

```shell
curl -sS http://127.0.0.1:3001/livez
```

## Expected result

A healthy gateway returns `200 OK` with the plain-text body `ok` on both listeners:

```text
ok
```

The body is intentionally minimal — the unauthenticated liveness route does not expose snapshot counts or registered providers. During shutdown the same routes return `500 Internal Server Error` with a body ending in `livez check failed`, which Kubernetes-style probes can match on.

For more detail, append `?verbose=1`. The verbose body is human-readable plain text suitable for `curl`, ending with `livez check passed` when the process is healthy.

For authenticated per-model operator health after boot, use `GET /admin/v1/health`. That endpoint returns the per-model `health` level (`0` healthy, `1` degraded, `2` down) for every model in the current snapshot. See [Health checks](../operations/health-checks.md) for the operator-facing routes.

:::note
This quickstart only verifies gateway bootstrap. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are managed after boot through the admin API.
:::

## Create traffic resources next

At this point, the gateway process is running but no model traffic can pass through it yet. Create the same minimum resources used by the main quickstart:

- a provider key for the upstream credential
- a model alias that callers send to AISIX
- a caller API key that is allowed to use that alias

Continue with [Step 6 of the Quickstart](../quickstart#step-6-export-your-local-variables), using the admin listener at `http://127.0.0.1:3001` and the proxy listener at `http://127.0.0.1:3000`.

## Cleanup

Stop the gateway process (Ctrl-C in its terminal) and remove the etcd container so you don't leak local state:

```shell
docker rm -f aisix-etcd
```

If you created admin resources later (models, API keys, provider keys), delete them through the admin API before stopping etcd, or remove the etcd `--prefix` keyspace if you want a clean slate.

## Next steps

Continue in this order:

1. [Quickstart Step 6](../quickstart#step-6-export-your-local-variables) to create a provider key, model alias, and caller API key.
2. [OpenAI SDK quickstart](openai-sdk.md) after you have a working model alias and caller API key.
3. [OpenAI-compatible API](../integration/openai-compatible-api.md) when you are ready to look at the full client contract.
