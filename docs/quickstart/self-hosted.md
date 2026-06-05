---
title: Run from Source
description: Build AISIX AI Gateway from the repository checkout, start it locally, and verify that the proxy and admin listeners are reachable.
sidebar_position: 16
toc_max_heading_level: 2
---

Build AISIX AI Gateway from the repository checkout, start it with the local example configuration, and verify that the proxy and admin listeners are reachable. For the fastest container-based path, start with the [Quickstart](../quickstart).

A source-built gateway still needs the same provider key, model alias, and AISIX API key as the container quickstart before it can proxy model traffic.

## Prerequisites

* Install Git to clone the AISIX AI Gateway repository.
* Install [Docker](https://docs.docker.com/get-docker/) to start local **etcd**.
* Install [cURL](https://curl.se/) to verify the proxy and admin listeners.
* Install Rust 1.93 or newer with `cargo`. Install Rust with [rustup](https://rustup.rs). The repository pins the required Rust version through `rust-toolchain.toml`.

## Prepare the Source Checkout

Clone the repository, start a local etcd container, and create the bootstrap configuration before running the gateway from source.

### Clone the Repository

Clone the AI Gateway repository and enter the checkout:

```shell
git clone https://github.com/api7/ai-gateway.git
cd ai-gateway
```

### Start Etcd

Start etcd in Docker for this local source-run path:

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

### Create the Bootstrap Config

Create a local `config.yaml` based on the example config:

```shell
cp config.example.yaml config.yaml
```

The example configuration points at local etcd and binds:

The proxy listener binds `0.0.0.0:3000`, the admin listener binds `127.0.0.1:3001`, and the local admin key is `admin-local-only-change-me`.

If either port is already in use on your machine, update `proxy.addr` or `admin.addr` in `config.yaml` before starting the gateway.

## Start the Gateway

Start the gateway with the local configuration file:

```shell
cargo run -p aisix-server -- --config config.yaml
```

The package defines `aisix` as its binary, so `cargo run -p aisix-server` starts the gateway. The first run compiles the Rust workspace and can take several minutes; later runs are incremental and much faster.

Leave the gateway process running. In a new terminal, verify the proxy listener at `http://127.0.0.1:3000` and the admin listener at `http://127.0.0.1:3001`.

## Verify Installation

Both listeners expose an unauthenticated liveness route at `/livez`. The proxy and admin handlers share the same response format, so you can probe either with the same expectation.

Verify the proxy listener:

```shell
curl -sS "http://127.0.0.1:3000/livez"
```

Verify the admin listener:

```shell
curl -sS "http://127.0.0.1:3001/livez"
```

A healthy gateway returns `200 OK` with the plain-text body `ok` on both listeners:

```text
ok
```

The liveness route confirms that the listener is reachable. For verbose liveness output, shutdown behavior, and per-model health, see [Health checks](../operations/health-checks.md).

:::note
This source-run path only verifies gateway bootstrap. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are managed after boot through the admin API.
:::

## Create Traffic Resources

At this point, the gateway process is running but no model traffic can pass through it yet. Create the same minimum resources used by the main quickstart: a provider key for the upstream credential, a model alias for the model name clients send to AISIX, and an AISIX API key that can use that alias.

Continue with [Export Local Variables](../quickstart#export-local-variables) in the Quickstart, using the admin listener at `http://127.0.0.1:3001` and the proxy listener at `http://127.0.0.1:3000`.

## Clean Up

Stop the gateway process (Ctrl-C in its terminal) and remove the etcd container so local state is not left behind:

```shell
docker rm -f aisix-etcd
```

If you created admin resources such as models, API keys, or provider keys, delete them through the admin API before stopping etcd, or remove the etcd `--prefix` keyspace if you want a clean slate.

## Next Steps

You have now prepared the local self-hosted gateway runtime. From here, follow the [Quickstart](../quickstart) to create the minimum provider key, model alias, and AISIX API key. To call the gateway from application code, continue with [OpenAI SDK Quickstart](openai-sdk.md) or [Anthropic SDK Quickstart](anthropic-sdk.md).
