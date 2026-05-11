---
title: Self-Hosted Quickstart
description: Deploy a self-hosted AISIX AI Gateway instance and verify that the proxy and admin listeners are reachable.
sidebar_position: 10
---

This guide shows how to start a self-hosted AISIX AI Gateway instance with the local example configuration and verify that both the proxy and admin surfaces are reachable.

## Prerequisites

- Docker
- a reachable etcd instance

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

Create a local `config.yaml` based on the example config.

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
cargo run -- --config config.yaml
```

In another terminal, you should now have:

- proxy listener on `http://127.0.0.1:3000`
- admin listener on `http://127.0.0.1:3001`

## Step 4: Verify the listeners

Verify the proxy listener:

```bash title="Check proxy health"
curl -s http://127.0.0.1:3000/health
```

Verify the admin listener:

```bash title="Check admin health"
curl -s http://127.0.0.1:3001/health
```

## Expected Result

The proxy health response should include a JSON body like:

```json
{
  "status": "ok",
  "models": 0,
  "apikeys": 0,
  "providers": 0
}
```

The admin health response should include a JSON body like:

```json
{
  "status": "ok",
  "models": 0,
  "apikeys": 0
}
```

:::note
This quickstart only verifies gateway bootstrap. Dynamic resources such as models, API keys, provider keys, guardrails, cache policies, and observability exporters are managed after boot through the admin API.
:::

## Next Steps

- Review [What Is AISIX AI Gateway](../overview/what-is-aisix-ai-gateway.md).
- Compare [Deployment Modes](../overview/deployment-modes.md).
- Track the next quickstart and integration pages in the [Roadmap](../roadmap.md).
