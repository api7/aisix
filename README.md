# ai-gateway

## Prerequisites

- Rust (latest stable/nightly version)

## Build and Run

```bash
RUST_LOG=info cargo run
```

## Provision config data

### Config file (config.yaml)

```
deployment:
  etcd:
    host:
      - "http://127.0.0.1:2379"
    prefix: /aisix
    timeout: 30
```

### ETCD

```bash
etcdctl put /aisix/apikeys/user1 '{"key":"user1","allowed_models": ["@my-ds/chat", "mock"]}'

etcdctl put /aisix/apikeys/user2 '{"key":"user2","allowed_models": []}'

etcdctl put /aisix/models/deepseek-chat '{"name":"@my-ds/chat","model":"deepseek/deepseek-chat","provider_config":{"api_key":"sk-<your_key>"}}'

etcdctl put /aisix/models/mock '{"name":"mock","model":"mock/mock","provider_config":{}}'
```
