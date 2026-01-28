# ai-gateway

## Prerequisites

- Rust (latest stable/nightly version)

## Build and Run

```bash
RUST_LOG=info cargo run

## Or enable OTel-based tracing support
docker run --rm --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 5778:5778 \
  -p 9411:9411 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.14.0

RUST_LOG=info cargo run --features trace
```

## Provision config data

### Config file (config.yaml)

```yaml
deployment:
  etcd:
    host:
      - "http://127.0.0.1:2379"
    prefix: /aisix
    timeout: 30
```

### ETCD

#### Chat Completions

```bash
etcdctl put /aisix/apikeys/user1 '{"key":"user1","allowed_models": ["@my-ds/chat","@my-gemini/gemini-2.5-flash","mock", "@my-gemini/embed"]}'

etcdctl put /aisix/apikeys/user2 '{"key":"user2","allowed_models": []}'

etcdctl put /aisix/models/deepseek-chat '{"name":"@my-ds/chat","model":"deepseek/deepseek-chat","provider_config":{"api_key":"<your_key>"}}'

etcdctl put /aisix/models/mock '{"name":"mock","model":"mock/mock","provider_config":{}}'

etcdctl put /aisix/models/gemini-2_5-flash '{"name":"@my-gemini/gemini-2.5-flash","model":"gemini/gemini-2.5-flash","provider_config":{"api_key":"<your_key>"}}'
```

#### Embeddings

```bash
etcdctl put /aisix/models/gemini-embedding '{"name":"@my-gemini/embed","model":"gemini/gemini-embedding-001","provider_config":{"api_key":"<your_key>"}}'
```
