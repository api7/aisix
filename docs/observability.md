# Observability

AI Gateway exports traces and metrics via OpenTelemetry (OTLP).

```
AI Gateway ──OTLP gRPC──► Jaeger    (traces)   :4317
            ──OTLP HTTP──► Prometheus (metrics)  :9090
                                         └──► Grafana  (dashboard) :3100
```

## Prerequisites

Build with trace feature enabled:

```bash
cargo build --release --features trace
```

Metrics are always enabled. Tracing requires `--features trace`.

## Quick Start

### 1. Jaeger (Traces)

```bash
docker run --rm --name jaeger -d \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.16.0
```

### 2. Prometheus (Metrics)

```bash
docker run --rm --name prometheus -d \
  -p 9090:9090 \
  quay.io/prometheus/prometheus:v3.1.0 \
  --web.enable-otlp-receiver
```

### 3. Grafana (Dashboard)

```bash
docker run --rm --name grafana -d \
  --add-host=host.docker.internal:host-gateway \
  -p 3100:3000 \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning:ro \
  -v $(pwd)/grafana/dashboards:/var/lib/grafana/dashboards:ro \
  grafana/grafana:latest
```

> `--add-host` is required on Linux. macOS/Windows Docker Desktop resolves `host.docker.internal` automatically.

### 4. Start AI Gateway

```bash
RUST_LOG=info ./target/release/ai-gateway
```

## Web UI

| Service | URL |
|---------|-----|
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3100 (admin / admin) |

## Metrics Reference

AI Gateway exports the following metrics. Prometheus metric names include unit suffixes added by the OTLP exporter.

### Counters

| Prometheus Metric | Labels | Description |
|---|---|---|
| `aisix_request_count_total` | `method`, `endpoint`, `status_code` | Total HTTP requests processed |
| `aisix_token_count_total` | `model`, `type` | Token usage. `type` values: `prompt`, `completion`, `total` |

**Label values:**
- `method` — HTTP method (e.g., `POST`)
- `endpoint` — Matched route path (e.g., `/v1/chat/completions`)
- `status_code` — HTTP response status code (e.g., `200`, `400`, `500`)
- `model` — Model name (alias configured via Admin API)
- `type` — Token type: `prompt` (input tokens), `completion` (output tokens), `total` (sum)

### Histograms

Each histogram produces three series: `_bucket` (for quantiles), `_count` (number of observations), `_sum` (total value).

| Prometheus Metric | Labels | Description |
|---|---|---|
| `aisix_request_latency_milliseconds_bucket` | `method`, `endpoint`, `status_code` | End-to-end request latency (from request received to response body fully sent) |
| `aisix_request_latency_milliseconds_count` | `method`, `endpoint`, `status_code` | |
| `aisix_request_latency_milliseconds_sum` | `method`, `endpoint`, `status_code` | |
| `aisix_llm_latency_milliseconds_bucket` | `model` | LLM provider latency (from calling upstream to receiving response) |
| `aisix_llm_latency_milliseconds_count` | `model` | |
| `aisix_llm_latency_milliseconds_sum` | `model` | |
| `aisix_llm_first_token_latency_milliseconds_bucket` | `model` | Time to first token (streaming requests only, from request start to first token received) |
| `aisix_llm_first_token_latency_milliseconds_count` | `model` | |
| `aisix_llm_first_token_latency_milliseconds_sum` | `model` | |

## Grafana Dashboard

The `grafana/` directory contains provisioning configs that auto-load a pre-built dashboard:

```
grafana/
├── provisioning/
│   ├── datasources/datasource.yml
│   └── dashboards/dashboards.yml
└── dashboards/ai-gateway.json
```

Dashboard panels:

- **Overview** — Total requests, total tokens, avg request latency, avg LLM latency
- **Request Throughput** — QPS by endpoint, QPS by status code
- **Latency** — Request P50/P90/P99, LLM P50/P90/P99, first token latency by model, LLM latency by model
- **Token Usage** — Prompt vs completion rate, usage by model, cumulative tokens, distribution pie chart

## Verify

### Traces

1. Send a request to the proxy API
2. Open Jaeger UI (http://localhost:16686) → select service `ai-gateway` → Find Traces

### Metrics

**Prometheus UI:**

1. Open http://localhost:9090
2. Enter query in the expression input, e.g., `aisix_request_count_total`
3. Click "Execute" to view results

**Grafana Dashboard:**

1. Open http://localhost:3100 (login: admin / admin)
2. Navigate to Dashboards → "AI Gateway"
3. View pre-built panels for requests, latency, and token usage

## Stop

```bash
docker stop jaeger prometheus grafana
```
