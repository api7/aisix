# Observability

AI Gateway exports traces and metrics through OpenTelemetry (OTLP).

```
AI Gateway ──OTLP gRPC──► Jaeger    (traces)   :4317
            ──OTLP HTTP──► Prometheus (metrics)  :9090
                                         └──► Grafana  (dashboard) :3100
```

## At a Glance

- Metrics are pushed from AI Gateway to Prometheus OTLP HTTP receiver.
- Traces are exported from AI Gateway to Jaeger OTLP gRPC receiver.
- Grafana reads from Prometheus and provides dashboards.

### Important Semantics

- `http://127.0.0.1:9090/api/v1/otlp/v1/metrics` is a write endpoint for OTLP push.
- It is not a Prometheus scrape endpoint.
- `scrape_configs` in `prometheus.yml` is only for pull-based targets (for example, Prometheus itself).

## Prerequisites

Build AI Gateway with trace feature enabled:

```bash
cargo build --release --features trace
```

Metrics are always enabled. Tracing requires `--features trace`.

## Quick Start (Local)

### 1. Jaeger (Traces)

```bash
docker run --rm --name jaeger -d \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.16.0
```

### 2. Prometheus (Metrics)

Create a minimal `prometheus.yml`:

```bash
cat > prometheus.yml <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
EOF
```

Run Prometheus with OTLP receiver enabled:

```bash
docker run --rm --name prometheus -d \
  -p 9090:9090 \
  quay.io/prometheus/prometheus:v3.1.0 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  --web.enable-otlp-receiver
```

`--web.enable-otlp-receiver` is required so Prometheus can accept pushed OTLP metrics.

### 3. Grafana (Dashboard)

```bash
docker run --rm --name grafana -d \
  -p 3100:3000 \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning:ro \
  -v $(pwd)/grafana/dashboards:/var/lib/grafana/dashboards:ro \
  --add-host=host.docker.internal:host-gateway \
  grafana/grafana:latest
```

`--add-host` is required on Linux. macOS/Windows Docker Desktop resolves `host.docker.internal` automatically.

### 4. Start AI Gateway

```bash
RUST_LOG=info ./target/release/aisix
```

## Service URLs

| Service | URL |
|---------|-----|
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3100 (admin / admin) |

## Verify

### Verify Traces

1. Send a request to the proxy API.
2. Open Jaeger at `http://localhost:16686`.
3. Select service `aisix`, then click `Find Traces`.

### Verify Metrics

1. Open Prometheus at `http://localhost:9090`.
2. Run a query such as `aisix_request_count_total`.
3. Open Grafana at `http://localhost:3100` (admin/admin).
4. Go to Dashboards and open `AI Gateway`.

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

The `grafana/` directory contains provisioning configs that auto-load a built-in dashboard:

```
grafana/
├── provisioning/
│   ├── datasources/datasource.yml
│   └── dashboards/dashboards.yml
└── dashboards/aisix.json
```

Dashboard panels:

- **Overview** — Total requests, total tokens, avg request latency, avg LLM latency
- **Request Throughput** — QPS by endpoint, QPS by status code
- **Latency** — Request P50/P90/P99, LLM P50/P90/P99, first token latency by model, LLM latency by model
- **Token Usage** — Prompt vs completion rate, usage by model, cumulative tokens, distribution pie chart

## Stop

```bash
docker stop jaeger prometheus grafana
```
