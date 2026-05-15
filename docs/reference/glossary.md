---
title: Glossary
description: Glossary of key AISIX AI Gateway and AISIX Cloud terms.
sidebar_position: 65
---

- `AISIX AI Gateway`: the gateway product documented in this repo
- `AISIX Cloud`: the managed control-plane extension on top of the gateway
- `Model`: a caller-visible model alias or routing model
- `ProviderKey`: reusable upstream credential resource
- `ApiKey`: caller-facing gateway credential resource
- `Guardrail`: content-policy resource applied on chat paths
- `CachePolicy`: dynamic cache-control resource
- `ObservabilityExporter`: dynamic OTLP exporter resource
- `RateLimitPolicy`: standalone rate-limit rule stored in etcd, scoped by `api_key`, `model`, `team`, or `member`
- `Snapshot`: the in-memory config view used by the proxy hot path
- `Watch Supervisor`: the task that keeps the snapshot current from etcd
- `Managed DP`: a managed data plane operating under AISIX Cloud control-plane workflows

## Reading Tips

- when a page says `caller-facing`, it refers to the client contract on `/v1/*`
- when a page says `operator-facing`, it refers to the standalone admin or Cloud control surface
- when a page says `projected`, it means control-plane state has been materialized into the data-plane snapshot
