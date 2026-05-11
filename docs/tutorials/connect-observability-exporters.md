---
title: Connect Observability Exporters
description: Send AISIX AI Gateway telemetry to an OTLP/HTTP endpoint through the observability exporter resource.
sidebar_position: 84
---

This tutorial shows how to add an OTLP/HTTP exporter and verify that the data plane sends telemetry to it.

## Goal

Configure `kind: "otlp_http"` and confirm that gateway traffic produces exporter traffic.

## Flow

1. create an observability exporter
2. wait for propagation
3. send a real gateway request
4. verify span or telemetry reception on the receiver side

## Related Pages

- [Observability Exporters](../configuration/observability-exporters.md)
- [Metrics And Logs](../operations/metrics-and-logs.md)
- [AISIX Cloud Overview](../cloud/overview.md)
