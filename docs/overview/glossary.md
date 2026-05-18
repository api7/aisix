---
title: Glossary
description: Operational definitions of the cross-cutting AISIX AI Gateway and AISIX Cloud terms used throughout the documentation.
sidebar_position: 5
---

This page collects the cross-cutting terms used across AISIX AI Gateway and AISIX Cloud documentation. Each entry is a 1–3 sentence operational definition — what the term *does at runtime*, not marketing prose.

Page-local codenames and one-off identifiers (for example, the `bedrock` guardrail kind or the `YOUR_ADMIN_KEY` placeholder format) are explained inline at first use on the page where they appear.

## gateway

The AISIX runtime binary that accepts caller traffic on the proxy listener and forwards it to upstream model providers. Synonym: data plane.

## data plane

The request-handling tier — the gateway itself. Receives caller traffic, applies caching, guardrails, budgets, and routing, forwards to upstream providers, and returns responses.

## control plane

The management tier. Stores model, API-key, provider-key, guardrail, and cache-policy rows in [etcd](#etcd); the data plane reads from etcd. In standalone deployments the control plane is the gateway's own admin listener. In AISIX Cloud the control plane is a separate hosted service that projects state down to the gateway via etcd-over-TLS.

## AISIX Cloud

The managed control-plane service operated separately from the OSS gateway. Provides multi-tenant team, project, and budget concepts that don't exist in standalone-only mode. See [Deployment Modes](deployment-modes.md) for the comparison.

## API key

Also called the **caller key**. The bearer token your clients send in the `Authorization` header on the proxy listener. Created via the admin API's `POST /admin/v1/apikeys`. The data plane stores `key_hash`, not plaintext — the caller chooses (or generates) the plaintext bearer and SHA-256-hashes it locally before submission, so the gateway never sees or returns the plaintext at create time. The only endpoint that emits a server-generated plaintext is `POST /admin/v1/apikeys/:id/rotate`, which returns the new plaintext exactly once.

## provider key

The upstream provider's credential (for example an OpenAI `sk-...` key) the gateway uses to authenticate to the provider on outbound requests. Created via the admin API's `POST /admin/v1/provider_keys`. Distinct from the [API key](#api-key) your callers send to the gateway.

## guardrail

A request- or response-policy object applied by the gateway. Configured via the admin API's `/admin/v1/guardrails`. Current schema supports the in-process `keyword` backend and an AWS Bedrock backend behind a feature flag. See [Core Concepts § Guardrail](core-concepts.md#guardrail) for the full kind list.

## Observability Exporter

A per-row admin resource that ships per-request span telemetry — derived from gateway `UsageEvent` records — over OTLP/HTTP to an external backend such as Grafana Tempo, Honeycomb, or Langfuse via OTLP. Configure one when you want a per-request trace of gateway proxy activity forwarded to your existing tracing backend. Distinct from process-wide bootstrap observability (service name, log level, Prometheus scrape endpoint) configured in [Bootstrap Configuration](../configuration/bootstrap-config.md).

## etcd

The key-value store the gateway uses for control-plane state. The admin listener writes dynamic resources (models, API keys, provider keys, guardrails, cache policies, observability exporters) into etcd; the data plane watches etcd for live config updates, so restart-free changes ride this path. The admin-side etcd client lives in `crates/aisix-admin/src/etcd_store.rs`.

## Related pages

- [What Is AISIX AI Gateway](what-is-aisix-ai-gateway.md)
- [Core Concepts](core-concepts.md)
- [Deployment Modes](deployment-modes.md)
- [Bootstrap Configuration](../configuration/bootstrap-config.md)
