---
title: Glossary
description: Operational definitions of the cross-cutting AISIX AI Gateway and AISIX Cloud terms used throughout the documentation.
sidebar_position: 5
---

This page defines the terms that appear across AISIX AI Gateway and AISIX Cloud docs. Each entry explains what the term does at runtime.

Page-local identifiers, such as the `bedrock` guardrail kind or the `YOUR_ADMIN_KEY` placeholder format, are explained inline at first use on the page where they appear.

## gateway

The AISIX runtime binary that accepts caller traffic on the proxy listener and forwards it to upstream model providers. Synonym: data plane.

## data plane

The request-handling tier. It receives caller traffic, applies authentication, model access checks, routing, rate limits, cache policies, guardrails, budget checks when enabled, and observability, then forwards requests to upstream providers.

## control plane

The management tier. In standalone deployments, the control plane is the gateway's admin listener, which writes dynamic resources to [etcd](#etcd). In AISIX Cloud, the control plane is a hosted service that projects environment-scoped configuration to managed data planes.

## AISIX Cloud

The managed control-plane service operated separately from the gateway runtime. It provides hosted environment management and Cloud-only controls such as budget checks. See [Deployment modes](deployment-modes.md) for the comparison.

## model

The caller-facing model alias clients send in the request body. A direct model maps that alias to an upstream provider key and upstream model name. A routing model maps the alias to one or more target models and lets the gateway choose the target at request time.

## API key

Also called the **caller key**. The bearer token your clients send in the `Authorization` header on the proxy listener.

API keys are created through `POST /admin/v1/apikeys`. The data plane stores `key_hash`, not plaintext. In the standalone admin API, the operator chooses or generates the plaintext bearer, hashes it, and writes the SHA-256 hash to the API-key resource.

The rotate endpoint, `POST /admin/v1/apikeys/:id/rotate`, is the only endpoint that returns a server-generated plaintext key. It returns that plaintext once.

## provider key

The upstream provider credential the gateway uses on outbound requests. Created via the admin API's `POST /admin/v1/provider_keys`. Distinct from the [API key](#api-key) your callers send to the gateway.

## rate-limit policy

A standalone rate-limit rule that targets a scope such as API key, model, team, or member. The proxy evaluates matching policy rows together with inline limits on API keys and models.

## guardrail

A request- or response-policy object applied by the gateway. Configured via the admin API's `/admin/v1/guardrails`. Current kinds include local `keyword` guardrails and remote guardrails for AWS Bedrock and Azure Content Safety. See [Core concepts § Guardrail](core-concepts.md#guardrail) for the current boundary.

## cache policy

A policy object that controls when chat-completion response cache lookup and storage apply. The runtime cache backend is selected from bootstrap configuration; a policy controls matching and TTL, not the process-level backend.

## observability exporter

An admin resource that ships per-request span telemetry over OTLP/HTTP to an external backend such as Grafana Tempo, Honeycomb, or Langfuse via OTLP.

Configure an observability exporter when you want request traces from gateway proxy activity in your tracing backend. This is separate from process-wide bootstrap observability such as service name, log level, and the Prometheus scrape endpoint. See [Bootstrap configuration](../configuration/bootstrap-config.md).

## etcd

The key-value store the gateway uses for dynamic configuration. The data plane watches etcd for live resource updates, so most configuration changes do not require a gateway restart.

## Next steps

- [Core concepts](core-concepts.md) — see how the main resources fit together.
- [Deployment modes](deployment-modes.md) — compare standalone and managed data-plane operation.
- [Bootstrap configuration](../configuration/bootstrap-config.md) — configure the gateway process.
