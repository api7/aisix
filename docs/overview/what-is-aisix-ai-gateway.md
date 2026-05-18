---
title: What Is AISIX AI Gateway
description: Learn what AISIX AI Gateway is, what problems it solves, and how it differs from a direct provider integration.
sidebar_position: 1
---

AISIX AI Gateway is an AI [gateway](glossary.md#gateway) that sits between your applications and upstream LLM providers. It gives platform teams a single operational layer for routing, governing, and observing model traffic without forcing application teams to manage every provider integration directly.

## What Problems It Solves

Use AISIX AI Gateway when you need to:

- expose a consistent OpenAI-compatible API to internal applications or AI agents
- route requests to multiple upstream providers through one gateway surface
- control access with gateway API keys and model allowlists
- enforce per-key rate limits and spend budgets
- add cache, guardrail, and observability layers at the gateway boundary
- separate operator configuration from application integration

Instead of embedding provider credentials and traffic policy into every client, you configure those concerns once at the gateway layer.

## What It Looks Like In Practice

At runtime, AISIX AI Gateway serves two main surfaces:

- a **proxy surface** for client traffic
- an **admin surface** for operator-managed configuration

The proxy surface currently includes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/messages`
- `POST /v1/rerank`
- `POST /v1/responses`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`
- `POST /v1/images/generations`
- `ANY /passthrough/:provider/*rest`

The admin surface currently manages:

- models
- API keys
- provider keys
- guardrails
- cache policies
- observability exporters

## Who It Is For

### Platform Engineers

AISIX AI Gateway gives platform teams a place to centralize:

- provider credentials
- model routing
- authentication and authorization
- cost and traffic controls
- observability

### AI Agent Developers

AISIX AI Gateway lets AI agent developers target a stable client-facing API instead of coupling directly to every provider's native integration details.

Today, that includes:

- OpenAI-compatible usage through `/v1/chat/completions` and related endpoints
- Anthropic-style usage through `/v1/messages`
- provider-specific escape hatches through `/passthrough/:provider/*rest`

## Supported Providers Today

The current provider enum includes:

- `openai`
- `anthropic`
- `google`
- `deepseek`
- `cohere`
- `jina`

Provider support is not identical across every endpoint. The current high-level support summary is captured in the [Feature Matrix](feature-matrix.md), and the current provider-oriented reference lives in [Provider Compatibility](../reference/provider-compatibility.md).

## Deployment Modes

AISIX AI Gateway can be used in two modes:

### Self-Hosted Gateway

You run the gateway directly and manage bootstrap configuration, dynamic resources, and deployment yourself.

### AISIX Cloud Managed Data Plane

[AISIX Cloud](glossary.md#aisix-cloud) adds a managed [control plane](glossary.md#control-plane) for environments, certificates, and Cloud workflows while the [data plane](glossary.md#data-plane) still runs as AISIX AI Gateway.

See [Deployment Modes](deployment-modes.md) for the comparison.

## Current Product Boundary

`AISIX AI Gateway` is the primary product documented in this repo.

`AISIX Cloud` is the managed extension that adds environment management, certificate issuance, projection, usage-event collection, and Cloud-specific workflows.

:::note
Main docs describe current, verified behavior. Planned capabilities are tracked in the [Roadmap](../roadmap.md).
:::

## Related Pages

- [Deployment Modes](deployment-modes.md)
- [Core Concepts](core-concepts.md)
- [Feature Matrix](feature-matrix.md)
- [Provider Compatibility](../reference/provider-compatibility.md)
- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
