---
title: AISIX AI Gateway Documentation
description: Official documentation for AISIX AI Gateway and AISIX Cloud, including quickstarts, integration guides, configuration, operations, API reference, and roadmap.
sidebar_position: 1
---

AISIX AI Gateway is an AI gateway for platform engineers and AI agent developers who need a consistent way to route, govern, and observe LLM traffic across multiple providers. AISIX Cloud extends that gateway with a managed control plane and managed data-plane workflows.

This documentation set is organized for two primary audiences:

- **Platform engineers** who deploy, configure, and operate the gateway.
- **AI agent developers** who integrate through OpenAI-compatible or Anthropic-style APIs.

## Choose Your Path

### I want to evaluate the product

- Start with [What Is AISIX AI Gateway](overview/what-is-aisix-ai-gateway.md).
- Compare [Deployment Modes](overview/deployment-modes.md).
- Review the [Feature Matrix](overview/feature-matrix.md).

### I want to get a gateway running quickly

- Follow the [Self-Hosted Quickstart](quickstart/self-hosted.md).
- Then continue with the first model and API key setup once the next quickstart pages land.
- If you are evaluating the managed control plane, use the [Deployment Modes](overview/deployment-modes.md) and [Roadmap](roadmap.md) pages first.

### I want to integrate an SDK or client

- Start with [What Is AISIX AI Gateway](overview/what-is-aisix-ai-gateway.md) and the current gateway surface summary on this page.
- Full integration guides are part of the upcoming docs expansion tracked in the [Roadmap](roadmap.md).

### I want to operate the gateway in production

- Start with the [Self-Hosted Quickstart](quickstart/self-hosted.md).
- Use the [Feature Matrix](overview/feature-matrix.md) to understand current coverage.
- Follow the [Roadmap](roadmap.md) for upcoming operations pages.

## Documentation Structure

- [Overview](overview/what-is-aisix-ai-gateway.md)
- [Quickstart](quickstart/self-hosted.md)
- Client Integration
- Gateway Configuration
- AISIX Cloud
- Operations
- Reference
- Tutorials
- [Roadmap](roadmap.md)

## Product Boundary

`AISIX AI Gateway` is the primary product documented here.

`AISIX Cloud` is the managed extension that adds control-plane, environment, certificate, and billing workflows on top of the gateway.

:::note
Main documentation pages describe current, verified behavior. Planned or not-yet-implemented capabilities belong in the [Roadmap](roadmap.md).
:::

## Current Gateway Surface

The gateway currently exposes these client-facing routes:

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

Detailed integration and reference pages are part of the ongoing docs rebuild tracked in the [Roadmap](roadmap.md).

## Current Admin Surface

The standalone gateway admin listener currently supports:

- models
- API keys
- provider keys
- guardrails
- cache policies
- observability exporters
- health
- metrics
- OpenAPI
- in-process playground

Detailed configuration and reference pages are part of the ongoing docs rebuild tracked in the [Roadmap](roadmap.md).

## Next Steps

- Read [What Is AISIX AI Gateway](overview/what-is-aisix-ai-gateway.md).
- Set up a [Self-Hosted Gateway](quickstart/self-hosted.md).
- Review the [Roadmap](roadmap.md).
