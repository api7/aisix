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
- Continue with [First Model, First Key, First Request](quickstart/first-model-first-key-first-request.md).
- If you are evaluating the managed control plane, use the [Deployment Modes](overview/deployment-modes.md) and [Roadmap](roadmap.md) pages first.

### I want to integrate an SDK or client

- Start with the [OpenAI-Compatible API](integration/openai-compatible-api.md).
- Use the quickstarts to configure a working model and caller key first.

### I want to operate the gateway in production

- Start with the [Self-Hosted Quickstart](quickstart/self-hosted.md).
- Review [Bootstrap Configuration](configuration/bootstrap-config.md).
- Use [Admin API](configuration/admin-api.md) to manage dynamic resources.
- Continue with the dedicated configuration pages for models, keys, routing, caching, and guardrails.
- Use the [Feature Matrix](overview/feature-matrix.md) to understand current coverage.
- Use the operations, reference, cloud, and tutorial sections for the current docs set.

## Documentation Structure

- [Overview](overview/what-is-aisix-ai-gateway.md)
- [Quickstart](quickstart/self-hosted.md)
- [Client Integration](integration/openai-compatible-api.md)
- [Gateway Configuration](configuration/bootstrap-config.md)
- [Models](configuration/models.md)
- [API Keys](configuration/api-keys.md)
- [AISIX Cloud](cloud/overview.md)
- [Operations](operations/production-deployment.md)
- [Reference](reference/proxy-api-reference.md)
- [Tutorials](tutorials/build-a-virtual-model-with-failover.md)
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

Start with [OpenAI-Compatible API](integration/openai-compatible-api.md) and the [Reference](reference/proxy-api-reference.md) section.

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

Start with [Admin API](configuration/admin-api.md), [Bootstrap Configuration](configuration/bootstrap-config.md), and the dedicated configuration pages.

## Next Steps

- Read [What Is AISIX AI Gateway](overview/what-is-aisix-ai-gateway.md).
- Set up a [Self-Hosted Gateway](quickstart/self-hosted.md).
- Review the [Roadmap](roadmap.md).
