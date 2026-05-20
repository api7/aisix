---
title: AISIX AI Gateway Documentation
description: Official documentation for AISIX AI Gateway and AISIX Cloud, including quickstarts, integration guides, configuration, operations, API reference, and roadmap.
sidebar_position: 1
---

AISIX AI Gateway is an AI [gateway](overview/glossary.md#gateway) for platform engineers and AI agent developers who need a consistent way to route, govern, and observe LLM traffic across multiple providers. [AISIX Cloud](overview/glossary.md#aisix-cloud) extends that gateway with a managed [control plane](overview/glossary.md#control-plane) and managed [data plane](overview/glossary.md#data-plane) workflows.

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
- Use [OpenAI SDK Quickstart](quickstart/openai-sdk.md) if you already have an OpenAI client.
- Use [Anthropic SDK Quickstart](quickstart/anthropic-sdk.md) if you need the Anthropic-style `messages` API.
- If you signed up for AISIX Cloud and want to point a hosted control plane at your own gateway, follow the [Deployment Modes](overview/deployment-modes.md) and [Roadmap](roadmap.md) pages. If you want to run everything locally on your own machine, start with the [Self-Hosted Quickstart](quickstart/self-hosted.md) instead.
- For current Cloud bootstrap behavior, review [AISIX Cloud Managed Data Plane Quickstart](quickstart/aisix-cloud-managed-dp.md).

### I want to integrate an SDK or client

- Start with the [OpenAI-Compatible API](integration/openai-compatible-api.md).
- Use [Streaming](integration/streaming.md) for SSE behavior.
- Use [Tool Calling](integration/tool-calling.md) for agent-style integrations.
- Use [Anthropic Messages](integration/anthropic-messages.md) for Claude-style clients.
- Use [Errors And Retries](integration/errors-and-retries.md) for shared failure handling.
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
- [Streaming](integration/streaming.md)
- [Errors And Retries](integration/errors-and-retries.md)
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
- `ANY /passthrough/:provider/*rest` (`ANY` here means the route accepts every HTTP verb: GET, POST, PUT, DELETE, etc.)

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
