---
title: Core Concepts
description: Understand the core AISIX AI Gateway and AISIX Cloud concepts, including models, provider keys, API keys, routing models, guardrails, cache policies, and observability exporters.
sidebar_position: 3
---

This page defines the core objects and terms used across AISIX AI Gateway and AISIX Cloud.

## Model

A `Model` is the resource clients target through the gateway.

For direct models, a model includes:

- `display_name`
- `provider`
- `model_name`
- `provider_key_id`
- optional timeout, rate limit, and cost metadata

The current provider enum includes:

- `openai`
- `anthropic`
- `gemini`
- `deepseek`

## Provider Key

A `Provider Key` stores an upstream provider credential and optional base URL override.

It exists so multiple models can reuse one upstream credential instead of embedding provider secrets per model.

Current provider key fields include:

- `display_name`
- `secret`
- optional `api_base`

## API Key

An `API Key` is the caller-facing credential used to access the gateway.

Current data-plane behavior is based on `key_hash`, not plaintext storage. The proxy hashes the incoming bearer token and resolves it against the stored `key_hash`.

An API key also carries:

- `allowed_models`
- optional `rate_limit`
- optional `max_budget_usd`

An empty `allowed_models` list denies access to every model. A wildcard entry `"*"` allows access to every model in scope.

## Routing Model

A routing model, sometimes called a virtual model, is a model with a `routing` block instead of direct provider fields.

Current routing strategies include:

- `failover`
- `round_robin`
- `weighted`

The gateway resolves the routing model to one of its target models at request time.

## Guardrail

A `Guardrail` is a request or response policy object applied by the gateway.

Current schema supports:

- `keyword`
- `bedrock`

Important current boundary:

- `keyword` guardrails are the current in-process guardrail path.
- `bedrock` appears in the schema, but should not be documented as generally available runtime behavior yet.

## Cache Policy

A `Cache Policy` controls when prompt-response cache lookup and storage apply.

Current fields include:

- `name`
- `enabled`
- `backend`
- `ttl_seconds`
- `applies_to`

Current `applies_to` matching understands:

- `all`
- `model:<name>`
- `api_key:<id>`

Important current boundary:

- `memory` is the current enforced cache backend.
- `redis` is present in schema and planning shape, but should not be documented as fully available runtime behavior yet.

## Observability Exporter

An `Observability Exporter` is a resource that configures external telemetry export from the gateway.

Use this concept when documenting external metrics, traces, or event forwarding behavior.

## Environment

An `Environment` is a first-class AISIX Cloud control-plane concept.

The managed data plane watches configuration scoped to its environment. In Cloud mode, projection rules ensure the data plane only sees the resources intended for that environment.

## Managed Data Plane

The managed data plane is still `AISIX AI Gateway`, but it runs under AISIX Cloud control-plane workflows.

In this mode:

- the admin listener is not bound
- dynamic resources come from the Cloud-managed etcd path
- control-plane communication uses mTLS-authenticated `/dp/*` endpoints

## Playground

There are two different playground concepts:

- the standalone gateway has an in-process playground endpoint on the admin listener
- AISIX Cloud has a control-plane playground path that currently talks directly to the upstream provider and is **not** data-plane-equivalent

Do not treat these as the same behavior.

## Related Pages

- [What Is AISIX AI Gateway](what-is-aisix-ai-gateway.md)
- [Deployment Modes](deployment-modes.md)
- [Feature Matrix](feature-matrix.md)
- [Roadmap](../roadmap.md)
