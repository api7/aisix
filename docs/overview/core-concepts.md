---
title: Core Concepts
description: Understand the core AISIX AI Gateway and AISIX Cloud concepts, including models, provider keys, API keys, routing models, guardrails, cache policies, and observability exporters.
sidebar_position: 3
---

This page defines the core objects and terms used across AISIX AI Gateway and AISIX Cloud.

## Model

A `Model` is the resource clients target through the [gateway](glossary.md#gateway).

For direct models, a model includes:

- `display_name` — the alias your callers send as the API request's `model` field
- `provider`
- `model_name` — the upstream model ID the gateway forwards to the provider (for example, `"gpt-4o"`)
- `provider_key_id`
- optional timeout, rate limit, and cost metadata

The current provider enum includes:

- `openai`
- `anthropic`
- `google`
- `deepseek`
- `cohere`
- `jina`

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

The plaintext bearer is chosen (or generated) by the caller and SHA-256-hashed locally before submission — the gateway never sees or returns the plaintext at create time, only the stored `key_hash`. The one endpoint that emits a server-generated plaintext is `POST /admin/v1/apikeys/:id/rotate`, which returns the new plaintext exactly once in its response under a `plaintext` field; capture it then, because subsequent reads only include the hash.

An API key also carries:

- `allowed_models`
- optional `rate_limit`
- optional `team_id` and `user_id`

`team_id` and `user_id` are bucket identifiers consumed by `team`-scoped and `member`-scoped [`RateLimitPolicy`](#rate-limit-policy) rows and by `team` / `member` budget rows. They are not access controls on their own.

An empty `allowed_models` list denies access to every model. A wildcard entry `"*"` allows access to every model in scope.

## Rate Limit Policy

A `RateLimitPolicy` is a standalone rate-limit rule stored in [etcd](glossary.md#etcd). Each policy targets a single subject through `(scope, scope_ref)`:

- `api_key` — match by `ApiKey` entry id
- `model` — match by `Model` entry id
- `team` — match by `ApiKey.team_id`
- `member` — match by `ApiKey.user_id`

The proxy enforces all matching policies alongside the inline `ApiKey.rate_limit` and `Model.rate_limit` layers. Any layer with a configured limit can reject a request with `429`. See [Rate Limits](../configuration/rate-limits.md) for the full enforcement model.

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
- `bedrock` (the codename for the AWS Bedrock guardrails integration — see the Guardrails reference for setup)

Important current boundary:

- `keyword` guardrails are the current in-process guardrail path.
- `bedrock` has runtime implementation behind the `bedrock` feature and should be treated as an advanced capability with support and deployment caveats rather than as a planned-only feature.

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

- `memory` is the default cache backend.
- `redis` has runtime connection and backend selection logic today, but should be treated as a limited capability until the broader cache documentation and support boundaries are fully written down.

## Observability Exporter

An `Observability Exporter` ships per-request span telemetry — derived from gateway `UsageEvent` records — to an OTLP/HTTP-compatible backend (Grafana Tempo, Honeycomb, Langfuse via OTLP, and so on). Configure one when you want a per-request trace of gateway proxy activity forwarded to your existing tracing backend.

## Environment

An `Environment` is a first-class [AISIX Cloud](glossary.md#aisix-cloud) [control-plane](glossary.md#control-plane) concept.

The managed data plane watches configuration scoped to its environment. In Cloud mode, projection rules ensure the data plane only sees the resources intended for that environment.

## Managed Data Plane

The managed data plane is still `AISIX AI Gateway`, but it runs under AISIX Cloud control-plane workflows.

In this mode:

- the admin listener is not bound
- dynamic resources come from the Cloud-managed etcd path
- control-plane communication uses mTLS-authenticated `/dp/*` endpoints

## Playground

The playground is an in-process proxy endpoint mounted on the admin listener that forwards requests to the proxy router so model rows can be smoke-tested without a network hop. Auth uses a proxy API key (not the admin key); the full proxy middleware stack — auth, rate limit, bridge, guardrails — still runs.

There are two different playground concepts:

- the standalone gateway has an in-process playground endpoint on the admin listener
- AISIX Cloud has a control-plane playground path that currently talks directly to the upstream provider and is **not** data-plane-equivalent

Do not treat these as the same behavior.

## Related Pages

- [What Is AISIX AI Gateway](what-is-aisix-ai-gateway.md)
- [Deployment Modes](deployment-modes.md)
- [Feature Matrix](feature-matrix.md)
- [Roadmap](../roadmap.md)
