---
title: Troubleshooting
description: Diagnose the most common startup, configuration, upstream, policy, and managed-path failures in AISIX AI Gateway.
sidebar_position: 56
---

This troubleshooting guide starts from symptoms that appear after a deployment is running but does not behave as
expected. Start with the symptom, then narrow the failure to startup,
configuration propagation, caller access, policy, upstream provider, or
managed data-plane connectivity.

## Fast triage order

When you are not sure where to start:

1. Check proxy `GET /livez`.
2. In standalone mode, check admin `GET /livez` and
   `GET /admin/v1/health`.
3. Check whether the model alias appears in `GET /v1/models` for the
   caller key.
4. Send one real request to the endpoint that fails.
5. Use response headers, logs, metrics, and usage events to identify the
   failing layer.

## Startup or etcd connectivity problems

Symptoms:

- process fails during startup
- watch freshness stalls
- errors mention etcd transport, DNS, TLS, or connection failure

Check:

- `etcd.endpoints`
- etcd network reachability from the gateway host or container
- etcd TLS certificate paths and file permissions
- whether etcd is reachable before the gateway starts

In standalone mode, etcd reachability is a hard dependency for dynamic
resource state. Treat it as part of the gateway control-plane boundary.

## Configuration propagation problems

Symptoms:

- a new model does not appear in `GET /v1/models`
- a request fails immediately after creating resources
- a model resolves but referenced resources are missing
- errors mention an unknown `provider_key_id`

Common cause:

- the watch-driven snapshot has not caught up yet, or the resource was
  rejected before entering the live snapshot

Check:

1. Confirm the admin write succeeded.
2. Poll `GET /v1/models` or the target endpoint instead of sleeping.
3. Inspect `GET /admin/v1/health` for snapshot freshness in standalone
   mode.
4. Check heartbeat or health state for rejected resources.

## Caller access problems

Symptoms:

- request is rejected before reaching the provider
- model discovery does not show the expected alias
- one API key works but another does not

Check:

- caller API key value and authorization header
- `allowed_models` on the API key
- model alias spelling
- team or user scope if rate-limit or budget policy depends on those
  fields

## Policy problems

### Guardrail blocking

Symptoms:

- proxy returns `422`
- error type is `content_filter`

Check:

- enabled keyword guardrails
- `hook_point`
- prompt or response content that triggered the rule

Current live guardrail behavior applies to `POST /v1/chat/completions` and
`POST /v1/messages`.

### Rate-limit or budget denial

Symptoms:

- proxy returns `429`
- response includes `Retry-After` or rate-limit headers
- managed deployment denies traffic after budget evaluation

Check:

- API-key and model-level rate limits
- scoped `RateLimitPolicy` resources
- Cloud budget policy in managed mode
- whether multiple proxy replicas affect in-process counters

## Upstream provider problems

Symptoms:

- model health degrades
- requests fail after model resolution succeeds
- provider-specific auth or network errors appear in logs

Check:

- provider key secret and `api_base`
- upstream model id
- provider-specific auth shape
- outbound network path from the data plane
- provider outage or quota state

## Managed data-plane problems

Symptoms:

- managed heartbeat fails
- Cloud shows resources that live traffic does not use
- budget checks fail or appear unavailable
- Cloud playground succeeds but live traffic differs

Check:

- certificate bundle and trust root
- `AISIX_MANAGED__CP_BASE_URL`
- data-plane-manager `/dp/*` reachability
- resource environment scope
- projection status
- whether the request was sent through the live managed data plane

## Playground issue

Symptom:

- admin playground returns `playground not wired: proxy router not configured`

Meaning:

- the standalone playground is unavailable in this process

Check:

- whether the gateway is running in managed mode
- whether the deployment binds the standalone admin listener
- whether normal proxy requests work on `/v1/chat/completions`

## Next steps

- [Health checks](/ai-gateway/operations/health-checks) explains the
  health endpoints.
- [Testing and verification](/ai-gateway/operations/testing-and-verification)
  gives a production smoke-test flow.
- [Configuration propagation](/ai-gateway/configuration/configuration-propagation)
  explains snapshot propagation.
