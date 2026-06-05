---
title: Troubleshooting
description: Diagnose the most common startup, configuration, upstream, policy, and managed-path failures in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 56
---

When a running deployment does not behave as expected, start with the symptom and narrow the failure to startup, configuration propagation, caller access, policy, upstream provider, or managed data-plane connectivity.

## Start with the Failing Layer

When you are not sure where to start, first confirm proxy `GET /livez`. In standalone mode, also check admin `GET /livez` and `GET /admin/v1/health`. Then verify whether the model alias appears in `GET /v1/models` for the caller key and send one real request to the endpoint that fails.

Use response headers, logs, metrics, and usage events to identify the failing layer before changing configuration.

## Startup and Configuration

| Symptom | Check |
| --- | --- |
| Process fails during startup. | `etcd.endpoints`, network reachability from the gateway host or container, and etcd TLS certificate paths and permissions. |
| Watch freshness stalls. | etcd connectivity, TLS configuration, and configuration watch health. |
| Errors mention etcd transport, DNS, TLS, or connection failure. | Whether etcd is reachable before the gateway starts. |

In standalone mode, etcd reachability is a hard dependency for dynamic resource state. Treat it as part of the gateway control plane.

### Configuration Propagation

Common symptoms include a new model missing from `GET /v1/models`, a request that fails immediately after creating resources, a model that resolves with missing referenced resources, or an error that mentions an unknown `provider_key_id`.

The usual cause is a watch-driven snapshot that has not caught up, or a resource that was rejected before entering the live snapshot.

Confirm that the admin write succeeded, then poll `GET /v1/models` or the target endpoint instead of sleeping. In standalone mode, inspect `GET /admin/v1/health` for snapshot freshness. Check heartbeat or health state when a resource may have been rejected.

## Request Failures

Use the caller-visible status code and the failing request path to narrow the problem. Check caller access before provider credentials when the request is rejected before reaching the upstream provider.

### Caller Access

| Symptom | Check |
| --- | --- |
| Request is rejected before reaching the provider. | Caller API key value, authorization header, and `allowed_models`. |
| Model discovery does not show the expected alias. | Model alias spelling and whether the alias should appear in `/v1/models`. |
| One API key works but another does not. | Key-specific `allowed_models`, team scope, and user scope when policy depends on those fields. |

### Guardrail Blocking

When the proxy returns `422` with error type `content_filter`, check enabled keyword guardrails, `hook_point`, and the prompt or response content that triggered the rule.

Guardrail blocking applies to `POST /v1/chat/completions` and `POST /v1/messages`.

### Rate-Limit or Budget Denial

When the proxy returns `429`, includes `Retry-After` or rate-limit headers, or denies traffic after a managed budget check, inspect API-key and model-level rate limits, matching `RateLimitPolicy` resources, Cloud budget policy in managed mode, and whether multiple proxy replicas affect in-process counters.

### Upstream Provider

| Symptom | Check |
| --- | --- |
| Model health degrades. | Provider outage, quota state, and data-plane outbound network path. |
| Requests fail after model resolution succeeds. | Provider key secret, `api_base`, and upstream model id. |
| Provider-specific auth or network errors appear in logs. | Provider-specific authentication behavior and outbound connectivity from the data plane. |

### Admin Playground

When the admin playground returns `playground not wired: proxy router not configured`, the standalone playground is unavailable in that gateway instance.

Check whether the gateway is running in managed mode, whether the deployment binds the standalone admin listener, and whether normal proxy requests work on `/v1/chat/completions`.

## Managed Data Plane

| Symptom | Check |
| --- | --- |
| Managed heartbeat fails. | Certificate bundle, trust root, `AISIX_MANAGED__CP_BASE_URL`, and data-plane-manager `/dp/*` reachability. |
| Cloud shows resources that live traffic does not use. | Resource environment scope and projection status. |
| Budget checks fail or appear unavailable. | Managed control-plane connectivity and budget-check reachability. |
| Cloud playground succeeds but live traffic differs. | Whether the request was sent through the live managed data plane. |

## Related Reading

[Health checks](/ai-gateway/operations/health-checks) covers health endpoints, [Testing and verification](/ai-gateway/operations/testing-and-verification) covers production smoke tests, and [Configuration propagation](/ai-gateway/configuration/configuration-propagation) explains snapshot propagation.
