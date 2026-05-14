---
title: Troubleshooting
description: Diagnose the most common startup, configuration, upstream, and managed-path failures in AISIX AI Gateway.
sidebar_position: 55
---

## Config Propagation Problems

Symptoms:

- a new model does not appear on `/v1/models`
- a request fails right after admin writes
- a model resolves but referenced resources are still missing

Common cause:

- the watch-driven snapshot has not caught up yet

Typical signal:

- errors around an unknown `provider_key_id`

What to do:

- poll the target endpoint or `/v1/models`
- inspect `/admin/v1/health` for per-model health and optional freshness data

First-response checklist:

1. confirm the admin write succeeded
2. check admin health freshness
3. poll instead of re-writing the same resource immediately

## etcd Connectivity Problems

Symptoms:

- startup failure
- watch staleness
- transport or DNS-looking etcd errors

What to check:

- `etcd.endpoints`
- etcd TLS files
- network path from gateway to etcd

If startup is involved, treat etcd reachability as a hard dependency, not as an optional background component.

## Guardrail Blocking

Symptoms:

- proxy returns `422`
- error type is `content_filter`

What to check:

- enabled keyword guardrails
- `hook_point`
- the prompt or response content that triggered the rule

Remember that current live guardrail behavior is centered on the chat-completions path.

## Managed Budget Or mTLS Issues

Symptoms:

- budget checks silently disabled at boot
- managed heartbeat or control-plane paths fail

What to check:

- mTLS bundle files exist and are readable
- managed bootstrap produced the expected bundle
- control-plane URL and trust roots are correct

Also check whether you are diagnosing a managed-mode deployment at all. These symptoms do not apply the same way to standalone mode.

## Playground Issues

Symptom:

- admin playground returns `playground not wired: proxy router not configured`

Meaning:

- the admin surface does not have a proxy router wired into the same process state

## Fast Triage Order

When you are not sure where to start:

1. check `GET /health`
2. check admin-listener `GET /health`, then `GET /admin/v1/health` in standalone mode
3. identify whether the symptom is startup, propagation, upstream, or policy related
4. inspect the most specific signal next: logs, metrics, headers, or admin health freshness

## Related Pages

- [Health Checks](health-checks.md)
- [Testing And Verification](testing-and-verification.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
