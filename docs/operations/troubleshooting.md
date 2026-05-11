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
- inspect `/admin/v1/health` for snapshot freshness

## etcd Connectivity Problems

Symptoms:

- startup failure
- watch staleness
- transport or DNS-looking etcd errors

What to check:

- `etcd.endpoints`
- etcd TLS files
- network path from gateway to etcd

## Guardrail Blocking

Symptoms:

- proxy returns `422`
- error type is `content_filter`

What to check:

- enabled keyword guardrails
- `hook_point`
- the prompt or response content that triggered the rule

## Managed Budget Or mTLS Issues

Symptoms:

- budget checks silently disabled at boot
- managed heartbeat or control-plane paths fail

What to check:

- mTLS bundle files exist and are readable
- managed bootstrap produced the expected bundle
- control-plane URL and trust roots are correct

## Playground Issues

Symptom:

- admin playground returns `playground not wired: proxy router not configured`

Meaning:

- the admin surface does not have a proxy router wired into the same process state

## Related Pages

- [Health Checks](health-checks.md)
- [Testing And Verification](testing-and-verification.md)
- [Configuration Propagation](../configuration/configuration-propagation.md)
