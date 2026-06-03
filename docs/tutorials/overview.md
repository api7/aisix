---
title: Tutorials Overview
sidebar_label: Overview
description: Follow scenario-based AISIX AI Gateway tutorials for protocol translation, failover, guardrails, and response caching.
sidebar_position: 79
---

Tutorials show complete operator scenarios from setup to verification. Use them
after you finish the [Quickstart](../quickstart) and understand
the basic resource flow:

```text
provider key -> model -> caller API key -> proxy request
```

Each tutorial creates real gateway resources, sends proxy traffic, verifies the
observable result, and includes cleanup steps.

## Before you start

You should have:

- a running standalone gateway from the [Quickstart](../quickstart)
- an admin key for `Authorization: Bearer YOUR_ADMIN_KEY`
- a caller key such as `sk-demo-caller`
- at least one provider credential you can use for live upstream traffic

Admin writes propagate asynchronously to the proxy snapshot. The tutorials use
polling where propagation matters, so do not replace those checks with fixed
sleep calls in automation.

## Choose a tutorial

| Goal | Tutorial | What you verify |
| --- | --- | --- |
| Keep an OpenAI-style client while using an Anthropic upstream | [Use an OpenAI client with an Anthropic upstream](openai-client-to-anthropic-upstream.md) | The caller receives an OpenAI-shaped response while the gateway calls Anthropic Messages upstream. |
| Keep one stable model alias across primary and secondary targets | [Build a virtual model with failover](build-a-virtual-model-with-failover.md) | A broken primary target fails over to the secondary target and reports `x-aisix-served-by`. |
| Block forbidden prompt content at the gateway boundary | [Add keyword guardrails](add-keyword-guardrails.md) | A clean request passes and a forbidden request returns `422 content_filter`. |
| Reuse identical chat-completion responses | [Enable response caching](enable-response-caching.md) | The first request returns `x-aisix-cache: miss`; the repeated request returns `x-aisix-cache: hit`. |

## Suggested order

If you are learning the product for the first time, run the tutorials in this
order:

1. [Use an OpenAI client with an Anthropic upstream](openai-client-to-anthropic-upstream.md)
2. [Build a virtual model with failover](build-a-virtual-model-with-failover.md)
3. [Add keyword guardrails](add-keyword-guardrails.md)
4. [Enable response caching](enable-response-caching.md)

This order starts with the gateway's core value — stable caller contracts in
front of provider-specific upstreams — then adds resilience, policy, and
optimization.

## Where to go next

- [Configuration overview](../configuration/overview.md) explains the resource
  model behind the tutorials.
- [Operations](../operations/production-deployment.md) covers production
  deployment, telemetry, health checks, and troubleshooting.
- [Reference](../reference/proxy-api-reference.md) documents the proxy API
  surface and generated resource schemas.
