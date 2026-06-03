---
title: Architecture Overview
sidebar_label: Overview
description: Understand the AISIX AI Gateway architecture details that affect operation, scale, and provider compatibility.
sidebar_position: 1
---

This section explains the runtime behavior behind AISIX AI Gateway
features, including why a configuration change is not visible yet, how
limits behave under load, and what happens when a client protocol is
routed to a different provider family.

These pages are not required for the first request. They are useful when
you are operating AISIX in production, designing a rollout, or debugging
behavior that depends on the gateway's internal request path.

## When to use this section

Use these pages when you need to answer operational questions such as:

- Why did an admin write succeed, but the proxy still serves the previous
  configuration?
- How are request, token, and concurrency limits charged when provider
  usage is only known after the upstream response?
- What changes when an Anthropic Messages request is routed through a
  non-Anthropic upstream provider?

## Start with the right topic

| Topic | Read this when you need to understand |
| --- | --- |
| [Snapshot and watch](snapshot-and-watch.md) | How dynamic resources propagate from etcd into each proxy instance, and why request handling does not call etcd directly. |
| [Two-phase rate limit](two-phase-rate-limit.md) | How AISIX reserves request and concurrency capacity before dispatch, then records provider-reported token usage after the response. |
| [Protocol translation](protocol-translation.md) | How AISIX serves Anthropic Messages traffic through Anthropic passthrough or cross-provider translation paths. |

## Related operational guides

Most operators can configure AISIX without reading every architecture
page first. If you are troubleshooting production behavior, pair this
section with the operational docs:

- [Configuration propagation](/ai-gateway/configuration/configuration-propagation)
  explains the user-visible propagation model.
- [Rate limits](/ai-gateway/configuration/rate-limits) explains the
  policy fields that feed the two-phase limiter.
- [Troubleshooting](/ai-gateway/operations/troubleshooting) starts from
  symptoms and points back to the relevant architecture detail.
