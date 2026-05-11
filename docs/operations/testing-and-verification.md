---
title: Testing And Verification
description: Verify AISIX AI Gateway deployments with health checks, propagation probes, and end-to-end request tests.
sidebar_position: 57
---

Production verification should check more than process startup.

## Minimum Verification Flow

1. confirm proxy health
2. confirm admin health in standalone mode
3. write or inspect the expected dynamic resources
4. verify snapshot propagation on a real proxy path
5. send one real end-to-end request to the upstream

## Prefer Positive Probes

Current test harness and runtime comments show that propagation is asynchronous and can vary under load.

Prefer:

- polling `/v1/models` for model visibility
- polling the exact endpoint you care about until a known propagation error disappears

Over:

- relying only on a fixed sleep

## What To Assert

For each critical path, verify:

- expected HTTP status
- expected response shape
- expected upstream hit behavior when relevant
- operational headers such as `x-aisix-cache` or `x-aisix-call-id` when those are part of your workflow

## Related Pages

- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md)
- [Troubleshooting](troubleshooting.md)
