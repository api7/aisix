---
title: Testing And Verification
description: Verify AISIX AI Gateway deployments with health checks, propagation probes, and end-to-end request tests.
sidebar_position: 57
---

Production verification should check more than process startup.

Use this page as the minimum validation standard before saying a deployment is healthy.

## Minimum Verification Flow

1. confirm proxy health
2. confirm admin health in standalone mode
3. write or inspect the expected dynamic resources
4. verify snapshot propagation on a real proxy path
5. send one real end-to-end request to the upstream

That final step matters most. A healthy process with bad caller-to-upstream behavior is still a failed deployment.

## Prefer Positive Probes

Current test harness and runtime comments show that propagation is asynchronous and can vary under load.

Prefer:

- polling `/v1/models` for model visibility
- polling the exact endpoint you care about until a known propagation error disappears

Over:

- relying only on a fixed sleep

This is especially important after creating several dependent resources in sequence.

## What To Assert

For each critical path, verify:

- expected HTTP status
- expected response shape
- expected upstream hit behavior when relevant
- operational headers such as `x-aisix-cache`, `x-aisix-call-id`, or `x-aisix-request-id` when those are part of your workflow

## Practical Test Set

For a production-minded smoke test, include:

1. one auth check
2. one model-discovery check
3. one happy-path request per critical endpoint family
4. one policy behavior check if you depend on cache, guardrails, or rate limits

## Troubleshooting

### Health checks pass but smoke tests fail

Trust the smoke tests. They are closer to real user behavior than process liveness alone.

## Related Pages

- [Self-Hosted Quickstart](../quickstart/self-hosted.md)
- [First Model, First Key, First Request](../quickstart/first-model-first-key-first-request.md)
- [Troubleshooting](troubleshooting.md)
