---
title: Testing and Verification
description: Verify AISIX AI Gateway deployments with health checks, propagation probes, and end-to-end request tests.
toc_max_heading_level: 2
sidebar_position: 55
---

Production verification should check the full caller-to-provider path, not only process startup.

## Verification Flow

Start with proxy liveness, then confirm admin liveness in standalone mode. After the expected provider key, model, and caller API key exist, verify configuration propagation on the proxy path and send one real request to the upstream provider. Finish by confirming that logs, metrics, headers, or usage events show the request.

The final request matters most. A gateway can be alive while caller authentication, model resolution, provider credentials, or upstream network access is still broken.

Configuration propagation is asynchronous. Prefer probes that confirm the desired state rather than fixed sleeps.

Use positive probes that confirm the expected state is visible. Poll `/v1/models` until the expected model alias appears, poll the exact endpoint you use until a known propagation error disappears, and check `GET /admin/v1/health` in standalone mode when you need to confirm that the standalone admin API sees a fresh snapshot.

Avoid fixed sleeps after admin writes, process liveness alone, or a Cloud playground result when the live managed data-plane path is what you need to verify.

## Verify Request Behavior

For each critical path, verify both the caller-visible result and the management signals. Check the expected HTTP status, response format, and model alias behavior; provider-side logs, metrics, or errors where they matter; gateway headers such as `x-aisix-cache`, `x-aisix-call-id`, `x-aisix-request-id`, or `Retry-After`; and operational telemetry such as logs, metrics, usage events, or exporter output.

For a production-minded smoke test, cover both accepted and rejected traffic. Use a valid caller API key to confirm the authenticated request path, then use an invalid or unauthorized key to confirm caller access is enforced. Check `GET /v1/models`, send one successful request for each endpoint family in use, and verify any cache, guardrail, budget, or rate-limit policy that affects production traffic.

## Deployment Mode Differences

In standalone mode, include admin API and admin health checks. In managed mode, include managed data-plane heartbeat, projection, and live proxy request checks.

Do not use the Cloud playground as the only signal for live managed traffic. The playground is a Cloud UI check and does not exercise every managed data-plane feature.

## Troubleshooting

Use these checks when basic health checks pass but end-to-end smoke tests fail.

### Health Checks Pass but Smoke Tests Fail

Trust the smoke tests. They are closer to real user behavior than process liveness. Check model visibility, provider-key references, caller API-key access, and upstream provider connectivity.

### A Fixed Sleep Works Locally but Flakes in Production

Replace the sleep with a positive probe, such as polling `/v1/models` or the exact endpoint that must become ready.

## Related Reading

Run the first local end-to-end request with the [Quickstart](/ai-gateway/quickstart/). For health endpoints, see [Health checks](/ai-gateway/operations/health-checks). For symptom-oriented diagnosis, see [Troubleshooting](/ai-gateway/operations/troubleshooting).
