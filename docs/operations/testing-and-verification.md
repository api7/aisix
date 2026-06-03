---
title: Testing and Verification
description: Verify AISIX AI Gateway deployments with health checks, propagation probes, and end-to-end request tests.
sidebar_position: 55
---

Production verification should check the full caller-to-provider path,
not only process startup.

## Start with this verification flow

1. Confirm proxy liveness.
2. Confirm admin liveness in standalone mode.
3. Create or inspect the expected provider key, model, and caller API
   key.
4. Verify configuration propagation on the proxy path.
5. Send one real request to the upstream provider.
6. Confirm logs, metrics, headers, or usage events show the request.

The final request matters most. A gateway can be alive while caller
authentication, model resolution, provider credentials, or upstream
network access is still broken.

## Prefer positive probes over fixed waits

Configuration propagation is asynchronous. Prefer probes that confirm the
desired state rather than fixed sleeps.

Use:

- polling `/v1/models` until the expected model alias appears
- polling the exact endpoint you use until a known propagation error
  disappears
- checking `GET /admin/v1/health` in standalone mode for snapshot
  freshness

Avoid relying only on:

- a fixed sleep after admin writes
- process liveness alone
- a Cloud preview surface when the live managed data-plane path is what
  you need to verify

## Verify these signals

For each critical path, verify:

- expected HTTP status
- expected response shape
- expected model alias behavior
- expected upstream-provider behavior where it matters
- relevant operational headers, such as `x-aisix-cache`,
  `x-aisix-call-id`, `x-aisix-request-id`, or `Retry-After`
- relevant logs, metrics, usage events, or exporter output

## Build a practical smoke test set

For a production-minded smoke test, include:

1. One authentication check with a valid caller API key.
2. One denied request with an invalid or unauthorized key.
3. One model-discovery check with `GET /v1/models`.
4. One successful request for each endpoint family you use.
5. One policy check if you depend on cache, guardrails, budgets, or rate
   limits.

## Standalone and managed differences

In standalone mode, include admin API and admin health checks. In managed
mode, include managed data-plane heartbeat, projection, and live proxy
request checks.

Do not use the Cloud playground as proof that live managed traffic is
healthy. The playground is a preview surface and does not exercise every
managed data-plane feature.

## Troubleshooting

### Health checks pass but smoke tests fail

Trust the smoke tests. They are closer to real user behavior than
process liveness. Check model visibility, provider-key references, caller
API-key access, and upstream provider connectivity.

### A fixed sleep works locally but flakes in production

Replace the sleep with a positive probe, such as polling `/v1/models` or
the exact endpoint that must become ready.

## Next steps

- [Quickstart](/ai-gateway/quickstart/) gives the first local
  end-to-end request.
- [Health checks](/ai-gateway/operations/health-checks) explains health
  surfaces.
- [Troubleshooting](/ai-gateway/operations/troubleshooting) gives a
  symptom-oriented diagnosis flow.
