---
title: Gateway Certificates And Managed DP
description: Set up and understand the current certificate-based managed data-plane flow in AISIX Cloud.
sidebar_position: 72
---

Current AISIX Cloud managed bootstrap is certificate-based.

This is the key bootstrapping contract for current Cloud-managed data planes.

## Current Flow

At a high level:

1. create or select an environment
2. issue a gateway certificate bundle through the control plane
3. provision the data plane with that certificate bundle
4. let the data plane authenticate to `/dp/*` with mTLS
5. observe heartbeat and config propagation

This flow replaces older mental models that assumed bearer-token registration on `/dp/register`.

The current `/dp/*` managed surface includes:

- `POST /dp/heartbeat`
- `POST /dp/telemetry`
- `POST /dp/rotate-cert`
- `GET /dp/budget_check`

Each endpoint has a different purpose:

- `heartbeat` proves liveness and identity from the data plane
- `telemetry` sends usage-oriented data to the control plane
- `rotate-cert` supports certificate lifecycle management
- `budget_check` supports managed budget enforcement decisions

## Important Boundary

The legacy bearer-auth `/dp/register` path is no longer the current Cloud bootstrap contract. Treat the certificate bundle flow as authoritative for current Cloud docs.

## Operational Meaning

When diagnosing managed bootstrap, think certificate bundle, trust roots, and mTLS identity first.

Do not start with bearer-token assumptions unless your deployment intentionally uses a legacy or self-managed path.

## Troubleshooting

### The data plane never appears healthy in Cloud

Check certificate bundle correctness, trust roots, and mTLS connectivity before looking at higher-level projection issues.

### `/dp/*` calls fail after initial success

Inspect certificate rotation and trust-chain changes, not just application-level configuration.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Offline Resilience](offline-resilience.md)
- [TLS And mTLS](../operations/tls-and-mtls.md)
