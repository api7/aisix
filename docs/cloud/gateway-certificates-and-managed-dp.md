---
title: Gateway Certificates And Managed DP
description: Set up and understand the current certificate-based managed data-plane flow in AISIX Cloud.
sidebar_position: 72
---

Current AISIX Cloud managed bootstrap is certificate-based.

## Current Flow

At a high level:

1. create or select an environment
2. issue a gateway certificate bundle through the control plane
3. provision the data plane with that certificate bundle
4. let the data plane authenticate to `/dp/*` with mTLS
5. observe heartbeat and config propagation

The current `/dp/*` managed surface includes:

- `POST /dp/heartbeat`
- `POST /dp/telemetry`
- `POST /dp/rotate-cert`
- `GET /dp/budget_check`

## Important Boundary

The legacy bearer-auth `/dp/register` path is no longer the current Cloud bootstrap contract. Treat the certificate bundle flow as authoritative for current Cloud docs.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Offline Resilience](offline-resilience.md)
- [TLS And mTLS](../operations/tls-and-mtls.md)
