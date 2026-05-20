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

## Runtime Configuration Notes

`AISIX_MANAGED__CP_BASE_URL` must point at the control-plane
data-plane-manager endpoint that serves `/dp/heartbeat`,
`/dp/telemetry`, `/dp/rotate-cert`, and `/dp/budget_check`.

For example:

- `https://dpm.example.com:7944` for an externally reachable DPM
- `https://dpm:7944` when the DP joins the AISIX Cloud Compose network

Do not use the browser-facing dashboard/cp-api origin such as
`http://api:8080`; that service does not own the `/dp/*` heartbeat
surface.

The DP accepts either inline PEM values:

- `AISIX_MANAGED__CP_CERT_PEM`
- `AISIX_MANAGED__CP_KEY_PEM`
- `AISIX_MANAGED__CP_CA_PEM`

or file paths:

- `AISIX_MANAGED__CP_CERT_FILE`
- `AISIX_MANAGED__CP_KEY_FILE`
- `AISIX_MANAGED__CP_CA_FILE`

If file paths are used from a container, make sure the process user can
read those files and can write the runtime state directory
`/var/lib/aisix`. The state directory contains the persisted mTLS
bundle and sidecar files such as the DP identity used on restart.

## Troubleshooting

### The data plane never appears healthy in Cloud

Check certificate bundle correctness, trust roots, and mTLS connectivity before looking at higher-level projection issues.

### `/dp/*` calls fail after initial success

Inspect certificate rotation and trust-chain changes, not just application-level configuration.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Offline Resilience](offline-resilience.md)
- [TLS And mTLS](../operations/tls-and-mtls.md)
