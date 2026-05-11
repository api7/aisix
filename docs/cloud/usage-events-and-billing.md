---
title: Usage Events And Billing
description: Understand the current AISIX Cloud usage-event ingestion and billing-oriented control-plane workflows.
sidebar_position: 75
---

AISIX Cloud collects usage information from the managed data plane and exposes customer-facing usage and billing workflows above that telemetry.

Current documented behavior includes:

- `/dp/telemetry` ingestion on the control-plane side
- usage-event views surfaced from the control plane
- managed budget enforcement and budget-driven `429` outcomes on real DP traffic

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Offline Resilience](offline-resilience.md)
- [Budgets](../configuration/budgets.md)
