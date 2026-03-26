---
slug: /aisix/core-concepts/dynamic-configuration
title: 'Dynamic Configuration'
description: Discover how AISIX achieves real-time configuration updates without restarts.
---

A key feature of AISIX is its dynamic configuration. You can modify Models and API Keys at any time, and the changes take effect instantly across the gateway cluster without restarts. This is possible due to its decoupled architecture, which separates the control plane from the data plane.

## Control Plane vs. Data Plane

AISIX has two main components:

-   **Data Plane (DP)**: The core proxy engine. It is a lightweight, high-performance component that handles client traffic, executes the hook pipeline, and forwards requests to upstream LLM providers.

-   **Control Plane (CP)**: The brain of the system. It stores and manages configuration data, such as Models and API Keys, and acts as the single source of truth.

In AISIX, these roles are filled by:

-   **etcd**: The configuration store (part of the CP).
-   The **Admin API**: The management interface (part of the CP).
-   **AISIX proxy instances**: The Data Plane.

```mermaid
graph TD
    subgraph Control Plane
        A[Admin API] --> B(etcd);
    end

    subgraph Data Plane
        C1(AISIX Instance 1)
        C2(AISIX Instance 2)
        C3(AISIX Instance ...)
    end

    B -- "Watch for Changes" --> C1;
    B -- "Watch for Changes" --> C2;
    B -- "Watch for Changes" --> C3;

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C1 fill:#ccf,stroke:#333,stroke-width:2px
    style C2 fill:#ccf,stroke:#333,stroke-width:2px
    style C3 fill:#ccf,stroke:#333,stroke-width:2px
```

## Real-time Updates via `watch`

Dynamic configuration works as follows:

1.  **Configuration Changes**: When you create a Model using the Admin API, the configuration is written to etcd.

2.  **Watching for Updates**: Each AISIX data plane instance maintains a persistent connection to etcd and uses its `watch` mechanism to monitor for changes to the configuration data.

3.  **Instant Propagation**: When a change is written to etcd, etcd notifies all watching AISIX instances.

4.  **In-Memory Cache Update**: Upon receiving a notification, each AISIX instance fetches the updated configuration and refreshes its in-memory cache. This update is atomic and lock-free, designed to minimize performance impact.

This architecture ensures that configuration changes are propagated to all gateway nodes in near real-time, providing a responsive and manageable system.
