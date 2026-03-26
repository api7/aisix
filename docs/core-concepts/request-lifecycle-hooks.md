---
slug: /core-concepts/request-lifecycle-hooks
title: 'Request Lifecycle and Hooks: AI Gateway Processing Pipeline'
description: 'Understand how AISIX processes every LLM request through a hook-based pipeline — covering authentication, model validation, rate limiting, and metrics collection.'
keywords: ['LLM request pipeline', 'AI gateway hooks', 'request lifecycle', 'AI gateway middleware', 'LLM proxy pipeline']
---

At the heart of AISIX is a flexible request processing pipeline built on a system of **Hooks**. This section explains the journey of an AI request through the gateway and how hooks implement key features.

## The Journey of a Request

An AI request passes through several stages as it is processed by AISIX. The following diagram illustrates this lifecycle:

```mermaid
graph TD
    A[Client Request] --> B{Authentication Middleware};
    B --> C{Pre-Call Hooks};
    C --> D{Provider API Call};
    D --> E{Post-Call Hooks};
    E --> F[Client Response];

    subgraph Pre-Call Hooks
        C1[Validate Model Hook]
        C2[Rate Limit Hook (Pre-Check)]
        C3[...]
    end

    subgraph Post-Call Hooks
        E1[Rate Limit Hook (Post-Check)]
        E2[Metrics Hook]
        E3[...]
    end

    style B fill:#cfc,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
```

1.  **Client Request**: A client sends an OpenAI-compatible API request to AISIX.
2.  **Authentication Middleware**: Before the hook pipeline, authentication middleware validates the API key from the `Authorization` header. If the key is missing or invalid, the request is rejected with `401 Unauthorized`.
3.  **Pre-Call Hooks**: Before the request is sent to the upstream LLM provider, it passes through `pre_call` hooks that perform tasks like model validation and rate limiting.
4.  **Provider API Call**: If all pre-call hooks pass, AISIX forwards the request to the upstream provider defined in the requested Model.
5.  **Post-Call Hooks**: After receiving a response from the provider, the data passes through `post_call` hooks. These are used for tasks that depend on the provider's response, such as recording token usage for metrics.
6.  **Client Response**: AISIX sends the processed response to the client.

## Hooks: The Extensible Pipeline

Hooks are the building blocks of AISIX's request processing logic. They are modular components that execute at specific stages of the request lifecycle. This design makes the gateway extensible, allowing new functionalities to be added without altering the core proxying logic.

AISIX includes several built-in hooks that are enabled by default:

### Default Hooks

| Hook | Stage(s) | Description |
| :--- | :--- | :--- |
| `ValidateModelHook` | `pre_call` | Validates the request contains a `model` field, the model exists, and the API key is authorized to access it. Returns `400 Bad Request` if the model field is missing or the model is not found, or `403 Forbidden` if access is denied. |
| `RateLimitHook` | `pre_call`, `post_call` | Enforces rate limits. In `pre_call`, it checks if the request count exceeds the limit. In `post_call`, it updates the token usage counters. |
| `MetricHook` | `post_call` | Collects and exposes metrics (e.g., token counts, request latency) for observability via Prometheus. |

### Hook Stages

The hook system is divided into two stages:

-   **`pre_call`**: Runs **before** the request is sent to the upstream LLM provider. It is used for validation, authentication, and pre-emptive checks. If a hook terminates the request, it can return an immediate response, and subsequent hooks and the provider call are skipped.

-   **`post_call`**: Runs **after** a response is received from the provider. It is used for logging, metrics collection, and logic that needs to inspect the final response data.

This phased approach ensures a clean separation of concerns.

## Related Docs

- [Rate Limiting](../guides/rate-limiting.md) — How the `RateLimitHook` enforces RPM, TPM, and concurrency limits
- [Authentication](../guides/authentication.md) — How the authentication middleware and `ValidateModelHook` secure LLM access
- [Observability](../observability.md) — How the `MetricHook` exposes LLM metrics to Prometheus
