---
slug: /ai-gateway/core-concepts/model-and-api-key
title: 'First-Class Citizens: Model and API Key'
description: Learn about the core entities in AISIX: Models and API Keys.
---

In AISIX, **Models** and **API Keys** are the two fundamental resources, or *first-class citizens*, that drive its functionality. Understanding their roles and relationship is key to managing your AI services.

## Model

A **Model** is the central entity in AISIX. It represents a callable upstream AI model from a provider like OpenAI, Google Gemini, or DeepSeek, and serves as the primary unit for routing, authentication, and policy enforcement.

When you create a Model, you define a new endpoint for your clients. Each Model is configured with details about the upstream service and any policies to apply, such as rate limiting.

### Model Fields

The following table describes the Model configuration fields:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `name` | String | A unique, human-readable name for the Model. This is the identifier clients use in their requests. | `my-gpt4-mini` |
| `model` | String | The identifier for the upstream model, formatted as `{provider}/{model_id}`. The `provider` part tells AISIX which driver to use, and `model_id` is passed to the provider. | `openai/gpt-4.1-mini` |
| `provider_config` | Object | Provider-specific configuration for credentials. For current providers, this is where you place the API key for the upstream service. | `{"api_key": "sk-..."}` |
| `rate_limit` | Object | (Optional) Rules to limit the request rate and token usage for this Model. See [Rate Limiting](./rate-limiting.md) for details. | `{"rpm": 100, "tpm": 10000}` |

### Model as a Routing Target

Clients direct requests to a Model by setting the `model` field in their request body to the `name` of the Model configured in AISIX. AISIX uses this `name` to look up the Model entity, retrieve its configuration, and route the request to the correct provider.

For example, a client sends a request:

```json
{
  "model": "my-gpt4-mini",
  "messages": [...]
}
```

AISIX finds the Model named `my-gpt4-mini`, sees it maps to `openai/gpt-4.1-mini`, and forwards the request to the OpenAI API using the credentials in `provider_config`.

## API Key

An **API Key** is a credential used to authenticate clients. It is the primary identifier for a consumer and is associated with permissions that dictate which Models it can access.

### API Key Fields

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `key` | String | The secret key string that clients must provide in the `Authorization` header. | `aisix-user-key-xxxxxxxx` |
| `allowed_models` | Array of Strings | A list of Model `name`s this API Key is permitted to access. This is a strict whitelist. | `["my-gpt4-mini", "my-gemini-pro"]` |
| `rate_limit` | Object | (Optional) Rules to limit the request rate and token usage for this API Key. | `{"rpd": 1000, "tpd": 1000000}` |

### The Whitelist Model: `allowed_models`

AISIX uses an explicit whitelist for access control. The `allowed_models` array defines which Models an API Key can use. 

> **Important**: If `allowed_models` is an empty array (`[]`), the API Key cannot access **any** Model. There is no implicit "allow all" behavior.

### Authentication Flow

1. A client sends a request with an `Authorization: Bearer <key>` header.
2. AISIX extracts the `<key>` and finds the corresponding API Key entity.
3. It checks the `model` field from the client's request body.
4. It verifies if the requested model `name` is in the API Key's `allowed_models` list.

If the model is in the list, the request proceeds. Otherwise, AISIX rejects the request with a `403 Forbidden` error.
