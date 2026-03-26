---
slug: /aisix/guides/model-management
title: Model Management
description: Learn how to configure and manage models in AISIX.
---

In AISIX, a **Model** is a virtual entity that represents a specific upstream AI model. This guide explains how to create and manage `Model` entities.

## The Role of a Model Entity

A `Model` entity serves several purposes:

-   **Routing Target**: It acts as the identifier for routing. When a client sends a request with `"model": "my-gpt4-mini"`, AISIX looks for a `Model` with that name.
-   **Upstream Configuration**: It holds the configuration to connect to the upstream LLM, including the provider type, model identifier, and credentials.
-   **Access Control**: It is the resource against which API key permissions are checked.
-   **Rate Limiting**: It can have its own rate limits applied to all its traffic.

## Creating and Managing Models

Models are managed via the Admin API, which listens on `127.0.0.1:3001` by default.

### Create a Model

To create a new model, send a `POST` request to the `/aisix/admin/models` endpoint.

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/models \
  -H "Authorization: Bearer your-strong-admin-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpt4-mini",
    "model": "openai/gpt-4.1-mini",
    "provider_config": {
      "api_key": "<YOUR_OPENAI_API_KEY>"
    }
  }'
```

A successful request returns a JSON object confirming the creation, including the etcd key.

### Key Fields

-   `name`: A unique, human-readable name for this model entity. This is the identifier your clients use.
-   `model`: A string that specifies the provider and the upstream model ID, formatted as `{provider}/{model_id}`. Supported providers are `openai`, `gemini`, `deepseek`, and `anthropic`.
-   `provider_config`: A JSON object with credentials for the upstream provider. For OpenAI, Gemini, DeepSeek, and Anthropic, this must contain an `api_key`.
-   `rate_limit` (optional): A JSON object specifying rate limits for this model. See the [Rate Limiting](./rate-limiting.md) guide.

### List All Models

To see all configured models, send a `GET` request:

```bash
curl http://127.0.0.1:3001/aisix/admin/models
```

### Get a Specific Model

To retrieve a single model by its ID (the UUID assigned on creation), use a `GET` request with the ID in the path:

```bash
# First, get the ID from the list endpoint
MODEL_ID=$(curl -s http://127.0.0.1:3001/aisix/admin/models | jq -r ".list[] | select(.value.name == \"my-gpt4-mini\") | .key" | cut -d/ -f3)

# Then, get the model by ID
curl http://127.0.0.1:3001/aisix/admin/models/$MODEL_ID
```

### Update a Model (Upsert)

The `PUT` endpoint performs an **upsert**: it updates the model if it exists, or creates it if it does not. Send a `PUT` request with the full model configuration to the model's ID endpoint. The response returns `201 Created` for new models or `200 OK` for updates.

```bash
curl -X PUT http://127.0.0.1:3001/aisix/admin/models/$MODEL_ID \
  -H "Authorization: Bearer your-strong-admin-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-gpt4-mini-updated",
    "model": "openai/gpt-4.1-mini",
    "provider_config": {
      "api_key": "<YOUR_NEW_API_KEY>"
    }
  }'
```

### Delete a Model

To delete a model, send a `DELETE` request:

```bash
curl -X DELETE http://127.0.0.1:3001/aisix/admin/models/$MODEL_ID
```

## Model Validation

The `ValidateModelHook` is a default hook that ensures every request targets a valid and accessible model. It performs two checks:

1.  **Model Exists**: It verifies that the model name in the client's request body corresponds to an existing `Model` entity.
2.  **Access is Allowed**: It checks if the client's API key is authorized to use this model (see the [Authentication](./authentication.md) guide).

If the model is not found, it returns a `400 Bad Request` error:

```json
{
  "error": {
    "message": "Model my-nonexistent-model not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```
