---
slug: /guides/authentication
title: 'Authentication'
description: 'Secure your AI gateway with API key authentication and model allowlists. Learn how AISIX validates every LLM request and enforces per-key access control.'
keywords: ['LLM API security', 'AI gateway authentication', 'API key validation', 'LLM access control', 'AI gateway authorization']
---

Authentication is the first line of defense for your AI services. AISIX uses API keys to ensure that only authorized clients can access your models. This guide covers creating and managing API keys and how authentication and model access validation work.

## How Authentication Works

Every request to the AISIX data plane must include a valid API key. Authentication is handled by middleware that runs before the hook pipeline.

The process is as follows:

1.  **Extract API Key**: The authentication middleware inspects the `Authorization` header to extract the API key. It supports `Authorization: <key>` and `Authorization: Bearer <key>` formats.

2.  **Find API Key**: It looks up the key in its in-memory cache of `ApiKey` entities.

If either step fails, the request is rejected with a `401 Unauthorized` error.

3.  **Validate Model Access**: After authentication, the `ValidateModelHook` checks if the `allowed_models` list of the `ApiKey` contains the model name requested by the client. If the model is not in the list, the request is rejected with a `403 Forbidden` error.

## Creating an API Key

API keys are managed via the Admin API, providing a consistent RESTful interface for all configuration.

To create an API key, send a `POST` request to the `/aisix/admin/apikeys` endpoint. The request body must contain the `key` and the `allowed_models`.

```bash
curl -X POST http://127.0.0.1:3001/aisix/admin/apikeys \
  -H "Authorization: Bearer your-strong-admin-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "key": "my-secret-key",
    "allowed_models": ["openai-gpt4-mini"]
  }'
```

### Key Fields

The Admin API will automatically assign a unique ID to this key.

:::info[Security Best Practice]
For production environments, always use a long, randomly generated string as your `admin_key` to secure the Admin API. The example key `your-strong-admin-key-here` is for demonstration only.
:::
-   `key`: The secret key your client application will use. It must be unique.
-   `allowed_models`: A JSON array of strings. This is a **strict whitelist** of the models (by `name`) this key can access. An empty array `[]` means no access.

## Using the API Key

Your client application must include the API key in the `Authorization` header of every request.

Example using `curl`:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-gpt4-mini",
    "messages": [...]
  }'
```

If the key is valid and authorized for the model, the request proceeds. Otherwise, it is rejected.

## Error Responses

If authentication fails, AISIX returns a `401 Unauthorized` error with one of the following messages in the response body:

-   `Missing API key in request`: If the `Authorization` header is not provided.
-   `Invalid API key`: If the provided API key does not exist in the configuration.

If the key is valid but not authorized for the requested model, the `ValidateModelHook` returns a `403 Forbidden` error with a detailed JSON body:

```json
{
  "error": {
    "message": "Access to model 'openai-gpt4-mini' is forbidden",
    "type": "invalid_request_error",
    "code": "model_access_forbidden"
  }
}
```

## Related Docs

- [Models and API Keys](../core-concepts/model-and-api-key.md) — Deep dive into the Model and API Key entities and their relationship
- [Rate Limiting](./rate-limiting.md) — Add usage limits to API keys alongside access control
