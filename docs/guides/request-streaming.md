---
slug: /aisix/guides/request-streaming
title: Request Streaming
description: Learn how to handle real-time, streaming responses in AISIX.
---

AISIX supports streaming chat completions, allowing you to build real-time applications where the AI's response is streamed to the client token by token. This is essential for a good user experience in chatbots and other conversational AI interfaces.

## How Streaming Works

To initiate a streaming request, your client sets the `stream` parameter to `true` in the chat completion request body. This is the standard way to request streaming in the OpenAI API.

When AISIX receives a request with `"stream": true`, it performs these steps:

1.  **Pre-Call Hooks**: All `pre_call` hooks (e.g., `ValidateModelHook`, `RateLimitHook`) are executed as usual.

2.  **Initiate Upstream Stream**: AISIX opens a streaming connection to the upstream LLM provider.

3.  **Stream Chunks to Client**: As AISIX receives Server-Sent Events (SSE) chunks from the upstream provider, it forwards them to the client in the standard OpenAI streaming format.

4.  **Post-Call Hooks**: During the stream, `post_call` hooks are active. The `RateLimitHook`, for example, accumulates the token count from the `usage` field in the final chunk to update token-based rate limits.

## Client Implementation

From the client's perspective, interacting with a streaming endpoint in AISIX is identical to OpenAI's streaming API. You can use any OpenAI-compatible library.

### Example with `curl`

You can test streaming with `curl`:

```bash
curl -N http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-gpt4-mini",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a long story about a brave little API gateway."
      }
    ],
    "stream": true
  }'
```

The `-N` flag in `curl` disables buffering, which is important for seeing response chunks as they arrive.

You will see a series of `data:` lines, each with a JSON object representing a response chunk, followed by a final `data: [DONE]` message.

### Example with Python

This is a simple example using Python's `requests` library:

```python
import requests
import json

headers = {
    "Authorization": "Bearer my-secret-key",
    "Content-Type": "application/json",
}

data = {
    "model": "my-gpt4-mini",
    "messages": [
        {
            "role": "user",
            "content": "Tell me a long story about a brave little API gateway."
        }
    ],
    "stream": True,
}

response = requests.post(
    "http://localhost:3000/v1/chat/completions",
    headers=headers,
    json=data,
    stream=True,
)

for line in response.iter_lines():
    if line:
        line_str = line.decode("utf-8")
        if line_str.startswith("data:"):
            json_data = line_str[5:].strip()
            if json_data == "[DONE]":
                print("\nStream finished.")
                break
            try:
                chunk = json.loads(json_data)
                if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                    print(chunk["choices"][0]["delta"]["content"], end="")
            except json.JSONDecodeError:
                print(f"\nCould not decode: {json_data}")

```

This script makes a streaming request and prints the content of each chunk to the console as it arrives.
