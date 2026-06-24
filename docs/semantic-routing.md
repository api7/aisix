# Semantic routing

Semantic routing lets one virtual model dispatch each request to a different
upstream model based on the **meaning** of the request, instead of a fixed
model name, a weight, or a health check. Callers always send the same
`model`; the gateway reads the latest user message, embeds it, compares it to
example utterances you attach to each route, and forwards the request to the
best-matching route's target — or to a `default` model when nothing matches.

It is a fourth virtual-model shape alongside direct models, routing groups
(load balancing), and ensembles. Like those, it is just a `Model` with a
config block — here, a `semantic` block.

```text
caller sends  model: "prod-chat"
   │
   ├─ embed the latest user message  (via the configured embedding model)
   ├─ cosine-score it against every route's example embeddings
   ├─ aggregate per route (max), keep the highest route that clears its threshold
   └─ dispatch to that route's target  ·  or to `default` when none clears
```

## Pieces you configure

Semantic routing needs two model kinds:

1. **An embedding model** — a normal direct model that points at an
   OpenAI-compatible `/v1/embeddings` endpoint, plus an `embedding` block that
   records its output dimensionality. It can also be called directly via
   `/v1/embeddings`.
2. **A semantic router** — a model with a `semantic` block that references the
   embedding model and lists the routes.

### Embedding model

```jsonc
{
  "display_name": "bge-m3",
  "provider": "openai",
  "model_name": "bge-m3",            // the upstream model id
  "provider_key_id": "<provider key for the embeddings endpoint>",
  "embedding": {
    "dimensions": 1024,               // required: output vector size
    "normalize": true                 // default true; set false if the
                                      // endpoint does not return unit vectors
  }
}
```

The endpoint must speak the OpenAI `/v1/embeddings` shape and return float
vectors (`encoding_format: "float"`). Self-hosted runners such as
[TEI](https://github.com/huggingface/text-embeddings-inference) and Ollama, as
well as hosted providers, all qualify.

### Semantic router

```jsonc
{
  "display_name": "prod-chat",        // callers send model: "prod-chat"
  "semantic": {
    "embedding_model": "bge-m3",      // alias of the embedding model above
    "routes": [
      {
        "name": "legal",              // surfaced in x-aisix-route + logs
        "target": "claude-opus",      // a direct model alias
        "description": "Contract & legal risk analysis",  // optional, docs only
        "examples": [                 // >= 1; the route is matched on these
          "analyze this contract for legal risk",
          "review this NDA for liability exposure",
          "这条赔偿条款合法吗"
        ],
        "threshold": 0.8              // optional per-route override
      },
      { "name": "translate", "target": "gpt-4o-mini", "examples": ["translate this"] }
    ],
    "default": "gpt-4o",              // used when no route clears its threshold
    "match": {
      "distance_metric": "cosine",   // default cosine (only value today)
      "aggregation": "max",          // default max (a route's score is its best example)
      "threshold": 0.75              // default threshold for routes without one
    },
    "embedding_timeout_ms": 500,     // optional per-call embedding deadline
    "on_embedding_failure": "default" // see below
  }
}
```

## How matching works

- Only the **latest user message** is embedded (multimodal text blocks are
  concatenated; non-text content is ignored). System/assistant/tool turns do
  not affect routing.
- Each route's score is the **maximum** cosine similarity between the request
  and that route's example utterances. Higher is more similar / stricter.
- A route matches when its score is `>=` its effective threshold (its own
  `threshold`, else `match.threshold`). Among matching routes, the highest
  score wins. If none match, the request goes to `default`.
- Example utterances are embedded once and cached in the data plane (keyed by
  embedding model, dimensions, and text), so the steady-state cost of a
  request is a single embedding call for the prompt plus local arithmetic.

Cross-lingual matching works out of the box with a multilingual embedding
model — a Chinese prompt can match English examples and vice-versa. Real
cosine scores for related-but-not-identical text typically fall in the
`0.4–0.65` range, so tune thresholds against your own examples rather than
assuming high cutoffs.

## What the caller sees

The response body reports the **resolved** upstream model, and two headers
expose the decision:

- `x-aisix-route: <route name>` — the route that matched (absent when the
  request fell through to `default`).
- `x-aisix-served-by: <target display name>` — the direct model that served
  the request.

## When embedding fails

If the embedding call errors or times out (or the `embedding_model` /
`target` / `default` reference is missing), the router applies
`on_embedding_failure`:

| value             | behavior                                   |
|-------------------|--------------------------------------------|
| `"default"`       | route to the router's `default` model (the default) |
| `"fail"`          | reject the request with `503`              |
| `{ "target": "<alias>" }` | route to a specific safe model      |

## Notes

- A semantic router is mutually exclusive with the direct-upstream fields and
  with `routing` / `ensemble`; the `embedding` block is only valid on a direct
  model. The Admin API rejects invalid combinations.
- Route example vectors are recomputed automatically when you change an
  example's text, the embedding model, or its dimensions.
- Routing decisions can be influenced by adversarial prompts, so run input
  guardrails *before* routing (the gateway does) and treat similarity scores
  as operator-only signals.
