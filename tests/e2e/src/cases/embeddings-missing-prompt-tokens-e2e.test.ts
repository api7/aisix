import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: OpenAI client → OpenAI-compatible upstream — `/v1/embeddings`
// tolerates a usage block that omits `prompt_tokens`.
//
// Pins issue #474: Jina's OpenAI-compatible `/v1/embeddings` returns
// `usage.total_tokens` only, omitting `prompt_tokens`. The gateway's
// `OpenAiEmbedUsage` declared `prompt_tokens` as a required field, so
// serde rejected the body and the gateway surfaced
// `502 upstream_decode_error` even though the upstream returned 200.
//
// What this spec proves: an embeddings response whose `usage` has only
// `total_tokens` returns 200 to the SDK with the vectors intact.

const CALLER_PLAINTEXT = "sk-openai-embed-missing-pt-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FLOAT_VECTOR = [0.1, -0.2, 0.3];

// Jina-shape response: `usage` carries `total_tokens` only.
const JINA_SHAPE_RESPONSE = {
  object: "list",
  model: "jina-embeddings-v5-text-small",
  data: [{ index: 0, object: "embedding", embedding: FLOAT_VECTOR }],
  usage: { total_tokens: 6 },
};

describe("OpenAI /v1/embeddings tolerates missing usage.prompt_tokens (#474)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      scriptedResponses: [
        // Readiness probe via chat — consumes scripted step 0.
        {
          nonStreamBody: {
            id: "chatcmpl-ready",
            object: "chat.completion",
            created: 0,
            model: "jina-embeddings-v5-text-small",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: "ready" },
                finish_reason: "stop",
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        },
        // Embeddings call — usage without prompt_tokens.
        { nonStreamBody: JINA_SHAPE_RESPONSE },
      ],
      nonStreamBody: JINA_SHAPE_RESPONSE,
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "jina-embed-pk",
      secret: "sk-jina-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "jina-embed",
      provider: "openai",
      model_name: "jina-embeddings-v5-text-small",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["jina-embed"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("usage with only total_tokens → 200, not 502", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "jina-embed",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    // Pre-#474 this threw a 502 upstream_decode_error.
    const resp = await client.embeddings.create({
      model: "jina-embed",
      input: ["hello", "world"],
      encoding_format: "float",
    });

    expect(resp.data).toHaveLength(1);
    expect(resp.data[0]?.object).toBe("embedding");
    expect(Array.isArray(resp.data[0]?.embedding)).toBe(true);
    expect(resp.usage.total_tokens).toBe(6);
    // prompt_tokens was absent upstream → surfaces as 0, not an error.
    expect(resp.usage.prompt_tokens).toBe(0);
  });
});
