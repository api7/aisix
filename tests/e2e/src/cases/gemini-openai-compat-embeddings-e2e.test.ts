import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: /v1/embeddings against a Google AI Studio (`provider: "google"`)
// Model, which the gateway reaches through Gemini's OpenAI-compatible
// `/v1beta/openai/embeddings` endpoint.
//
// That endpoint serialises from proto3, so it omits every zero-valued
// field: `data[].index` is absent on the first (or only) vector, and the
// response carries no `usage` block. The bodies below are the shapes
// Gemini returned for `gemini-embedding-001` (captured from live calls),
// shortened to two floats.

const CALLER_PLAINTEXT = "sk-gemini-compat-embed-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const MODEL = "gemini-compat-embed";

const SINGLE_RESPONSE = {
  object: "list",
  model: "gemini-embedding-001",
  data: [{ object: "embedding", embedding: [0.11, -0.22] }],
};

const BATCH_RESPONSE = {
  object: "list",
  model: "gemini-embedding-001",
  data: [
    { object: "embedding", embedding: [0.11, -0.22] },
    { object: "embedding", index: 1, embedding: [0.33, -0.44] },
  ],
};

describe("Gemini OpenAI-compatible /v1/embeddings", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let proxy: ProxyClient;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Consumed in test order: single input first, then batch.
    upstream = await startOpenAiUpstream({
      scriptedResponses: [{ nonStreamBody: SINGLE_RESPONSE }, { nonStreamBody: BATCH_RESPONSE }],
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "gemini-compat-embed-pk",
      provider: "google",
      secret: "gemini-mock-key",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "google",
      model_name: "gemini-embedding-001",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL],
    });

    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await proxy.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function embed(input: string | string[]) {
    const res = await fetch(`${app!.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: MODEL, input }),
    });
    return { status: res.status, body: (await res.json()) as Record<string, any> };
  }

  test("single input: a vector without `index` is index 0", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const res = await embed("hi");

    expect(res.status, JSON.stringify(res.body)).toBe(200);
    expect(res.body.data).toEqual([
      { index: 0, object: "embedding", embedding: [0.11, -0.22] },
    ]);
    expect(res.body.usage).toEqual({ prompt_tokens: 0, total_tokens: 0 });

    const sent = upstream!.receivedRequests;
    expect(sent).toHaveLength(1);
    expect(sent[0]!.path).toBe("/embeddings");
    expect(JSON.parse(sent[0]!.body)).toEqual({ model: "gemini-embedding-001", input: "hi" });
  });

  test("batch input: the omitted zero index and the explicit one both survive", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const res = await embed(["a", "b"]);

    expect(res.status, JSON.stringify(res.body)).toBe(200);
    expect(res.body.data).toEqual([
      { index: 0, object: "embedding", embedding: [0.11, -0.22] },
      { index: 1, object: "embedding", embedding: [0.33, -0.44] },
    ]);
  });
});
