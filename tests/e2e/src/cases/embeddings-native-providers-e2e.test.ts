import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: /v1/embeddings native provider translation (#723,
// AISIX-Cloud#873 §⑤ "跨厂商 embeddings"). Pre-#723 only OpenAI-shaped
// upstreams worked — a Vertex/Gemini or Bedrock Model on /v1/embeddings
// returned 501. These cases drive the real `aisix` binary through the
// OpenAI SDK and pin the provider-native upstream wire:
//
//   1. Vertex/Gemini: google-publisher `:predict` with the
//      `instances[{content}]` body and OAuth bearer; per-prediction
//      `statistics.token_count` sums into OpenAI-shape usage.
//   2. Bedrock Titan: one SigV4-signed InvokeModel per input text
//      (`{"inputText": …}`), vectors reassembled in input order.

const CALLER_PLAINTEXT = "sk-nembed-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface RecordingUpstream {
  baseUrl: string;
  received: { method: string; path: string; body: string; headers: Record<string, string> }[];
  close(): Promise<void>;
}

async function startJsonUpstream(
  reply: (path: string, body: string) => unknown,
): Promise<RecordingUpstream> {
  const received: RecordingUpstream["received"] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      const path = (req.url ?? "/").split("?")[0];
      received.push({
        method: req.method ?? "GET",
        path,
        body,
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [
            k,
            Array.isArray(v) ? v.join(",") : (v ?? ""),
          ]),
        ),
      });
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(reply(path, body)));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

describe("native-provider embeddings e2e (#723)", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: RecordingUpstream[] = [];
  let client: OpenAI;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("vertex/gemini model embeds via :predict with OAuth bearer + summed usage", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const upstream = await startJsonUpstream(() => ({
      predictions: [
        { embeddings: { values: [0.11, 0.22], statistics: { token_count: 3 } } },
        { embeddings: { values: [0.33, 0.44], statistics: { token_count: 2 } } },
      ],
    }));
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "vertex-embed-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.e2e-test",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "vertex-embed-model",
      provider: "google",
      model_name: "text-embedding-005",
      provider_key_id: pk.id,
    });

    // Gate on the DP snapshot via /v1/models — authenticates only once the
    // caller key has propagated, lists the model only once the snapshot has
    // it, and touches no upstream.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "vertex-embed-model");
    });

    const resp = await client.embeddings.create({
      model: "vertex-embed-model",
      input: ["hello", "world"],
    });
    expect(resp.data.length).toBe(2);
    expect(resp.data[0].embedding).toEqual([0.11, 0.22]);
    expect(resp.data[1].embedding).toEqual([0.33, 0.44]);
    expect(resp.usage.prompt_tokens).toBe(5);

    const seen = upstream.received[0];
    expect(seen.path).toBe(
      "/v1/projects/proj-e2e/locations/us-central1/publishers/google/models/text-embedding-005:predict",
    );
    expect(seen.headers.authorization).toBe("Bearer ya29.e2e-test");
    const sent = JSON.parse(seen.body) as { instances: { content: string }[] };
    expect(sent.instances).toEqual([{ content: "hello" }, { content: "world" }]);
  });

  test("bedrock titan model embeds via one SigV4 InvokeModel per input", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    let call = 0;
    const upstream = await startJsonUpstream(() => {
      call += 1;
      return { embedding: [call * 0.1, call * 0.2], inputTextTokenCount: 4 };
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "bedrock-embed-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-e2e",
        secret_access_key: "sk-e2e",
        region: "us-west-2",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "bedrock-embed-model",
      provider: "bedrock",
      model_name: "amazon.titan-embed-text-v1",
      provider_key_id: pk.id,
    });

    // Same DP-snapshot gate as the vertex case above.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "bedrock-embed-model");
    });

    const resp = await client.embeddings.create({
      model: "bedrock-embed-model",
      input: ["a", "b"],
    });
    expect(resp.data.length).toBe(2);
    expect(resp.usage.prompt_tokens).toBe(8);

    // Titan embeds one text per call: two invokes on the model URL,
    // SigV4-signed, bodies in input order.
    const invokes = upstream.received.filter((r) =>
      r.path.includes("/model/amazon.titan-embed-text-v1/invoke"),
    );
    expect(invokes.length).toBe(2);
    expect(JSON.parse(invokes[0].body)).toEqual({ inputText: "a" });
    expect(JSON.parse(invokes[1].body)).toEqual({ inputText: "b" });
    expect(invokes[0].headers.authorization).toContain("AWS4-HMAC-SHA256");
  });
});
