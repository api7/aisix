import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
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
import { pickFreePort } from "../harness/ports.js";

// E2E: the debug response headers on /v1/chat/completions are opt-in
// (AISIX-Cloud#1917). A caller of a model group must not learn from the
// response which model behind the group answered, so by default none of
// `x-aisix-call-id`, `x-aisix-served-by`, `x-aisix-route`, `x-aisix-cache`,
// `x-aisix-cache-layer`, `x-aisix-cache-similarity` is sent. With
// `proxy.expose_debug_headers: true` they are sent as before. The
// correlation id and the server token are not debug headers and are sent
// either way.

const CALLER_PLAINTEXT = "sk-debug-headers-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const DEBUG_HEADERS = [
  "x-aisix-call-id",
  "x-aisix-served-by",
  "x-aisix-route",
  "x-aisix-cache",
  "x-aisix-cache-layer",
  "x-aisix-cache-similarity",
];

// "contract" prompts embed onto the legal route's example; anything else
// is orthogonal to it.
async function startEmbeddingMock(): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((text, index) => ({
            object: "embedding",
            index,
            embedding: text.toLowerCase().includes("contract") ? [1, 0] : [0, 1],
          })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe.each([
  { expose: undefined, label: "default config" },
  { expose: true, label: "proxy.expose_debug_headers: true" },
])("debug response headers ($label)", ({ expose }) => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let embed: Awaited<ReturnType<typeof startEmbeddingMock>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    embed = await startEmbeddingMock();
    app = await spawnApp(expose === undefined ? {} : { exposeDebugHeaders: expose });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "dh-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "dh-direct",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const embedPk = await seed.createProviderKey({
      display_name: "dh-embed-pk",
      secret: "sk-mock",
      api_base: `${embed.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "dh-embed",
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: embedPk.id,
      embedding: { dimensions: 2, normalize: true },
    });
    await seed.createModel({
      display_name: "dh-group",
      routing: { strategy: "failover", targets: [{ model: "dh-direct" }] },
    });
    await seed.createModel({
      display_name: "dh-semantic",
      semantic: {
        embedding_model: "dh-embed",
        routes: [
          {
            name: "legal",
            target: "dh-direct",
            examples: ["review this contract"],
            threshold: 0.5,
          },
        ],
        default: "dh-direct",
        match: { threshold: 0.5 },
      },
    });
    await seed.createCachePolicy({ name: "dh-cache", enabled: true, applies_to: "all" });
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await embed?.close();
  });

  async function chat(
    model: string,
    prompt: string,
    extraHeaders: Record<string, string> = {},
  ): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        ...extraHeaders,
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
    });
    await res.text();
    expect(res.status).toBe(200);
    expect(res.headers.get("x-aisix-request-id")).toBeTruthy();
    expect(res.headers.get("server")).toMatch(/^AISIX\//);
    return res;
  }

  function expectNoDebugHeaders(res: Response): void {
    for (const name of DEBUG_HEADERS) {
      expect(res.headers.get(name), name).toBeNull();
    }
  }

  test("model group: served-by and call-id", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await chat("dh-group", "hello from the group");
    if (expose) {
      expect(res.headers.get("x-aisix-served-by")).toBe("dh-direct");
      expect(res.headers.get("x-aisix-call-id")).toBeTruthy();
    } else {
      expectNoDebugHeaders(res);
    }
  });

  test("semantic router hit: route", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await chat("dh-semantic", "please check the contract terms");
    if (expose) {
      expect(res.headers.get("x-aisix-route")).toBe("legal");
    } else {
      expectNoDebugHeaders(res);
    }
  });

  test("cache miss, hit and bypass: cache headers", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const before = upstream.receivedRequests.length;
    const miss = await chat("dh-direct", "cache me please");
    const hit = await chat("dh-direct", "cache me please");
    // The cache works the same either way; only the headers differ.
    expect(upstream.receivedRequests.length).toBe(before + 1);
    const bypass = await chat("dh-direct", "cache me please", { "cache-control": "no-cache" });
    expect(upstream.receivedRequests.length).toBe(before + 2);
    if (expose) {
      expect(miss.headers.get("x-aisix-cache")).toBe("miss");
      expect(hit.headers.get("x-aisix-cache")).toBe("hit");
      expect(hit.headers.get("x-aisix-cache-layer")).toBe("exact");
      expect(bypass.headers.get("x-aisix-cache")).toBe("bypass");
    } else {
      expectNoDebugHeaders(miss);
      expectNoDebugHeaders(hit);
      expectNoDebugHeaders(bypass);
    }
  });
});
