import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { connect } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// A Redis cache hit treats the stored usage exactly as the response that
// was stored: a count the upstream reported stands, `0` included, and only
// a count it left out is estimated. The client and the usage record agree,
// and an estimated record carries no total.
//
// Pinned on the shared Redis backend, both layers: the exact layer, and the
// semantic layer (a rephrased request served by the vector index). The
// in-process stores hold the response itself; the Redis stores round-trip
// it through JSON, which is where "reported 0" and "left out" used to
// collapse into one.

const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

const CALLER_PLAINTEXT = "sk-cache-hit-usage-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "cache-hit-usage-proj";
const LOGSTORE = "cache-hit-usage-store";

const OUT = 2;
const PROMPT_ZERO = { prompt_tokens: 0, completion_tokens: OUT, total_tokens: OUT };
const PROMPT_LEFT_OUT = { completion_tokens: OUT, total_tokens: OUT };

const chatBody = (usage: Record<string, number>) => ({
  id: "chatcmpl-cache-hit-usage",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "relay-compat-x",
  choices: [
    { index: 0, message: { role: "assistant", content: "Hello world" }, finish_reason: "stop" },
  ],
  usage,
});

/** `null` when the server is unreachable or the answer proves nothing. */
async function redisVectorSupport(url: string): Promise<boolean | null> {
  if (/^rediss:\/\//.test(url)) return null;
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return null;
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  return new Promise((resolve) => {
    const sock = connect({ host, port }, () => sock.write("FT._LIST\r\n"));
    const done = (v: boolean | null) => {
      sock.destroy();
      resolve(v);
    };
    sock.once("data", (buf) => {
      const head = buf.toString();
      if (head.startsWith("*")) return done(true);
      if (/^-ERR unknown command/i.test(head)) return done(false);
      done(null);
    });
    sock.once("error", () => done(null));
    sock.setTimeout(1000, () => done(null));
  });
}

/** Every input embeds to the same vector, so any rephrasing is a semantic
 * match within its model's scope. */
async function startEmbeddingMock(): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input ?? ""];
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((_, index) => ({ object: "embedding", index, embedding: [1, 0, 0, 0] })),
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

/** Model display name → the usage its upstream reports, and whether its
 * cache policy carries a semantic layer. */
const MODELS: Record<string, { usage: Record<string, number>; semantic: boolean }> = {
  "hit-exact-zero": { usage: PROMPT_ZERO, semantic: false },
  "hit-exact-absent": { usage: PROMPT_LEFT_OUT, semantic: false },
  "hit-sem-zero": { usage: PROMPT_ZERO, semantic: true },
  "hit-sem-absent": { usage: PROMPT_LEFT_OUT, semantic: true },
};

describe("a redis cache hit estimates only the usage counts its upstream left out", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let embed: { baseUrl: string; close(): Promise<void> } | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let ready = false;
  let vectorReady = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    // The semantic layer on Redis needs the vector query engine (redis:8+,
    // what CI provisions); the exact layer runs on any reachable Redis.
    const vector = await redisVectorSupport(REDIS_URL);
    if (vector === null) return;
    vectorReady = vector;

    sls = await startMockSls();
    embed = await startEmbeddingMock();
    app = await spawnApp({
      exposeDebugHeaders: true,
      extra: { cache: { backend: "memory", redis: { url: REDIS_URL } } },
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "cache-hit-usage-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const embedPk = await seed.createProviderKey({
      display_name: "cache-hit-usage-embed-pk",
      api_key: "sk-mock",
      api_base: `${embed.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "cache-hit-usage-embed",
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: embedPk.id,
      embedding: { dimensions: 4, normalize: true },
    });
    // A non-OpenAI upstream model name, so the estimator's fallback
    // encoding applies.
    for (const [display, { usage, semantic }] of Object.entries(MODELS)) {
      if (semantic && !vectorReady) continue;
      const upstream = await startOpenAiUpstream({ nonStreamBody: chatBody(usage) });
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        api_key: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: display,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
      await seed.createCachePolicy({
        name: `${display}-policy`,
        backend: "redis",
        applies_to: `model:${display}`,
        ttl_seconds: 600,
        ...(semantic
          ? { semantic: { embedding_model: "cache-hit-usage-embed", threshold: 0.85 } }
          : {}),
      });
    }
    // Caller key last: it authenticating implies the whole seed set is in
    // the snapshot (tests/e2e/AGENTS.md).
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    ready = true;
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await embed?.close();
    for (const u of upstreams) await u.close();
  });

  async function chat(model: string, prompt: string) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }] }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return {
      requestId,
      cache: res.headers.get("x-aisix-cache"),
      layer: res.headers.get("x-aisix-cache-layer"),
      usage: (JSON.parse(text) as { usage: Record<string, number> }).usage,
    };
  }

  const record = (requestId: string) =>
    waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, `usage row for ${requestId}`, 15_000);

  /** Store the upstream's response with one request, then answer a second
   * one from the cache on `layer`. */
  async function hit(model: string, layer: "exact" | "semantic") {
    // Redis outlives the test process: a run-unique prompt starts cold.
    const runTag = `${process.pid}-${Date.now()}-${model}`;
    const first = await chat(model, `cache-hit-usage question ${runTag}`);
    expect(first.cache).toBe("miss");
    const second = await chat(
      model,
      layer === "exact" ? `cache-hit-usage question ${runTag}` : `cache-hit-usage rephrased ${runTag}`,
    );
    expect(second.cache).toBe("hit");
    expect(second.layer).toBe(layer);
    return second;
  }

  for (const layer of ["exact", "semantic"] as const) {
    const prefix = layer === "exact" ? "hit-exact" : "hit-sem";

    const runs = () => ready && (layer === "exact" || vectorReady);

    test(`${layer} hit: a reported prompt_tokens 0 stands`, async (ctx) => {
      if (!runs()) return ctx.skip();
      const served = await hit(`${prefix}-zero`, layer);
      expect(served.usage.prompt_tokens).toBe(0);
      expect(served.usage.completion_tokens).toBe(OUT);
      expect(served.usage.total_tokens).toBe(OUT);
      const log = await record(served.requestId);
      expect(log.get("cache_status")).toBe("hit");
      expect(log.get("usage_estimated")).toBeUndefined();
      expect(log.get("prompt_tokens") ?? "0").toBe("0");
      expect(log.get("completion_tokens")).toBe(String(OUT));
      expect(log.get("total_tokens")).toBe(String(OUT));
    });

    test(`${layer} hit: a prompt_tokens left out is estimated`, async (ctx) => {
      if (!runs()) return ctx.skip();
      const served = await hit(`${prefix}-absent`, layer);
      const log = await record(served.requestId);
      expect(log.get("cache_status")).toBe("hit");
      expect(log.get("usage_estimated")).toBe("true");
      expect(log.get("total_tokens")).toBeUndefined();
      expect(log.get("completion_tokens")).toBe(String(OUT));
      const prompt = Number(log.get("prompt_tokens") ?? "0");
      expect(prompt).toBeGreaterThan(0);
      expect(served.usage.prompt_tokens).toBe(prompt);
      expect(served.usage.completion_tokens).toBe(OUT);
      expect(served.usage.total_tokens).toBe(prompt + OUT);
    });
  }
});
