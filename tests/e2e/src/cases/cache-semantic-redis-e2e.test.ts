import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { connect } from "node:net";
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

// E2E: semantic cache on the SHARED redis backend (AISIX-Cloud#558,
// follow-up to the in-process store). Two DP replicas of one
// environment (same etcd prefix) share one vector-capable Redis: an
// entry stored by replica A must serve replica B both exactly AND
// semantically, and a purge must invalidate across replicas.
//
// Capability-conditional: the vector suite runs iff AISIX_E2E_REDIS
// speaks the vector-search command family (redis:8+ — what CI
// provisions); the degradation suite runs iff it does NOT (a plain
// redis 6/7), pinning that `semantic` on a backend=redis policy then
// stays exact-only without failing traffic. Each suite skips honestly
// when its capability precondition doesn't hold.

const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

const CALLER_PLAINTEXT = "sk-semredis-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** RESP-level probe: does the server speak FT.* (vector search)?
 *  `null` = server unreachable (both suites skip). */
async function redisVectorSupport(url: string): Promise<boolean | null> {
  // Plaintext RESP probe: a rediss:// target cannot be probed this way
  // — that is "unknown", not "unsupported", so both suites skip
  // honestly rather than mislabeling a TLS server as vector-less.
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
      if (head.startsWith("*")) return done(true); // array reply = supported
      // Only an explicit unknown-command error PROVES the capability is
      // absent; anything else (auth required, loading, protocol noise)
      // proves nothing either way.
      if (/^-ERR unknown command/i.test(head)) return done(false);
      done(null);
    });
    sock.once("error", () => done(null));
    sock.setTimeout(1000, () => done(null));
  });
}

function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  if (t.includes("shared-topic")) return [1, 0, 0, 0];
  if (t.includes("purge-topic")) return [0, 1, 0, 0];
  return [0, 0, 0, 1];
}

interface EmbeddingMock {
  baseUrl: string;
  close(): Promise<void>;
}

async function startEmbeddingMock(): Promise<EmbeddingMock> {
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input)
        ? body.input
        : [body.input ?? ""];
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data: inputs.map((text, index) => ({
            object: "embedding",
            index,
            embedding: keywordVector(text),
          })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

interface ChatResult {
  status: number;
  content: string | undefined;
  cache: string | null;
  layer: string | null;
  similarity: number | null;
}

async function chatOn(
  proxyUrl: string,
  model: string,
  prompt: string,
): Promise<ChatResult> {
  const res = await fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  let content: string | undefined;
  if (res.status === 200) {
    const json = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    content = json.choices?.[0]?.message?.content;
  } else {
    await res.text();
  }
  const sim = res.headers.get("x-aisix-cache-similarity");
  return {
    status: res.status,
    content,
    cache: res.headers.get("x-aisix-cache"),
    layer: res.headers.get("x-aisix-cache-layer"),
    similarity: sim === null ? null : Number(sim),
  };
}

/** Seed the full fixture set (embed model, chat model, redis+semantic
 *  policy, caller key LAST per tests/e2e/AGENTS.md), returning the
 *  policy handle for the purge test. */
async function seedFixtures(
  seed: SeedClient,
  embedBase: string,
  upstreamBase: string,
): Promise<{ policyId: string; policyBody: Record<string, unknown> }> {
  const embedPk = await seed.createProviderKey({
    display_name: "semredis-embed-pk",
    secret: "sk-mock",
    api_base: `${embedBase}/v1`,
  });
  await seed.createModel({
    display_name: "embed-redis",
    provider: "openai",
    model_name: "embed-mock",
    provider_key_id: embedPk.id,
    embedding: { dimensions: 4, normalize: true },
  });
  const chatPk = await seed.createProviderKey({
    display_name: "semredis-chat-pk",
    secret: "sk-mock",
    api_base: `${upstreamBase}/v1`,
  });
  await seed.createModel({
    display_name: "chat-redis",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: chatPk.id,
  });
  const policy = await seed.createCachePolicy({
    name: "sem-redis",
    backend: "redis",
    applies_to: "model:chat-redis",
    ttl_seconds: 600,
    semantic: { embedding_model: "embed-redis", threshold: 0.85 },
  });
  await seed.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: ["*"],
  });
  return { policyId: policy.id, policyBody: policy.value };
}

async function waitKeyLive(proxyUrl: string): Promise<void> {
  const probe = new ProxyClient(proxyUrl, CALLER_PLAINTEXT);
  await waitConfigPropagation(
    async () => (await probe.listModels()).status === 200,
  );
}

describe("semantic cache on shared redis (vector-capable)", () => {
  let appA: SpawnedApp | undefined;
  let appB: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: OpenAiUpstream | undefined;
  let embedMock: EmbeddingMock | undefined;
  let policyId: string;
  let policyBody: Record<string, unknown>;
  let ready = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    if ((await redisVectorSupport(REDIS_URL)) !== true) return;

    embedMock = await startEmbeddingMock();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-shared",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "answer-shared-redis" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 },
      },
    });
    const redisExtra = {
      cache: { backend: "memory", redis: { url: REDIS_URL } },
    };
    appA = await spawnApp({ extra: redisExtra });
    appB = await spawnApp({ extra: redisExtra, etcdPrefix: appA.etcdPrefix });
    seed = new SeedClient(etcd, appA.etcdPrefix);
    ({ policyId, policyBody } = await seedFixtures(
      seed,
      embedMock.baseUrl,
      upstream.baseUrl,
    ));
    await waitKeyLive(appA.proxyUrl);
    await waitKeyLive(appB.proxyUrl);
    ready = true;
  });

  afterAll(async () => {
    await appA?.exit();
    await appB?.exit();
    await upstream?.close();
    await embedMock?.close();
  });

  test("replica A's entry serves replica B exactly AND semantically", async (ctx) => {
    if (!ready) {
      ctx.skip();
      return;
    }
    const baseline = upstream!.receivedRequests.length;
    // Unique-enough prompt per run: redis outlives the test process.
    const runTag = `${process.pid}-${Date.now()}`;
    const prompt = `shared-topic question ${runTag}`;

    const first = await chatOn(appA!.proxyUrl, "chat-redis", prompt);
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");
    expect(upstream!.receivedRequests.length).toBe(baseline + 1);

    // Same wording on the OTHER replica: exact layer through redis.
    const exactB = await chatOn(appB!.proxyUrl, "chat-redis", prompt);
    expect(exactB.cache).toBe("hit");
    expect(exactB.layer).toBe("exact");
    expect(exactB.content).toBe("answer-shared-redis");

    // Different wording, same meaning, on the OTHER replica: the
    // semantic layer itself is shared — the entry was stored by A.
    const semanticB = await chatOn(
      appB!.proxyUrl,
      "chat-redis",
      `shared-topic rephrased ${runTag}`,
    );
    expect(semanticB.cache).toBe("hit");
    expect(semanticB.layer).toBe("semantic");
    expect(semanticB.similarity).not.toBeNull();
    expect(semanticB.similarity!).toBeGreaterThanOrEqual(0.85);
    expect(semanticB.content).toBe("answer-shared-redis");
    expect(upstream!.receivedRequests.length).toBe(baseline + 1);
  });

  test("purge invalidates across replicas", async (ctx) => {
    if (!ready) {
      ctx.skip();
      return;
    }
    const runTag = `${process.pid}-${Date.now()}`;
    const prompt = `purge-topic question ${runTag}`;
    const seeded = await chatOn(appA!.proxyUrl, "chat-redis", prompt);
    expect(seeded.cache).toBe("miss");
    expect((await chatOn(appB!.proxyUrl, "chat-redis", prompt)).cache).toBe(
      "hit",
    );

    await seed!.update("cache_policies", policyId, {
      ...policyBody,
      purge_generation: 1,
    });
    // Seal the update on BOTH replicas: a canary key seeded after the
    // update authenticating proves the new generation landed
    // (tests/e2e/AGENTS.md).
    const canary = `sk-semredis-canary-${runTag}`;
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    for (const app of [appA!, appB!]) {
      const probe = new ProxyClient(app.proxyUrl, canary);
      await waitConfigPropagation(
        async () => (await probe.listModels()).status === 200,
      );
    }

    // Semantic first, on the axis nothing has re-stored yet: the
    // pre-purge entry is unreachable from the OTHER replica.
    const semanticB = await chatOn(
      appB!.proxyUrl,
      "chat-redis",
      `purge-topic rephrased ${runTag}`,
    );
    expect(semanticB.cache).toBe("miss");
    // The pre-purge WORDING on replica A is no longer exact-served: it
    // matches the rephrased request's fresh entry SEMANTICALLY (layer
    // discriminates — an un-purged exact entry would win as `exact`).
    const oldWording = await chatOn(appA!.proxyUrl, "chat-redis", prompt);
    expect(oldWording.cache).toBe("hit");
    expect(oldWording.layer).toBe("semantic");
  });
});

// A dedicated vector-LESS redis for this suite when provided (CI sets
// AISIX_E2E_REDIS_PLAIN to a redis:7 service); otherwise fall back to
// the main URL and run only when IT happens to lack vector support.
const PLAIN_REDIS_URL = process.env.AISIX_E2E_REDIS_PLAIN ?? REDIS_URL;

describe("semantic on backend=redis degrades to exact-only without vector support", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let embedMock: EmbeddingMock | undefined;
  let ready = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    if ((await redisVectorSupport(PLAIN_REDIS_URL)) !== false) return;

    embedMock = await startEmbeddingMock();
    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      extra: { cache: { backend: "memory", redis: { url: PLAIN_REDIS_URL } } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seedFixtures(seed, embedMock.baseUrl, upstream.baseUrl);
    await waitKeyLive(app.proxyUrl);
    ready = true;
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await embedMock?.close();
  });

  test("exact matching keeps working; similar wording pays the upstream", async (ctx) => {
    if (!ready) {
      ctx.skip();
      return;
    }
    const runTag = `${process.pid}-${Date.now()}`;
    const prompt = `shared-topic question ${runTag}`;
    const first = await chatOn(app!.proxyUrl, "chat-redis", prompt);
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");

    const exact = await chatOn(app!.proxyUrl, "chat-redis", prompt);
    expect(exact.cache).toBe("hit");
    expect(exact.layer).toBe("exact");

    // Same meaning, different wording: no semantic layer available —
    // the request pays the upstream and still succeeds.
    const similar = await chatOn(
      app!.proxyUrl,
      "chat-redis",
      `shared-topic rephrased ${runTag}`,
    );
    expect(similar.status).toBe(200);
    expect(similar.cache).toBe("miss");
  });
});
