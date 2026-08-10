import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { pickFreePort } from "../harness/ports.js";

// E2E: semantic cache (AISIX-Cloud#558). A CachePolicy carrying a
// `semantic` block serves an exact-fingerprint (L1) hit first and, on an
// L1 miss, embeds the request and serves the nearest stored entry at or
// above the policy's cosine threshold (L2). Real `aisix` binary + etcd +
// mock chat upstreams + a deterministic mock embedding endpoint; no CP.
//
// The embedding mock maps input text to fixed 8-dim vectors by keyword,
// so similarity is fully deterministic. Each test owns an orthogonal
// axis (entries persist across tests within a policy+scope partition,
// so unrelated tests must never share a vector):
//   "topic-a-near" -> 0.9*topic-a + spill   (cos 0.9 vs topic-a)
//   "topic-c-near" -> 0.9*topic-c + spill   (same pair on another axis)
//   "topic-a" / "topic-b" / "topic-c" / "no-store" / "refresh" /
//   "purge" / default -> one distinct axis each
//
// Observable contract under test (headers are the wire surface):
//   x-aisix-cache: hit | miss | bypass
//   x-aisix-cache-layer: exact | semantic   (hits only)
//   x-aisix-cache-similarity: <float>       (semantic hits only)

const CALLER_PLAINTEXT = "sk-semcache-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const OTHER_CALLER_PLAINTEXT = "sk-semcache-other";
const OTHER_CALLER_KEY_HASH = createHash("sha256")
  .update(OTHER_CALLER_PLAINTEXT)
  .digest("hex");

function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  const spill = Math.sqrt(0.19); // makes the *-near vectors unit-length
  // Longest keyword first: "topic-a-near" contains "topic-a".
  if (t.includes("topic-a-near")) return [0.9, 0, 0, 0, spill, 0, 0, 0];
  if (t.includes("topic-a")) return [1, 0, 0, 0, 0, 0, 0, 0];
  if (t.includes("topic-b")) return [0, 1, 0, 0, 0, 0, 0, 0];
  if (t.includes("topic-c-near")) return [0, 0, 0.9, 0, spill, 0, 0, 0];
  if (t.includes("topic-c")) return [0, 0, 1, 0, 0, 0, 0, 0];
  if (t.includes("no-store")) return [0, 0, 0, 0, 0, 1, 0, 0];
  if (t.includes("refresh")) return [0, 0, 0, 0, 0, 0, 1, 0];
  if (t.includes("purge")) return [0, 0, 0, 0, 0, 0, 0, 1];
  return [0, 0, 0, 1, 0, 0, 0, 0];
}

interface EmbeddingMock {
  baseUrl: string;
  callCount(): number;
  close(): Promise<void>;
}

async function startEmbeddingMock(
  opts: { fail?: boolean } = {},
): Promise<EmbeddingMock> {
  let calls = 0;
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      if (!req.url?.includes("/embeddings")) {
        res.statusCode = 404;
        res.end("{}");
        return;
      }
      calls++;
      if (opts.fail) {
        res.statusCode = 500;
        res.setHeader("content-type", "application/json");
        res.end(
          JSON.stringify({ error: { message: "embedding upstream down" } }),
        );
        return;
      }
      const body = JSON.parse(raw || "{}") as { input?: string | string[] };
      const inputs = Array.isArray(body.input)
        ? body.input
        : [body.input ?? ""];
      const data = inputs.map((text, index) => ({
        object: "embedding",
        index,
        embedding: keywordVector(text),
      }));
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          object: "list",
          model: "embed-mock",
          data,
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
    callCount: () => calls,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

function chatUpstreamReplying(content: string): Promise<OpenAiUpstream> {
  return startOpenAiUpstream({
    nonStreamBody: {
      id: `cmpl-${content}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "gpt-4o-mini",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 },
    },
  });
}

interface ChatResult {
  status: number;
  content: string | undefined;
  cache: string | null;
  layer: string | null;
  similarity: number | null;
}

describe("semantic cache e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];
  const embedMocks: EmbeddingMock[] = [];
  let embed: EmbeddingMock;
  let upstreamA: OpenAiUpstream;
  let sharedPolicyId: string | undefined;
  let sharedPolicyBody: Record<string, unknown>;

  async function createDirectModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed not ready");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  }

  async function createEmbeddingModel(
    displayName: string,
    mock: EmbeddingMock,
  ): Promise<void> {
    if (!seed) throw new Error("seed not ready");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${mock.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "embed-mock",
      provider_key_id: pk.id,
      embedding: { dimensions: 8, normalize: true },
    });
  }

  async function chat(
    model: string,
    prompt: string,
    opts: {
      bearer?: string;
      cacheControl?: string;
      temperature?: number;
      contentBlocks?: unknown[];
    } = {},
  ): Promise<ChatResult> {
    const headers: Record<string, string> = {
      "content-type": "application/json",
      authorization: `Bearer ${opts.bearer ?? CALLER_PLAINTEXT}`,
    };
    if (opts.cacheControl) headers["cache-control"] = opts.cacheControl;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "user",
            content: opts.contentBlocks ?? prompt,
          },
        ],
        ...(opts.temperature !== undefined
          ? { temperature: opts.temperature }
          : {}),
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
    await seed.createApiKey({
      key_hash: OTHER_CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    embed = await startEmbeddingMock();
    embedMocks.push(embed);
    await createEmbeddingModel("embed-cache", embed);

    upstreamA = await chatUpstreamReplying("answer-a");
    upstreams.push(upstreamA);
    await createDirectModel("chat-a", upstreamA);
    const created = await seed.createCachePolicy({
      name: "sem-default",
      backend: "memory",
      applies_to: "model:chat-a",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-cache", threshold: 0.85 },
    });
    sharedPolicyId = created.id;
    sharedPolicyBody = created.value;

    // Gate open = the policy propagated (header present on a covered
    // request).
    await waitConfigPropagation(async () => {
      try {
        const r = await chat("chat-a", "warmup probe please");
        return r.status === 200 && r.cache !== null;
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(embedMocks.map((m) => m.close()));
  });

  test("paraphrase above threshold hits semantically; same wording then hits exactly", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const first = await chat("chat-a", "tell me about topic-a");
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");
    expect(first.content).toBe("answer-a");

    const upstreamCallsAfterMiss = upstreamA.receivedRequests.length;

    // Different wording, same meaning (same mock vector): L1 misses,
    // L2 serves the stored answer without an upstream call.
    const paraphrase = await chat("chat-a", "please explain topic-a to me");
    expect(paraphrase.status).toBe(200);
    expect(paraphrase.cache).toBe("hit");
    expect(paraphrase.layer).toBe("semantic");
    expect(paraphrase.content).toBe("answer-a");
    expect(paraphrase.similarity).not.toBeNull();
    expect(paraphrase.similarity!).toBeGreaterThanOrEqual(0.85);
    expect(paraphrase.similarity!).toBeLessThanOrEqual(1.0);
    expect(upstreamA.receivedRequests.length).toBe(upstreamCallsAfterMiss);

    // The semantic hit backfilled the exact layer: the SAME paraphrase
    // again is now an exact hit (no embedding call either).
    const embedCallsBefore = embed.callCount();
    const repeat = await chat("chat-a", "please explain topic-a to me");
    expect(repeat.cache).toBe("hit");
    expect(repeat.layer).toBe("exact");
    expect(repeat.content).toBe("answer-a");
    expect(embed.callCount()).toBe(embedCallsBefore);
    expect(upstreamA.receivedRequests.length).toBe(upstreamCallsAfterMiss);
  });

  test("unrelated prompt misses and goes upstream", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const before = upstreamA.receivedRequests.length;
    const r = await chat("chat-a", "tell me about topic-b");
    expect(r.status).toBe(200);
    expect(r.cache).toBe("miss");
    expect(upstreamA.receivedRequests.length).toBe(before + 1);
  });

  test("same meaning but different sampling params never matches", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // "topic-a" entries exist from the first test at default params; the
    // same meaning with temperature set must not be served from them —
    // the stored answer was generated under different parameters.
    const r = await chat("chat-a", "tell me about topic-a", {
      temperature: 0.9,
    });
    expect(r.status).toBe(200);
    expect(r.cache).toBe("miss");
  });

  test("below-threshold similarity misses", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const strictUpstream = await chatUpstreamReplying("answer-strict");
    upstreams.push(strictUpstream);
    await createDirectModel("chat-strict", strictUpstream);
    await seed.createCachePolicy({
      name: "sem-strict",
      backend: "memory",
      applies_to: "model:chat-strict",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-cache", threshold: 0.95 },
    });
    await waitConfigPropagation(async () => {
      try {
        const r = await chat("chat-strict", "warmup probe please");
        return r.status === 200 && r.cache !== null;
      } catch {
        return false;
      }
    });

    const seeded = await chat("chat-strict", "tell me about topic-a");
    expect(seeded.cache).toBe("miss");
    // cos(topic-a-near, topic-a) = 0.9 < 0.95 -> upstream again.
    const near = await chat("chat-strict", "tell me about topic-a-near");
    expect(near.status).toBe(200);
    expect(near.cache).toBe("miss");

    // Sibling check on the looser policy (0.85): the same pair DOES
    // match semantically, pinning that 0.9 sits between the two
    // thresholds rather than being flattened to 1.0 or 0.0.
    const seededLoose = await chat("chat-a", "note down topic-c");
    expect(seededLoose.cache).toBe("miss");
    const nearLoose = await chat("chat-a", "note down topic-c-near");
    expect(nearLoose.cache).toBe("hit");
    expect(nearLoose.layer).toBe("semantic");
    expect(nearLoose.similarity!).toBeGreaterThanOrEqual(0.85);
    expect(nearLoose.similarity!).toBeLessThan(0.95);
  });

  test("scope defaults to api_key: another caller never sees the entry; scope env shares it", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Default scope (api_key): the other caller's identical + similar
    // requests both miss.
    const otherExact = await chat("chat-a", "tell me about topic-a", {
      bearer: OTHER_CALLER_PLAINTEXT,
    });
    expect(otherExact.status).toBe(200);
    expect(otherExact.cache).toBe("miss");

    // scope: env policy on a separate model: caller 1 stores, caller 2
    // hits — both exactly and semantically.
    const sharedUpstream = await chatUpstreamReplying("answer-shared");
    upstreams.push(sharedUpstream);
    await createDirectModel("chat-shared", sharedUpstream);
    await seed.createCachePolicy({
      name: "sem-shared",
      backend: "memory",
      applies_to: "model:chat-shared",
      ttl_seconds: 600,
      scope: "env",
      semantic: { embedding_model: "embed-cache", threshold: 0.85 },
    });
    await waitConfigPropagation(async () => {
      try {
        const r = await chat("chat-shared", "warmup probe please");
        return r.status === 200 && r.cache !== null;
      } catch {
        return false;
      }
    });

    const store = await chat("chat-shared", "tell me about topic-a");
    expect(store.cache).toBe("miss");
    const crossExact = await chat("chat-shared", "tell me about topic-a", {
      bearer: OTHER_CALLER_PLAINTEXT,
    });
    expect(crossExact.cache).toBe("hit");
    expect(crossExact.layer).toBe("exact");
    expect(crossExact.content).toBe("answer-shared");
    const crossSemantic = await chat(
      "chat-shared",
      "explain topic-a in short",
      { bearer: OTHER_CALLER_PLAINTEXT },
    );
    expect(crossSemantic.cache).toBe("hit");
    expect(crossSemantic.layer).toBe("semantic");
    expect(crossSemantic.content).toBe("answer-shared");
  });

  test("requests with non-text content never match semantically", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const blocks = (url: string) => [
      { type: "text", text: "what is in this picture of topic-a" },
      { type: "image_url", image_url: { url } },
    ];
    const first = await chat("chat-a", "", {
      contentBlocks: blocks("https://example.com/cat.jpg"),
    });
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");

    // Same question about a DIFFERENT image: same text, so a text
    // embedding could not tell them apart — must go upstream, not match.
    const otherImage = await chat("chat-a", "", {
      contentBlocks: blocks("https://example.com/dog.jpg"),
    });
    expect(otherImage.cache).toBe("miss");

    // Identical multimodal request still hits the exact layer.
    const exactRepeat = await chat("chat-a", "", {
      contentBlocks: blocks("https://example.com/cat.jpg"),
    });
    expect(exactRepeat.cache).toBe("hit");
    expect(exactRepeat.layer).toBe("exact");
  });

  test("Cache-Control: no-store keeps the response out of both layers", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const first = await chat("chat-a", "no-store probe unique", {
      cacheControl: "no-store",
    });
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");
    // Nothing was stored: the identical request misses again.
    const repeat = await chat("chat-a", "no-store probe unique");
    expect(repeat.cache).toBe("miss");
  });

  test("Cache-Control: no-cache bypasses the read path and refreshes the entry", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const seeded = await chat("chat-a", "refresh probe unique");
    expect(seeded.cache).toBe("miss");
    const cachedNow = await chat("chat-a", "refresh probe unique");
    expect(cachedNow.cache).toBe("hit");

    const before = upstreamA.receivedRequests.length;
    const bypass = await chat("chat-a", "refresh probe unique", {
      cacheControl: "no-cache",
    });
    expect(bypass.status).toBe(200);
    expect(bypass.cache).toBe("bypass");
    expect(bypass.layer).toBeNull();
    expect(upstreamA.receivedRequests.length).toBe(before + 1);

    // The bypass refreshed (re-stored) the entry — still served after.
    const after = await chat("chat-a", "refresh probe unique");
    expect(after.cache).toBe("hit");
  });

  test("embedding failure degrades to exact-only, never fails the request", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const brokenEmbed = await startEmbeddingMock({ fail: true });
    embedMocks.push(brokenEmbed);
    await createEmbeddingModel("embed-broken", brokenEmbed);
    const upstream = await chatUpstreamReplying("answer-degraded");
    upstreams.push(upstream);
    await createDirectModel("chat-broken", upstream);
    await seed.createCachePolicy({
      name: "sem-broken",
      backend: "memory",
      applies_to: "model:chat-broken",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-broken", threshold: 0.85 },
    });
    await waitConfigPropagation(async () => {
      try {
        const r = await chat("chat-broken", "warmup probe please");
        return r.status === 200 && r.cache !== null;
      } catch {
        return false;
      }
    });

    const first = await chat("chat-broken", "tell me about topic-a");
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");
    expect(first.content).toBe("answer-degraded");
    // Similar wording cannot match (no embeddings) -> upstream.
    const similar = await chat("chat-broken", "explain topic-a briefly");
    expect(similar.status).toBe(200);
    expect(similar.cache).toBe("miss");
    // The exact layer still works.
    const exact = await chat("chat-broken", "tell me about topic-a");
    expect(exact.cache).toBe("hit");
    expect(exact.layer).toBe("exact");
  });

  test("purge_generation bump invalidates both layers at once", async (ctx) => {
    if (!etcdReachable || !app || !seed || !sharedPolicyId) {
      ctx.skip();
      return;
    }
    const seeded = await chat("chat-a", "purge probe unique");
    expect(seeded.cache).toBe("miss");
    expect(
      (await chat("chat-a", "purge probe unique")).cache,
    ).toBe("hit");

    await seed.update("cache_policies", sharedPolicyId, {
      ...sharedPolicyBody,
      purge_generation: 1,
    });
    // Propagation probe on an UNRELATED axis (the keyword-free warmup
    // text): its pre-purge entry stops being served once the new
    // generation lands. Probing with the purge-axis text would re-store
    // a fresh same-axis entry and hand the paraphrase below a
    // legitimate new-generation hit, masking what this test pins.
    await waitConfigPropagation(async () => {
      try {
        const r = await chat("chat-a", "warmup probe please");
        return r.status === 200 && r.cache === "miss";
      } catch {
        return false;
      }
    });

    // Both layers are gone: a paraphrase of the pre-purge entry misses
    // (nothing on its axis was re-stored since the purge)…
    const paraphrase = await chat("chat-a", "purge probe reworded");
    expect(paraphrase.cache).toBe("miss");
    // …and the cache works normally under the new generation.
    const rewarm = await chat("chat-a", "purge probe reworded");
    expect(rewarm.cache).toBe("hit");
  });
});
