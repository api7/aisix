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

// E2E: semantic cache (AISIX-Cloud#558). A CachePolicy carrying a
// `semantic` block serves an exact-fingerprint (L1) hit first and, on an
// L1 miss, embeds the request and serves the nearest stored entry at or
// above the policy's cosine threshold (L2). Real `aisix` binary + etcd +
// mock chat upstreams + a deterministic mock embedding endpoint; no CP.
//
// The embedding mock maps input text to fixed 12-dim vectors by keyword,
// so similarity is fully deterministic. Each test owns an orthogonal
// axis (entries persist across tests within a policy+scope partition,
// so unrelated tests must never share a vector):
//   "topic-a-near" -> 0.9*topic-a + spill   (cos 0.9 vs topic-a)
//   "topic-c-near" -> 0.9*topic-c + spill   (same pair on another axis)
//   "topic-a" / "topic-b" / "topic-c" / "no-store" / "refresh" /
//   "purge" / "params-x" / "scope-x" / default -> one distinct axis each
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

const DIMS = 12;

function axis(i: number, weight = 1): number[] {
  const v = new Array<number>(DIMS).fill(0);
  v[i] = weight;
  return v;
}

function keywordVector(text: string): number[] {
  const t = text.toLowerCase();
  const spill = Math.sqrt(0.19); // makes the *-near vectors unit-length
  // Longest keyword first: "topic-a-near" contains "topic-a".
  if (t.includes("topic-a-near")) {
    const v = axis(0, 0.9);
    v[4] = spill;
    return v;
  }
  if (t.includes("topic-a")) return axis(0);
  if (t.includes("topic-b")) return axis(1);
  if (t.includes("topic-c-near")) {
    const v = axis(2, 0.9);
    v[4] = spill;
    return v;
  }
  if (t.includes("topic-c")) return axis(2);
  if (t.includes("no-store")) return axis(5);
  if (t.includes("refresh")) return axis(6);
  if (t.includes("purge")) return axis(7);
  if (t.includes("params-x")) return axis(8);
  if (t.includes("scope-x")) return axis(9);
  return axis(3);
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

interface CountingUpstream {
  baseUrl: string;
  callCount(): number;
  close(): Promise<void>;
}

/**
 * A chat upstream whose reply body CHANGES on every call
 * (`reply-1`, `reply-2`, …). Refresh contracts need this: with a
 * fixed-body mock, "the entry was re-stored" and "the old entry
 * survived" are indistinguishable.
 */
async function countingChatUpstream(): Promise<CountingUpstream> {
  let calls = 0;
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    req.on("data", () => {});
    req.on("end", () => {
      calls++;
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: `cmpl-${calls}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: "gpt-4o-mini",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: `reply-${calls}` },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 },
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
  const countingUpstreams: CountingUpstream[] = [];
  const embedMocks: EmbeddingMock[] = [];
  let embed: EmbeddingMock;
  let upstreamA: OpenAiUpstream;
  let canarySeq = 0;

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
      embedding: { dimensions: 12, normalize: true },
    });
  }

  // Convention (tests/e2e/AGENTS.md): resources land as etcd watch
  // events in revision order, so a throwaway key seeded AFTER a batch
  // authenticating proves the whole batch is in the snapshot. The gate
  // is the non-throwing ProxyClient.listModels — never the cache
  // behavior under test — and a non-200 is the normal transient state.
  async function sealSeedBatch(): Promise<void> {
    if (!seed || !app) throw new Error("seed not ready");
    const plaintext = `sk-semcache-canary-${++canarySeq}`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(plaintext).digest("hex"),
      allowed_models: ["*"],
    });
    const probe = new ProxyClient(app.proxyUrl, plaintext);
    await waitConfigPropagation(
      async () => (await probe.listModels()).status === 200,
    );
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

    embed = await startEmbeddingMock();
    embedMocks.push(embed);
    await createEmbeddingModel("embed-cache", embed);

    upstreamA = await chatUpstreamReplying("answer-a");
    upstreams.push(upstreamA);
    await createDirectModel("chat-a", upstreamA);
    await seed.createCachePolicy({
      name: "sem-default",
      backend: "memory",
      applies_to: "model:chat-a",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-cache", threshold: 0.85 },
    });

    // Caller keys are seeded LAST: once one authenticates, revision
    // order implies every resource above is in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    await seed.createApiKey({
      key_hash: OTHER_CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    const probe = new ProxyClient(app.proxyUrl, OTHER_CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await probe.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(countingUpstreams.map((u) => u.close()));
    await Promise.all(embedMocks.map((m) => m.close()));
  });

  test("paraphrase above threshold hits semantically; same wording then hits exactly", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const embedCallsBeforeMiss = embed.callCount();
    const first = await chat("chat-a", "tell me about topic-a");
    expect(first.status).toBe(200);
    expect(first.cache).toBe("miss");
    expect(first.content).toBe("answer-a");
    // One embedding call covers BOTH the L2 lookup and the store —
    // a miss must never pay twice.
    expect(embed.callCount()).toBe(embedCallsBeforeMiss + 1);

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
    // Self-seeded on a dedicated axis: store at default params, then
    // the same meaning with temperature set must not be served from it
    // — the stored answer was generated under different parameters.
    const seeded = await chat("chat-a", "params-x probe question");
    expect(seeded.cache).toBe("miss");
    const r = await chat("chat-a", "params-x probe question", {
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
    await sealSeedBatch();

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
    // Default scope (api_key), self-seeded on a dedicated axis:
    // caller 1 stores, then the other caller's identical request must
    // miss BOTH layers (scoped exact key + scoped semantic partition).
    const mine = await chat("chat-a", "scope-x probe question");
    expect(mine.cache).toBe("miss");
    const otherExact = await chat("chat-a", "scope-x probe question", {
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
    await sealSeedBatch();

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

  test("Cache-Control: no-cache bypasses the read path and refreshes BOTH layers", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Dedicated model on a counting upstream: every upstream call
    // returns a DIFFERENT body, so "refreshed" vs "stale entry
    // survived" is distinguishable in the served content.
    const counting = await countingChatUpstream();
    countingUpstreams.push(counting);
    const pk = await seed.createProviderKey({
      display_name: "chat-refresh-pk",
      secret: "sk-mock",
      api_base: `${counting.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "chat-refresh",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createCachePolicy({
      name: "sem-refresh",
      backend: "memory",
      applies_to: "model:chat-refresh",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-cache", threshold: 0.85 },
    });
    await sealSeedBatch();

    const seeded = await chat("chat-refresh", "refresh probe unique");
    expect(seeded.cache).toBe("miss");
    const originalBody = seeded.content!;
    const cachedNow = await chat("chat-refresh", "refresh probe unique");
    expect(cachedNow.cache).toBe("hit");
    expect(cachedNow.content).toBe(originalBody);

    const upstreamBefore = counting.callCount();
    const bypass = await chat("chat-refresh", "refresh probe unique", {
      cacheControl: "no-cache",
    });
    expect(bypass.status).toBe(200);
    expect(bypass.cache).toBe("bypass");
    expect(bypass.layer).toBeNull();
    expect(counting.callCount()).toBe(upstreamBefore + 1);
    const refreshedBody = bypass.content!;
    expect(refreshedBody).not.toBe(originalBody);

    // The bypass refreshed the EXACT layer: the identical request now
    // serves the NEW body, not the pre-bypass one.
    const exactAfter = await chat("chat-refresh", "refresh probe unique");
    expect(exactAfter.cache).toBe("hit");
    expect(exactAfter.layer).toBe("exact");
    expect(exactAfter.content).toBe(refreshedBody);

    // …and the SEMANTIC layer: the bypass path re-embedded and
    // upserted, so a paraphrase serves the new body too (upsert —
    // the stale entry was replaced, not shadowed).
    const semanticAfter = await chat("chat-refresh", "refresh probe reworded");
    expect(semanticAfter.cache).toBe("hit");
    expect(semanticAfter.layer).toBe("semantic");
    expect(semanticAfter.content).toBe(refreshedBody);
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
    await sealSeedBatch();

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

  test("swapping the embedding model orphans old entries even at identical vectors", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Second embedding model on the SAME mock: identical vectors, new
    // identity. If the candidate partition keyed only on vectors, the
    // swap would keep serving old entries; the contract is that vectors
    // from a different embedding model are never compared.
    await createEmbeddingModel("embed-cache-b", embed);
    const upstream = await chatUpstreamReplying("answer-swap");
    upstreams.push(upstream);
    await createDirectModel("chat-swap", upstream);
    const policy = await seed.createCachePolicy({
      name: "sem-swap",
      backend: "memory",
      applies_to: "model:chat-swap",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-cache", threshold: 0.85 },
    });
    await sealSeedBatch();

    const seeded = await chat("chat-swap", "tell me about topic-a");
    expect(seeded.cache).toBe("miss");
    const baseline = await chat("chat-swap", "explain topic-a please");
    expect(baseline.cache).toBe("hit");
    expect(baseline.layer).toBe("semantic");

    await seed.update("cache_policies", policy.id, {
      ...policy.value,
      semantic: { embedding_model: "embed-cache-b", threshold: 0.85 },
    });
    await sealSeedBatch();

    // A FRESH same-meaning wording (L1-cold) misses once the swap has
    // landed — the old partition is orphaned even though both models
    // produce identical vectors.
    const fresh = await chat("chat-swap", "describe topic-a now");
    expect(fresh.status).toBe(200);
    expect(fresh.cache).toBe("miss");
    // …and the new partition warms normally.
    const rewarmed = await chat("chat-swap", "topic-a summary please");
    expect(rewarmed.cache).toBe("hit");
    expect(rewarmed.layer).toBe("semantic");
    expect(rewarmed.content).toBe("answer-swap");
  });

  test("purge_generation bump invalidates both layers at once", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Dedicated model + policy: purging must not disturb (or depend
    // on) any other test's partition.
    const upstream = await chatUpstreamReplying("answer-purge");
    upstreams.push(upstream);
    await createDirectModel("chat-purge", upstream);
    const policy = await seed.createCachePolicy({
      name: "sem-purge",
      backend: "memory",
      applies_to: "model:chat-purge",
      ttl_seconds: 600,
      semantic: { embedding_model: "embed-cache", threshold: 0.85 },
    });
    await sealSeedBatch();

    const seeded = await chat("chat-purge", "purge probe unique");
    expect(seeded.cache).toBe("miss");
    expect((await chat("chat-purge", "purge probe unique")).cache).toBe("hit");

    await seed.update("cache_policies", policy.id, {
      ...policy.value,
      purge_generation: 1,
    });
    await sealSeedBatch();

    // Both layers are gone: a paraphrase of the pre-purge entry misses
    // (nothing on its axis exists in the new generation)…
    const paraphrase = await chat("chat-purge", "purge probe reworded");
    expect(paraphrase.cache).toBe("miss");
    // …the pre-purge WORDING is no longer exact-served either: it now
    // matches the paraphrase's fresh entry SEMANTICALLY, proving the
    // old exact entry (which would win as `layer: exact`) is gone…
    const oldWording = await chat("chat-purge", "purge probe unique");
    expect(oldWording.cache).toBe("hit");
    expect(oldWording.layer).toBe("semantic");
    // …and the cache works normally under the new generation.
    const rewarm = await chat("chat-purge", "purge probe reworded");
    expect(rewarm.cache).toBe("hit");
    expect(rewarm.layer).toBe("exact");
  });
});
