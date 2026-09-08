import { createHash } from "node:crypto";
import OpenAI from "openai";
import { WebSocketServer, type WebSocket as WsSocket } from "ws";
import { WebSocket } from "undici";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: shared pricing documents (AISIX-Cloud#1546) against a real `aisix`
// binary, a real etcd and mock upstreams.
//
// A model can take its per-token price from a `pricing` document instead
// of carrying `cost` inline. The gateway watches two prefixes for these:
// its own environment's, and the shared `<base>/global/` catalog. What
// the cases below pin is the resolution order and its consequences —
// environment over global over inline, a price edit taking effect with no
// model document rewritten, the catalog carrying nothing but prices, and
// the same chain pricing the usage events.

const CALLER_PLAINTEXT = "sk-global-pricing-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

function okBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("global pricing e2e: pricing documents drive least_cost", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let global: SeedClient | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    // The shared catalog: `<base>/global/`, the one collection the
    // gateway reads from outside its own environment prefix.
    global = new SeedClient(etcd, `${app.etcdPrefix}/global`);
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
  ): Promise<{ id: string }> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    return seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
      ...extra,
    });
  }

  function client(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  /** Upstream-neutral readiness: `/v1/models` lists the named rows. */
  async function waitForModels(...names: string[]): Promise<void> {
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const ids =
        ((await res.json()) as { data?: Array<{ id?: string }> }).data?.map((m) => m.id) ?? [];
      return names.every((n) => ids.includes(n));
    });
  }

  /**
   * Wait until the two pricing tables together hold `total` rows.
   *
   * `waitForModels` cannot stand in for this: the catalog is a separate
   * prefix on its own watch, so a model can be listed while the price
   * that ranks it has not landed. Counting rows is the only signal the
   * gateway exposes for the catalog, and it is upstream-neutral.
   */
  async function waitForPricingRows(total: number): Promise<void> {
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.metricsUrl}/status/config`);
      if (!res.ok) return false;
      const counts =
        ((await res.json()) as { applied?: { resource_counts?: Record<string, number> } }).applied
          ?.resource_counts ?? {};
      return (counts.pricing ?? 0) + (counts.global_pricing ?? 0) >= total;
    });
  }

  /** Which upstream served, by the content the mock answers with. */
  async function ask(model: string, prompt: string): Promise<string | null | undefined> {
    const completion = await client().chat.completions.create({
      model,
      messages: [{ role: "user", content: prompt }],
    });
    return completion.choices[0]?.message.content;
  }

  test("a global pricing document orders least_cost between two targets", async (ctx) => {
    if (!etcdReachable || !app || !seed || !global) {
      ctx.skip();
      return;
    }

    const cheap = await startOpenAiUpstream({ nonStreamBody: okBody("g-cheap-served") });
    const pricey = await startOpenAiUpstream({ nonStreamBody: okBody("g-pricey-served") });
    upstreams.push(cheap, pricey);

    // The catalog says cheap < pricey. The inline `cost` on each model
    // says the OPPOSITE, so a resolver that consulted `cost` first would
    // serve the other upstream and fail this case — without the
    // conflicting values, omitting `cost` entirely would let such a
    // resolver pass.
    await global.createPricing({ key: "vendor/g-cheap", input_per_1k: 0.1, output_per_1k: 0.1 });
    await global.createPricing({ key: "vendor/g-pricey", input_per_1k: 10, output_per_1k: 10 });

    // Expensive target declared FIRST: the price has to reorder it.
    await seed.createModel({
      display_name: "g-virtual",
      routing: {
        strategy: "least_cost",
        targets: [{ model: "g-pricey" }, { model: "g-cheap" }],
      },
    });
    await createOpenAiModel("g-cheap", cheap, {
      pricing_key: "vendor/g-cheap",
      cost: { input_per_1k: 50, output_per_1k: 50 },
    });
    await createOpenAiModel("g-pricey", pricey, {
      pricing_key: "vendor/g-pricey",
      cost: { input_per_1k: 0.01, output_per_1k: 0.01 },
    });
    await waitForModels("g-cheap", "g-pricey");
    await waitForPricingRows(2);

    const cheapBaseline = cheap.receivedRequests.length;
    const priceyBaseline = pricey.receivedRequests.length;

    expect(await ask("g-virtual", "cheapest please")).toBe("g-cheap-served");
    expect(cheap.receivedRequests.length - cheapBaseline).toBe(1);
    expect(pricey.receivedRequests.length - priceyBaseline).toBe(0);
  });

  test("an environment pricing document wins over the global one with the same key", async (ctx) => {
    if (!etcdReachable || !app || !seed || !global) {
      ctx.skip();
      return;
    }

    const a = await startOpenAiUpstream({ nonStreamBody: okBody("env-a-served") });
    const b = await startOpenAiUpstream({ nonStreamBody: okBody("env-b-served") });
    upstreams.push(a, b);

    // Globally, A is the cheap one. The environment overrides A's price
    // upward, which must flip the order — so the assertion below is the
    // OPPOSITE of what the catalog alone would produce.
    await global.createPricing({ key: "vendor/env-a", input_per_1k: 0.1, output_per_1k: 0.1 });
    await global.createPricing({ key: "vendor/env-b", input_per_1k: 1, output_per_1k: 1 });
    await seed.createPricing({ key: "vendor/env-a", input_per_1k: 50, output_per_1k: 50 });

    await seed.createModel({
      display_name: "env-virtual",
      routing: {
        strategy: "least_cost",
        targets: [{ model: "env-a" }, { model: "env-b" }],
      },
    });
    await createOpenAiModel("env-a", a, { pricing_key: "vendor/env-a" });
    await createOpenAiModel("env-b", b, { pricing_key: "vendor/env-b" });
    await waitForModels("env-a", "env-b");
    // Three more rows than the previous case left behind.
    await waitForPricingRows(5);

    const aBaseline = a.receivedRequests.length;
    const bBaseline = b.receivedRequests.length;

    expect(await ask("env-virtual", "override applies")).toBe("env-b-served");
    expect(b.receivedRequests.length - bBaseline).toBe(1);
    expect(a.receivedRequests.length - aBaseline).toBe(0);
  });

  test("with no pricing document the model's inline cost still ranks it", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const cheap = await startOpenAiUpstream({ nonStreamBody: okBody("inline-cheap-served") });
    const pricey = await startOpenAiUpstream({ nonStreamBody: okBody("inline-pricey-served") });
    upstreams.push(cheap, pricey);

    // `pricing_key` names a document neither prefix carries, so the chain
    // has to fall all the way through to `cost`.
    await seed.createModel({
      display_name: "inline-virtual",
      routing: {
        strategy: "least_cost",
        targets: [{ model: "inline-pricey" }, { model: "inline-cheap" }],
      },
    });
    await createOpenAiModel("inline-cheap", cheap, {
      pricing_key: "vendor/never-written",
      cost: { input_per_1k: 0.1, output_per_1k: 0.1 },
    });
    await createOpenAiModel("inline-pricey", pricey, {
      pricing_key: "vendor/never-written-either",
      cost: { input_per_1k: 10, output_per_1k: 10 },
    });
    await waitForModels("inline-cheap", "inline-pricey");

    const cheapBaseline = cheap.receivedRequests.length;
    const priceyBaseline = pricey.receivedRequests.length;

    expect(await ask("inline-virtual", "inline fallback")).toBe("inline-cheap-served");
    expect(cheap.receivedRequests.length - cheapBaseline).toBe(1);
    expect(pricey.receivedRequests.length - priceyBaseline).toBe(0);
  });

  test("editing the global price flips the order without rewriting a model", async (ctx) => {
    if (!etcdReachable || !app || !seed || !global || !etcd) {
      ctx.skip();
      return;
    }

    const a = await startOpenAiUpstream({ nonStreamBody: okBody("flip-a-served") });
    const b = await startOpenAiUpstream({ nonStreamBody: okBody("flip-b-served") });
    upstreams.push(a, b);

    const priceA = await global.createPricing({
      key: "vendor/flip-a",
      input_per_1k: 0.1,
      output_per_1k: 0.1,
    });
    await global.createPricing({ key: "vendor/flip-b", input_per_1k: 1, output_per_1k: 1 });

    await seed.createModel({
      display_name: "flip-virtual",
      routing: {
        strategy: "least_cost",
        targets: [{ model: "flip-a" }, { model: "flip-b" }],
      },
    });
    const modelA = await createOpenAiModel("flip-a", a, { pricing_key: "vendor/flip-a" });
    await createOpenAiModel("flip-b", b, { pricing_key: "vendor/flip-b" });
    await waitForModels("flip-a", "flip-b");
    await waitForPricingRows(7);

    const before = await seed.raw("models", modelA.id);
    expect(await ask("flip-virtual", "a is cheaper")).toBe("flip-a-served");

    // One write, to the pricing document alone.
    await etcd!.put(
      `${app.etcdPrefix}/global/pricing/${priceA.id}`,
      JSON.stringify({ key: "vendor/flip-a", input_per_1k: 100, output_per_1k: 100 }),
    );

    // The order flips on a later request; poll because the write reaches
    // the snapshot through the watch stream.
    await waitConfigPropagation(async () => (await ask("flip-virtual", "b now")) === "flip-b-served");

    // The model document is byte-identical: nothing repriced it but the
    // pricing row, which is the point of the reference.
    expect(await seed.raw("models", modelA.id)).toBe(before);
  });

  test("the global prefix carries prices only — another kind there is never served", async (ctx) => {
    if (!etcdReachable || !app || !seed || !global) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({ nonStreamBody: okBody("legit-served") });
    upstreams.push(upstream);

    // Model rows already in the snapshot from the cases above — this
    // describe shares one gateway, so the assertion below has to be a
    // delta rather than an absolute count.
    async function modelCount(): Promise<number> {
      const res = await fetch(`${app!.metricsUrl}/status/config`);
      const body = (await res.json()) as {
        applied?: { resource_counts?: Record<string, number> };
      };
      return body.applied?.resource_counts?.models ?? 0;
    }
    const modelsBefore = await modelCount();

    // A price alongside the smuggled row, so this case proves BOTH
    // halves. Without it, "the model is not served" is equally true of a
    // gateway that never reads the global prefix at all, and the case
    // survives deleting the whole feature.
    await global.createPricing({
      key: "vendor/allowlist-probe",
      input_per_1k: 1,
      output_per_1k: 1,
    });

    // A well-formed model document, written under the global prefix. If
    // the gateway loaded kinds other than `pricing` from there, this
    // model would be servable — by a writer outside the environment.
    await global.createModel({
      display_name: "smuggled-global-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: "11111111-1111-1111-1111-111111111111",
    });
    await createOpenAiModel("legit-env-model", upstream);
    // Gating on the environment model written AFTER the smuggled one:
    // watch events apply in revision order, so once this is visible the
    // global write has been processed too.
    await waitForModels("legit-env-model");

    // The prefix IS being read: its pricing document reached the
    // snapshot, under its own table.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.metricsUrl}/status/config`);
      if (!res.ok) return false;
      const body = (await res.json()) as {
        applied?: { resource_counts?: Record<string, number> };
      };
      return (body.applied?.resource_counts?.global_pricing ?? 0) >= 1;
    });

    // …and the model written beside it did not: exactly one model row
    // was added, the environment one.
    expect((await modelCount()) - modelsBefore).toBe(1);

    const res = await fetch(`${app.proxyUrl}/v1/models`, {
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
    });
    const ids =
      ((await res.json()) as { data?: Array<{ id?: string }> }).data?.map((m) => m.id) ?? [];
    expect(ids).not.toContain("smuggled-global-model");

    const refused = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "smuggled-global-model",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(refused.status).toBeGreaterThanOrEqual(400);
  });
});

/** Mock OpenAI Realtime upstream answering one usage-bearing frame. */
async function startRealtimeUpstream(input: number, output: number) {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.on("message", () => {
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: {
            usage: {
              input_tokens: input,
              output_tokens: output,
              input_token_details: { cached_tokens: 0 },
            },
          },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    close: () =>
      new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("global pricing e2e: usage events price through the same chain", () => {
  // The realtime session is one of the two paths that compute `cost_usd`
  // on the data plane rather than leaving it to the control plane, so it
  // is where a divergence between "what least_cost ranks by" and "what
  // the usage event bills" would show up. Nine input and four output
  // tokens against 1.0/2.0 per 1K is 0.017 USD — 17000 micro-USD.
  const INPUT_TOKENS = 9;
  const OUTPUT_TOKENS = 4;
  const CALLER = "sk-global-pricing-realtime-caller";
  const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let global: SeedClient | undefined;
  let upstream: Awaited<ReturnType<typeof startRealtimeUpstream>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startRealtimeUpstream(INPUT_TOKENS, OUTPUT_TOKENS);
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    global = new SeedClient(etcd, `${app.etcdPrefix}/global`);

    await global.createPricing({ key: "vendor/rt", input_per_1k: 1.0, output_per_1k: 2.0 });
    const pk = await seed.createProviderKey({
      display_name: "rt-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${upstream.port}/v1`,
    });
    // Priced by the catalog. The inline `cost` is deliberately absent, so
    // a spend counter that moves at all proves the reference resolved.
    await seed.createModel({
      display_name: "rt-catalog",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: pk.id,
      pricing_key: "vendor/rt",
    });
    // Same session shape, priced inline instead — the fallback half.
    await seed.createModel({
      display_name: "rt-inline",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: pk.id,
      pricing_key: "vendor/not-written",
      cost: { input_per_1k: 1.0, output_per_1k: 2.0 },
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  /** One realtime session against `model`; resolves when it closes. */
  async function runSession(model: string): Promise<void> {
    const ws = new WebSocket(`${app!.proxyUrl.replace("http", "ws")}/v1/realtime?model=${model}`, [
      "realtime",
      `openai-insecure-api-key.${CALLER}`,
    ]);
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("realtime session timed out")), 15000);
      ws.addEventListener("open", () => ws.send(JSON.stringify({ type: "response.create" })));
      ws.addEventListener("message", () => {
        clearTimeout(timer);
        ws.close();
        resolve();
      });
      ws.addEventListener("error", (e) => {
        clearTimeout(timer);
        reject(new Error(`realtime session errored: ${String(e)}`));
      });
    });
  }

  async function spendFor(model: string): Promise<number> {
    return sumMetric(
      await scrapeMetrics(app!.metricsUrl),
      "aisix_llm_spend_micro_usd_total",
      (labels) => labels.model === model,
    );
  }

  test("a realtime session bills at the pricing document's rate", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      if (res.status !== 200) return false;
      const ids =
        ((await res.json()) as { data?: Array<{ id?: string }> }).data?.map((m) => m.id) ?? [];
      return ids.includes("rt-catalog") && ids.includes("rt-inline");
    });

    const before = await spendFor("rt-catalog");
    await runSession("rt-catalog");
    // The usage event is emitted as the session tears down.
    await waitConfigPropagation(async () => (await spendFor("rt-catalog")) > before);
    expect(Math.round((await spendFor("rt-catalog")) - before)).toBe(17000);
  });

  test("without a pricing document the session bills at the model's inline cost", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const before = await spendFor("rt-inline");
    await runSession("rt-inline");
    await waitConfigPropagation(async () => (await spendFor("rt-inline")) > before);
    expect(Math.round((await spendFor("rt-inline")) - before)).toBe(17000);
  });
});
