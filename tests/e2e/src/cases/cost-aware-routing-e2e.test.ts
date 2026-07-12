import { createHash } from "node:crypto";
import OpenAI from "openai";
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

const CALLER_PLAINTEXT = "sk-cost-routing-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function okBody(content: string) {
  return {
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
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("cost-aware (least_cost) routing e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

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
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
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

  test("ranks the cheapest target first regardless of declaration order", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const cheap = await startOpenAiUpstream({ nonStreamBody: okBody("cheap-served") });
    const pricey = await startOpenAiUpstream({ nonStreamBody: okBody("pricey-served") });
    upstreams.push(cheap, pricey);

    // Declare the expensive target FIRST — least_cost must reorder by price,
    // not honor declaration order. The router document is written BEFORE its
    // targets: watch events apply in revision order, so once both targets are
    // visible the router is in the snapshot too (virtual models don't appear
    // in /v1/models themselves).
    await seed.createModel({
      display_name: "cost-virtual",
      routing: {
        strategy: "least_cost",
        targets: [{ model: "cost-pricey" }, { model: "cost-cheap" }],
      },
    });
    // Cheap total unit price = 0.2/1K; pricey = 20/1K.
    await createOpenAiModel("cost-cheap", cheap, {
      cost: { input_per_1k: 0.1, output_per_1k: 0.1 },
    });
    await createOpenAiModel("cost-pricey", pricey, {
      cost: { input_per_1k: 10, output_per_1k: 10 },
    });

    // Gate on the DP snapshot via /v1/models — it only authenticates once
    // the caller key has propagated and only lists the targets once the
    // snapshot has them, without dispatching to a target (which would skew
    // the per-target counts below).
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const ids = ((await res.json()) as { data?: Array<{ id?: string }> }).data?.map((m) => m.id) ?? [];
      return ids.includes("cost-cheap") && ids.includes("cost-pricey");
    });

    const cheapBaseline = cheap.receivedRequests.length;
    const priceyBaseline = pricey.receivedRequests.length;

    const completion = await client().chat.completions.create({
      model: "cost-virtual",
      messages: [{ role: "user", content: "cheapest please" }],
    });

    expect(completion.choices[0]?.message.content).toBe("cheap-served");
    expect(cheap.receivedRequests.length - cheapBaseline).toBe(1);
    expect(pricey.receivedRequests.length - priceyBaseline).toBe(0);
  });

  test("falls forward to the next-cheapest when the cheapest fails", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const cheap = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "cheapest down", type: "server_error" } },
    });
    const mid = await startOpenAiUpstream({ nonStreamBody: okBody("mid-served") });
    const pricey = await startOpenAiUpstream({ nonStreamBody: okBody("pricey-served") });
    upstreams.push(cheap, mid, pricey);

    // Router BEFORE targets (revision-order gate, as above).
    await seed.createModel({
      display_name: "cost-ff-virtual",
      routing: {
        strategy: "least_cost",
        targets: [
          { model: "cost-ff-pricey" },
          { model: "cost-ff-mid" },
          { model: "cost-ff-cheap" },
        ],
      },
    });
    // Keep the failing cheapest in rotation so the assertion sees it attempted
    // (cooldown would take it out after the first 503).
    await createOpenAiModel("cost-ff-cheap", cheap, {
      cost: { input_per_1k: 0.1, output_per_1k: 0.1 },
      cooldown: { enabled: false },
    });
    await createOpenAiModel("cost-ff-mid", mid, {
      cost: { input_per_1k: 1, output_per_1k: 1 },
    });
    await createOpenAiModel("cost-ff-pricey", pricey, {
      cost: { input_per_1k: 10, output_per_1k: 10 },
    });

    // Same DP-snapshot gate as above — upstream-neutral readiness probe.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const ids = ((await res.json()) as { data?: Array<{ id?: string }> }).data?.map((m) => m.id) ?? [];
      return ["cost-ff-cheap", "cost-ff-mid", "cost-ff-pricey"].every((id) => ids.includes(id));
    });

    const cheapBaseline = cheap.receivedRequests.length;
    const midBaseline = mid.receivedRequests.length;
    const priceyBaseline = pricey.receivedRequests.length;

    const completion = await client().chat.completions.create({
      model: "cost-ff-virtual",
      messages: [{ role: "user", content: "cheapest then fall forward" }],
    });

    // Cheapest tried first (503), falls forward to next-cheapest (mid). The
    // pricey target is never reached.
    expect(completion.choices[0]?.message.content).toBe("mid-served");
    expect(cheap.receivedRequests.length - cheapBaseline).toBe(1);
    expect(mid.receivedRequests.length - midBaseline).toBe(1);
    expect(pricey.receivedRequests.length - priceyBaseline).toBe(0);
  });
});
