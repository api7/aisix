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
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
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
    if (!admin) throw new Error("admin client not initialized");
    const providerKey = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
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
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const cheap = await startOpenAiUpstream({ nonStreamBody: okBody("cheap-served") });
    const pricey = await startOpenAiUpstream({ nonStreamBody: okBody("pricey-served") });
    upstreams.push(cheap, pricey);

    // Cheap total unit price = 0.2/1K; pricey = 20/1K.
    await createOpenAiModel("cost-cheap", cheap, {
      cost: { input_per_1k: 0.1, output_per_1k: 0.1 },
    });
    await createOpenAiModel("cost-pricey", pricey, {
      cost: { input_per_1k: 10, output_per_1k: 10 },
    });
    // Declare the expensive target FIRST — least_cost must reorder by price,
    // not honor declaration order.
    await admin.createModel({
      display_name: "cost-virtual",
      routing: {
        strategy: "least_cost",
        targets: [{ model: "cost-pricey" }, { model: "cost-cheap" }],
      },
    });

    // Gate on the routing model reaching the DP snapshot (probe would be
    // fine here since both targets are healthy, but listModels avoids any
    // per-target request skew before baselines are taken). listModels reads
    // etcd directly and shouldn't fail during propagation — let a genuine
    // admin failure surface instead of masking it as a 30s timeout.
    await waitConfigPropagation(async () => {
      const models = await admin!.listModels();
      return models.some((m) => m.display_name === "cost-virtual");
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
    if (!etcdReachable || !app || !admin) {
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
    await admin.createModel({
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

    await waitConfigPropagation(async () => {
      const models = await admin!.listModels();
      return models.some((m) => m.display_name === "cost-ff-virtual");
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
