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

const CALLER_PLAINTEXT = "sk-canary-routing-e2e-caller";
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

describe("sticky (A/B / canary) weighted routing e2e", () => {
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
    });
  }

  function client(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  async function askWithKey(key: string): Promise<string | null> {
    const completion = await client().chat.completions.create(
      { model: "canary-router", messages: [{ role: "user", content: "hi" }] },
      { headers: { "x-aisix-routing-key": key } },
    );
    return completion.choices[0]?.message.content ?? null;
  }

  test("pins a stability key to one target while splitting across keys", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const stable = await startOpenAiUpstream({ nonStreamBody: okBody("stable-served") });
    const canary = await startOpenAiUpstream({ nonStreamBody: okBody("canary-served") });
    upstreams.push(stable, canary);
    await createOpenAiModel("canary-stable", stable);
    await createOpenAiModel("canary-new", canary);
    await admin.createModel({
      display_name: "canary-router",
      routing: {
        strategy: "weighted",
        sticky: true,
        targets: [
          { model: "canary-stable", weight: 50 },
          { model: "canary-new", weight: 50 },
        ],
      },
    });

    await waitConfigPropagation(async () => {
      const models = await admin!.listModels();
      return models.some((m) => m.display_name === "canary-router");
    });

    // Same key → same target on every request (sticky).
    const repeated = await Promise.all(
      Array.from({ length: 6 }, () => askWithKey("user-A")),
    );
    expect(new Set(repeated).size).toBe(1);

    // Distinct keys spread across both targets (the split is honored, not a
    // single funnel). Deterministic hashing keeps this stable across runs.
    const served = new Set<string | null>();
    for (let i = 0; i < 32; i++) {
      served.add(await askWithKey(`user-${i}`));
    }
    expect(served).toEqual(new Set(["stable-served", "canary-served"]));
  });
});
