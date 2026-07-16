import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER_PLAINTEXT = "sk-runtime-mixed-filtering-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("runtime mixed filtering e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let unhealthyUpstream: OpenAiUpstream | undefined;
  let cooldownUpstream: OpenAiUpstream | undefined;
  let healthyUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    unhealthyUpstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "unhealthy target", type: "server_error" } },
    });
    cooldownUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          status: 502,
          errorBody: { error: { message: "cooldown target failed", type: "server_error" } },
        },
        {
          nonStreamBody: {
            id: "cmpl-cooldown-recovered",
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: "gpt-4o-mini",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: "should not be selected second" },
                finish_reason: "stop",
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        },
      ],
    });
    healthyUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-healthy-mixed",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "healthy candidate won" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const unhealthyPk = await seed.createProviderKey({
      display_name: "mixed-unhealthy-pk",
      secret: "sk-mock",
      api_base: `${unhealthyUpstream.baseUrl}/v1`,
    });
    const cooldownPk = await seed.createProviderKey({
      display_name: "mixed-cooldown-pk",
      secret: "sk-mock",
      api_base: `${cooldownUpstream.baseUrl}/v1`,
    });
    const healthyPk = await seed.createProviderKey({
      display_name: "mixed-healthy-pk",
      secret: "sk-mock",
      api_base: `${healthyUpstream.baseUrl}/v1`,
    });

    await seed.createModel({
      display_name: "mixed-unhealthy",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: unhealthyPk.id,
      background_model_check: {
        enabled: true,
        interval_seconds: 5,
        timeout_seconds: 10,
        prompt: "Respond with OK",
        max_tokens: 8,
        ignore_statuses: [408, 429],
        stale_after_seconds: 90,
      },
    });
    await seed.createModel({
      display_name: "mixed-cooldown",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: cooldownPk.id,
    });
    await seed.createModel({
      display_name: "mixed-healthy",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: healthyPk.id,
    });

    await seed.createModel({
      display_name: "mixed-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "mixed-unhealthy" },
          { model: "mixed-cooldown" },
          { model: "mixed-healthy" },
        ],
        max_fallbacks: 2,
      },
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["mixed-router", "mixed-cooldown", "mixed-healthy"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await unhealthyUpstream?.close();
    await cooldownUpstream?.close();
    await healthyUpstream?.close();
  });

  test("routing skips unhealthy first, then cooldown, and lands on healthy candidate", async (ctx) => {
    if (!etcdReachable || !app || !admin || !unhealthyUpstream || !cooldownUpstream || !healthyUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      const statuses = await admin!.listModelStatuses();
      const unhealthy = statuses.find((row) => row.display_name === "mixed-unhealthy");
      return unhealthy?.status === "unhealthy";
    });

    const first = await client.chat.completions.create({
      model: "mixed-router",
      messages: [{ role: "user", content: "trip cooldown in middle candidate" }],
    });
    expect(first.choices[0]?.message.content).toBe("healthy candidate won");

    const unhealthyBaseline = unhealthyUpstream.receivedRequests.length;
    const cooldownBaseline = cooldownUpstream.receivedRequests.length;
    const healthyBaseline = healthyUpstream.receivedRequests.length;

    const second = await client.chat.completions.create({
      model: "mixed-router",
      messages: [{ role: "user", content: "mixed filtering second pass" }],
    });
    expect(second.choices[0]?.message.content).toBe("healthy candidate won");

    expect(unhealthyUpstream.receivedRequests.length - unhealthyBaseline).toBe(0);
    expect(cooldownUpstream.receivedRequests.length - cooldownBaseline).toBe(0);
    expect(healthyUpstream.receivedRequests.length - healthyBaseline).toBe(1);
  });
});
