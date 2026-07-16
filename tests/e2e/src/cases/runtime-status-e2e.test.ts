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

const CALLER_PLAINTEXT = "sk-runtime-status-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("runtime status e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let flakyUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let flakyModelID = "";
  let stableModelID = "";
  let routerModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    flakyUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          status: 502,
          errorBody: { error: { message: "temporary upstream failure", type: "server_error" } },
        },
        {
          nonStreamBody: {
            id: "cmpl-flaky-recovered",
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: "gpt-4o-mini",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: "flaky recovered" },
                finish_reason: "stop",
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        },
      ],
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "stable fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const flakyPk = await seed.createProviderKey({
      display_name: "runtime-flaky-pk",
      secret: "sk-mock",
      api_base: `${flakyUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "runtime-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });
    const flakyModel = await seed.createModel({
      display_name: "runtime-flaky",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: flakyPk.id,
    });
    flakyModelID = flakyModel.id;
    const stableModel = await seed.createModel({
      display_name: "runtime-stable",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stablePk.id,
    });
    stableModelID = stableModel.id;
    const routerModel = await seed.createModel({
      display_name: "runtime-router",
      routing: {
        strategy: "failover",
        targets: [{ model: "runtime-flaky" }, { model: "runtime-stable" }],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    routerModelID = routerModel.id;
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["runtime-router", "runtime-stable", "runtime-flaky"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await flakyUpstream?.close();
    await stableUpstream?.close();
  });

  test("retryable failure cools down the direct model, routing skips it, and admin surfaces runtime status", async (ctx) => {
    if (!etcdReachable || !app || !admin || !flakyUpstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "runtime-stable",
          messages: [{ role: "user", content: "ready-runtime-stable" }],
        });
        return probe.choices[0]?.message.content === "stable fallback";
      } catch {
        return false;
      }
    });

    await waitConfigPropagation();

    const first = await client.chat.completions.create({
      model: "runtime-router",
      messages: [{ role: "user", content: "trip cooldown" }],
    });
    expect(first.choices[0]?.message.content).toBe("stable fallback");
    expect(flakyUpstream.receivedRequests.length).toBeGreaterThanOrEqual(1);
    expect(stableUpstream.receivedRequests.length).toBeGreaterThanOrEqual(1);

    const statusesAfterFirst = await admin.listModelStatuses();
    const flakyAfterFirst = statusesAfterFirst.find((row) => row.id === flakyModelID)!;
    expect(flakyAfterFirst.status).toBe("cooldown");
    // After PR #268 H1 contract: cooldown reason is per-error-category.
    // A 502 from the upstream maps to `upstream_server_error`.
    expect(flakyAfterFirst.status_reason).toBe("upstream_server_error");

    const flakyBaseline = flakyUpstream.receivedRequests.length;
    const stableBaseline = stableUpstream.receivedRequests.length;

    const second = await client.chat.completions.create({
      model: "runtime-router",
      messages: [{ role: "user", content: "skip cooled target" }],
    });
    expect(second.choices[0]?.message.content).toBe("stable fallback");
    expect(flakyUpstream.receivedRequests.length - flakyBaseline).toBe(0);
    expect(stableUpstream.receivedRequests.length - stableBaseline).toBe(1);

    const statuses = await admin.listModelStatuses();
    const flaky = statuses.find((row) => row.id === flakyModelID)!;
    const stable = statuses.find((row) => row.id === stableModelID)!;
    const router = statuses.find((row) => row.id === routerModelID)!;

    expect(flaky.status).toBe("cooldown");
    expect(flaky.status_reason).toBe("upstream_server_error");
    expect(flaky.cooldown_until).toBeTruthy();
    expect(stable.status).toBe("healthy");
    expect(router.status).toBe("not_applicable");
  });
});
