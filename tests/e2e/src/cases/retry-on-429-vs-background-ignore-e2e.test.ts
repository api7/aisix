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

const CALLER_PLAINTEXT = "sk-retry-429-vs-bg-ignore";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("retry_on_429 vs background ignore e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let rateLimitedUpstream: OpenAiUpstream | undefined;
  let fallbackUpstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    rateLimitedUpstream = await startOpenAiUpstream({
      scriptedResponses: [
        {
          status: 429,
          errorBody: { error: { message: "background ignore", type: "rate_limit_error" } },
        },
        {
          status: 429,
          errorBody: { error: { message: "request path 429", type: "rate_limit_error" } },
        },
        {
          status: 429,
          errorBody: { error: { message: "request path 429 retry", type: "rate_limit_error" } },
        },
      ],
    });
    fallbackUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-429-fallback",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "request-path 429 fallback" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const primaryPk = await seed.createProviderKey({
      display_name: "retry-429-primary-pk",
      secret: "sk-mock",
      api_base: `${rateLimitedUpstream.baseUrl}/v1`,
    });
    const fallbackPk = await seed.createProviderKey({
      display_name: "retry-429-fallback-pk",
      secret: "sk-mock",
      api_base: `${fallbackUpstream.baseUrl}/v1`,
    });

    await seed.createModel({
      display_name: "retry-429-primary",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: primaryPk.id,
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
      display_name: "retry-429-fallback",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: fallbackPk.id,
    });
    await seed.createModel({
      display_name: "retry-429-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "retry-429-primary" },
          { model: "retry-429-fallback" },
        ],
        retries: 1,
        max_fallbacks: 1,
        retry_on_429: true,
      },
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["retry-429-router", "retry-429-fallback"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await rateLimitedUpstream?.close();
    await fallbackUpstream?.close();
  });

  test("background 429 stays healthy while request-path 429 still retries and fails over", async (ctx) => {
    if (!etcdReachable || !app || !admin || !rateLimitedUpstream || !fallbackUpstream) {
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
      const row = statuses.find((item) => item.display_name === "retry-429-primary");
      return row?.status === "healthy" && row?.last_check_status === 429;
    });

    const completion = await client.chat.completions.create({
      model: "retry-429-router",
      messages: [{ role: "user", content: "429 request path should still retry" }],
    });

    expect(completion.choices[0]?.message.content).toBe("request-path 429 fallback");

    const statuses = await admin.listModelStatuses();
    const row = statuses.find((item) => item.display_name === "retry-429-primary")!;
    expect(row.status).toBe("cooldown");
    // After PR #268 H1/M1 contract: cooldown reason reflects the
    // upstream HTTP semantic, not the gateway-internal retry flag.
    // 429 maps to upstream_rate_limited regardless of whether
    // retry_on_429 was true (here) or false (see m1_decoupling test).
    expect(row.status_reason).toBe("upstream_rate_limited");
  });
});
