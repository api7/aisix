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

const CALLER_PLAINTEXT = "sk-routing-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("routing strategies and retry behavior e2e", () => {
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

  async function waitUntilModelResponds(
    model: string,
    content: string,
  ): Promise<void> {
    await waitConfigPropagation(async () => {
      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app?.proxyUrl}/v1`,
        maxRetries: 0,
      });
      try {
        const probe = await client.chat.completions.create({
          model,
          messages: [{ role: "user", content: `ready-${model}` }],
        });
        return probe.choices[0]?.message.content === content;
      } catch {
        return false;
      }
    });
  }

  test("failover retries the current target before moving to the next target", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const primary = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "retry primary down", type: "server_error" } },
    });
    const secondary = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-routing-retry-fallback",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "after retries" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreams.push(primary, secondary);

    await createOpenAiModel("routing-retry-primary", primary);
    await createOpenAiModel("routing-retry-secondary", secondary);
    await waitUntilModelResponds("routing-retry-secondary", "after retries");
    await admin.createModel({
      display_name: "routing-retry-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "routing-retry-primary" },
          { model: "routing-retry-secondary" },
        ],
        retries: 1,
        max_fallbacks: 1,
      },
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "routing-retry-virtual",
          messages: [{ role: "user", content: "ready-routing-retry" }],
        });
        return probe.choices[0]?.message.content === "after retries";
      } catch {
        return false;
      }
    });

    const primaryBaseline = primary.receivedRequests.length;
    const secondaryBaseline = secondary.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "routing-retry-virtual",
      messages: [{ role: "user", content: "retry then fallback" }],
    });

    expect(completion.choices[0]?.message.content).toBe("after retries");
    expect(primary.receivedRequests.length - primaryBaseline).toBe(2);
    expect(secondary.receivedRequests.length - secondaryBaseline).toBe(1);
  });

  test("retry_on_429 lets 429 participate in retry and failover", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const primary = await startOpenAiUpstream({
      status: 429,
      errorBody: { error: { message: "too many requests", type: "rate_limit_error" } },
    });
    const secondary = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-routing-429-fallback",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "429 fallback worked" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreams.push(primary, secondary);

    await createOpenAiModel("routing-429-primary", primary);
    await createOpenAiModel("routing-429-secondary", secondary);
    await waitUntilModelResponds("routing-429-secondary", "429 fallback worked");
    await admin.createModel({
      display_name: "routing-429-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "routing-429-primary" },
          { model: "routing-429-secondary" },
        ],
        retries: 1,
        max_fallbacks: 1,
        retry_on_429: true,
      },
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "routing-429-virtual",
          messages: [{ role: "user", content: "ready-routing-429" }],
        });
        return probe.choices[0]?.message.content === "429 fallback worked";
      } catch {
        return false;
      }
    });

    const primaryBaseline = primary.receivedRequests.length;
    const secondaryBaseline = secondary.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "routing-429-virtual",
      messages: [{ role: "user", content: "429 should retry and fail over" }],
    });

    expect(completion.choices[0]?.message.content).toBe("429 fallback worked");
    expect(primary.receivedRequests.length - primaryBaseline).toBe(2);
    expect(secondary.receivedRequests.length - secondaryBaseline).toBe(1);
  });

  test("round_robin rotates the starting target between requests", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const first = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-routing-rr-a",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "rr-a" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    const second = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-routing-rr-b",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "rr-b" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreams.push(first, second);

    await createOpenAiModel("routing-rr-a", first);
    await createOpenAiModel("routing-rr-b", second);
    await waitUntilModelResponds("routing-rr-a", "rr-a");
    await waitUntilModelResponds("routing-rr-b", "rr-b");
    await admin.createModel({
      display_name: "routing-rr-virtual",
      routing: {
        strategy: "round_robin",
        targets: [{ model: "routing-rr-a" }, { model: "routing-rr-b" }],
        max_fallbacks: 0,
      },
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "routing-rr-virtual",
          messages: [{ role: "user", content: "ready-routing-rr" }],
        });
        return ["rr-a", "rr-b"].includes(probe.choices[0]?.message.content ?? "");
      } catch {
        return false;
      }
    });

    const firstBaseline = first.receivedRequests.length;
    const secondBaseline = second.receivedRequests.length;
    const contents: string[] = [];
    for (let i = 0; i < 4; i++) {
      const completion = await client.chat.completions.create({
        model: "routing-rr-virtual",
        messages: [{ role: "user", content: `round-robin-${i}` }],
      });
      contents.push(completion.choices[0]?.message.content ?? "");
    }

    expect(new Set(contents)).toEqual(new Set(["rr-a", "rr-b"]));
    expect(contents[0]).not.toBe(contents[1]);
    expect(contents[1]).not.toBe(contents[2]);
    expect(contents[2]).not.toBe(contents[3]);
    expect(first.receivedRequests.length - firstBaseline).toBe(2);
    expect(second.receivedRequests.length - secondBaseline).toBe(2);
  });

  test("weighted picks the positive-weight target first and falls forward from there", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const zeroWeightBefore = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-routing-weighted-before",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "should-not-run" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    const weightedPrimary = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "weighted primary down", type: "server_error" } },
    });
    const forwardFallback = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-routing-weighted-fallback",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "weighted fallback worked" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreams.push(zeroWeightBefore, weightedPrimary, forwardFallback);

    await createOpenAiModel("routing-weighted-before", zeroWeightBefore);
    await createOpenAiModel("routing-weighted-primary", weightedPrimary);
    await createOpenAiModel("routing-weighted-fallback", forwardFallback);
    await waitUntilModelResponds("routing-weighted-before", "should-not-run");
    await waitUntilModelResponds("routing-weighted-fallback", "weighted fallback worked");
    await admin.createModel({
      display_name: "routing-weighted-virtual",
      routing: {
        strategy: "weighted",
        targets: [
          { model: "routing-weighted-before", weight: 0 },
          { model: "routing-weighted-primary", weight: 1 },
          { model: "routing-weighted-fallback", weight: 0 },
        ],
        max_fallbacks: 1,
      },
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "routing-weighted-virtual",
          messages: [{ role: "user", content: "ready-routing-weighted" }],
        });
        return probe.choices[0]?.message.content === "weighted fallback worked";
      } catch {
        return false;
      }
    });

    const beforeBaseline = zeroWeightBefore.receivedRequests.length;
    const primaryBaseline = weightedPrimary.receivedRequests.length;
    const fallbackBaseline = forwardFallback.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "routing-weighted-virtual",
      messages: [{ role: "user", content: "weighted routing request" }],
    });

    expect(completion.choices[0]?.message.content).toBe("weighted fallback worked");
    expect(zeroWeightBefore.receivedRequests.length - beforeBaseline).toBe(0);
    expect(weightedPrimary.receivedRequests.length - primaryBaseline).toBe(1);
    expect(forwardFallback.receivedRequests.length - fallbackBaseline).toBe(1);
  });
});
