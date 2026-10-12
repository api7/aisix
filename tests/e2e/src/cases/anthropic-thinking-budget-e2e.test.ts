import { createHash } from "node:crypto";
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

// E2E: an OpenAI-shape reasoning effort sent to a Claude family that
// rejects `output_config.effort` becomes a thinking budget, on every
// bridge that carries it into an Anthropic body.
//
// Haiku 4.5, Sonnet 4.5 and every earlier Claude answer
// `output_config.effort` with a 400 ("This model does not support the
// effort parameter"). Those families take
// `thinking: {type: enabled, budget_tokens}`, and Anthropic requires
// `max_tokens` above the budget. Every model that accepts effort keeps
// receiving exactly the body it received before.
//
// Assertions read what the mock upstream received: Anthropic's own
// API, Bedrock `/invoke` and `/converse-stream`, and Vertex
// `:rawPredict` all build their body from the same translation.

const CALLER_PLAINTEXT = "sk-thinking-budget-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const ANTHROPIC_MESSAGE = {
  id: "msg_budget_01",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "hello" }],
  model: "claude-sonnet-4-5",
  stop_reason: "end_turn",
  usage: { input_tokens: 5, output_tokens: 4 },
};

describe("reasoning effort → Claude families without effort support: thinking budget on every Anthropic-body bridge", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_MESSAGE });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const anthropicPk = await seed.createProviderKey({
      display_name: "budget-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    for (const [alias, name] of [
      ["budget-sonnet", "claude-sonnet-4-5"],
      ["budget-haiku", "claude-haiku-4-5-20251001"],
      ["budget-sonnet-37", "claude-3-7-sonnet-20250219"],
      ["effort-opus-45", "claude-opus-4-5"],
      ["effort-sonnet-46", "claude-sonnet-4-6"],
      ["effort-unversioned", "glm-4.6"],
    ]) {
      await seed.createModel({
        display_name: alias,
        provider: "anthropic",
        model_name: name,
        provider_key_id: anthropicPk.id,
      });
    }

    const bedrockPk = await seed.createProviderKey({
      display_name: "budget-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-budget-e2e",
        secret_access_key: "sk-budget-e2e",
        region: "us-west-2",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "budget-bedrock",
      provider: "bedrock",
      model_name: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      provider_key_id: bedrockPk.id,
    });

    const vertexPk = await seed.createProviderKey({
      display_name: "budget-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.budget-e2e",
        project: "proj-e2e",
        region: "us-east5",
      }),
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "budget-vertex",
      provider: "google",
      model_name: "claude-sonnet-4-5@20250929",
      provider_key_id: vertexPk.id,
    });

    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) {
        await res.text();
        return false;
      }
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "budget-vertex");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  /** POST to the gateway and return the body the upstream received. */
  async function upstreamBody(
    endpoint: "/v1/chat/completions" | "/v1/responses",
    body: Record<string, unknown>,
    pathPattern: RegExp,
    expectOk = true,
  ): Promise<Record<string, any>> {
    const baseline = upstream!.receivedRequests.length;
    const res = await fetch(`${app!.proxyUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    if (expectOk) expect(res.status, text).toBe(200);
    const sent = upstream!.receivedRequests
      .slice(baseline)
      .find((r) => pathPattern.test(r.path));
    expect(sent, `no upstream request matched ${pathPattern}`).toBeDefined();
    return JSON.parse(sent!.body) as Record<string, any>;
  }

  function chat(model: string, extra: Record<string, unknown>) {
    return {
      model,
      messages: [{ role: "user", content: "probe" }],
      ...extra,
    };
  }

  const MESSAGES = /\/v1\/messages$/;

  test("chat → Sonnet 4.5: each tier becomes a budget and max_tokens leaves room for the answer", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    for (const [effort, budget] of [
      ["minimal", 1024],
      ["low", 1024],
      ["medium", 2048],
      ["high", 4096],
      ["xhigh", 8192],
      ["max", 16384],
    ] as const) {
      const body = await upstreamBody(
        "/v1/chat/completions",
        chat("budget-sonnet", { reasoning_effort: effort }),
        MESSAGES,
      );
      expect(body.thinking, effort).toEqual({ type: "enabled", budget_tokens: budget });
      expect(body.output_config, effort).toBeUndefined();
      expect(body.reasoning_effort, effort).toBeUndefined();
      expect(body.max_tokens, effort).toBe(budget + 4096);
    }
  });

  test("Haiku 4.5 and Claude 3.7 take the budget too", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    for (const model of ["budget-haiku", "budget-sonnet-37"]) {
      const body = await upstreamBody(
        "/v1/chat/completions",
        chat(model, { reasoning_effort: "low" }),
        MESSAGES,
      );
      expect(body.thinking, model).toEqual({ type: "enabled", budget_tokens: 1024 });
      expect(body.output_config, model).toBeUndefined();
    }
  });

  test("models that accept effort keep the effort-only body", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    // Opus 4.5, Claude 4.6+ and names with no readable Claude version
    // (an Anthropic-compatible endpoint serving another vendor's model).
    for (const model of ["effort-opus-45", "effort-sonnet-46", "effort-unversioned"]) {
      const body = await upstreamBody(
        "/v1/chat/completions",
        chat(model, { reasoning_effort: "high" }),
        MESSAGES,
      );
      expect(body.output_config, model).toEqual({ effort: "high" });
      expect(body.thinking, model).toBeUndefined();
      expect(body.max_tokens, model).toBe(4096);
      expect(body.reasoning_effort, model).toBeUndefined();
    }
    const responses = await upstreamBody(
      "/v1/responses",
      { model: "effort-sonnet-46", input: "probe", reasoning: { effort: "low" } },
      MESSAGES,
    );
    expect(responses.output_config).toEqual({ effort: "low" });
    expect(responses.thinking).toBeUndefined();
    expect(responses.max_tokens).toBe(4096);
  });

  test("a caller's max_tokens is kept and the budget is capped below it", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const capped = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-sonnet", { reasoning_effort: "high", max_tokens: 2000 }),
      MESSAGES,
    );
    expect(capped.max_tokens).toBe(2000);
    expect(capped.thinking).toEqual({ type: "enabled", budget_tokens: 1999 });

    const roomy = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-sonnet", { reasoning_effort: "medium", max_tokens: 10000 }),
      MESSAGES,
    );
    expect(roomy.max_tokens).toBe(10000);
    expect(roomy.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });

    // At or below Anthropic's 1024-token minimum budget there is no room
    // to think at all.
    const tiny = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-sonnet", { reasoning_effort: "high", max_tokens: 1024 }),
      MESSAGES,
    );
    expect(tiny.max_tokens).toBe(1024);
    expect(tiny.thinking).toBeUndefined();
    expect(tiny.output_config).toBeUndefined();
  });

  test("none stays the disabled thinking mode on every model", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    for (const model of ["budget-sonnet", "effort-opus-45", "effort-unversioned"]) {
      const body = await upstreamBody(
        "/v1/chat/completions",
        chat(model, { reasoning_effort: "none" }),
        MESSAGES,
      );
      expect(body.thinking, model).toEqual({ type: "disabled" });
      expect(body.output_config, model).toBeUndefined();
      expect(body.max_tokens, model).toBe(4096);
    }
  });

  test("streaming chat takes the same translation", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const body = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-sonnet", { reasoning_effort: "medium", stream: true }),
      MESSAGES,
      false,
    );
    expect(body.stream).toBe(true);
    expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });
    expect(body.output_config).toBeUndefined();
  });

  test("/v1/responses reasoning.effort → Sonnet 4.5 becomes a budget", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const body = await upstreamBody(
      "/v1/responses",
      { model: "budget-sonnet", input: "probe", reasoning: { effort: "medium" } },
      MESSAGES,
    );
    expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });
    expect(body.output_config).toBeUndefined();
    expect(body.max_tokens).toBe(2048 + 4096);

    const capped = await upstreamBody(
      "/v1/responses",
      {
        model: "budget-sonnet",
        input: "probe",
        reasoning: { effort: "high" },
        max_output_tokens: 3000,
      },
      MESSAGES,
    );
    expect(capped.max_tokens).toBe(3000);
    expect(capped.thinking).toEqual({ type: "enabled", budget_tokens: 2999 });
  });

  test("Bedrock /invoke carries the budget", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const body = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-bedrock", { reasoning_effort: "low" }),
      /\/invoke$/,
    );
    expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 1024 });
    expect(body.output_config).toBeUndefined();
    expect(body.max_tokens).toBe(1024 + 4096);
  });

  test("Bedrock /converse-stream carries the budget and the max_tokens it needs", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    // The mock answers JSON rather than an AWS event stream, so only the
    // request is asserted.
    const body = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-bedrock", { reasoning_effort: "medium", stream: true }),
      /\/converse-stream$/,
      false,
    );
    expect(body.additionalModelRequestFields).toEqual({
      thinking: { type: "enabled", budget_tokens: 2048 },
    });
    expect(body.inferenceConfig?.maxTokens).toBe(2048 + 4096);
  });

  test("Vertex :rawPredict carries the budget", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();
    const body = await upstreamBody(
      "/v1/chat/completions",
      chat("budget-vertex", { reasoning_effort: "high" }),
      /:rawPredict$/,
    );
    expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 4096 });
    expect(body.output_config).toBeUndefined();
    expect(body.max_tokens).toBe(4096 + 4096);
  });
});
