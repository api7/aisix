import { createHash } from "node:crypto";
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

// E2E: the reasoning-effort knob survives cross-protocol translation in
// both directions (AISIX-Cloud#1474).
//
// Anthropic moved the control off `thinking.budget_tokens` and onto
// `output_config.effort`: the budget is deprecated on Opus 4.6 and
// rejected outright from 4.7, so a current client sends adaptive
// thinking plus an effort tier, or an effort tier alone. The gateway
// translated only `thinking`, so `output_config` fell into the
// drop-everything-else branch and the caller's tier never reached the
// upstream — an adaptive request arrived byte-identical whichever tier
// it asked for.
//
// These assert on what the mock upstream actually received, which is
// the only place the bug was visible: SLS `content_mode = full` records
// the client-side body from before translation, so an `output_config`
// there proves nothing about what went upstream.

const ANTHROPIC_IN_PLAINTEXT = "sk-effort-anthropic-in";
const ANTHROPIC_IN_KEY_HASH = createHash("sha256")
  .update(ANTHROPIC_IN_PLAINTEXT)
  .digest("hex");

describe("Anthropic Messages → OpenAI upstream: effort tier reaches the upstream (#1474)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-effort-01",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hello" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "effort-xprov-pk",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "effort-xprov",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: ANTHROPIC_IN_KEY_HASH,
      allowed_models: ["effort-xprov"],
    });

    // The caller key is seeded last, so it authenticating implies the
    // whole seed set is in the snapshot. Gating on a `/v1/messages`
    // call instead would fail by timeout rather than by assertion when
    // the translation under test breaks.
    const proxy = new ProxyClient(app.proxyUrl, ANTHROPIC_IN_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  /** Send an Anthropic body and return the body the upstream received. */
  async function upstreamBodyFor(
    extra: Record<string, unknown>,
    stream = false,
  ): Promise<Record<string, unknown>> {
    const baseline = upstream!.receivedRequests.length;
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_IN_PLAINTEXT,
      },
      body: JSON.stringify({
        model: "effort-xprov",
        max_tokens: 16000,
        messages: [{ role: "user", content: "probe" }],
        ...(stream ? { stream: true } : {}),
        ...extra,
      }),
    });
    expect(res.ok).toBe(true);
    // Drain a streaming body so the upstream request is complete before
    // the assertions read it.
    await res.text();
    const req = upstream!.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/chat/completions");
    expect(req).toBeDefined();
    return JSON.parse(req!.body) as Record<string, unknown>;
  }

  test("output_config.effort becomes reasoning_effort, and outranks thinking", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // The reported request: adaptive thinking + an explicit tier. The
    // tier is what the caller chose, so the tier is what goes upstream.
    const withThinking = await upstreamBodyFor({
      thinking: { type: "adaptive" },
      output_config: { effort: "max" },
    });
    expect(withThinking.reasoning_effort).toBe("max");
    expect(withThinking.output_config).toBeUndefined();
    expect(withThinking.thinking).toBeUndefined();

    // The reported variant: a tier with no `thinking` at all, which is
    // how a current client asks, since thinking is on by default.
    const effortOnly = await upstreamBodyFor({
      output_config: { effort: "xhigh" },
    });
    expect(effortOnly.reasoning_effort).toBe("xhigh");
    expect(effortOnly.output_config).toBeUndefined();

    // Adaptive with no tier takes Anthropic's own default, not a
    // middle tier the caller never asked for.
    const adaptiveOnly = await upstreamBodyFor({
      thinking: { type: "adaptive" },
    });
    expect(adaptiveOnly.reasoning_effort).toBe("high");

    // The legacy budget shape still maps, for clients that predate the
    // effort field.
    const legacyBudget = await upstreamBodyFor({
      thinking: { type: "enabled", budget_tokens: 8000 },
    });
    expect(legacyBudget.reasoning_effort).toBe("high");
  });

  test("streaming takes the same translation as non-streaming", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const body = await upstreamBodyFor(
      { thinking: { type: "adaptive" }, output_config: { effort: "max" } },
      true,
    );
    expect(body.reasoning_effort).toBe("max");
    expect(body.output_config).toBeUndefined();
  });

  test("output_config.format becomes a strict response_format", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const body = await upstreamBodyFor({
      output_config: {
        format: {
          type: "json_schema",
          schema: {
            type: "object",
            properties: { city: { type: "string" } },
          },
        },
      },
    });
    expect(body.output_config).toBeUndefined();
    expect(body.response_format).toEqual({
      type: "json_schema",
      json_schema: {
        name: "structured_output",
        strict: true,
        schema: {
          type: "object",
          properties: { city: { type: "string" } },
          additionalProperties: false,
          required: ["city"],
        },
      },
    });
  });
});

// ─── Reverse direction ──────────────────────────────────────────────

const OPENAI_IN_PLAINTEXT = "sk-effort-openai-in";
const OPENAI_IN_KEY_HASH = createHash("sha256")
  .update(OPENAI_IN_PLAINTEXT)
  .digest("hex");

describe("OpenAI Chat → Anthropic upstream: reasoning_effort becomes output_config (#1474)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_effort_01",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "hello" }],
        model: "claude-opus-4-6",
        stop_reason: "end_turn",
        usage: { input_tokens: 5, output_tokens: 4 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // The Anthropic bridge appends `/v1/messages` itself, so api_base
    // is the bare host — the opposite of the OpenAI bridge convention.
    const pk = await seed.createProviderKey({
      display_name: "effort-anth-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: "effort-anth",
      provider: "anthropic",
      model_name: "claude-opus-4-6",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: OPENAI_IN_KEY_HASH,
      allowed_models: ["effort-anth"],
    });

    const proxy = new ProxyClient(app.proxyUrl, OPENAI_IN_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function upstreamBodyFor(
    extra: Record<string, unknown>,
    stream = false,
  ): Promise<Record<string, unknown>> {
    const baseline = upstream!.receivedRequests.length;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${OPENAI_IN_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "effort-anth",
        messages: [{ role: "user", content: "probe" }],
        ...(stream ? { stream: true } : {}),
        ...extra,
      }),
    });
    expect(res.ok).toBe(true);
    await res.text();
    const req = upstream!.receivedRequests
      .slice(baseline)
      .find((r) => r.path.endsWith("/v1/messages"));
    expect(req).toBeDefined();
    return JSON.parse(req!.body) as Record<string, unknown>;
  }

  test("reasoning_effort is translated, never forwarded verbatim", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // `reasoning_effort` is not an Anthropic field: forwarding it
    // verbatim reaches `/v1/messages` as an unknown parameter.
    const tiered = await upstreamBodyFor({ reasoning_effort: "xhigh" });
    expect(tiered.reasoning_effort).toBeUndefined();
    expect(tiered.output_config).toEqual({ effort: "xhigh" });
    // The caller asked for a depth, not a thinking mode — the model
    // applies its own.
    expect(tiered.thinking).toBeUndefined();

    // Anthropic's vocabulary has no `minimal`; `low` is its floor.
    const minimal = await upstreamBodyFor({ reasoning_effort: "minimal" });
    expect(minimal.output_config).toEqual({ effort: "low" });

    // ...and no `none`: asking for no reasoning is the disabled mode.
    const none = await upstreamBodyFor({ reasoning_effort: "none" });
    expect(none.output_config).toBeUndefined();
    expect(none.thinking).toEqual({ type: "disabled" });
  });

  test("the streaming call site takes the same translation", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    // The Anthropic bridge calls `build_request` from two places, one
    // per streaming mode. They share the translation today; this keeps
    // the streaming one from drifting out of it.
    const body = await upstreamBodyFor({ reasoning_effort: "xhigh" }, true);
    expect(body.reasoning_effort).toBeUndefined();
    expect(body.output_config).toEqual({ effort: "xhigh" });
    expect(body.stream).toBe(true);
  });

  test("a carrier output_config keeps its other keys and gains the tier", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const body = await upstreamBodyFor({
      reasoning_effort: "high",
      output_config: { task_budget: { type: "tokens", total: 64000 } },
    });
    expect(body.output_config).toEqual({
      task_budget: { type: "tokens", total: 64000 },
      effort: "high",
    });
  });
});
