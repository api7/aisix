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

// E2E for ai-gateway#471: a Model Group (routing model) accessed via
// /v1/messages used to 400 with
//   model "<MG>" has no provider_key_id (routing models can't be dispatched directly)
// because the Anthropic Messages handler dispatched the virtual model
// directly instead of walking `routing.targets` like
// /v1/chat/completions does.
//
// The customer's group mixed an Anthropic and an OpenAI provider, so
// these tests exercise both per-target dispatch paths through
// /v1/messages — Anthropic passthrough and cross-provider translation —
// plus cross-protocol failover.
//
// The mock-upstream harness is path-agnostic: `startOpenAiUpstream`
// doubles as the Anthropic mock when fed an Anthropic-shaped
// `nonStreamBody`, and the Anthropic bridge appends `/v1/messages` to
// the api_base on its own.

const CALLER_PLAINTEXT = "sk-mg-messages-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function openAiChatBody(content: string) {
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
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  };
}

function anthropicMessageBody(text: string) {
  return {
    id: `msg_${text}`,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: "claude-3-5-haiku-20241022",
    stop_reason: "end_turn",
    usage: { input_tokens: 5, output_tokens: 4 },
  };
}

type MessagesResult = {
  status: number;
  body: {
    type?: string;
    content?: Array<{ type?: string; text?: string }>;
    error?: { message?: string };
  };
};

describe("model group via passthrough endpoints e2e (#471)", () => {
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
    const pk = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-openai-mock",
      // OpenAI bridge convention: host + /v1.
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  }

  async function createAnthropicModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!admin) throw new Error("admin client not initialized");
    const pk = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-ant-mock",
      // Anthropic bridge appends /v1/messages, so point at the bare host.
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: displayName,
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
  }

  // Warm a direct target so the routing health filter keeps it.
  async function waitUntilModelResponds(model: string): Promise<void> {
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
        return probe.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });
  }

  async function callMessages(model: string): Promise<MessagesResult> {
    const res = await fetch(`${app?.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 64,
        messages: [{ role: "user", content: "你是谁" }],
      }),
    });
    return { status: res.status, body: (await res.json()) as MessagesResult["body"] };
  }

  async function callCountTokens(
    model: string,
  ): Promise<{ status: number; body: { input_tokens?: number; error?: unknown } }> {
    const res = await fetch(`${app?.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "count me" }],
      }),
    });
    return { status: res.status, body: (await res.json()) as { input_tokens?: number } };
  }

  async function callResponses(
    model: string,
  ): Promise<{ status: number; body: { object?: string; error?: unknown } }> {
    const res = await fetch(`${app?.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, input: "hello" }),
    });
    return { status: res.status, body: (await res.json()) as { object?: string } };
  }

  async function waitUntilGroupRoutable(virtual: string): Promise<void> {
    // Gate on the virtual record reaching the DP snapshot without
    // sending traffic through it, so failover hit-counts below start clean.
    await waitConfigPropagation(async () => {
      try {
        const models = await admin!.listModels();
        return models.some((m) => m.display_name === virtual);
      } catch {
        return false;
      }
    });
  }

  test("Anthropic target in a group is reachable via /v1/messages", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const anthropic = await startOpenAiUpstream({
      nonStreamBody: anthropicMessageBody("claude-in-group"),
    });
    const openai = await startOpenAiUpstream({
      nonStreamBody: openAiChatBody("gpt-in-group"),
    });
    upstreams.push(anthropic, openai);

    await createAnthropicModel("mg-an-anthropic", anthropic);
    await createOpenAiModel("mg-an-openai", openai);
    await waitUntilModelResponds("mg-an-anthropic");
    await waitUntilModelResponds("mg-an-openai");
    await admin.createModel({
      display_name: "mg-anthropic-first",
      routing: {
        strategy: "failover",
        targets: [
          { model: "mg-an-anthropic" },
          { model: "mg-an-openai" },
        ],
        max_fallbacks: 1,
      },
    });

    let result: MessagesResult | undefined;
    await waitConfigPropagation(async () => {
      result = await callMessages("mg-anthropic-first");
      return result.status === 200;
    });

    // Pre-fix this 400'd with "has no provider_key_id".
    expect(result?.status).toBe(200);
    expect(result?.body.type).toBe("message");
    expect(result?.body.content?.[0]?.text).toBe("claude-in-group");
  });

  test("OpenAI target in a group is reachable via /v1/messages (cross-provider)", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const openai = await startOpenAiUpstream({
      nonStreamBody: openAiChatBody("gpt-served-via-messages"),
    });
    upstreams.push(openai);

    await createOpenAiModel("mg-op-openai", openai);
    await waitUntilModelResponds("mg-op-openai");
    await admin.createModel({
      display_name: "mg-openai-first",
      routing: {
        strategy: "failover",
        targets: [{ model: "mg-op-openai" }],
        max_fallbacks: 0,
      },
    });

    let result: MessagesResult | undefined;
    await waitConfigPropagation(async () => {
      result = await callMessages("mg-openai-first");
      return result.status === 200;
    });

    // Anthropic-in → OpenAI upstream → Anthropic-out to the caller.
    expect(result?.status).toBe(200);
    expect(result?.body.type).toBe("message");
    expect(result?.body.content?.[0]?.text).toBe("gpt-served-via-messages");
  });

  test("cross-protocol failover: OpenAI target down falls over to Anthropic target", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const openaiDown = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "openai target down", type: "server_error" } },
    });
    const anthropicGood = await startOpenAiUpstream({
      nonStreamBody: anthropicMessageBody("failover-to-claude"),
    });
    upstreams.push(openaiDown, anthropicGood);

    await createOpenAiModel("mg-fo-openai", openaiDown);
    await createAnthropicModel("mg-fo-anthropic", anthropicGood);
    // Only the healthy target can be probe-warmed; the 502 target is
    // left unprobed so its cooldown doesn't exclude it from the attempt
    // list before the measured call below.
    await waitUntilModelResponds("mg-fo-anthropic");
    await admin.createModel({
      display_name: "mg-failover",
      routing: {
        strategy: "failover",
        targets: [
          { model: "mg-fo-openai" },
          { model: "mg-fo-anthropic" },
        ],
        max_fallbacks: 1,
      },
    });
    await waitUntilGroupRoutable("mg-failover");

    const openaiBaseline = openaiDown.receivedRequests.length;
    const anthropicBaseline = anthropicGood.receivedRequests.length;

    const result = await callMessages("mg-failover");

    expect(result.status).toBe(200);
    expect(result.body.type).toBe("message");
    expect(result.body.content?.[0]?.text).toBe("failover-to-claude");
    // First target (OpenAI) attempted once, then failover to the
    // Anthropic target which served the response.
    expect(openaiDown.receivedRequests.length - openaiBaseline).toBe(1);
    expect(anthropicGood.receivedRequests.length - anthropicBaseline).toBe(1);
  });

  test("Anthropic group is reachable via /v1/messages/count_tokens", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const anthropic = await startOpenAiUpstream({
      nonStreamBody: { input_tokens: 42 },
    });
    upstreams.push(anthropic);

    await createAnthropicModel("mg-ct-anthropic", anthropic);
    await admin.createModel({
      display_name: "mg-count-tokens",
      routing: {
        strategy: "failover",
        targets: [{ model: "mg-ct-anthropic" }],
        max_fallbacks: 0,
      },
    });

    let result: { status: number; body: { input_tokens?: number } } | undefined;
    await waitConfigPropagation(async () => {
      result = await callCountTokens("mg-count-tokens");
      return result.status === 200;
    });

    // Pre-fix this 400'd: a routing model has no provider, so the
    // Anthropic-only gate rejected it before walking targets.
    expect(result?.status).toBe(200);
    expect(result?.body.input_tokens).toBe(42);
  });

  test("OpenAI group is reachable via /v1/responses", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const openai = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_mg_01",
        object: "response",
        status: "completed",
        model: "gpt-4o-mini",
        output: [
          {
            id: "msg_mg_01",
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "grouped response" }],
          },
        ],
        usage: { input_tokens: 5, output_tokens: 6, total_tokens: 11 },
      },
    });
    upstreams.push(openai);

    await createOpenAiModel("mg-resp-openai", openai);
    await admin.createModel({
      display_name: "mg-responses",
      routing: {
        strategy: "failover",
        targets: [{ model: "mg-resp-openai" }],
        max_fallbacks: 0,
      },
    });

    let result: { status: number; body: { object?: string } } | undefined;
    await waitConfigPropagation(async () => {
      result = await callResponses("mg-responses");
      return result.status === 200;
    });

    // Pre-fix this 400'd: a routing model has no provider, so the
    // OpenAI-only gate rejected it before walking targets.
    expect(result?.status).toBe(200);
    expect(result?.body.object).toBe("response");
  });
});
