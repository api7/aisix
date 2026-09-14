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

// AISIX-Cloud#1614: a `tool_choice` that reaches an upstream without an
// accompanying `tools` list is rejected — OpenAI-compatible endpoints
// answer 400 "'tool_choice' is only allowed when 'tools' are specified",
// and Anthropic refuses the same pair. The Responses API accepts it, and
// the Codex CLI serialises its context-compaction call exactly that way
// (`"tools": []`, `"tool_choice": "auto"`, `"parallel_tool_calls": false`),
// so every compaction through a bridged model used to fail.
//
// Each protocol converter must therefore drop `tool_choice` whenever the
// tool list translates to nothing — an empty list, or one holding only
// tools the target protocol cannot express. The three directions pinned
// here are the three converters that emit the two fields:
//
//   1. Responses → chat completions
//   2. Anthropic Messages → chat completions
//   3. chat completions → Anthropic Messages

const CALLER_PLAINTEXT = "sk-tool-choice-1614";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CHAT_REPLY = {
  id: "chatcmpl-1614",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "bridged-chat-model",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "compacted" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 8, completion_tokens: 4, total_tokens: 12 },
};

const ANTHROPIC_REPLY = {
  id: "msg_1614",
  type: "message",
  role: "assistant",
  model: "claude-3-haiku-20240307",
  content: [{ type: "text", text: "compacted" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 9, output_tokens: 3 },
};

const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

interface SentBody {
  tools?: unknown;
  tool_choice?: unknown;
}

/** Parse the last request the mock received on `path`. */
function lastBodyOn(
  upstream: OpenAiUpstream,
  baseline: number,
  path: string,
): SentBody {
  const calls = upstream.receivedRequests
    .slice(baseline)
    .filter((r) => r.path === path);
  expect(calls.length).toBeGreaterThan(0);
  return JSON.parse(calls.at(-1)!.body) as SentBody;
}

describe("tool_choice without tools is dropped at every bridge (#1614)", () => {
  let app: SpawnedApp | undefined;
  let chatUpstream: OpenAiUpstream | undefined;
  let anthropicUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    chatUpstream = await startOpenAiUpstream({ nonStreamBody: CHAT_REPLY });
    anthropicUpstream = await startOpenAiUpstream({
      nonStreamBody: ANTHROPIC_REPLY,
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // An OpenAI-compatible upstream that serves chat completions only, so
    // both `/v1/responses` and `/v1/messages` reach it through a bridge.
    const chatPk = await seed.createProviderKey({
      display_name: "tc1614-chat-pk",
      provider: "deepseek",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${chatUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "tc1614-chat",
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: chatPk.id,
    });

    // An Anthropic upstream for the opposite direction; `api_base` is the
    // bare host because the bridge composes `/v1/messages` itself.
    const anthropicPk = await seed.createProviderKey({
      display_name: "tc1614-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: anthropicUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: "tc1614-anthropic",
      provider: "anthropic",
      model_name: "claude-3-haiku-20240307",
      provider_key_id: anthropicPk.id,
    });

    // Seeded last: the key authenticating implies the whole seed set is
    // in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["tc1614-chat", "tc1614-anthropic"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await chatUpstream?.close();
    await anthropicUpstream?.close();
  });

  async function ready(): Promise<void> {
    const probe = new ProxyClient(app!.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const r = await probe.listModels();
      return r.status === 200;
    });
  }

  test("/v1/responses: the Codex compaction shape reaches a chat upstream with neither key", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) {
      ctx.skip();
      return;
    }
    await ready();

    const baseline = chatUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      // Byte-for-byte the shape the Codex CLI sends when it compacts its
      // context: an empty tool list next to a tool choice.
      body: JSON.stringify({
        model: "tc1614-chat",
        input: "Summarise: hello",
        max_output_tokens: 32,
        tools: [],
        tool_choice: "auto",
        parallel_tool_calls: false,
      }),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as { status?: unknown };
    expect(body.status).toBe("completed");

    const sent = lastBodyOn(chatUpstream, baseline, "/v1/chat/completions");
    expect(sent).not.toHaveProperty("tool_choice");
    expect(sent).not.toHaveProperty("tools");
  });

  test("/v1/messages: an Anthropic tool_choice whose tools all filter out does not reach a chat upstream", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) {
      ctx.skip();
      return;
    }
    await ready();

    const baseline = chatUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "tc1614-chat",
        max_tokens: 32,
        messages: [{ role: "user", content: "Summarise: hello" }],
        // A non-empty list that translates to nothing — the Anthropic →
        // chat converter keeps only entries carrying a `name`. Both keys
        // must therefore be absent upstream, which an empty list here
        // would not have proved.
        tools: [{ description: "no name" }],
        tool_choice: { type: "auto" },
      }),
    });

    expect(res.status).toBe(200);
    const sent = lastBodyOn(chatUpstream, baseline, "/v1/chat/completions");
    expect(sent).not.toHaveProperty("tool_choice");
    expect(sent).not.toHaveProperty("tools");
  });

  test("/v1/chat/completions: an empty tool list drops the choice on the way to Anthropic", async (ctx) => {
    if (!etcdReachable || !app || !anthropicUpstream) {
      ctx.skip();
      return;
    }
    await ready();

    const baseline = anthropicUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "tc1614-anthropic",
        max_tokens: 32,
        messages: [{ role: "user", content: "Summarise: hello" }],
        tools: [],
        tool_choice: "auto",
      }),
    });

    expect(res.status).toBe(200);
    const sent = lastBodyOn(anthropicUpstream, baseline, "/v1/messages");
    expect(sent).not.toHaveProperty("tool_choice");
    expect(sent).not.toHaveProperty("tools");
  });
});
