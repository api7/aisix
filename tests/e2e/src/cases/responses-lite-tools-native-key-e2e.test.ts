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
import type { OpenAiUpstreamOptions } from "../harness/upstream-openai.js";

// Codex's lite mode declares its tools in an `additional_tools` input item
// of a `/v1/responses` request. Only OpenAI's own Responses API reads that
// item type; another vendor's native Responses endpoint ignores it, and the
// model then sees no tools at all. So on a key whose provider is neither
// OpenAI, Azure OpenAI nor bring-your-own, such a request goes through the
// chat-completions bridge (which reads the item) even when the key declares
// a native Responses surface. Every other request keeps the native path.

const CALLER = "sk-responses-lite-tools-native-key";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const EXEC_TOOL = {
  type: "custom",
  name: "exec",
  description: "Run a shell command.",
  format: { type: "grammar", syntax: "lark", definition: "start: /.+/" },
};
const ADDITIONAL_TOOLS = {
  type: "additional_tools",
  role: "developer",
  tools: [{ type: "namespace", name: "functions", description: "Agent tools.", tools: [EXEC_TOOL] }],
};
const USER_TURN = { type: "message", role: "user", content: [{ type: "input_text", text: "list the files" }] };
const LITE_INPUT = [ADDITIONAL_TOOLS, USER_TURN];

const CHAT_BODY = {
  id: "chatcmpl-lite",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "vendor-model",
  choices: [{ index: 0, message: { role: "assistant", content: "done" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 12, completion_tokens: 1, total_tokens: 13 },
};
const CHAT_STREAM = [
  JSON.stringify({
    id: "chatcmpl-lite-s",
    object: "chat.completion.chunk",
    model: "vendor-model",
    choices: [{ index: 0, delta: { role: "assistant", content: "done" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-lite-s",
    object: "chat.completion.chunk",
    model: "vendor-model",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 12, completion_tokens: 1, total_tokens: 13 },
  }),
  "[DONE]",
];
const RESPONSES_BODY = {
  id: "resp_native",
  object: "response",
  created_at: 1_700_000_000,
  status: "completed",
  model: "vendor-model",
  output: [
    {
      type: "message",
      id: "msg_native",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "done", annotations: [] }],
    },
  ],
  usage: { input_tokens: 12, output_tokens: 1, total_tokens: 13 },
};

interface KeySpec {
  upstream: OpenAiUpstreamOptions;
  provider: string;
  adapter: string;
}

// Each model gets its own key and its own upstream, so the path and body
// an upstream received belong to exactly one model.
const MODELS: Record<string, KeySpec> = {
  deepseek: { upstream: { nonStreamBody: CHAT_BODY }, provider: "deepseek", adapter: "openai" },
  "deepseek-stream": { upstream: { streamEvents: CHAT_STREAM }, provider: "deepseek", adapter: "openai" },
  "deepseek-plain": { upstream: { nonStreamBody: RESPONSES_BODY }, provider: "deepseek", adapter: "openai" },
  openai: { upstream: { nonStreamBody: RESPONSES_BODY }, provider: "openai", adapter: "openai" },
  byo: { upstream: { nonStreamBody: RESPONSES_BODY }, provider: "byo", adapter: "openai" },
  "member-openai": {
    upstream: { status: 500, errorBody: { error: { message: "upstream down", type: "server_error" } } },
    provider: "openai",
    adapter: "openai",
  },
  "member-deepseek": { upstream: { nonStreamBody: CHAT_BODY }, provider: "deepseek", adapter: "openai" },
};
const WILDCARD_UPSTREAM = "wildcard";

describe("/v1/responses lite-mode tools on a key with a native Responses surface", () => {
  let app: SpawnedApp | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const keyFor = async (name: string, spec: KeySpec) => {
      const upstream = await startOpenAiUpstream(spec.upstream);
      upstreams[name] = upstream;
      return seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        provider: spec.provider,
        adapter: spec.adapter,
        apis: { responses: {} },
      });
    };
    for (const [model, spec] of Object.entries(MODELS)) {
      const pk = await keyFor(model, spec);
      await seed.createModel({
        display_name: model,
        provider: spec.provider,
        model_name: "vendor-model",
        provider_key_id: pk.id,
      });
    }
    const wildPk = await keyFor(WILDCARD_UPSTREAM, {
      upstream: { nonStreamBody: CHAT_BODY },
      provider: "deepseek",
      adapter: "openai",
    });
    await seed.createModel({
      display_name: "ds-wild/*",
      provider: "deepseek",
      model_name: "*",
      provider_key_id: wildPk.id,
    });
    await seed.createModel({
      display_name: "group-openai-then-deepseek",
      routing: {
        strategy: "failover",
        targets: [{ model: "member-openai" }, { model: "member-deepseek" }],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of Object.values(upstreams)) await u.close();
  });

  async function responses(body: Record<string, unknown>): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  function received(name: string): { path: string; body: Record<string, any> }[] {
    return upstreams[name]!.receivedRequests.map((r) => ({ path: r.path, body: JSON.parse(r.body) }));
  }

  function expectBridgedWithTools(sent: { path: string; body: Record<string, any> }): void {
    expect(sent.path).toBe("/v1/chat/completions");
    expect((sent.body.tools ?? []).map((t: { function: { name: string } }) => t.function.name)).toEqual([
      "functions__exec",
    ]);
    expect(JSON.stringify(sent.body.messages)).not.toContain("additional_tools");
  }

  function expectNativeVerbatim(sent: { path: string; body: Record<string, any> }): void {
    expect(sent.path).toBe("/v1/responses");
    expect(sent.body.input).toEqual(LITE_INPUT);
    expect(sent.body.tools).toBeUndefined();
  }

  test("a non-OpenAI vendor key gets the request bridged, with the tools", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "deepseek", input: LITE_INPUT });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, any>;
    expect(body.object).toBe("response");
    const sent = received("deepseek");
    expect(sent).toHaveLength(1);
    expectBridgedWithTools(sent[0]!);
  });

  test("streaming: a non-OpenAI vendor key gets the request bridged, with the tools", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "deepseek-stream", stream: true, input: LITE_INPUT });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("response.completed");
    const sent = received("deepseek-stream");
    expect(sent).toHaveLength(1);
    expectBridgedWithTools(sent[0]!);
    expect(sent[0]!.body.stream).toBe(true);
  });

  test("a wildcard row on a non-OpenAI vendor key bridges with the caller's model name", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "ds-wild/deepseek-chat", input: LITE_INPUT });
    expect(res.status).toBe(200);
    const sent = received(WILDCARD_UPSTREAM);
    expect(sent).toHaveLength(1);
    expectBridgedWithTools(sent[0]!);
    expect(sent[0]!.body.model).toBe("deepseek-chat");
  });

  test("a request without additional_tools keeps the native path on the same kind of key", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const tools = [{ type: "function", name: "lookup", parameters: { type: "object" } }];
    const res = await responses({ model: "deepseek-plain", input: [USER_TURN], tools });
    expect(res.status).toBe(200);
    const sent = received("deepseek-plain");
    expect(sent).toHaveLength(1);
    expect(sent[0]!.path).toBe("/v1/responses");
    expect(sent[0]!.body.input).toEqual([USER_TURN]);
    expect(sent[0]!.body.tools).toEqual(tools);
  });

  test("an OpenAI key keeps the native path with additional_tools unchanged", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "openai", input: LITE_INPUT });
    expect(res.status).toBe(200);
    const sent = received("openai");
    expect(sent).toHaveLength(1);
    expectNativeVerbatim(sent[0]!);
  });

  test("a bring-your-own key keeps the native path with additional_tools unchanged", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "byo", input: LITE_INPUT });
    expect(res.status).toBe(200);
    const sent = received("byo");
    expect(sent).toHaveLength(1);
    expectNativeVerbatim(sent[0]!);
  });

  test("a group fails over from an OpenAI member sent natively to a vendor member sent bridged", async (ctx) => {
    if (!etcdReachable || !app) return void ctx.skip();
    const res = await responses({ model: "group-openai-then-deepseek", input: LITE_INPUT });
    expect(res.status).toBe(200);
    const first = received("member-openai");
    expect(first).toHaveLength(1);
    expectNativeVerbatim(first[0]!);
    const second = received("member-deepseek");
    expect(second).toHaveLength(1);
    expectBridgedWithTools(second[0]!);
  });
});
