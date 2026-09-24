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

// E2E for the tool half of the `/v1/responses` → chat-completions bridge,
// and for its counterpart on the chat → Anthropic converter.
//
// Three contracts are pinned here.
//
// 1. Every tool parameter a Responses caller sets reaches the upstream in a
//    shape a plain OpenAI-compatible endpoint accepts: a `custom` (freeform)
//    tool as a function tool, `parallel_tool_calls` verbatim, and an
//    `allowed_tools` choice narrowed to its bare mode. All three used to be
//    dropped, so a caller that asked for a forced call to one of two tools
//    got an unconstrained answer instead.
//
// 2. A tool result that is a JSON object reaches the upstream as that JSON
//    serialised. The chat `tool` role carries a string, and the bridge used
//    to render anything that was not a string as an empty one — the model
//    saw a tool that had returned nothing.
//
// 3. `parallel_tool_calls` survives the chat → Anthropic converter, which
//    has no top-level field of that name: it becomes
//    `tool_choice.disable_parallel_tool_use`. Flattened onto the body as it
//    was, Anthropic rejects it as an unknown parameter.
//
// The chat bridge is reached by declaring an EMPTY `apis` map on the
// provider key — the operator saying "this OpenAI-compatible endpoint has
// no `/v1/responses`".

const CALLER_PLAINTEXT = "sk-responses-bridge-tool-parameters";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

const CHAT_REPLY = {
  id: "chatcmpl-tool-parameters",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "relay-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "done" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 },
};

const ANTHROPIC_REPLY = {
  id: "msg_tool_parameters",
  type: "message",
  role: "assistant",
  model: "claude-3-haiku-20240307",
  content: [{ type: "text", text: "done" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 9, output_tokens: 3 },
};

/** Parse the last request the mock received on `path` after `baseline`. */
function lastBodyOn(
  upstream: OpenAiUpstream,
  baseline: number,
  path: string,
): Record<string, any> {
  const calls = upstream.receivedRequests
    .slice(baseline)
    .filter((r) => r.path === path);
  expect(calls.length).toBeGreaterThan(0);
  return JSON.parse(calls.at(-1)!.body) as Record<string, any>;
}

describe("bridged tool parameters: custom tools, tool_choice forms, parallel_tool_calls", () => {
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

    const chatPk = await seed.createProviderKey({
      display_name: "tool-params-chat-pk",
      secret: "sk-mock",
      api_base: `${chatUpstream.baseUrl}/v1`,
      apis: {},
    });
    await seed.createModel({
      display_name: "tool-params-chat",
      provider: "openai",
      model_name: "relay-compat-x",
      provider_key_id: chatPk.id,
    });

    // `api_base` is the bare host: the Anthropic bridge composes
    // `/v1/messages` itself.
    const anthropicPk = await seed.createProviderKey({
      display_name: "tool-params-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: anthropicUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: "tool-params-anthropic",
      provider: "anthropic",
      model_name: "claude-3-haiku-20240307",
      provider_key_id: anthropicPk.id,
    });

    // Seeded last: the key authenticating implies the whole seed set is
    // in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["tool-params-chat", "tool-params-anthropic"],
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

  test("/v1/responses: a custom tool, parallel_tool_calls and an allowed_tools choice all reach the chat upstream", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) {
      ctx.skip();
      return;
    }
    await ready();

    const baseline = chatUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "tool-params-chat",
        input: "patch the file",
        max_output_tokens: 32,
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Look up the weather",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
          {
            type: "custom",
            name: "apply_patch",
            description: "Edit a file",
            format: { type: "grammar", syntax: "lark", definition: "start: TEXT" },
          },
        ],
        parallel_tool_calls: false,
        tool_choice: {
          type: "allowed_tools",
          mode: "required",
          tools: [{ type: "function", name: "get_weather" }],
        },
      }),
    });
    expect(res.status).toBe(200);

    const sent = lastBodyOn(chatUpstream, baseline, "/v1/chat/completions");

    // Both tools arrive as function tools — the custom one taking the
    // single string that stands in for its freeform input. Pre-fix the
    // list held only `get_weather`.
    expect(sent.tools).toHaveLength(2);
    expect(sent.tools[0].function.name).toBe("get_weather");
    expect(sent.tools[1]).toEqual({
      type: "function",
      function: {
        name: "apply_patch",
        description:
          "Edit a file\n\nFormat:\n```lark\nstart: TEXT\n```",
        parameters: {
          type: "object",
          properties: {
            content: {
              type: "string",
              description: "The apply_patch content following the specified format",
            },
          },
          required: ["content"],
        },
      },
    });

    // Pre-fix both of these were absent: the request was sent with no
    // constraint on which tool the model called, or how many at once.
    expect(sent.parallel_tool_calls).toBe(false);
    expect(sent.tool_choice).toBe("required");
  });

  test("/v1/responses: a function_call_output that is a JSON object reaches the upstream as a JSON string", async (ctx) => {
    if (!etcdReachable || !app || !chatUpstream) {
      ctx.skip();
      return;
    }
    await ready();

    const baseline = chatUpstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "tool-params-chat",
        max_output_tokens: 32,
        input: [
          { role: "user", content: "weather in Paris?" },
          {
            type: "function_call",
            call_id: "call_1",
            name: "get_weather",
            arguments: '{"city":"Paris"}',
          },
          {
            type: "function_call_output",
            call_id: "call_1",
            output: { temp_c: 21, conditions: "sunny" },
          },
        ],
      }),
    });
    expect(res.status).toBe(200);

    const sent = lastBodyOn(chatUpstream, baseline, "/v1/chat/completions");
    const toolTurn = sent.messages.find((m: any) => m.role === "tool");
    // Pre-fix this was "" — the model was told the tool returned nothing.
    expect(JSON.parse(toolTurn.content)).toEqual({
      temp_c: 21,
      conditions: "sunny",
    });
    expect(toolTurn.tool_call_id).toBe("call_1");
  });

  test("/v1/chat/completions: parallel_tool_calls false becomes disable_parallel_tool_use on the way to Anthropic", async (ctx) => {
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
        model: "tool-params-anthropic",
        max_tokens: 32,
        messages: [{ role: "user", content: "weather in Paris?" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              parameters: {
                type: "object",
                properties: { city: { type: "string" } },
              },
            },
          },
        ],
        parallel_tool_calls: false,
      }),
    });
    expect(res.status).toBe(200);

    const sent = lastBodyOn(anthropicUpstream, baseline, "/v1/messages");
    // Anthropic rejects unknown top-level parameters, so the OpenAI key
    // must not survive — it is re-expressed on the choice instead, which
    // the caller sent none of and so defaults to Anthropic's own.
    expect(sent).not.toHaveProperty("parallel_tool_calls");
    expect(sent.tool_choice).toEqual({
      type: "auto",
      disable_parallel_tool_use: true,
    });
  });
});
