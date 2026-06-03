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

// E2E: tool / function calling cross-provider translation. Per
// gateway docs `docs/api-proxy.md` §6 outbound-axis table:
//
//   > Anthropic | /v1/messages | /v1/chat/completions (full translation)
//   > | The Hub maps content blocks ↔ messages, **tool_use ↔ tool_calls**,
//   > | system extraction, cache_control passthrough, stop_reason
//   > | normalisation
//
// User journey: a caller using the OpenAI SDK targets a Model whose
// provider is `anthropic`. The caller passes OpenAI-shape `tools`
// in the request. The gateway must:
//
//   1. Translate OpenAI `tools` → Anthropic `tools` on the way out
//      (`function.name/description/parameters` → `name/description/
//      input_schema`).
//   2. Translate Anthropic `content: [{type: "tool_use", id, name,
//      input}]` → OpenAI `tool_calls: [{id, type: "function",
//      function: {name, arguments}}]` on the way back.
//   3. Translate Anthropic `stop_reason: "tool_use"` → OpenAI
//      `finish_reason: "tool_calls"`.
//
// This is the cornerstone of agent-style workflows: every modern
// agent framework (LangChain, LlamaIndex, Vercel AI SDK, etc.)
// drives tool dispatch via the OpenAI tool-call shape, so a
// regression here breaks every agent built on top of an
// Anthropic-backed gateway Model.
//
// Prior to this file, the gateway had **zero** e2e coverage on
// cross-provider tool translation — only Rust unit tests at the
// translator level.
//
// References:
// - OpenAI Chat Completions tools spec
//   <https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools>
// - Anthropic Messages tools spec
//   <https://docs.anthropic.com/en/api/messages#parameter-tools>
// - Gateway's own translation contract: `docs/api-proxy.md` §6

const CALLER_PLAINTEXT = "sk-tools-xprov-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("tools cross-provider e2e: OpenAI tools → Anthropic upstream tool_use → OpenAI tool_calls back", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Mock Anthropic upstream returns a Messages-shape response
    // where the assistant chose to call `get_weather`. The body
    // shape mirrors Anthropic's documented tool_use response per
    // <https://docs.anthropic.com/en/api/messages#example-of-tool-use>.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_tool_use_01",
        type: "message",
        role: "assistant",
        model: "claude-3-5-sonnet-20241022",
        content: [
          {
            type: "tool_use",
            id: "toolu_xprov_01",
            name: "get_weather",
            input: { location: "San Francisco, CA", unit: "celsius" },
          },
        ],
        stop_reason: "tool_use",
        usage: { input_tokens: 12, output_tokens: 8 },
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Anthropic api_base = bare host; bridge composes /v1/messages
    // (mirrors anthropic-upstream-e2e convention).
    const pk = await admin.createProviderKey({
      display_name: "tools-xprov-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "tools-xprov",
      provider: "anthropic",
      model_name: "claude-3-5-sonnet-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["tools-xprov"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("OpenAI tools spec → Anthropic tools_use upstream → OpenAI tool_calls back to caller", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Readiness gate: same probe pattern as anthropic-upstream-e2e.
    // Probe doesn't include tools (the propagation we're gating on
    // is just snapshot loading; tool translation is a request-level
    // concern not a propagation concern).
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "tools-xprov",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    // OpenAI-shape tools spec per
    // <https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools>.
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "get_weather",
          description: "Get the current weather in a given location.",
          parameters: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "The city and state, e.g. San Francisco, CA",
              },
              unit: {
                type: "string",
                enum: ["celsius", "fahrenheit"],
              },
            },
            required: ["location"],
          },
        },
      },
    ];

    const completion = await client.chat.completions.create({
      model: "tools-xprov",
      messages: [{ role: "user", content: "What's the weather in SF?" }],
      tools,
    });

    // ── Caller-side: response is OpenAI-shape with tool_calls ────

    // The assistant message carries `tool_calls`, not `content` text.
    // Per OpenAI Chat Completions spec, when finish_reason is
    // "tool_calls" the message.content is null and tool_calls
    // populated.
    expect(completion.choices[0]?.message.role).toBe("assistant");
    expect(completion.choices[0]?.message.tool_calls).toBeDefined();
    expect(completion.choices[0]?.message.tool_calls).toHaveLength(1);

    const toolCall = completion.choices[0]?.message.tool_calls?.[0];
    // OpenAI tool_call.id propagates from upstream's tool_use.id
    // (Anthropic uses `toolu_...` prefix). Pin the exact upstream
    // value — a regression that re-issued ids would break agent
    // frameworks that submit tool_results referencing the original
    // id.
    expect(toolCall?.id).toBe("toolu_xprov_01");
    expect(toolCall?.type).toBe("function");
    expect(toolCall?.function.name).toBe("get_weather");
    // OpenAI's `arguments` is a JSON-encoded STRING (not an object),
    // unlike Anthropic's `input` which is a parsed JSON object.
    // The translation must encode upstream's input → JSON string.
    const args = JSON.parse(toolCall!.function.arguments) as {
      location?: unknown;
      unit?: unknown;
    };
    expect(args.location).toBe("San Francisco, CA");
    expect(args.unit).toBe("celsius");

    // OpenAI finish_reason "tool_calls" comes from translating
    // Anthropic stop_reason "tool_use". A regression that left
    // "tool_use" through would break any caller doing
    // `if (finish_reason === "tool_calls") { ... }`.
    expect(completion.choices[0]?.finish_reason).toBe("tool_calls");

    // ── Upstream-side: the gateway spoke Anthropic Messages with
    //    Anthropic-shape tools spec ────────────────────────────────

    const messagesReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(messagesReq).toBeDefined();

    const sentBody = JSON.parse(messagesReq!.body) as {
      tools?: Array<{
        name?: string;
        description?: string;
        input_schema?: { type?: string; properties?: unknown; required?: unknown };
      }>;
      messages?: Array<{ role?: string; content?: unknown }>;
    };

    // Tools translation: OpenAI's nested {type, function: {name,
    // description, parameters}} flattens to Anthropic's {name,
    // description, input_schema}. A regression that left the
    // OpenAI nested shape on the wire would 400 against real
    // Anthropic.
    expect(sentBody.tools).toHaveLength(1);
    expect(sentBody.tools?.[0]?.name).toBe("get_weather");
    expect(sentBody.tools?.[0]?.description).toBe(
      "Get the current weather in a given location.",
    );
    // The JSON Schema in `parameters` becomes `input_schema`
    // verbatim per Anthropic's tools doc.
    expect(sentBody.tools?.[0]?.input_schema?.type).toBe("object");
    expect(sentBody.tools?.[0]?.input_schema?.required).toEqual(["location"]);

    // Caller's user message reaches upstream intact.
    expect(sentBody.messages?.[0]?.role).toBe("user");
  });

  test("agent-loop turn 2: caller posts {role:'tool', tool_call_id} → Anthropic gets tool_result", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // After turn 1 (user → assistant tool_use), the OpenAI-shape
    // agent loop posts the tool's output via {role:"tool",
    // tool_call_id, content}. Without translation the Anthropic
    // bridge would 400. With translation the upstream receives
    // {role:"user", content:[{type:"tool_result", tool_use_id, content}]}
    // and the conversation continues.
    const baseline = upstream.receivedRequests.length;
    await client.chat.completions.create({
      model: "tools-xprov",
      messages: [
        { role: "user", content: "What's the weather in SF?" },
        // (real callers would also include the assistant's
        // tool_calls turn here; we skip it because Anthropic only
        // requires the tool_use_id to be referenced in the next
        // user-side tool_result.)
        {
          role: "tool",
          tool_call_id: "toolu_xprov_01",
          content: "72F, sunny",
        },
      ],
    });

    const messagesReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(messagesReq).toBeDefined();

    const sentBody = JSON.parse(messagesReq!.body) as {
      messages?: Array<{
        role?: string;
        content?: Array<{
          type?: string;
          tool_use_id?: string;
          content?: string;
        }>;
      }>;
    };
    // Last message should be a user-role turn with a tool_result
    // content block (Anthropic's shape per docs).
    const lastMsg = sentBody.messages?.[sentBody.messages.length - 1];
    expect(lastMsg?.role).toBe("user");
    const block = lastMsg?.content?.[0];
    expect(block?.type).toBe("tool_result");
    expect(block?.tool_use_id).toBe("toolu_xprov_01");
    expect(block?.content).toBe("72F, sunny");
  });

  test("tool_choice translates from OpenAI shape to Anthropic shape", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const baseline = upstream.receivedRequests.length;
    await client.chat.completions.create({
      model: "tools-xprov",
      messages: [{ role: "user", content: "Call something." }],
      // OpenAI's most-common tool_choice value. Forwarded
      // verbatim to Anthropic, this would 400 (Anthropic expects
      // an object, not a string).
      tool_choice: "auto",
      tools: [
        {
          type: "function",
          function: {
            name: "noop",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    });

    const messagesReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(messagesReq).toBeDefined();

    const sentBody = JSON.parse(messagesReq!.body) as {
      tool_choice?: { type?: string };
    };
    expect(sentBody.tool_choice).toEqual({ type: "auto" });
  });
});
