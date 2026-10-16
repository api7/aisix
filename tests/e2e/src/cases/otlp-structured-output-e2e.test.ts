import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// E2E for ticket-analysis#1051: under `content_mode: full` the OTLP export's
// `gen_ai.output.messages` reports what the caller received the way the
// OpenTelemetry GenAI conventions (v1.41) shape it — one assistant message
// whose ordered parts keep reasoning, text and tool calls apart — and
// `gen_ai.tool.definitions` carries each tool's description and parameter
// schema. Driven through every family that carries an assistant turn, both
// streamed and buffered, both served natively and bridged from another
// protocol, against a real `aisix` binary and mock upstreams.
//
// An Aliyun-SLS exporter in full mode rides alongside: its flat `response`
// column must keep reading exactly what it always did.

const CALLER = "sk-otlp-structured-output";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "structured-output";

const REASONING = "The user wants the weather. ";
const ANSWER = "Checking the weather.";
const TOOL = "get_weather";
const ARGS = { city: "Paris" };

const CHAT_TOOLS = [
  {
    type: "function",
    function: {
      name: TOOL,
      description: "Look up the current weather",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
  },
];
const ANTHROPIC_TOOLS = [
  {
    name: TOOL,
    description: "Look up the current weather",
    input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  },
];

const EXPECTED_PARTS = [
  { type: "reasoning", content: REASONING },
  { type: "text", content: ANSWER },
  { type: "tool_call", id: "call_wx", name: TOOL, arguments: ARGS },
];

function chatChunk(delta: Record<string, unknown>, extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    id: "chatcmpl-structured",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: null }],
    ...extra,
  });
}

/** A chat stream: reasoning, then text, then one tool call in two pieces. */
const CHAT_STREAM = [
  chatChunk({ role: "assistant", reasoning_content: REASONING }),
  chatChunk({ content: ANSWER }),
  chatChunk({
    tool_calls: [
      { index: 0, id: "call_wx", type: "function", function: { name: TOOL, arguments: '{"city":' } },
    ],
  }),
  chatChunk({ tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }] }),
  JSON.stringify({
    id: "chatcmpl-structured",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
  }),
  JSON.stringify({
    id: "chatcmpl-structured",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [],
    usage: { prompt_tokens: 9, completion_tokens: 12, total_tokens: 21 },
  }),
  "[DONE]",
];

const CHAT_BODY = {
  id: "chatcmpl-structured-2",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        reasoning_content: REASONING,
        content: ANSWER,
        tool_calls: [
          { id: "call_wx", type: "function", function: { name: TOOL, arguments: JSON.stringify(ARGS) } },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 9, completion_tokens: 12, total_tokens: 21 },
};

const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

/** An Anthropic stream: thinking (signed), a redacted block, text, tool_use. */
const ANTHROPIC_STREAM = [
  sse("message_start", {
    type: "message_start",
    message: {
      id: "msg_structured",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-5",
      content: [],
      usage: { input_tokens: 9, output_tokens: 1 },
    },
  }),
  sse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }),
  sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: REASONING } }),
  sse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "sig" } }),
  sse("content_block_stop", { type: "content_block_stop", index: 0 }),
  sse("content_block_start", { type: "content_block_start", index: 1, content_block: { type: "redacted_thinking", data: "REDACTED-BLOB" } }),
  sse("content_block_stop", { type: "content_block_stop", index: 1 }),
  sse("content_block_start", { type: "content_block_start", index: 2, content_block: { type: "text", text: "" } }),
  sse("content_block_delta", { type: "content_block_delta", index: 2, delta: { type: "text_delta", text: ANSWER } }),
  sse("content_block_stop", { type: "content_block_stop", index: 2 }),
  sse("content_block_start", {
    type: "content_block_start",
    index: 3,
    content_block: { type: "tool_use", id: "call_wx", name: TOOL, input: {} },
  }),
  sse("content_block_delta", { type: "content_block_delta", index: 3, delta: { type: "input_json_delta", partial_json: '{"city":' } }),
  sse("content_block_delta", { type: "content_block_delta", index: 3, delta: { type: "input_json_delta", partial_json: '"Paris"}' } }),
  sse("content_block_stop", { type: "content_block_stop", index: 3 }),
  sse("message_delta", { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 12 } }),
  sse("message_stop", { type: "message_stop" }),
];

const REASONING_ITEM = {
  id: "rs_1",
  type: "reasoning",
  summary: [{ type: "summary_text", text: REASONING }],
  encrypted_content: "ENCRYPTED-REASONING",
};
const MESSAGE_ITEM = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text: ANSWER, annotations: [] }],
};
const CALL_ITEM = {
  id: "fc_1",
  type: "function_call",
  call_id: "call_wx",
  name: TOOL,
  arguments: JSON.stringify(ARGS),
  status: "completed",
};
const RESPONSES_USAGE = { input_tokens: 9, output_tokens: 12, total_tokens: 21 };

/** A Responses stream: reasoning summary, message text, a function call. */
const RESPONSES_STREAM = [
  sse("response.created", { type: "response.created", response: { id: "resp_1", status: "in_progress", output: [] } }),
  sse("response.output_item.added", { type: "response.output_item.added", output_index: 0, item: { ...REASONING_ITEM, summary: [] } }),
  sse("response.reasoning_summary_text.delta", { type: "response.reasoning_summary_text.delta", output_index: 0, summary_index: 0, delta: REASONING }),
  sse("response.output_item.done", { type: "response.output_item.done", output_index: 0, item: REASONING_ITEM }),
  sse("response.output_item.added", { type: "response.output_item.added", output_index: 1, item: { ...MESSAGE_ITEM, content: [] } }),
  sse("response.output_text.delta", { type: "response.output_text.delta", output_index: 1, content_index: 0, delta: ANSWER }),
  sse("response.output_item.done", { type: "response.output_item.done", output_index: 1, item: MESSAGE_ITEM }),
  sse("response.output_item.added", { type: "response.output_item.added", output_index: 2, item: { ...CALL_ITEM, arguments: "" } }),
  sse("response.function_call_arguments.delta", { type: "response.function_call_arguments.delta", output_index: 2, delta: '{"city":"Paris"}' }),
  sse("response.output_item.done", { type: "response.output_item.done", output_index: 2, item: CALL_ITEM }),
  sse("response.completed", {
    type: "response.completed",
    response: { id: "resp_1", status: "completed", output: [REASONING_ITEM, MESSAGE_ITEM, CALL_ITEM], usage: RESPONSES_USAGE },
  }),
];

const RESPONSES_INCOMPLETE_BODY = {
  id: "resp_3",
  object: "response",
  status: "incomplete",
  incomplete_details: { reason: "max_output_tokens" },
  model: "gpt-4o-mini",
  output: [MESSAGE_ITEM],
  usage: RESPONSES_USAGE,
};

const RESPONSES_BODY = {
  id: "resp_2",
  object: "response",
  status: "completed",
  model: "gpt-4o-mini",
  output: [REASONING_ITEM, MESSAGE_ITEM, CALL_ITEM],
  usage: RESPONSES_USAGE,
};

const KIND_SERVER = 2;

describe("OTLP output messages keep reasoning, text and tool calls apart", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let otlp: MockOtlp | undefined;
  let sls: MockSls | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstreams.chatStream = await startOpenAiUpstream({ streamEvents: CHAT_STREAM });
    upstreams.chatBody = await startOpenAiUpstream({ nonStreamBody: CHAT_BODY });
    upstreams.anthropic = await startOpenAiUpstream({ rawStreamFrames: ANTHROPIC_STREAM });
    upstreams.responsesStream = await startOpenAiUpstream({ rawStreamFrames: RESPONSES_STREAM });
    upstreams.responsesBody = await startOpenAiUpstream({ nonStreamBody: RESPONSES_BODY });
    upstreams.responsesIncomplete = await startOpenAiUpstream({ nonStreamBody: RESPONSES_INCOMPLETE_BODY });
    otlp = await startMockOtlp();
    sls = await startMockSls();

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "structured-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
      content_max_bytes: 65536,
    });
    await seed.createObservabilityExporter({
      name: "structured-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
    });
    const model = async (name: string, upstream: OpenAiUpstream, pk: Record<string, unknown> = {}) => {
      const key = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
        ...pk,
      });
      await seed.createModel({
        display_name: name,
        provider: (pk.provider as string | undefined) ?? "openai",
        model_name: pk.provider === "anthropic" ? "claude-sonnet-4-5" : "gpt-4o-mini",
        provider_key_id: key.id,
      });
    };
    await model("so-chat-stream", upstreams.chatStream);
    await model("so-chat-body", upstreams.chatBody);
    await model("so-anthropic", upstreams.anthropic, {
      provider: "anthropic",
      adapter: "anthropic",
      api_base: upstreams.anthropic.baseUrl,
    });
    await model("so-responses-stream", upstreams.responsesStream);
    await model("so-responses-body", upstreams.responsesBody);
    await model("so-responses-incomplete", upstreams.responsesIncomplete);
    // No `/v1/responses` on this key: the gateway bridges to chat.
    await model("so-responses-bridged", upstreams.chatStream, { apis: {} });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await sls?.close();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
  });

  async function post(path: string, body: unknown): Promise<{ status: number; requestId: string; text: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        ...(path === "/v1/messages" ? { "x-api-key": CALLER } : { authorization: `Bearer ${CALLER}` }),
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId, `${path} returned a request id`).not.toBe("");
    return { status: res.status, requestId, text };
  }

  /** The span carrying the request's GenAI attributes. */
  async function operationSpan(requestId: string): Promise<CapturedSpan> {
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      const done = otlp!.spans.some((s) => s.attributes["aisix.request_id"] === requestId && s.kind === KIND_SERVER);
      const span = otlp!.spans.find(
        (s) => s.attributes["aisix.request_id"] === requestId && "gen_ai.output.messages" in s.attributes,
      );
      if (done && span) return span;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`no span with gen_ai.output.messages for ${requestId}`);
  }

  function outputParts(span: CapturedSpan): unknown[] {
    const messages = JSON.parse(String(span.attributes["gen_ai.output.messages"]));
    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe("assistant");
    // The conventions' enum, whatever raw value the protocol reported
    // (`tool_calls`, `tool_use`, or the one derived for Responses).
    expect(messages[0].finish_reason).toBe("tool_call");
    return messages[0].parts;
  }

  async function slsRow(requestId: string): Promise<Map<string, string>> {
    return waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, "SLS row");
  }

  async function slsResponse(requestId: string): Promise<string | undefined> {
    return (await slsRow(requestId)).get("response");
  }

  test("chat, streamed: ordered parts, parsed arguments, full tool definitions", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: "so-chat-stream",
      stream: true,
      stream_options: { include_usage: true },
      tools: CHAT_TOOLS,
      messages: [{ role: "user", content: "weather in Paris?" }],
    });
    expect(status).toBe(200);
    const span = await operationSpan(requestId);
    expect(outputParts(span)).toEqual(EXPECTED_PARTS);
    expect(JSON.parse(String(span.attributes["gen_ai.tool.definitions"]))).toEqual([
      {
        type: "function",
        name: TOOL,
        description: "Look up the current weather",
        parameters: CHAT_TOOLS[0].function.parameters,
      },
    ]);
    // The flat column reads the answer text alone, as it always has.
    expect(await slsResponse(requestId)).toBe(ANSWER);
  });

  test("chat, buffered", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/chat/completions", {
      model: "so-chat-body",
      tools: CHAT_TOOLS,
      messages: [{ role: "user", content: "weather in Paris?" }],
    });
    expect(status).toBe(200);
    expect(outputParts(await operationSpan(requestId))).toEqual(EXPECTED_PARTS);
    expect(await slsResponse(requestId)).toBe(ANSWER);
  });

  test("messages, native stream: thinking and tool_use kept, redacted thinking left out", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId, text } = await post("/v1/messages", {
      model: "so-anthropic",
      max_tokens: 256,
      stream: true,
      tools: ANTHROPIC_TOOLS,
      messages: [{ role: "user", content: "weather in Paris?" }],
    });
    expect(status).toBe(200);
    expect(text).toContain("REDACTED-BLOB");
    const span = await operationSpan(requestId);
    expect(outputParts(span)).toEqual(EXPECTED_PARTS);
    expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("REDACTED-BLOB");
    expect(JSON.parse(String(span.attributes["gen_ai.tool.definitions"]))).toEqual([
      {
        type: "function",
        name: TOOL,
        description: "Look up the current weather",
        parameters: ANTHROPIC_TOOLS[0].input_schema,
      },
    ]);
    const flat = await slsResponse(requestId);
    expect(flat).toContain(ANSWER);
    expect(flat).not.toContain(REASONING);
  });

  test("messages, bridged stream from a chat upstream", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/messages", {
      model: "so-chat-stream",
      max_tokens: 256,
      stream: true,
      tools: ANTHROPIC_TOOLS,
      messages: [{ role: "user", content: "weather in Paris?" }],
    });
    expect(status).toBe(200);
    expect(outputParts(await operationSpan(requestId))).toEqual(EXPECTED_PARTS);
  });

  test("messages, bridged buffered", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/messages", {
      model: "so-chat-body",
      max_tokens: 256,
      tools: ANTHROPIC_TOOLS,
      messages: [{ role: "user", content: "weather in Paris?" }],
    });
    expect(status).toBe(200);
    expect(outputParts(await operationSpan(requestId))).toEqual(EXPECTED_PARTS);
  });

  test("responses, native stream: readable summary only, never the encrypted content", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/responses", {
      model: "so-responses-stream",
      stream: true,
      input: "weather in Paris?",
      tools: [{ type: "function", name: TOOL, description: "Look up the current weather", parameters: CHAT_TOOLS[0].function.parameters }],
    });
    expect(status).toBe(200);
    const span = await operationSpan(requestId);
    expect(outputParts(span)).toEqual(EXPECTED_PARTS);
    expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("ENCRYPTED-REASONING");
    // A Responses result names no finish reason; the event's is derived
    // from it in chat's vocabulary, and the span reports that raw value.
    expect(span.attributes["gen_ai.response.finish_reasons"]).toBe("tool_calls");
    expect((await slsRow(requestId)).get("finish_reason")).toBe("tool_calls");
    expect(JSON.parse(String(span.attributes["gen_ai.tool.definitions"]))).toEqual([
      {
        type: "function",
        name: TOOL,
        description: "Look up the current weather",
        parameters: CHAT_TOOLS[0].function.parameters,
      },
    ]);
  });

  test("responses, native buffered", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/responses", {
      model: "so-responses-body",
      input: "weather in Paris?",
    });
    expect(status).toBe(200);
    expect(outputParts(await operationSpan(requestId))).toEqual(EXPECTED_PARTS);
    expect((await slsRow(requestId)).get("finish_reason")).toBe("tool_calls");
  });

  test("responses, incomplete for max_output_tokens: finish_reason length", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/responses", {
      model: "so-responses-incomplete",
      input: "weather in Paris?",
    });
    expect(status).toBe(200);
    expect((await slsRow(requestId)).get("finish_reason")).toBe("length");
    const span = await operationSpan(requestId);
    const messages = JSON.parse(String(span.attributes["gen_ai.output.messages"]));
    expect(messages[0].finish_reason).toBe("length");
    expect(messages[0].parts).toEqual([{ type: "text", content: ANSWER }]);
  });

  test("responses, bridged stream from a chat upstream", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { status, requestId } = await post("/v1/responses", {
      model: "so-responses-bridged",
      stream: true,
      input: "weather in Paris?",
      tools: [{ type: "function", name: TOOL, parameters: CHAT_TOOLS[0].function.parameters }],
    });
    expect(status).toBe(200);
    expect(outputParts(await operationSpan(requestId))).toEqual(EXPECTED_PARTS);
    expect((await slsRow(requestId)).get("finish_reason")).toBe("tool_calls");
  });
});
