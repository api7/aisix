import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  spawnApp,
  waitConfigPropagation,
  waitForLogLine,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the Amazon Bedrock provider against a mock bedrock-runtime host.
//
// - Grok and OpenAI GPT 5.6+ addressed through a cross-region inference
//   profile are served by bedrock-runtime's OpenAI-compatible Chat
//   Completions endpoint (SigV4-signed), and fall back to Converse when the
//   request needs something only Converse offers.
// - Converse requests follow the chat request: alternating roles, stop
//   sequences, the output cap, `response_format`, `reasoning_effort` per
//   publisher, Bedrock-native request blocks, model-specific fields, media
//   inside tool results, and ARN model ids.
// - The caller receives the upstream's own error message, on Bedrock and on
//   the other bridges.
// - `x-amzn-requestid` is the recorded provider request id.
// - Cohere embeddings take the caller's `input_type`.

const CALLER = "sk-bedrock-converse-cc-e2e";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const PNG_1PX =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

type Received = { path: string; headers: Record<string, string>; raw: string; body: any };
type Reply = { status?: number; headers?: Record<string, string>; body: unknown };
type Responder = (r: Received) => Reply;

const CONVERSE_TEXT = (text: string) => ({
  output: { message: { role: "assistant", content: [{ text }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 10, outputTokens: 3, totalTokens: 13 },
  metrics: { latencyMs: 5 },
});

const CONVERSE_STREAM_TEXT = (text: string) =>
  eventStream([
    ["messageStart", { role: "assistant" }],
    ["contentBlockDelta", { contentBlockIndex: 0, delta: { text } }],
    ["contentBlockStop", { contentBlockIndex: 0 }],
    ["messageStop", { stopReason: "end_turn" }],
    ["metadata", { usage: { inputTokens: 10, outputTokens: 3, totalTokens: 13 }, metrics: { latencyMs: 5 } }],
  ]);

const CHAT_COMPLETION = {
  id: "chatcmpl-bedrock-cc",
  object: "chat.completion",
  model: "global.openai.gpt-6.1-sol",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "call_0", type: "function", function: { name: "get_weather", arguments: "{}" } },
          { id: "call_1", type: "function", function: { name: "get_weather", arguments: "{\"day\":2}" } },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 },
};

/** The default answer for each Bedrock operation the gateway calls. */
const defaultReply: Responder = (r) => {
  if (r.path.endsWith("/openai/v1/chat/completions")) {
    if (r.body.stream) {
      const frames = [
        { id: "chatcmpl-s", choices: [{ index: 0, delta: { role: "assistant", content: "Hi " } }] },
        { id: "chatcmpl-s", choices: [{ index: 0, delta: { content: "there" }, finish_reason: "stop" }] },
        { id: "chatcmpl-s", choices: [], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } },
      ];
      return {
        headers: { "content-type": "text/event-stream" },
        body: frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("") + "data: [DONE]\n\n",
      };
    }
    return { body: CHAT_COMPLETION };
  }
  if (r.path.endsWith("/converse-stream")) {
    return {
      headers: { "content-type": "application/vnd.amazon.eventstream" },
      body: CONVERSE_STREAM_TEXT("streamed answer"),
    };
  }
  if (r.path.endsWith("/converse")) return { body: CONVERSE_TEXT("converse answer") };
  if (r.path.endsWith("/invoke")) {
    if (r.body.texts) {
      return { body: { id: "e1", response_type: "embeddings_by_type", embeddings: { float: [[0.1, 0.2], [0.3, 0.4]] }, meta: { billed_units: { input_tokens: 6 } } } };
    }
    return {
      body: {
        id: "msg_bdrk_1",
        type: "message",
        role: "assistant",
        model: "claude",
        content: [{ type: "text", text: "invoke answer" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 5, output_tokens: 2 },
      },
    };
  }
  return { status: 404, body: { message: `no route ${r.path}` } };
};

describe("Amazon Bedrock: Converse, Chat Completions and errors", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;
  const received: Received[] = [];
  let respond: Responder = defaultReply;

  async function chat(body: Record<string, unknown>) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    return { status: res.status, text, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  }

  /** The requests the upstream received while `run` was in flight. */
  async function capture<T>(run: () => Promise<T>): Promise<{ result: T; sent: Received[] }> {
    const before = received.length;
    const result = await run();
    return { result, sent: received.slice(before) };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req: IncomingMessage, res) => {
      let raw = "";
      req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
      req.on("end", () => {
        let body: any = {};
        try {
          body = JSON.parse(raw || "{}");
        } catch {
          body = {};
        }
        const r: Received = {
          path: decodeURIComponent(req.url ?? ""),
          headers: Object.fromEntries(
            Object.entries(req.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(",") : (v ?? "")]),
          ),
          raw,
          body,
        };
        received.push(r);
        const reply = respond(r);
        const headers = { "content-type": "application/json", ...(reply.headers ?? {}) };
        res.writeHead(reply.status ?? 200, headers);
        res.end(
          typeof reply.body === "string" || Buffer.isBuffer(reply.body)
            ? reply.body
            : JSON.stringify(reply.body),
        );
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");
    const base = `http://127.0.0.1:${addr.port}`;

    // The access-log line the request-id case reads is an INFO line.
    app = await spawnApp({ extraEnv: { RUST_LOG: "info" } });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "bedrock-cc-e2e-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-cc-e2e",
        secret_access_key: "sk-cc-e2e",
        region: "us-east-1",
      }),
      api_base: base,
    });
    const models: Array<[string, string]> = [
      ["cc-gpt", "global.openai.gpt-6.1-sol"],
      ["cc-gpt-bare", "openai.gpt-6-sol"],
      ["cc-grok", "us.xai.grok-4.6"],
      ["cc-llama", "meta.llama3-3-70b-instruct-v1:0"],
      ["cc-gptoss", "openai.gpt-oss-120b-1:0"],
      ["cc-nova2", "us.amazon.nova-2-lite-v1:0"],
      ["cc-claude", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"],
      ["cc-arn-app", "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz"],
      [
        "cc-arn-claude",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      ],
      ["cc-cohere", "cohere.embed-v4:0"],
    ];
    for (const [alias, name] of models) {
      await seed.createModel({
        display_name: alias,
        provider: "bedrock",
        model_name: name,
        provider_key_id: pk.id,
      });
    }
    const azure = await seed.createProviderKey({
      display_name: "bedrock-cc-e2e-azure",
      provider: "azure-openai",
      adapter: "azure-openai",
      secret: "az-key",
      api_base: base,
    });
    await seed.createModel({
      display_name: "cc-azure",
      provider: "azure-openai",
      model_name: "gpt-4o-mini",
      provider_key_id: azure.id,
    });
    const openai = await seed.createProviderKey({
      display_name: "bedrock-cc-e2e-openai",
      secret: "sk-openai",
      api_base: `${base}/oai`,
    });
    await seed.createModel({
      display_name: "cc-openai",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: openai.id,
    });
    // Seeded last: this key authenticating implies the whole seed set is in
    // the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  const skip = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return true;
    }
    respond = defaultReply;
    return false;
  };

  const TOOLS = [
    { type: "function", function: { name: "get_weather", parameters: { type: "object", properties: {} } } },
  ];

  // ── Chat Completions route ──────────────────────────────────────────

  test("a GPT 5.6-class model on a cross-region profile answers through Chat Completions, signed for bedrock", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-gpt",
        messages: [{ role: "user", content: "weather?" }],
        tools: TOOLS,
        reasoning_effort: "none",
        max_tokens: 64,
        stop: ["END"],
        n: 1,
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent).toHaveLength(1);
    expect(sent[0].path).toBe("/openai/v1/chat/completions");
    expect(sent[0].headers.authorization).toMatch(/^AWS4-HMAC-SHA256 Credential=AKIA-cc-e2e\/\d{8}\/us-east-1\/bedrock\/aws4_request/);
    expect(sent[0].body.model).toBe("global.openai.gpt-6.1-sol");
    expect(sent[0].body.max_completion_tokens).toBe(64);
    expect(sent[0].body.max_tokens).toBeUndefined();
    expect(sent[0].body.stop).toBeUndefined();
    expect(sent[0].body.n).toBeUndefined();
    expect(sent[0].body.tools).toHaveLength(1);
    // The positional `call_0` / `call_1` are replaced by unique ids, one per call.
    const body = JSON.parse(result.text);
    const ids = body.choices[0].message.tool_calls.map((c: any) => c.id);
    expect(ids).toHaveLength(2);
    for (const id of ids) expect(id).toMatch(/^call_[0-9a-f]{32}$/);
    expect(ids[0]).not.toBe(ids[1]);
  });

  test("a Chat Completions error keeps the endpoint's own message and code", async (ctx) => {
    if (skip(ctx)) return;
    respond = () => ({
      status: 400,
      body: {
        error: {
          type: "invalid_request_error",
          code: "context_length_exceeded",
          message: "This model's maximum context length is 400000 tokens.",
        },
      },
    });
    const res = await chat({ model: "cc-gpt", messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(400);
    const body = JSON.parse(res.text);
    expect(body.error.message).toBe("This model's maximum context length is 400000 tokens.");
    expect(body.error.code).toBe("context_length_exceeded");
  });

  test("a GPT model while it reasons drops the sampling parameters AWS refuses", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-gpt", messages: [{ role: "user", content: "hi" }], temperature: 0.2, top_p: 0.9 }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/openai/v1/chat/completions");
    expect(sent[0].body.temperature).toBeUndefined();
    expect(sent[0].body.top_p).toBeUndefined();
  });

  test("Chat Completions streams through the same endpoint", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-gpt", stream: true, messages: [{ role: "user", content: "hi" }] }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/openai/v1/chat/completions");
    expect(sent[0].body.stream).toBe(true);
    expect(result.text).toContain("Hi ");
    expect(result.text).toContain("there");
    expect(result.text).toContain("[DONE]");
  });

  test("a GPT model with tools at its default effort falls back to Converse", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-gpt", messages: [{ role: "user", content: "weather?" }], tools: TOOLS }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/model/global.openai.gpt-6.1-sol/converse");
    expect(result.text).toContain("converse answer");
  });

  test("a Bedrock-native block keeps a Chat Completions model on Converse, at the top level", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-gpt",
        messages: [{ role: "user", content: "hi" }],
        guardrailConfig: { guardrailIdentifier: "gr-1", guardrailVersion: "2", trace: "enabled" },
        performanceConfig: { latency: "optimized" },
        requestMetadata: { team: "e2e" },
        service_tier: "auto",
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/model/global.openai.gpt-6.1-sol/converse");
    expect(sent[0].body.guardrailConfig).toEqual({
      guardrailIdentifier: "gr-1",
      guardrailVersion: "2",
      trace: "enabled",
    });
    expect(sent[0].body.performanceConfig).toEqual({ latency: "optimized" });
    expect(sent[0].body.requestMetadata).toEqual({ team: "e2e" });
    expect(sent[0].body.serviceTier).toEqual({ type: "default" });
    // With a guardrail, the newest user text is what it evaluates.
    expect(sent[0].body.messages.at(-1).content).toEqual([{ guardContent: { text: { text: "hi" } } }]);
    expect(sent[0].body.additionalModelRequestFields).toBeUndefined();
  });

  test("Grok keeps tools on Chat Completions and drops the penalties AWS refuses", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-grok",
        messages: [{ role: "user", content: "weather?" }],
        tools: TOOLS,
        frequency_penalty: 0.5,
        presence_penalty: 0.5,
        reasoning_effort: "none",
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/openai/v1/chat/completions");
    expect(sent[0].body.model).toBe("us.xai.grok-4.6");
    expect(sent[0].body.frequency_penalty).toBeUndefined();
    expect(sent[0].body.presence_penalty).toBeUndefined();
    expect(sent[0].body.reasoning_effort).toBeUndefined();
  });

  test("an in-region GPT id stays on Converse", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-gpt-bare", messages: [{ role: "user", content: "hi" }] }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/model/openai.gpt-6-sol/converse");
  });

  // ── Converse request shape ──────────────────────────────────────────

  test("ConverseStream merges a tool result and the user turn after it into one user message", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-claude",
        stream: true,
        tools: TOOLS,
        messages: [
          { role: "user", content: "weather?" },
          {
            role: "assistant",
            content: null,
            tool_calls: [{ id: "tooluse_1", type: "function", function: { name: "get_weather", arguments: "{}" } }],
          },
          { role: "tool", tool_call_id: "tooluse_1", content: "sunny" },
          { role: "user", content: "and tomorrow?" },
        ],
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toMatch(/\/converse-stream$/);
    const roles = sent[0].body.messages.map((m: any) => m.role);
    expect(roles).toEqual(["user", "assistant", "user"]);
    const last = sent[0].body.messages[2].content;
    expect(last[0].toolResult.toolUseId).toBe("tooluse_1");
    expect(last.at(-1)).toEqual({ text: "and tomorrow?" });
  });

  test("a Claude thinking budget fits under max_completion_tokens", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-claude",
        stream: true,
        messages: [{ role: "user", content: "think" }],
        max_completion_tokens: 3000,
        reasoning_effort: "high",
      }),
    );
    expect(result.status, result.text).toBe(200);
    const body = sent[0].body;
    expect(body.inferenceConfig.maxTokens).toBe(3000);
    const budget = body.additionalModelRequestFields?.thinking?.budget_tokens;
    expect(budget).toBeGreaterThan(0);
    expect(budget).toBeLessThan(3000);
  });

  test("a non-streamed Claude request with requestMetadata is served by Converse", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-claude", messages: [{ role: "user", content: "hi" }], requestMetadata: { team: "e2e" } }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toMatch(/\/converse$/);
    expect(sent[0].body.requestMetadata).toEqual({ team: "e2e" });
  });

  test("stop and max_completion_tokens reach inferenceConfig", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-llama", messages: [{ role: "user", content: "hi" }], stop: "END", max_completion_tokens: 50 }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.inferenceConfig).toEqual({ maxTokens: 50, stopSequences: ["END"] });
  });

  // Converse refuses `stopSequences` beside an additional `stop_sequences`.
  test("ConverseStream to Claude lets a native stop_sequences win over stop", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-claude",
        stream: true,
        max_tokens: 60,
        stop: ["delta"],
        stop_sequences: ["beta"],
        messages: [{ role: "user", content: "hi" }],
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toMatch(/\/converse-stream$/);
    expect(sent[0].body.inferenceConfig).toEqual({ maxTokens: 60 });
    expect(sent[0].body.additionalModelRequestFields).toEqual({ stop_sequences: ["beta"] });
  });

  // Values Claude's Messages API refuses are the upstream's to refuse, as on
  // the Anthropic route: nothing is dropped or filtered on the way.
  test("a non-streamed Claude request forwards stop and max_completion_tokens as sent", async (ctx) => {
    if (skip(ctx)) return;
    const cases: Array<[Record<string, unknown>, Record<string, unknown>]> = [
      [{ stop: "" }, { stop_sequences: [""] }],
      [{ stop: ["", "gamma"] }, { stop_sequences: ["", "gamma"] }],
      [{ stop: 5 }, { stop: 5 }],
      [{ max_completion_tokens: "many" }, { max_completion_tokens: "many" }],
    ];
    for (const [extra, expected] of cases) {
      const { result, sent } = await capture(() =>
        chat({ model: "cc-claude", messages: [{ role: "user", content: "hi" }], ...extra }),
      );
      expect(result.status, result.text).toBe(200);
      expect(sent[0].path).toMatch(/\/invoke$/);
      expect(sent[0].body, JSON.stringify(extra)).toMatchObject(expected);
    }
  });

  test("an OpenAI GPT model on Converse gets reasoning.effort, no stop sequences, and a cap of at least 16", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-gpt-bare",
        messages: [{ role: "user", content: "hi" }],
        reasoning_effort: "high",
        stop: ["END"],
        max_tokens: 4,
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.additionalModelRequestFields).toEqual({ reasoning: { effort: "high" } });
    expect(sent[0].body.inferenceConfig).toEqual({ maxTokens: 16 });
  });

  test("Nova 2 takes reasoning_effort as reasoningConfig", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({ model: "cc-nova2", messages: [{ role: "user", content: "hi" }], reasoning_effort: "medium" }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.additionalModelRequestFields).toEqual({
      reasoningConfig: { type: "enabled", maxReasoningEffort: "medium" },
    });
  });

  test("gpt-oss gets reasoning_effort verbatim, a forced JSON tool for response_format, and model-specific fields", async (ctx) => {
    if (skip(ctx)) return;
    respond = (r) =>
      r.path.endsWith("/converse")
        ? {
            body: {
              output: {
                message: {
                  role: "assistant",
                  content: [{ toolUse: { toolUseId: "t1", name: "json_tool_call", input: { city: "Paris" } } }],
                },
              },
              stopReason: "tool_use",
              usage: { inputTokens: 10, outputTokens: 3, totalTokens: 13 },
              metrics: { latencyMs: 5 },
            },
          }
        : defaultReply(r);
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-gptoss",
        messages: [{ role: "user", content: "where?" }],
        reasoning_effort: "high",
        response_format: {
          type: "json_schema",
          json_schema: { name: "place", schema: { type: "object", properties: { city: { type: "string" } } } },
        },
        additionalModelRequestFields: { custom_a: 1 },
        custom_b: "two",
        seed: 7,
      }),
    );
    expect(result.status, result.text).toBe(200);
    const body = sent[0].body;
    expect(body.additionalModelRequestFields).toEqual({ custom_a: 1, custom_b: "two", reasoning_effort: "high" });
    expect(body.toolConfig.tools[0].toolSpec.name).toBe("json_tool_call");
    expect(body.toolConfig.toolChoice).toEqual({ tool: { name: "json_tool_call" } });
    const reply = JSON.parse(result.text);
    expect(JSON.parse(reply.choices[0].message.content)).toEqual({ city: "Paris" });
    expect(reply.choices[0].finish_reason).toBe("stop");
  });

  test("tool_choice none with tool history sends no toolConfig and the history as text", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "cc-llama",
        tools: TOOLS,
        tool_choice: "none",
        messages: [
          { role: "user", content: "weather?" },
          {
            role: "assistant",
            content: null,
            tool_calls: [{ id: "tooluse_9", type: "function", function: { name: "get_weather", arguments: "{}" } }],
          },
          { role: "tool", tool_call_id: "tooluse_9", content: "sunny" },
        ],
      }),
    );
    expect(result.status, result.text).toBe(200);
    const body = sent[0].body;
    expect(body.toolConfig).toBeUndefined();
    expect(JSON.stringify(body.messages)).not.toContain("toolUse");
    expect(body.messages[1].content).toEqual([{ text: "[tool call tooluse_9: get_weather({})]" }]);
    expect(body.messages[2].content).toEqual([{ text: "[tool result for tooluse_9: sunny]" }]);
  });

  test("an image in a tool result rides toolResult.content for Claude and beside it for GPT", async (ctx) => {
    if (skip(ctx)) return;
    const messages = [
      { role: "user", content: "look" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "tooluse_img", type: "function", function: { name: "get_weather", arguments: "{}" } }],
      },
      {
        role: "tool",
        tool_call_id: "tooluse_img",
        content: [
          { type: "text", text: "chart" },
          { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_1PX}` } },
        ],
      },
    ];
    const claude = await capture(() => chat({ model: "cc-claude", stream: true, tools: TOOLS, messages }));
    expect(claude.result.status, claude.result.text).toBe(200);
    const claudeResult = claude.sent[0].body.messages[2].content[0].toolResult;
    expect(claudeResult.content[0]).toEqual({ text: "chart" });
    expect(claudeResult.content[1].image.format).toBe("png");

    const gpt = await capture(() => chat({ model: "cc-gpt-bare", tools: TOOLS, messages }));
    expect(gpt.result.status, gpt.result.text).toBe(200);
    const turn = gpt.sent[0].body.messages[2].content;
    expect(turn[0].toolResult.content).toEqual([{ text: "chart" }]);
    expect(turn[1].image.format).toBe("png");
  });

  test("ARN model ids dispatch: an application inference profile through Converse, a Claude profile through InvokeModel", async (ctx) => {
    if (skip(ctx)) return;
    const app1 = await capture(() => chat({ model: "cc-arn-app", messages: [{ role: "user", content: "hi" }] }));
    expect(app1.result.status, app1.result.text).toBe(200);
    expect(app1.sent[0].path).toBe(
      "/model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz/converse",
    );
    const claude = await capture(() => chat({ model: "cc-arn-claude", messages: [{ role: "user", content: "hi" }] }));
    expect(claude.result.status, claude.result.text).toBe(200);
    expect(claude.sent[0].path).toMatch(/inference-profile\/us\.anthropic\.claude-sonnet-4-5-20250929-v1:0\/invoke$/);
    expect(claude.result.text).toContain("invoke answer");
  });

  // ── Errors and request ids ──────────────────────────────────────────

  test("a Bedrock error reaches the caller with its own message, and an overflow carries context_length_exceeded", async (ctx) => {
    if (skip(ctx)) return;
    respond = () => ({
      status: 400,
      headers: { "x-amzn-errortype": "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/" },
      body: { message: "Input is too long for requested model." },
    });
    const res = await chat({ model: "cc-llama", messages: [{ role: "user", content: "hi" }] });
    expect(res.status).toBe(400);
    const body = JSON.parse(res.text);
    expect(body.error.message).toBe("Input is too long for requested model.");
    expect(body.error.code).toBe("context_length_exceeded");
  });

  test("an Azure error and an OpenAI 5xx reach the caller with the upstream's own message", async (ctx) => {
    if (skip(ctx)) return;
    respond = (r) =>
      r.path.startsWith("/oai")
        ? { status: 503, body: { error: { message: "engine offline shard 47", type: "server_error" } } }
        : {
            status: 404,
            body: { error: { code: "DeploymentNotFound", message: "The API deployment for this resource does not exist." } },
          };
    const azure = await chat({ model: "cc-azure", messages: [{ role: "user", content: "hi" }] });
    expect(azure.status).toBe(404);
    expect(JSON.parse(azure.text).error.message).toBe("The API deployment for this resource does not exist.");
    const openai = await chat({ model: "cc-openai", messages: [{ role: "user", content: "hi" }] });
    expect(openai.status).toBe(502);
    expect(JSON.parse(openai.text).error.message).toBe("engine offline shard 47");
  });

  test("x-amzn-requestid is the recorded provider request id for a Converse call", async (ctx) => {
    if (skip(ctx)) return;
    respond = (r) => ({ ...defaultReply(r), headers: { "x-amzn-requestid": "bedrock-req-e2e-7f3a" } });
    const res = await chat({ model: "cc-llama", messages: [{ role: "user", content: "hi" }] });
    expect(res.status, res.text).toBe(200);
    expect(JSON.parse(res.text).id).toBe("bedrock-req-e2e-7f3a");
    const line = await waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && l.includes(`request_id="${res.requestId}"`),
      "the access-log line for the Converse request",
    );
    expect(line).toContain(`provider_request_id="bedrock-req-e2e-7f3a"`);
  });

  // ── Embeddings ──────────────────────────────────────────────────────

  test("Cohere embeddings take the caller's input_type and read the v4 response and its usage", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/embeddings`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
        body: JSON.stringify({ model: "cc-cohere", input: ["a", "b"], input_type: "search_query" }),
      });
      return { status: res.status, text: await res.text() };
    });
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/model/cohere.embed-v4:0/invoke");
    expect(sent[0].body).toEqual({ texts: ["a", "b"], input_type: "search_query" });
    const body = JSON.parse(result.text);
    expect(body.data.map((d: any) => d.embedding)).toEqual([[0.1, 0.2], [0.3, 0.4]]);
    expect(body.usage.prompt_tokens).toBe(6);
  });
});
