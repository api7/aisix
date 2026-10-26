import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

// Amazon Bedrock handles each request field by the target model's contract.
// OpenAI parameters with no Converse field reach the models whose Bedrock
// request body takes them as `additionalModelRequestFields` members (the
// OpenAI models, Cohere Command R, Jamba), `reasoning_effort` reaches Z.AI GLM
// and turns Nova 2 reasoning off, Titan Multimodal takes `dimensions`, and
// `additionalModelResponseFieldPaths` / `promptVariables` are top-level
// Converse fields. Nothing is rewritten or pre-validated for the upstream: the
// output cap, a request with no turn, a non-Claude JSON schema and
// temperature reach Bedrock as sent, and Bedrock's own error reaches the
// caller. The Chat Completions route keeps `developer` and serves `thinking`,
// `top_k` and `json_object` itself. A value OpenAI's own contract forbids is
// refused naming the field.

const CALLER = "sk-param-bedrock";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const AWS_REFUSAL = "A conversation must start with a user message. Try again with a conversation that starts with a user message.";

type Received = { path: string; headers: Record<string, string>; raw: string; body: any };
type Reply = { status?: number; headers?: Record<string, string>; body: unknown };

const CONVERSE_TEXT = {
  output: { message: { role: "assistant", content: [{ text: "converse answer" }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 10, outputTokens: 3, totalTokens: 13 },
  metrics: { latencyMs: 5 },
};

const CHAT_COMPLETION = {
  id: "chatcmpl-1",
  object: "chat.completion",
  model: "global.openai.gpt-6.1-sol",
  choices: [{ index: 0, message: { role: "assistant", content: "cc answer" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 },
};

function defaultReply(r: Received): Reply {
  if (r.path.endsWith("/openai/v1/chat/completions")) return { body: CHAT_COMPLETION };
  if (r.path.endsWith("/converse")) return { body: CONVERSE_TEXT };
  if (r.path.endsWith("/invoke")) {
    if (r.body.inputText !== undefined) return { body: { embedding: [0.1, 0.2], inputTextTokenCount: 2 } };
    return {
      body: {
        id: "msg_1",
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
}

describe("Amazon Bedrock: request fields by the target's contract", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;
  const received: Received[] = [];
  let respond: (r: Received) => Reply = defaultReply;

  async function post(path: string, body: Record<string, unknown>, headers: Record<string, string> = {}) {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify(body),
    });
    return { status: res.status, text: await res.text() };
  }
  const chat = (body: Record<string, unknown>) => post("/v1/chat/completions", body);

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
        res.writeHead(reply.status ?? 200, { "content-type": "application/json", ...(reply.headers ?? {}) });
        res.end(typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body));
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "param-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({ access_key_id: "AKIA-param", secret_access_key: "sk-param", region: "us-east-1" }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    const models: Array<[string, string]> = [
      ["pb-gptoss", "openai.gpt-oss-120b-1:0"],
      ["pb-gpt-bare", "openai.gpt-6-sol"],
      ["pb-gpt", "global.openai.gpt-6.1-sol"],
      ["pb-grok-bare", "xai.grok-4.6"],
      ["pb-kimi", "moonshotai.kimi-k3"],
      ["pb-cohere", "cohere.command-r-plus-v1:0"],
      ["pb-jamba", "ai21.jamba-1-5-large-v1:0"],
      ["pb-glm", "zai.glm-5.3"],
      ["pb-nova2", "us.amazon.nova-2-lite-v1:0"],
      ["pb-nova", "us.amazon.nova-lite-v1:0"],
      ["pb-llama", "meta.llama3-3-70b-instruct-v1:0"],
      ["pb-claude", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"],
      ["pb-titan-mm", "amazon.titan-embed-image-v1"],
    ];
    for (const [alias, name] of models) {
      await seed.createModel({ display_name: alias, provider: "bedrock", model_name: name, provider_key_id: pk.id });
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
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

  const HI = [{ role: "user", content: "hi" }];

  test("the OpenAI models on Converse get every untranslated OpenAI field in additionalModelRequestFields, but not n or logprobs", async (ctx) => {
    if (skip(ctx)) return;
    const forwarded = {
      seed: 7,
      frequency_penalty: 0.5,
      presence_penalty: 0.25,
      user: "end-user-1",
      safety_identifier: "sid-1",
      logit_bias: { "50256": -100 },
      parallel_tool_calls: false,
      response_format: { type: "json_object" },
      metadata: { tag: "a" },
    };
    for (const model of ["pb-gptoss", "pb-gpt-bare", "pb-grok-bare"]) {
      const { result, sent } = await capture(() =>
        chat({ model, messages: HI, ...forwarded, n: 2, logprobs: true, top_logprobs: 2 }),
      );
      expect(result.status, `${model}: ${result.text}`).toBe(200);
      expect(sent[0].path, model).toMatch(/\/converse$/);
      expect(sent[0].body.additionalModelRequestFields, model).toEqual(forwarded);
    }
  });

  test("Cohere Command R, Jamba, Z.AI GLM and Kimi K3 get the fields their Bedrock request body defines", async (ctx) => {
    if (skip(ctx)) return;
    const cases: Array<[string, Record<string, unknown>, Record<string, unknown>]> = [
      ["pb-cohere", { seed: 3, frequency_penalty: 0.5, presence_penalty: 0.5, user: "u" }, { seed: 3, frequency_penalty: 0.5, presence_penalty: 0.5 }],
      ["pb-jamba", { seed: 3, frequency_penalty: 0.5, presence_penalty: 0.5 }, { frequency_penalty: 0.5, presence_penalty: 0.5 }],
      ["pb-glm", { reasoning_effort: "max" }, { reasoning_effort: "max" }],
      ["pb-kimi", { reasoning_effort: "high" }, { reasoning_effort: "high" }],
    ];
    for (const [model, extra, want] of cases) {
      const { result, sent } = await capture(() => chat({ model, messages: HI, ...extra }));
      expect(result.status, `${model}: ${result.text}`).toBe(200);
      expect(sent[0].body.additionalModelRequestFields, model).toEqual(want);
    }
  });

  test("Nova 2 turns reasoning off for reasoning_effort none and drops a tier it lacks", async (ctx) => {
    if (skip(ctx)) return;
    const off = await capture(() => chat({ model: "pb-nova2", messages: HI, reasoning_effort: "none" }));
    expect(off.result.status, off.result.text).toBe(200);
    expect(off.sent[0].body.additionalModelRequestFields).toEqual({ reasoningConfig: { type: "disabled" } });
    const xhigh = await capture(() => chat({ model: "pb-nova2", messages: HI, reasoning_effort: "xhigh" }));
    expect(xhigh.result.status, xhigh.result.text).toBe(200);
    expect(xhigh.sent[0].body.additionalModelRequestFields).toBeUndefined();
  });

  test("Titan Multimodal embeddings take dimensions as embeddingConfig.outputEmbeddingLength", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post("/v1/embeddings", { model: "pb-titan-mm", input: "hello", dimensions: 256 }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body).toEqual({ inputText: "hello", embeddingConfig: { outputEmbeddingLength: 256 } });
  });

  test("the output cap reaches Converse as sent, also below 16", async (ctx) => {
    if (skip(ctx)) return;
    for (const model of ["pb-gpt-bare", "pb-grok-bare", "pb-kimi"]) {
      const { result, sent } = await capture(() => chat({ model, messages: HI, max_tokens: 4 }));
      expect(result.status, `${model}: ${result.text}`).toBe(200);
      expect(sent[0].body.inferenceConfig, model).toEqual({ maxTokens: 4 });
    }
  });

  test("temperature and top_p reach Converse exactly as sent", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() => chat({ model: "pb-llama", messages: HI, temperature: 0.7, top_p: 0.9 }));
    expect(result.status, result.text).toBe(200);
    expect(sent[0].raw).toContain('"inferenceConfig":{"temperature":0.7,"topP":0.9}');
  });

  test("a request with no turn reaches Bedrock on every rail, and Bedrock's refusal reaches the caller", async (ctx) => {
    if (skip(ctx)) return;
    respond = () => ({
      status: 400,
      headers: { "x-amzn-errortype": "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/" },
      body: { message: AWS_REFUSAL },
    });
    const cases: Array<[string, Record<string, unknown>, RegExp]> = [
      ["/v1/chat/completions", { model: "pb-nova", messages: [{ role: "system", content: "terse" }] }, /\/converse$/],
      ["/v1/chat/completions", { model: "pb-claude", messages: [{ role: "system", content: "terse" }] }, /\/invoke$/],
      ["/v1/chat/completions", { model: "pb-gpt", messages: [{ role: "system", content: "terse" }] }, /\/openai\/v1\/chat\/completions$/],
      ["/v1/messages", { model: "pb-claude", max_tokens: 64, system: "terse", messages: [] }, /\/invoke$/],
    ];
    for (const [path, body, route] of cases) {
      const what = `${path} ${body.model}`;
      const { result, sent } = await capture(() => post(path, body));
      expect(sent, what).toHaveLength(1);
      expect(sent[0].path, what).toMatch(route);
      expect(result.status, `${what}: ${result.text}`).toBe(400);
      expect(result.text, what).toContain(AWS_REFUSAL);
    }
  });

  test("a non-Claude model gets the caller's JSON schema as sent on the tool route", async (ctx) => {
    if (skip(ctx)) return;
    const schema = { type: "object", properties: { name: { type: "string", maxLength: 20 } } };
    const { sent } = await capture(() =>
      chat({ model: "pb-llama", messages: HI, response_format: { type: "json_schema", json_schema: { name: "p", schema } } }),
    );
    expect(sent[0].body.toolConfig.tools[0].toolSpec.inputSchema.json).toEqual(schema);
  });

  test("a function's strict and the caller's json_schema.strict reach Converse toolSpec.strict", async (ctx) => {
    if (skip(ctx)) return;
    const parameters = { type: "object", properties: { city: { type: "string" } } };
    const tools = await capture(() =>
      chat({
        model: "pb-nova",
        messages: HI,
        tools: [
          { type: "function", function: { name: "strict_fn", strict: true, parameters } },
          { type: "function", function: { name: "loose_fn", parameters } },
        ],
      }),
    );
    expect(tools.result.status, tools.result.text).toBe(200);
    expect(tools.sent[0].body.toolConfig.tools.map((t: any) => [t.toolSpec.name, t.toolSpec.strict])).toEqual([
      ["strict_fn", true],
      ["loose_fn", undefined],
    ]);
    const schema = { type: "object", properties: { name: { type: "string" } }, required: ["name"], additionalProperties: false };
    const json = await capture(() =>
      chat({ model: "pb-nova", messages: HI, response_format: { type: "json_schema", json_schema: { name: "p", schema, strict: true } } }),
    );
    expect(json.result.status, json.result.text).toBe(200);
    expect(json.sent[0].body.toolConfig.tools[0].toolSpec.strict).toBe(true);
  });

  test("Chat Completions keeps developer and serves thinking, top_k and json_object itself", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      chat({
        model: "pb-gpt",
        messages: [{ role: "developer", content: "be terse" }, ...HI],
        thinking: { type: "enabled", budget_tokens: 1024 },
        top_k: 5,
        response_format: { type: "json_object" },
      }),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].path).toBe("/openai/v1/chat/completions");
    expect(sent[0].body.messages[0]).toEqual({ role: "developer", content: "be terse" });
    expect(sent[0].body).toMatchObject({
      thinking: { type: "enabled", budget_tokens: 1024 },
      top_k: 5,
      response_format: { type: "json_object" },
    });
  });

  test("additionalModelResponseFieldPaths and promptVariables are top-level Converse fields, Claude included", async (ctx) => {
    if (skip(ctx)) return;
    for (const model of ["pb-llama", "pb-claude"]) {
      const { result, sent } = await capture(() =>
        chat({
          model,
          messages: HI,
          additionalModelResponseFieldPaths: ["/stop_sequence"],
          promptVariables: { topic: { text: "cats" } },
        }),
      );
      expect(result.status, `${model}: ${result.text}`).toBe(200);
      expect(sent[0].path, model).toMatch(/\/converse$/);
      expect(sent[0].body.additionalModelResponseFieldPaths, model).toEqual(["/stop_sequence"]);
      expect(sent[0].body.promptVariables, model).toEqual({ topic: { text: "cats" } });
      expect(sent[0].body.additionalModelRequestFields, model).toBeUndefined();
    }
  });

  test("Claude on Bedrock takes the dev-full-thinking beta", async (ctx) => {
    if (skip(ctx)) return;
    const { result, sent } = await capture(() =>
      post(
        "/v1/messages",
        { model: "pb-claude", max_tokens: 64, messages: HI },
        { "anthropic-beta": "dev-full-thinking-2025-05-14" },
      ),
    );
    expect(result.status, result.text).toBe(200);
    expect(sent[0].body.anthropic_beta).toEqual(["dev-full-thinking-2025-05-14"]);
  });

  test("Claude gets safety_identifier over user as metadata.user_id, and neither raw key", async (ctx) => {
    if (skip(ctx)) return;
    const cases: Array<[Record<string, unknown>, string]> = [
      [{ user: "u", safety_identifier: "s" }, "s"],
      [{ user: "u", safety_identifier: "" }, "u"],
    ];
    for (const [extra, want] of cases) {
      const { result, sent } = await capture(() => chat({ model: "pb-claude", messages: HI, ...extra }));
      expect(result.status, result.text).toBe(200);
      expect(sent[0].path).toMatch(/\/invoke$/);
      expect(sent[0].body.metadata, JSON.stringify(extra)).toEqual({ user_id: want });
      expect(sent[0].body.user).toBeUndefined();
      expect(sent[0].body.safety_identifier).toBeUndefined();
    }
  });

  test("Converse refuses an illegal tool_choice, a malformed tool or history call, and a non-string service_tier, naming the field", async (ctx) => {
    if (skip(ctx)) return;
    const TOOL = { type: "function", function: { name: "f", parameters: { type: "object" } } };
    const cases: Array<[string, Record<string, unknown>]> = [
      ["tool_choice", { tools: [TOOL], tool_choice: "sometimes" }],
      ["tool_choice", { tools: [TOOL], tool_choice: { function: { name: "f" } } }],
      ["tool_choice", { tools: [TOOL], tool_choice: { type: "function", function: {} } }],
      ["tools[0].function", { tools: [{ type: "function" }] }],
      ["tools[0].function.name", { tools: [{ type: "function", function: { parameters: {} } }] }],
      ["tools[0].function.parameters", { tools: [{ type: "function", function: { name: "f", parameters: 1 } }] }],
      [
        "messages[1].tool_calls[0].id",
        {
          tools: [TOOL],
          messages: [
            ...HI,
            { role: "assistant", content: null, tool_calls: [{ type: "function", function: { name: "f", arguments: "{}" } }] },
            { role: "tool", tool_call_id: "x", content: "ok" },
          ],
        },
      ],
      ["service_tier", { service_tier: 5 }],
    ];
    for (const model of ["pb-llama", "pb-nova"]) {
      for (const [field, extra] of cases) {
        const what = `${model} ${JSON.stringify(extra)}`;
        const { result, sent } = await capture(() => chat({ model, messages: HI, ...extra }));
        expect(result.status, `${what}: ${result.text}`).toBe(400);
        expect(JSON.parse(result.text).error.message, what).toContain(field);
        expect(sent, what).toHaveLength(0);
      }
    }
  });
});
