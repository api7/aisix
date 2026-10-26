import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

// The Claude bridges handle each request field by the target's contract.
// chat → Claude: OpenAI-only fields with no Anthropic slot are dropped, the
// ones with a same-meaning slot are written there, Anthropic-native fields
// go through, and a value OpenAI's contract forbids is refused naming it.
// /v1/messages → chat: Anthropic server tools are not offered as functions,
// `top_k` reaches a target that takes it and no other, and a wrong-typed
// sampling field is refused. Stop reasons and refusals translate both ways,
// reasoning inside an open tool call survives the stream, an Anthropic
// error keeps its own message, the caller's `anthropic-beta` reaches
// Anthropic, and Claude below 4.5 on Vertex AI answers structured output on
// its native route.

const CALLER = "sk-param-claude-bridges";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const UPSTREAM_ERROR = "max_tokens: 9999999 > 8192, which is the maximum allowed";
const VERTEX_OLD_CLAUDE = "claude-3-5-sonnet-v2@20241022";

interface Received {
  path: string;
  headers: Record<string, string | string[] | undefined>;
  body: any;
}

function claudeReply(stopReason: string): Record<string, unknown> {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-5",
    content: [{ type: "text", text: "ok" }],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: 3, output_tokens: 1 },
  };
}

function openAiReply(message: Record<string, unknown>, finish: string): Record<string, unknown> {
  return {
    id: "chatcmpl-1",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "gpt-4o",
    choices: [{ index: 0, message: { role: "assistant", ...message }, finish_reason: finish }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  };
}

const chunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  `data: ${JSON.stringify({
    id: "chatcmpl-s",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o",
    choices: [{ index: 0, delta, finish_reason: finish }],
  })}\n\n`;

// A tool call whose arguments straddle a reasoning delta.
const TOOL_WITH_REASONING_STREAM = [
  chunk({ role: "assistant", tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "f", arguments: "{\"a\":" } }] }),
  chunk({ reasoning_content: "thinking mid-call" }),
  chunk({ tool_calls: [{ index: 0, function: { arguments: "1}" } }] }),
  chunk({}, "tool_calls"),
  `data: ${JSON.stringify({ id: "chatcmpl-s", object: "chat.completion.chunk", created: 1, model: "gpt-4o", choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } })}\n\n`,
  "data: [DONE]\n\n",
].join("");

/** Anthropic, OpenAI, Vertex AI and Bedrock routes on one server. */
async function startUpstream() {
  const received: Received[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = decodeURIComponent((req.url ?? "/").split("?")[0]);
      const raw = Buffer.concat(chunks).toString("utf8");
      const body = raw ? JSON.parse(raw) : {};
      received.push({ path, headers: req.headers, body });
      const json = (status: number, payload: unknown) => {
        res.writeHead(status, { "content-type": "application/json" });
        res.end(JSON.stringify(payload));
      };
      const lastText = JSON.stringify(body.messages?.at(-1) ?? "");
      if (path === "/v1/messages" || path === "/v1/messages/count_tokens") {
        if (lastText.includes("FAIL")) {
          return json(400, { type: "error", error: { type: "invalid_request_error", message: UPSTREAM_ERROR } });
        }
        if (path.endsWith("count_tokens")) return json(200, { input_tokens: 5 });
        const stop = /STOP:([a-z_]+)/.exec(lastText)?.[1] ?? "end_turn";
        return json(200, claudeReply(stop));
      }
      if (path === "/v1/chat/completions") {
        if (body.stream) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(TOOL_WITH_REASONING_STREAM);
          return;
        }
        if (lastText.includes("REFUSE")) return json(200, openAiReply({ content: null, refusal: "I can't help with that." }, "stop"));
        if (lastText.includes("FILTER")) return json(200, openAiReply({ content: "partial" }, "content_filter"));
        return json(200, openAiReply({ content: "ok" }, "stop"));
      }
      if (path.endsWith(":streamRawPredict")) {
        const sse = (event: Record<string, unknown>) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end(
          [
            sse({ type: "message_start", message: { id: "msg_vs", type: "message", role: "assistant", model: VERTEX_OLD_CLAUDE, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 7, output_tokens: 0 } } }),
            sse({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
            sse({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "streamed" } }),
            sse({ type: "content_block_stop", index: 0 }),
            sse({ type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 1 } }),
            sse({ type: "message_stop" }),
          ].join(""),
        );
        return;
      }
      if (path.endsWith(":rawPredict")) {
        return json(200, {
          id: "msg_vertex",
          type: "message",
          role: "assistant",
          model: VERTEX_OLD_CLAUDE,
          content: [
            { type: "text", text: "Sure." },
            { type: "tool_use", id: "toolu_json", name: "json_tool_call", input: { city: "Paris" } },
          ],
          stop_reason: "tool_use",
          stop_sequence: null,
          usage: { input_tokens: 7, output_tokens: 4 },
        });
      }
      if (path.endsWith("/converse")) {
        return json(200, {
          output: { message: { role: "assistant", content: [{ text: "ok" }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
          metrics: { latencyMs: 1 },
        });
      }
      json(404, { message: `no route ${path}` });
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () => new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

function frames(text: string): any[] {
  return text
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)));
}

describe("Claude bridges handle each field by the target's contract", () => {
  let app: SpawnedApp | undefined;
  let upstream: Awaited<ReturnType<typeof startUpstream>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const anthropicPk = await seed.createProviderKey({
      display_name: "pcb-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    const openaiPk = await seed.createProviderKey({
      display_name: "pcb-openai-pk",
      provider: "openai",
      adapter: "openai",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "pcb-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.pcb", project: "proj-e2e", region: "us-east5" }),
      api_base: upstream.baseUrl,
    });
    const bedrockPk = await seed.createProviderKey({
      display_name: "pcb-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({ access_key_id: "AKIA-pcb", secret_access_key: "sk-pcb", region: "us-west-2" }),
      api_base: upstream.baseUrl,
    });
    for (const [display_name, provider, model_name, pk] of [
      ["pcb-claude", "anthropic", "claude-sonnet-4-5", anthropicPk],
      ["pcb-gpt", "openai", "gpt-4o", openaiPk],
      ["pcb-vertex-old-claude", "google", VERTEX_OLD_CLAUDE, vertexPk],
      ["pcb-nova", "bedrock", "amazon.nova-pro-v1:0", bedrockPk],
    ] as const) {
      await seed.createModel({ display_name, provider, model_name, provider_key_id: pk.id });
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
    await upstream?.close();
  });

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return false;
    }
    return true;
  };

  async function post(path: string, body: unknown, headers: Record<string, string> = {}) {
    const before = upstream!.received.length;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        ...headers,
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    const sent = upstream!.received.slice(before);
    return { status: res.status, text, json: () => JSON.parse(text), sent };
  }

  test("chat → Claude drops OpenAI-only fields, maps user, keeps native ones and copies strict", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/chat/completions", {
      model: "pcb-claude",
      messages: [{ role: "user", content: "hi" }],
      seed: 7,
      n: 1,
      logprobs: true,
      presence_penalty: 0.2,
      store: true,
      user: "end-user-42",
      top_k: 5,
      tools: [{ type: "function", function: { name: "f", strict: true, parameters: { type: "object", properties: {} } } }],
    });
    expect(res.status, res.text).toBe(200);
    const body = res.sent.at(-1)!.body;
    for (const dropped of ["seed", "n", "logprobs", "presence_penalty", "store", "user"]) {
      expect(body, dropped).not.toHaveProperty(dropped);
    }
    expect(body.metadata).toEqual({ user_id: "end-user-42" });
    expect(body.top_k).toBe(5);
    expect(body.tools[0]).toMatchObject({ name: "f", strict: true });
  });

  test.for<[string, Record<string, unknown>, string]>([
    ["stop", { stop: 5 }, "stop"],
    ["max_completion_tokens", { max_completion_tokens: "many" }, "max_completion_tokens"],
    ["tool_choice", { tools: [{ type: "function", function: { name: "f" } }], tool_choice: "bogus" }, "tool_choice"],
    ["reasoning_effort", { reasoning_effort: 3 }, "reasoning_effort"],
    [
      "a history tool call without an id",
      {
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: null, tool_calls: [{ type: "function", function: { name: "f", arguments: "{}" } }] },
          { role: "user", content: "again" },
        ],
      },
      "messages[1].tool_calls[0].id",
    ],
  ])("chat → Claude refuses an illegal %s naming it", async ([, extra, field], ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/chat/completions", {
      model: "pcb-claude",
      messages: [{ role: "user", content: "hi" }],
      ...extra,
    });
    expect(res.status, res.text).toBe(400);
    expect(res.json().error.message).toContain(field);
    expect(res.sent).toHaveLength(0);
  });

  test.for<[string, string]>([
    ["refusal", "content_filter"],
    ["model_context_window_exceeded", "length"],
    ["pause_turn", "stop"],
  ])("Claude's %s ends a chat answer as %s", async ([stop, finish], ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/chat/completions", {
      model: "pcb-claude",
      messages: [{ role: "user", content: `STOP:${stop}` }],
    });
    expect(res.status, res.text).toBe(200);
    expect(res.json().choices[0].finish_reason).toBe(finish);
  });

  test("/v1/messages → chat drops server tools, carries strict and keeps top_k off the OpenAI wire", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/messages", {
      model: "pcb-gpt",
      max_tokens: 64,
      top_k: 5,
      messages: [{ role: "user", content: "hi" }],
      tools: [
        { type: "web_search_20250305", name: "web_search", max_uses: 2 },
        { name: "f", strict: true, input_schema: { type: "object", properties: {} } },
      ],
    });
    expect(res.status, res.text).toBe(200);
    const body = res.sent.at(-1)!.body;
    expect(body.tools).toEqual([
      { type: "function", function: { name: "f", parameters: { type: "object", properties: {} }, strict: true } },
    ]);
    expect(body).not.toHaveProperty("top_k");
  });

  test("/v1/messages → Converse carries top_k to Nova", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/messages", {
      model: "pcb-nova",
      max_tokens: 64,
      top_k: 7,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const body = res.sent.at(-1)!.body;
    expect(body.additionalModelRequestFields?.inferenceConfig?.topK).toBe(7);
  });

  test.for<[string, unknown]>([
    ["temperature", "hot"],
    ["top_p", true],
    ["max_tokens", 5_000_000_000],
    ["stream", "yes"],
  ])("/v1/messages → chat refuses a wrong-typed %s naming it", async ([field, value], ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/messages", {
      model: "pcb-gpt",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
      [field]: value,
    });
    expect(res.status, res.text).toBe(400);
    expect(res.json().error.message).toContain(`\`${field}\``);
    expect(res.sent).toHaveLength(0);
  });

  test("an OpenAI refusal or content filter ends a Messages answer as refusal", async (ctx) => {
    if (!ready(ctx)) return;
    const refused = await post("/v1/messages", {
      model: "pcb-gpt",
      max_tokens: 64,
      messages: [{ role: "user", content: "REFUSE" }],
    });
    expect(refused.status, refused.text).toBe(200);
    expect(refused.json().stop_reason).toBe("refusal");
    expect(refused.json().content).toEqual([{ type: "text", text: "I can't help with that." }]);

    const filtered = await post("/v1/messages", {
      model: "pcb-gpt",
      max_tokens: 64,
      messages: [{ role: "user", content: "FILTER" }],
    });
    expect(filtered.json().stop_reason).toBe("refusal");
  });

  test("reasoning streamed inside an open tool call reaches the Messages stream", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/messages", {
      model: "pcb-gpt",
      max_tokens: 64,
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status, res.text).toBe(200);
    const events = frames(res.text);
    const thinking = events
      .filter((e) => e.type === "content_block_delta" && e.delta.type === "thinking_delta")
      .map((e) => e.delta.thinking)
      .join("");
    expect(thinking).toBe("thinking mid-call");
    const args = events
      .filter((e) => e.type === "content_block_delta" && e.delta.type === "input_json_delta")
      .map((e) => e.delta.partial_json)
      .join("");
    expect(args).toBe("{\"a\":1}");
  });

  test("an Anthropic error keeps the upstream's message on /v1/messages and count_tokens", async (ctx) => {
    if (!ready(ctx)) return;
    for (const path of ["/v1/messages", "/v1/messages/count_tokens"]) {
      const res = await post(path, {
        model: "pcb-claude",
        max_tokens: 64,
        messages: [{ role: "user", content: "FAIL" }],
      });
      expect(res.status, `${path}: ${res.text}`).toBe(400);
      expect(res.json().error).toMatchObject({ type: "invalid_request_error", message: UPSTREAM_ERROR });
    }
  });

  test("the caller's anthropic-beta reaches Anthropic; anthropic-version stays the gateway's", async (ctx) => {
    if (!ready(ctx)) return;
    for (const path of ["/v1/messages", "/v1/messages/count_tokens"]) {
      const res = await post(
        path,
        { model: "pcb-claude", max_tokens: 64, messages: [{ role: "user", content: "hi" }] },
        { "anthropic-beta": "context-1m-2025-08-07", "anthropic-version": "2099-01-01" },
      );
      expect(res.status, `${path}: ${res.text}`).toBe(200);
      const headers = res.sent.at(-1)!.headers;
      expect(headers["anthropic-beta"], path).toBe("context-1m-2025-08-07");
      expect(headers["anthropic-version"], path).toBe("2023-06-01");
    }
  });

  test("Claude below 4.5 on Vertex AI answers structured output on its native route", async (ctx) => {
    if (!ready(ctx)) return;
    const request = (stream: boolean) => ({
      model: "pcb-vertex-old-claude",
      max_tokens: 64,
      top_k: 3,
      stream,
      messages: [{ role: "user", content: "weather?" }],
      output_format: {
        type: "json_schema",
        schema: { type: "object", properties: { city: { type: "string" } } },
      },
    });

    const res = await post("/v1/messages", request(false));
    expect(res.status, res.text).toBe(200);
    const sent = res.sent.at(-1)!;
    expect(sent.path).toMatch(/:rawPredict$/);
    expect(sent.body.top_k).toBe(3);
    expect(sent.body).not.toHaveProperty("output_format");
    expect(sent.body.tools.map((t: any) => t.name)).toEqual(["json_tool_call"]);
    expect(sent.body.tool_choice).toEqual({ type: "tool", name: "json_tool_call" });
    expect(res.json().content).toEqual([{ type: "text", text: "{\"city\":\"Paris\"}" }]);
    expect(res.json().stop_reason).toBe("end_turn");

    const streamed = await post("/v1/messages", request(true));
    expect(streamed.status, streamed.text).toBe(200);
    expect(streamed.sent.at(-1)!.path).toMatch(/:rawPredict$/);
    const events = frames(streamed.text);
    const text = events
      .filter((e) => e.type === "content_block_delta" && e.delta.type === "text_delta")
      .map((e) => e.delta.text)
      .join("");
    expect(text).toBe("{\"city\":\"Paris\"}");
    expect(events.find((e) => e.type === "message_delta")?.delta.stop_reason).toBe("end_turn");
  });

  test("a structured-output request with no object schema keeps streaming upstream", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/messages", {
      model: "pcb-vertex-old-claude",
      max_tokens: 64,
      stream: true,
      messages: [{ role: "user", content: "weather?" }],
      output_format: { type: "json_schema" },
    });
    expect(res.status, res.text).toBe(200);
    const sent = res.sent.at(-1)!;
    expect(sent.path).toMatch(/:streamRawPredict$/);
    expect(sent.body).not.toHaveProperty("tools");
    const text = frames(res.text)
      .filter((e) => e.type === "content_block_delta" && e.delta.type === "text_delta")
      .map((e) => e.delta.text)
      .join("");
    expect(text).toBe("streamed");
  });
});
