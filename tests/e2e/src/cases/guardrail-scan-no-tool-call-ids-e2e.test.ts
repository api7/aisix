import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// What a guardrail scans of a tool call: its name and arguments, never its
// id. An id is an opaque handle linking a call to its result — the caller's
// replayed ids on the way in, the upstream's freshly minted ones on the way
// out — so neither direction may hand one to a guardrail, on any surface,
// streamed or not. A moderation guardrail at both hook points records every
// text it is sent.

const CALLER = "sk-guardrail-scan-no-tool-call-ids";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

// Caller-replayed ids, one carrying a thought signature, and the ids the
// upstream mints. None may appear in any scanned text.
const IN_ID = "call_INIDMARK__thought__INSIGMARK";
const IN_ITEM_ID = "fc_INITEMMARK";
const OUT_CHAT_ID = "call_OUTIDCHAT";
const OUT_TOOLU_ID = "toolu_OUTIDMSGS";
const OUT_CALL_ID = "call_OUTIDRESP";
const OUT_ITEM_ID = "fc_OUTIDITEM";
const ID_MARKS = ["INIDMARK", "INSIGMARK", "INITEMMARK", "OUTID"];

const TOOL = "get_weather";
const IN_ARGS = JSON.stringify({ city: "INARGMARK" });
const OUT_ARGS = JSON.stringify({ city: "OUTARGMARK" });

const CHAT_TOOL = {
  type: "function",
  function: {
    name: TOOL,
    parameters: { type: "object", properties: { city: { type: "string" } } },
  },
};
const ANTHROPIC_TOOL = { name: TOOL, input_schema: CHAT_TOOL.function.parameters };
const RESPONSES_TOOL = { type: "function", name: TOOL, parameters: CHAT_TOOL.function.parameters };

const sse = (events: unknown[]) =>
  events.map((e) => (typeof e === "string" ? `data: ${e}\n\n` : `data: ${JSON.stringify(e)}\n\n`)).join("");

const anthropicSse = (events: Array<Record<string, unknown>>) =>
  events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join("");

function chatReply(stream: boolean): string {
  if (!stream) {
    return JSON.stringify({
      id: "chatcmpl-1",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [{ id: OUT_CHAT_ID, type: "function", function: { name: TOOL, arguments: OUT_ARGS } }],
          },
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    });
  }
  const chunk = (delta: Record<string, unknown>, finish_reason: string | null = null) => ({
    id: "chatcmpl-1",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o",
    choices: [{ index: 0, delta, finish_reason }],
  });
  return sse([
    chunk({ role: "assistant", content: null }),
    chunk({ tool_calls: [{ index: 0, id: OUT_CHAT_ID, type: "function", function: { name: TOOL, arguments: "" } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: OUT_ARGS } }] }),
    chunk({}, "tool_calls"),
    "[DONE]",
  ]);
}

function messagesReply(stream: boolean): string {
  if (!stream) {
    return JSON.stringify({
      id: "msg_1",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-5",
      content: [{ type: "tool_use", id: OUT_TOOLU_ID, name: TOOL, input: JSON.parse(OUT_ARGS) }],
      stop_reason: "tool_use",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
  }
  return anthropicSse([
    {
      type: "message_start",
      message: {
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5",
        content: [],
        stop_reason: null,
        usage: { input_tokens: 1, output_tokens: 1 },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: OUT_TOOLU_ID, name: TOOL, input: {} } },
    { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: OUT_ARGS } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 1 } },
    { type: "message_stop" },
  ]);
}

function responsesReply(stream: boolean): string {
  const item = { type: "function_call", id: OUT_ITEM_ID, call_id: OUT_CALL_ID, name: TOOL, arguments: OUT_ARGS, status: "completed" };
  const response = {
    id: "resp_1",
    object: "response",
    status: "completed",
    model: "gpt-4o",
    output: [item],
    usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
  };
  if (!stream) return JSON.stringify(response);
  return sse([
    { type: "response.created", response: { ...response, status: "in_progress", output: [] } },
    { type: "response.output_item.added", output_index: 0, item: { ...item, arguments: "", status: "in_progress" } },
    { type: "response.function_call_arguments.delta", item_id: OUT_ITEM_ID, output_index: 0, delta: OUT_ARGS },
    { type: "response.function_call_arguments.done", item_id: OUT_ITEM_ID, output_index: 0, name: TOOL, arguments: OUT_ARGS },
    { type: "response.output_item.done", output_index: 0, item },
    { type: "response.completed", response },
  ]);
}

describe("guardrails never scan a tool-call id", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let etcdReachable = false;
  const moderated: string[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // One server answers moderation, OpenAI Chat / Responses and Anthropic
    // Messages (+ count_tokens), streamed when the request asks for it.
    upstream = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        const path = (req.url ?? "/").split("?")[0];
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
        const stream = body.stream === true;
        let reply: string;
        if (path.endsWith("/moderations")) {
          moderated.push(JSON.stringify(body.input));
          reply = JSON.stringify({
            id: "modr-1",
            model: "omni-moderation-latest",
            results: [{ flagged: false, categories: {}, category_scores: {} }],
          });
        } else if (path.endsWith("/count_tokens")) {
          reply = JSON.stringify({ input_tokens: 12 });
        } else if (path.endsWith("/chat/completions")) {
          reply = chatReply(stream);
        } else if (path.endsWith("/messages")) {
          reply = messagesReply(stream);
        } else if (path.endsWith("/responses")) {
          reply = responsesReply(stream);
        } else {
          res.statusCode = 404;
          reply = "{}";
        }
        res.setHeader("content-type", stream ? "text/event-stream" : "application/json");
        res.end(reply);
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const base = `http://127.0.0.1:${(upstream.address() as { port: number }).port}`;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const openaiPk = await seed.createProviderKey({ display_name: "ids-openai-pk", secret: "sk-mock", api_base: `${base}/v1` });
    await seed.createModel({ display_name: "ids-gpt", provider: "openai", model_name: "gpt-4o", provider_key_id: openaiPk.id });
    const anthropicPk = await seed.createProviderKey({
      display_name: "ids-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: base,
    });
    await seed.createModel({
      display_name: "ids-claude",
      provider: "anthropic",
      model_name: "claude-sonnet-4-5",
      provider_key_id: anthropicPk.id,
    });
    await seed.createPassthroughRoute({
      name: "ids-pt",
      path_prefix: "/ids-pt",
      target_url: base,
      provider_key_id: openaiPk.id,
    });
    await seed.createGuardrail({
      name: "ids-moderation",
      enabled: true,
      kind: "openai_moderation",
      hook_point: "both",
      fail_open: false,
      api_key: "sk-moderation",
      endpoint: base,
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"], allowed_routes: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await res.text();
      return res.status === 200;
    });
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  /** Send one request and return every text the guardrail was sent for it. */
  async function scanned(path: string, body: Record<string, unknown>): Promise<string[]> {
    moderated.length = 0;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "x-api-key": CALLER, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    if (body.stream === true) expect(text).not.toContain("guardrail");
    return [...moderated];
  }

  function expectNoIds(texts: string[]) {
    for (const text of texts) {
      for (const mark of ID_MARKS) expect(text, `id ${mark} reached the guardrail`).not.toContain(mark);
    }
  }

  const chatHistory = [
    { role: "user", content: "Weather in Paris?" },
    {
      role: "assistant",
      content: null,
      tool_calls: [{ id: IN_ID, type: "function", function: { name: TOOL, arguments: IN_ARGS } }],
    },
    { role: "tool", tool_call_id: IN_ID, content: "sunny" },
  ];
  const messagesHistory = [
    { role: "user", content: "Weather in Paris?" },
    { role: "assistant", content: [{ type: "tool_use", id: IN_ID, name: TOOL, input: JSON.parse(IN_ARGS) }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: IN_ID, content: "sunny" }] },
  ];
  const responsesHistory = [
    { role: "user", content: "Weather in Paris?" },
    { type: "function_call", id: IN_ITEM_ID, call_id: IN_ID, name: TOOL, arguments: IN_ARGS },
    { type: "function_call_output", call_id: IN_ID, output: "sunny" },
  ];

  const question = "Weather in Paris?";
  const surfaces = [
    {
      name: "/v1/chat/completions",
      path: "/v1/chat/completions",
      body: (history: boolean) => ({
        model: "ids-gpt",
        messages: history ? chatHistory : [{ role: "user", content: question }],
        tools: [CHAT_TOOL],
      }),
    },
    {
      name: "/v1/messages",
      path: "/v1/messages",
      body: (history: boolean) => ({
        model: "ids-claude",
        max_tokens: 64,
        messages: history ? messagesHistory : [{ role: "user", content: question }],
        tools: [ANTHROPIC_TOOL],
      }),
    },
    {
      name: "/v1/chat/completions bridged to Anthropic",
      path: "/v1/chat/completions",
      body: (history: boolean) => ({
        model: "ids-claude",
        max_tokens: 64,
        messages: history ? chatHistory : [{ role: "user", content: question }],
        tools: [CHAT_TOOL],
      }),
    },
    {
      name: "/v1/messages bridged to OpenAI Chat",
      path: "/v1/messages",
      body: (history: boolean) => ({
        model: "ids-gpt",
        max_tokens: 64,
        messages: history ? messagesHistory : [{ role: "user", content: question }],
        tools: [ANTHROPIC_TOOL],
      }),
    },
    {
      name: "/v1/responses",
      path: "/v1/responses",
      body: (history: boolean) => ({ model: "ids-gpt", input: history ? responsesHistory : question, tools: [RESPONSES_TOOL] }),
    },
    {
      name: "passthrough chat envelope",
      path: "/ids-pt/v1/chat/completions",
      body: (history: boolean) => ({
        model: "gpt-4o",
        messages: history ? chatHistory : [{ role: "user", content: question }],
        tools: [CHAT_TOOL],
      }),
    },
    {
      name: "passthrough Responses envelope",
      path: "/ids-pt/v1/responses",
      body: (history: boolean) => ({ model: "gpt-4o", input: history ? responsesHistory : question, tools: [RESPONSES_TOOL] }),
    },
  ];

  for (const stream of [false, true]) {
    const mode = stream ? "streamed" : "buffered";
    for (const surface of surfaces) {
      // A fresh question: the only ids in play are the ones the upstream mints.
      test(`${surface.name} output (${mode})`, async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const texts = await scanned(surface.path, { ...surface.body(false), stream });
        expectNoIds(texts);
        expect(texts.some((t) => t.includes("OUTARGMARK")), "output arguments scanned").toBe(true);
      });

      test(`${surface.name} input (${mode})`, async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const texts = await scanned(surface.path, { ...surface.body(true), stream });
        expectNoIds(texts);
        expect(texts.some((t) => t.includes(TOOL) && t.includes("INARGMARK")), "input name and arguments scanned").toBe(
          true,
        );
      });
    }
  }

  // The Responses input carries other opaque handles beside call ids: an item
  // reference, MCP approval ids, and a replayed reasoning item's id and
  // ciphertext. Neither the typed route nor a passthrough route scans them.
  const handleHistory = [
    ...responsesHistory,
    { type: "item_reference", id: "fc_OUTIDREF" },
    { type: "mcp_approval_request", id: "mcpr_OUTIDAPPROVAL", server_label: "srv", name: "mcp_lookup", arguments: "{}" },
    { type: "mcp_approval_response", approval_request_id: "mcpr_OUTIDAPPROVAL", approve: true, reason: "APPROVEREASON" },
    {
      type: "reasoning",
      id: "rs_OUTIDREASONING",
      encrypted_content: "OUTIDCIPHERTEXT",
      summary: [{ type: "summary_text", text: "REASONSUMMARY" }],
    },
  ];
  for (const path of ["/v1/responses", "/ids-pt/v1/responses"]) {
    test(`${path} input leaves out item references, approval ids and reasoning ciphertext`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const model = path.startsWith("/ids-pt") ? "gpt-4o" : "ids-gpt";
      const texts = await scanned(path, { model, input: handleHistory, tools: [RESPONSES_TOOL] });
      expectNoIds(texts);
      for (const text of ["INARGMARK", "APPROVEREASON", "REASONSUMMARY"]) {
        expect(texts.some((t) => t.includes(text)), `${text} scanned`).toBe(true);
      }
    });
  }

  test("/v1/messages/count_tokens", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const texts = await scanned("/v1/messages/count_tokens", { model: "ids-claude", messages: messagesHistory, tools: [ANTHROPIC_TOOL] });
    expectNoIds(texts);
    expect(texts.some((t) => t.includes(TOOL) && t.includes("INARGMARK")), "input name and arguments scanned").toBe(true);
  });

  test("passthrough Anthropic body on the chat envelope", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const texts = await scanned("/ids-pt/v1/messages", { model: "claude-sonnet-4-5", max_tokens: 64, messages: messagesHistory });
    expectNoIds(texts);
    expect(texts.some((t) => t.includes("INARGMARK")), "tool_use input scanned").toBe(true);
    expect(texts.some((t) => t.includes("OUTARGMARK")), "output tool_use input scanned").toBe(true);
  });
});
