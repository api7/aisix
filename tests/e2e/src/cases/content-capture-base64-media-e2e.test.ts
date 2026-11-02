import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// Under `content_mode: full`, base64 media a client sends is reported as a part
// whose payload is a placeholder, on every path a record can take: the
// structured input messages, the flat prompt of a request over the cap, and each
// family's own media shapes. Nothing here depends on the payload's type or size.
// The image endpoints' `b64_json` answers are captured the same way, while the
// caller still receives them whole.

const CALLER = "sk-content-capture-base64-image";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "base64-image";
const CONTENT_MAX_BYTES = 64 * 1024;
const KIND_SERVER = 2;

// Base64 alphabet only; the marker is what must never reach a record.
const MARKER = "aW1hZ2VwYXlsb2FkbWFya2Vy";
const SMALL = MARKER.repeat(4);
const LARGE = MARKER.repeat(16 * 1024);
// Past the size at which the gateway stops waiting for a stream frame to end.
const OVERSIZED = MARKER.repeat(64 * 1024);
const dataUri = (payload: string) => `data:image/png;base64,${payload}`;
const placeholder = (bytes: number) => `[aisix: base64 data uri omitted, ${bytes} bytes]`;
const imagePart = (bytes: number) => ({
  type: "blob",
  modality: "image",
  mime_type: "image/png",
  content: placeholder(bytes),
});

const CHAT_BODY = {
  id: "chatcmpl-capture",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};
const CHAT_STREAM = [
  { ...CHAT_BODY, object: "chat.completion.chunk", usage: undefined, choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }] },
  { ...CHAT_BODY, object: "chat.completion.chunk", usage: undefined, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
];
// One of each media carrier an MCP tool result has, beside text that must stay.
const MCP_TEXT = `tool output, a payload named in prose stays: ${SMALL.slice(0, 8)}`;
const mcpResult = (image: string) => ({
  content: [
    { type: "text", text: MCP_TEXT },
    { type: "image", data: image, mimeType: "image/png" },
    { type: "audio", data: SMALL, mimeType: "audio/wav" },
    { type: "resource", resource: { uri: "file:///a.pdf", mimeType: "application/pdf", blob: SMALL } },
    { type: "resource", resource: { uri: "file:///a.log", mimeType: "text/plain", text: "a log line" } },
    { type: "resource_link", uri: "file:///b.png", name: "b.png" },
    { type: "text", text: dataUri(SMALL) },
  ],
  structuredContent: { thumbnail: dataUri(SMALL), width: 16 },
});
const mcpResultExported = (image: string) => ({
  content: [
    { type: "text", text: MCP_TEXT },
    { type: "image", data: placeholder(image.length), mimeType: "image/png" },
    { type: "audio", data: placeholder(SMALL.length), mimeType: "audio/wav" },
    { type: "resource", resource: { uri: "file:///a.pdf", mimeType: "application/pdf", blob: placeholder(SMALL.length) } },
    { type: "resource", resource: { uri: "file:///a.log", mimeType: "text/plain", text: "a log line" } },
    { type: "resource_link", uri: "file:///b.png", name: "b.png" },
    { type: "text", text: `data:image/png;base64,${placeholder(uriBytes(SMALL))}` },
  ],
  structuredContent: { thumbnail: `data:image/png;base64,${placeholder(uriBytes(SMALL))}`, width: 16 },
});
const IMAGE_BODY = {
  created: 1_700_000_000,
  data: [{ b64_json: SMALL, revised_prompt: "a small cat" }],
};
const RESPONSES_BODY = {
  id: "resp_capture",
  object: "response",
  status: "completed",
  model: "gpt-4o-mini",
  output: [
    { type: "message", id: "msg_1", role: "assistant", status: "completed", content: [{ type: "output_text", text: "ok" }] },
  ],
  usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
};
const ANTHROPIC_BODY = {
  id: "msg_capture",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-5",
  content: [{ type: "text", text: "ok" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 1, output_tokens: 1 },
};

type Entry = {
  name: string;
  path: string;
  model: string;
  /** A user turn with this text (none when undefined) and image payload. */
  body: (text: string | undefined, payload: string) => Record<string, unknown>;
  /** The placeholder's byte count for this payload. */
  bytes: (payload: string) => number;
};

const chatBody = (stream: boolean) => (text: string | undefined, payload: string) => ({
  model: "cap-openai",
  stream,
  messages: [
    {
      role: "user",
      content: [
        ...(text === undefined ? [] : [{ type: "text", text }]),
        { type: "image_url", image_url: { url: dataUri(payload) } },
      ],
    },
  ],
});
const messagesBody = (model: string) => (text: string | undefined, payload: string) => ({
  model,
  max_tokens: 16,
  messages: [
    {
      role: "user",
      content: [
        ...(text === undefined ? [] : [{ type: "text", text }]),
        { type: "image", source: { type: "base64", media_type: "image/png", data: payload } },
      ],
    },
  ],
});
const responsesBody = (model: string) => (text: string | undefined, payload: string) => ({
  model,
  input: [
    {
      role: "user",
      content: [
        ...(text === undefined ? [] : [{ type: "input_text", text }]),
        { type: "input_image", image_url: dataUri(payload) },
      ],
    },
  ],
});
const uriBytes = (payload: string) => dataUri(payload).length;

const ENTRIES: Entry[] = [
  { name: "chat", path: "/v1/chat/completions", model: "cap-openai", body: chatBody(false), bytes: uriBytes },
  { name: "chat streamed", path: "/v1/chat/completions", model: "cap-openai", body: chatBody(true), bytes: uriBytes },
  { name: "messages native", path: "/v1/messages", model: "cap-anthropic", body: messagesBody("cap-anthropic"), bytes: (p) => p.length },
  { name: "messages bridged", path: "/v1/messages", model: "cap-openai", body: messagesBody("cap-openai"), bytes: (p) => p.length },
  { name: "responses native", path: "/v1/responses", model: "cap-openai", body: responsesBody("cap-openai"), bytes: uriBytes },
  { name: "responses bridged", path: "/v1/responses", model: "cap-bridged", body: responsesBody("cap-bridged"), bytes: uriBytes },
];

describe("base64 media under content_mode full", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let sls: MockSls | undefined;
  let otlp: MockOtlp | undefined;
  let mcp: McpUpstream | undefined;
  let mcpLarge: McpUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        const path = (req.url ?? "").split("?")[0];
        if (path.endsWith("/pt/buffered")) {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(IMAGE_BODY));
          return;
        }
        if (path.endsWith("/pt/stream") || path.endsWith("/pt/stream-oversized")) {
          const payload = path.endsWith("/pt/stream") ? SMALL : OVERSIZED;
          res.writeHead(200, { "content-type": "text/event-stream" });
          void (async () => {
            const frame = `data: ${JSON.stringify({ type: "image_generation.completed", b64_json: payload })}\n\n`;
            for (let at = 0; at < frame.length; at += 64 * 1024) {
              res.write(frame.slice(at, at + 64 * 1024));
              await new Promise((r) => setImmediate(r));
            }
            res.end();
          })();
          return;
        }
        if (path.endsWith("/images/generations") || path.endsWith("/images/edits")) {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(IMAGE_BODY));
          return;
        }
        const streamed = JSON.parse(Buffer.concat(chunks).toString("utf8")).stream === true;
        if (path.endsWith("/chat/completions") && streamed) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(`${CHAT_STREAM.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("")}data: [DONE]\n\n`);
          return;
        }
        const body = path.endsWith("/responses")
          ? RESPONSES_BODY
          : path.endsWith("/messages")
            ? ANTHROPIC_BODY
            : CHAT_BODY;
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(body));
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");
    const base = `http://127.0.0.1:${addr.port}`;

    sls = await startMockSls();
    otlp = await startMockOtlp();
    mcp = await startMcpUpstream("media", { mediaResult: mcpResult(SMALL) });
    mcpLarge = await startMcpUpstream("large", { mediaResult: mcpResult(LARGE) });
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "cap-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
      content_max_bytes: CONTENT_MAX_BYTES,
    });
    await seed.createObservabilityExporter({
      name: "cap-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      content_mode: "full",
      content_max_bytes: CONTENT_MAX_BYTES,
    });
    const openai = await seed.createProviderKey({ display_name: "cap-openai-pk", secret: "sk-mock", api_base: `${base}/v1` });
    const bridged = await seed.createProviderKey({ display_name: "cap-bridged-pk", secret: "sk-mock", api_base: `${base}/v1`, apis: {} });
    const anthropic = await seed.createProviderKey({
      display_name: "cap-anthropic-pk",
      secret: "sk-mock",
      provider: "anthropic",
      adapter: "anthropic",
      api_base: base,
    });
    await seed.createModel({ display_name: "cap-openai", provider: "openai", model_name: "gpt-4o-mini", provider_key_id: openai.id });
    await seed.createModel({ display_name: "cap-bridged", provider: "openai", model_name: "gpt-4o-mini", provider_key_id: bridged.id });
    await seed.createModel({ display_name: "cap-anthropic", provider: "anthropic", model_name: "claude-sonnet-4-5", provider_key_id: anthropic.id });
    await seed.createPassthroughRoute({ name: "cap-pt", path_prefix: "/cap-pt", target_url: base, provider_key_id: openai.id });
    await seed.update("mcp_servers", randomUUID(), { display_name: "media", url: mcp.url, enabled: true });
    await seed.update("mcp_servers", randomUUID(), { display_name: "large", url: mcpLarge.url, enabled: true });
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
      mcp_access: { allow: ["*"] },
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await sls?.close();
    await mcp?.close();
    await mcpLarge?.close();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function send(path: string, body: unknown): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        ...(path === "/v1/messages" ? { "x-api-key": CALLER } : { authorization: `Bearer ${CALLER}` }),
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text.slice(0, 500)).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return requestId;
  }

  /** The request's SLS row and OTLP spans, with no image payload in either. */
  async function records(requestId: string): Promise<{ row: Map<string, string>; span: CapturedSpan }> {
    const row = await waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, "SLS row");
    const contentAttr = (s: CapturedSpan) => "gen_ai.input.messages" in s.attributes || "aisix.content.prompt" in s.attributes;
    const mine = () => otlp!.spans.filter((s) => s.attributes["aisix.request_id"] === requestId);
    await expect
      .poll(() => mine().some((s) => s.kind === KIND_SERVER) && mine().some(contentAttr), { timeout: 15_000 })
      .toBe(true);
    expect(JSON.stringify([...row.entries()]).includes(MARKER), "payload in the SLS row").toBe(false);
    expect(JSON.stringify(mine()).includes(MARKER), "payload in a span").toBe(false);
    return { row, span: mine().find(contentAttr)! };
  }

  function inputMessages(row: Map<string, string>, span: CapturedSpan): any[] {
    expect(row.has("prompt"), "the record is structured").toBe(false);
    expect(span.attributes["aisix.content.prompt"]).toBeUndefined();
    const fromRow = JSON.parse(row.get("input_messages") ?? "null");
    expect(JSON.parse(String(span.attributes["gen_ai.input.messages"]))).toEqual(fromRow);
    return fromRow;
  }

  describe.each(ENTRIES)("$name", ({ path, body, bytes }) => {
    test("a small image with text is an image part", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { row, span } = await records(await send(path, body("what is this?", SMALL)));
      expect(inputMessages(row, span)[0].parts).toEqual([
        { type: "text", content: "what is this?" },
        imagePart(bytes(SMALL)),
      ]);
    });

    test("an image over the content cap is an image part too", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      expect(LARGE.length).toBeGreaterThan(CONTENT_MAX_BYTES);
      const { row, span } = await records(await send(path, body("what is this?", LARGE)));
      expect(inputMessages(row, span)[0].parts).toEqual([
        { type: "text", content: "what is this?" },
        imagePart(bytes(LARGE)),
      ]);
    });

    test("an image with no text is the turn's only part", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { row, span } = await records(await send(path, body(undefined, SMALL)));
      expect(inputMessages(row, span)[0].parts).toEqual([imagePart(bytes(SMALL))]);
    });

    test("text over the cap leaves a flat prompt with the placeholder", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { row, span } = await records(await send(path, body("x".repeat(CONTENT_MAX_BYTES * 2), SMALL)));
      expect(row.has("input_messages")).toBe(false);
      expect(row.get("prompt")).toContain(placeholder(bytes(SMALL)));
      expect(span.attributes["aisix.content.prompt"]).toContain(placeholder(bytes(SMALL)));
    });
  });

  test("chat: a block's text is not counted twice against the cap", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const text = "y".repeat(Math.floor(CONTENT_MAX_BYTES * 0.7));
    const { row, span } = await records(await send("/v1/chat/completions", chatBody(false)(text, SMALL)));
    expect(inputMessages(row, span)[0].parts[0]).toEqual({ type: "text", content: text });
  });

  test("chat: an image URL stays a URL", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const url = "https://example.com/cat.png";
    const { row, span } = await records(
      await send("/v1/chat/completions", {
        model: "cap-openai",
        messages: [{ role: "user", content: [{ type: "image_url", image_url: { url } }] }],
      }),
    );
    expect(inputMessages(row, span)[0].parts).toEqual([{ type: "uri", modality: "image", uri: url }]);
  });

  test("chat: images an assistant turn replays are image parts", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { row, span } = await records(
      await send("/v1/chat/completions", {
        model: "cap-openai",
        messages: [
          { role: "user", content: "draw" },
          {
            role: "assistant",
            content: "here",
            images: [{ type: "image_url", image_url: { url: dataUri(SMALL) }, index: 0 }],
          },
          { role: "user", content: "again" },
        ],
      }),
    );
    expect(inputMessages(row, span)[1].parts).toEqual([
      { type: "text", content: "here" },
      imagePart(uriBytes(SMALL)),
    ]);
  });

  test("messages: an image inside a tool result is omitted", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { row, span } = await records(
      await send("/v1/messages", {
        model: "cap-anthropic",
        max_tokens: 16,
        messages: [
          { role: "user", content: "screenshot please" },
          { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "shot", input: {} }] },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "t1",
                content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: SMALL } }],
              },
            ],
          },
        ],
      }),
    );
    expect(JSON.stringify(inputMessages(row, span)[2])).toContain(placeholder(SMALL.length));
  });

  test("chat: audio and a file keep their parts, not their payloads", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const pdf = `data:application/pdf;base64,${SMALL}`;
    const { row, span } = await records(
      await send("/v1/chat/completions", {
        model: "cap-openai",
        messages: [
          {
            role: "user",
            content: [
              { type: "input_audio", input_audio: { data: SMALL, format: "wav" } },
              { type: "file", file: { file_data: pdf, filename: "a.pdf" } },
            ],
          },
        ],
      }),
    );
    expect(inputMessages(row, span)[0].parts).toEqual([
      { type: "blob", modality: "audio", mime_type: "audio/wav", content: placeholder(SMALL.length) },
      {
        type: "file",
        file: { file_data: `data:application/pdf;base64,${placeholder(pdf.length)}`, filename: "a.pdf" },
      },
    ]);
  });

  test("messages: a base64 document keeps its part and media type", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { row, span } = await records(
      await send("/v1/messages", {
        model: "cap-anthropic",
        max_tokens: 16,
        messages: [
          {
            role: "user",
            content: [{ type: "document", source: { type: "base64", media_type: "application/pdf", data: SMALL } }],
          },
        ],
      }),
    );
    expect(inputMessages(row, span)[0].parts).toEqual([
      { type: "blob", modality: "document", mime_type: "application/pdf", content: placeholder(SMALL.length) },
    ]);
  });

  test("responses: an input file keeps its part", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const pdf = `data:application/pdf;base64,${SMALL}`;
    const { row, span } = await records(
      await send("/v1/responses", {
        model: "cap-openai",
        input: [{ role: "user", content: [{ type: "input_file", filename: "a.pdf", file_data: pdf }] }],
      }),
    );
    expect(inputMessages(row, span)[0].parts).toEqual([
      { type: "input_file", filename: "a.pdf", file_data: `data:application/pdf;base64,${placeholder(pdf.length)}` },
    ]);
  });

  describe("MCP tool results", () => {
    async function callTool(server: string, args: Record<string, unknown>) {
      const res = await fetch(`${app!.proxyUrl}/mcp/${server}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER}`,
          "content-type": "application/json",
          accept: "application/json, text/event-stream",
        },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "tools/call",
          params: { name: "media", arguments: args },
        }),
      });
      const text = await res.text();
      expect(res.status, text.slice(0, 500)).toBe(200);
      return { reply: JSON.parse(text), requestId: res.headers.get("x-aisix-request-id") ?? "" };
    }

    /** The call's exported arguments and result, with no payload anywhere in its records. */
    async function exported(requestId: string) {
      expect(requestId).not.toBe("");
      const row = await waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, "SLS row");
      const mine = () => otlp!.spans.filter((s) => s.attributes["aisix.request_id"] === requestId);
      await expect.poll(() => mine().some((s) => "gen_ai.tool.call.result" in s.attributes), { timeout: 15_000 }).toBe(true);
      expect(JSON.stringify([...row.entries()]).includes(MARKER), "payload in the SLS row").toBe(false);
      expect(JSON.stringify(mine()).includes(MARKER), "payload in a span").toBe(false);
      const span = mine().find((s) => "gen_ai.tool.call.result" in s.attributes)!;
      const result = JSON.parse(row.get("response") ?? "null");
      expect(JSON.parse(String(span.attributes["gen_ai.tool.call.result"]))).toEqual(result);
      return { args: JSON.parse(row.get("prompt") ?? "null"), result };
    }

    test("image, audio, resource blob and data URIs keep their blocks, not their payloads", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { reply, requestId } = await callTool("media", { image: dataUri(SMALL) });
      // Only the export changes: the caller receives every payload.
      expect(reply.result.content).toEqual(mcpResult(SMALL).content);
      expect(reply.result.structuredContent).toEqual(mcpResult(SMALL).structuredContent);
      const { args, result } = await exported(requestId);
      expect(result.content).toEqual(mcpResultExported(SMALL).content);
      expect(result.structuredContent).toEqual(mcpResultExported(SMALL).structuredContent);
      expect(args).toEqual({ image: `data:image/png;base64,${placeholder(uriBytes(SMALL))}` });
    });

    test("an image over the content cap keeps every block", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      expect(LARGE.length).toBeGreaterThan(CONTENT_MAX_BYTES);
      const { reply, requestId } = await callTool("large", {});
      expect(reply.result.content[1].data).toBe(LARGE);
      const { result } = await exported(requestId);
      expect(result.content).toEqual(mcpResultExported(LARGE).content);
    });

    test("a result reporting a tool error is omitted the same way", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const { reply, requestId } = await callTool("media", { failed: true });
      expect(reply.result.isError).toBe(true);
      const { result } = await exported(requestId);
      expect(result.isError).toBe(true);
      expect(result.content).toEqual(mcpResultExported(SMALL).content);
    });
  });

  describe("passthrough routes", () => {
    const route = (suffix: string, body: unknown) =>
      fetch(`${app!.proxyUrl}/cap-pt/pt/${suffix}`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    /** The request's SLS row; `records` holds it and the spans to no payload. */
    const row = async (res: Response) => {
      const requestId = res.headers.get("x-aisix-request-id") ?? "";
      expect(requestId).not.toBe("");
      return (await records(requestId)).row;
    };

    test("a request body's media and a buffered image answer are omitted", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const res = await route("buffered", {
        contents: [{ parts: [{ inlineData: { mimeType: "image/png", data: SMALL } }, { text: "hello" }] }],
        image: dataUri(SMALL),
      });
      const answer = await res.text();
      expect(res.status, answer.slice(0, 500)).toBe(200);
      expect(JSON.parse(answer).data[0].b64_json).toBe(SMALL);
      const log = await row(res);
      const prompt = JSON.parse(log.get("prompt") ?? "null");
      expect(prompt.contents[0].parts).toEqual([
        { inlineData: { mimeType: "image/png", data: placeholder(SMALL.length) } },
        { text: "hello" },
      ]);
      expect(prompt.image).toBe(`data:image/png;base64,${placeholder(uriBytes(SMALL))}`);
      expect(JSON.parse(log.get("response") ?? "null").data).toEqual([
        { b64_json: placeholder(SMALL.length), revised_prompt: "a small cat" },
      ]);
    });

    test.each([
      { name: "small frame", suffix: "stream", payload: SMALL, body: { describe: "a cat", stream: true } },
      { name: "oversized frame", suffix: "stream-oversized", payload: OVERSIZED, body: { describe: "a cat", stream: true } },
      // The same frame on a route that reads the stream as a known protocol.
      { name: "oversized frame, known protocol", suffix: "stream-oversized", payload: OVERSIZED, body: { prompt: "a cat", stream: true } },
    ])("a streamed image answer is omitted: $name", async ({ suffix, payload, body }) => {
      if (!etcdReachable || !app) return;
      const res = await route(suffix, body);
      const answer = await res.text();
      expect(res.status, answer.slice(0, 500)).toBe(200);
      // Only the export changes: the caller receives every byte.
      expect(answer).toContain(payload);
      const log = await row(res);
      if ("describe" in body) expect(log.get("response")).toContain("[aisix: base64 data uri omitted, ");
    });
  });

  describe("image endpoints", () => {
    async function expectCapturedWithoutPayload(res: Response) {
      const text = await res.text();
      expect(res.status, text.slice(0, 500)).toBe(200);
      // Only the export changes: the caller receives the image.
      expect(JSON.parse(text).data[0].b64_json).toBe(SMALL);
      const requestId = res.headers.get("x-aisix-request-id") ?? "";
      const row = await waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, "SLS row");
      expect(JSON.stringify([...row.entries()]).includes(MARKER), "payload in the SLS row").toBe(false);
      expect(JSON.parse(row.get("response") ?? "null").data).toEqual([
        { b64_json: placeholder(SMALL.length), revised_prompt: "a small cat" },
      ]);
    }

    test("a generation's b64_json is not exported", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      await expectCapturedWithoutPayload(
        await fetch(`${app.proxyUrl}/v1/images/generations`, {
          method: "POST",
          headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
          body: JSON.stringify({ model: "cap-openai", prompt: "a cat", response_format: "b64_json" }),
        }),
      );
    });

    test("an edit's b64_json is not exported", async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const form = new FormData();
      form.set("model", "cap-openai");
      form.set("prompt", "a cat");
      form.set("image", new Blob([Buffer.from("not a real image")], { type: "image/png" }), "a.png");
      await expectCapturedWithoutPayload(
        await fetch(`${app.proxyUrl}/v1/images/edits`, {
          method: "POST",
          headers: { authorization: `Bearer ${CALLER}` },
          body: form,
        }),
      );
    });
  });

  test("responses: a replayed image generation call is omitted", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const { row, span } = await records(
      await send("/v1/responses", {
        model: "cap-openai",
        input: [
          { role: "user", content: [{ type: "input_text", text: "draw" }] },
          { type: "image_generation_call", id: "ig_1", status: "completed", result: SMALL },
        ],
      }),
    );
    expect(JSON.stringify(inputMessages(row, span)[1])).toContain(placeholder(SMALL.length));
  });
});
