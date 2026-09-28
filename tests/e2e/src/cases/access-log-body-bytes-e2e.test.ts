import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the access-log line reports how many request-body bytes the gateway
// read and how many response-body bytes it handed to the server. Every
// expected value below is what THIS client sent or received, byte for
// byte: SSE framing and keep-alive heartbeats are bytes the client
// receives, so they are in the count; a request refused before its body
// was read, or a response whose head was never written, has no size to
// report and must say nothing rather than `0`.

const KEY = "sk-access-log-body-bytes";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const BODY_LIMIT = 4096;
const HEARTBEAT_INTERVAL_S = 1;
const SPEECH_CHUNKS = ["ID3", "audio-part-1", "audio-part-2", "audio-part-3"];
const SPEECH_CHUNK_DELAY_MS = 300;

const chatChunk = (content: string) =>
  JSON.stringify({
    id: "c-bytes",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });
const chatFinish = () =>
  JSON.stringify({
    id: "c-bytes",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  });
const CHAT_COMPLETION = {
  id: "chatcmpl-bytes",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "hello there" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
};

const MESSAGES_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_bytes",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 3, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }),
  JSON.stringify({ type: "message_stop" }),
];

const RESPONSES_EVENTS = [
  JSON.stringify({ type: "response.created", response: { id: "resp-bytes", model: "gpt-4o-mini" } }),
  JSON.stringify({ type: "response.output_text.delta", delta: "hi" }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp-bytes",
      status: "completed",
      model: "gpt-4o-mini",
      usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
    },
  }),
  "[DONE]",
];

/** The value of an unsigned-integer field on the line, or `undefined`. */
function numField(line: string, name: string): number | undefined {
  const m = line.match(new RegExp(`(?:^| )${name}=(\\d+)(?: |$)`));
  return m ? Number(m[1]) : undefined;
}

describe("the access log reports request and response body sizes", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  const up = async (opts: Parameters<typeof startOpenAiUpstream>[0]) => {
    const u = await startOpenAiUpstream(opts);
    upstreams.push(u);
    return u;
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const chatUp = await up({ nonStreamBody: CHAT_COMPLETION });
    // Silent long enough for at least two heartbeats before the first frame.
    const slowUp = await up({ firstEventDelayMs: 2_500, streamEvents: [chatChunk("late"), chatFinish(), "[DONE]"] });
    const longUp = await up({
      eventDelayMs: 100,
      streamEvents: [...Array.from({ length: 30 }, (_, i) => chatChunk(`part-${i}`)), chatFinish(), "[DONE]"],
    });
    const hangUp = await up({ responseDelayMs: 5_000, nonStreamBody: CHAT_COMPLETION });
    const anthropicUp = await up({ streamEvents: MESSAGES_EVENTS });
    const responsesUp = await up({ streamEvents: RESPONSES_EVENTS });
    const speechUp = await up({
      rawBodyChunks: SPEECH_CHUNKS,
      rawContentType: "audio/mpeg",
      eventDelayMs: SPEECH_CHUNK_DELAY_MS,
    });
    const passthroughUp = await up({ nonStreamBody: CHAT_COMPLETION });

    app = await spawnApp({
      logLevel: "info",
      requestBodyLimitBytes: BODY_LIMIT,
      requestId: { accept_headers: ["x-request-id"] },
      extra: { downstream: { sse_keepalive_interval_secs: HEARTBEAT_INTERVAL_S } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const model = async (name: string, u: OpenAiUpstream, provider = "openai", modelName = "gpt-4o-mini") => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        provider,
        api_base: provider === "anthropic" ? u.baseUrl : `${u.baseUrl}/v1`,
      });
      await seed.createModel({ display_name: name, provider, model_name: modelName, provider_key_id: pk.id });
      return pk;
    };
    await model("bytes-chat", chatUp);
    await model("bytes-slow", slowUp);
    await model("bytes-long", longUp);
    await model("bytes-hang", hangUp);
    await model("bytes-claude", anthropicUp, "anthropic", "claude-3-5-haiku-20241022");
    await model("bytes-responses", responsesUp);
    await model("bytes-tts", speechUp, "openai", "tts-1");
    const ptrPk = await seed.createProviderKey({ display_name: "bytes-ptr-pk", secret: "sk-mock", api_base: "http://unused" });
    await seed.createPassthroughRoute({
      name: "bytes-ptr",
      path_prefix: "/bytes-ptr",
      target_url: `${passthroughUp.baseUrl}/v1`,
      provider_key_id: ptrPk.id,
    });
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"], allowed_routes: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  const headers = (requestId: string, extra: Record<string, string> = {}) => ({
    authorization: `Bearer ${KEY}`,
    "x-api-key": KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
    "x-request-id": requestId,
    ...extra,
  });

  /** Send a request, read the whole response, and return what crossed the wire. */
  async function send(
    method: string,
    path: string,
    body?: string,
    extraHeaders: Record<string, string> = {},
  ): Promise<{ status: number; requestId: string; sent: number; received: number; text: string }> {
    const requestId = randomUUID();
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method,
      headers: headers(requestId, extraHeaders),
      body,
    });
    const bytes = Buffer.from(await res.arrayBuffer());
    return {
      status: res.status,
      requestId,
      sent: body === undefined ? 0 : Buffer.byteLength(body),
      received: bytes.length,
      text: bytes.toString("utf8"),
    };
  }

  const lineFor = (requestId: string, what: string) =>
    waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && l.includes(`request_id="${requestId}"`),
      what,
      15_000,
    );

  test("a non-streamed chat completion reports both bodies exactly", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "bytes-chat", messages: [{ role: "user", content: "count me" }] });
    const r = await send("POST", "/v1/chat/completions", body);
    expect(r.status, r.text).toBe(200);
    const line = await lineFor(r.requestId, "the chat line");
    expect(numField(line, "request_body_bytes"), line).toBe(r.sent);
    expect(numField(line, "response_body_bytes"), line).toBe(r.received);
  });

  test("an SSE stream counts its framing and every keep-alive heartbeat", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "bytes-slow", stream: true, messages: [{ role: "user", content: "wait" }] });
    const r = await send("POST", "/v1/chat/completions", body);
    expect(r.status, r.text).toBe(200);
    // Premise: the client really did receive heartbeat frames, so a count
    // taken inside the heartbeat wrapper would come out short.
    expect(r.text.split(":\n\n").length - 1, r.text).toBeGreaterThanOrEqual(2);
    expect(r.text).toContain("data: [DONE]");
    const line = await lineFor(r.requestId, "the streamed chat line");
    expect(numField(line, "request_body_bytes"), line).toBe(r.sent);
    expect(numField(line, "response_body_bytes"), line).toBe(r.received);
  }, 30_000);

  test("streamed /v1/messages and /v1/responses count the same way", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    for (const [path, body] of [
      ["/v1/messages", { model: "bytes-claude", stream: true, max_tokens: 16, messages: [{ role: "user", content: "go" }] }],
      ["/v1/responses", { model: "bytes-responses", stream: true, input: "go" }],
    ] as const) {
      const r = await send("POST", path, JSON.stringify(body));
      expect(r.status, r.text).toBe(200);
      const line = await lineFor(r.requestId, `the ${path} line`);
      expect(numField(line, "request_body_bytes"), `${path}: ${line}`).toBe(r.sent);
      expect(numField(line, "response_body_bytes"), `${path}: ${line}`).toBe(r.received);
    }
  });

  test("a client that leaves mid-stream is charged what was handed over until then", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "bytes-long", stream: true, messages: [{ role: "user", content: "long" }] });
    // The same stream read to the end, for the size a completed one has.
    const full = await send("POST", "/v1/chat/completions", body);
    expect(full.status, full.text).toBe(200);

    const requestId = randomUUID();
    const abort = new AbortController();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: headers(requestId),
      body,
      signal: abort.signal,
    });
    expect(res.status).toBe(200);
    const reader = res.body!.getReader();
    const first = await reader.read();
    const received = first.value?.length ?? 0;
    expect(received).toBeGreaterThan(0);
    abort.abort();
    await reader.cancel().catch(() => {});

    const line = await waitForLogLine(
      app,
      (l) => l.includes("proxy request completed") && l.includes(`request_id="${requestId}"`),
      "the abandoned stream's line",
      15_000,
    );
    expect(line).toContain("status=499");
    expect(numField(line, "request_body_bytes"), line).toBe(Buffer.byteLength(body));
    const handed = numField(line, "response_body_bytes");
    expect(handed, line).toBeGreaterThanOrEqual(received);
    expect(handed!, line).toBeLessThan(full.received);
  }, 30_000);

  test("a caller that leaves before the response head has a request size and no response size", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "bytes-hang", messages: [{ role: "user", content: "never mind" }] });
    const requestId = randomUUID();
    const abort = new AbortController();
    const pending = fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: headers(requestId),
      body,
      signal: abort.signal,
    }).catch(() => undefined);
    await new Promise((r) => setTimeout(r, 800));
    abort.abort();
    await pending;
    const line = await lineFor(requestId, "the head-phase cancel line");
    expect(line).toContain("status=499");
    expect(numField(line, "request_body_bytes"), line).toBe(Buffer.byteLength(body));
    expect(line).not.toContain("response_body_bytes");
  }, 30_000);

  test("/v1/audio/speech writes its line after the relayed audio, with its size", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "bytes-tts", input: "say this", voice: "alloy" });
    const r = await send("POST", "/v1/audio/speech", body);
    expect(r.status, r.text).toBe(200);
    expect(r.received).toBe(Buffer.byteLength(SPEECH_CHUNKS.join("")));
    const line = await lineFor(r.requestId, "the speech line");
    expect(numField(line, "request_body_bytes"), line).toBe(r.sent);
    expect(numField(line, "response_body_bytes"), line).toBe(r.received);
    // The relay took at least the upstream's pauses between chunks, and the
    // line's duration spans it.
    expect(numField(line, "duration_ms"), line).toBeGreaterThanOrEqual(
      (SPEECH_CHUNKS.length - 1) * SPEECH_CHUNK_DELAY_MS,
    );
  });

  test("a passthrough route reports the bytes it relayed", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "anything", messages: [{ role: "user", content: "relay me" }] });
    const r = await send("POST", "/bytes-ptr/chat/completions", body);
    expect(r.status, r.text).toBe(200);
    const line = await lineFor(r.requestId, "the passthrough line");
    expect(numField(line, "request_body_bytes"), line).toBe(r.sent);
    expect(numField(line, "response_body_bytes"), line).toBe(r.received);
  });

  test("a 413 decided from Content-Length reports no request size", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const body = JSON.stringify({ model: "bytes-chat", messages: [{ role: "user", content: "x".repeat(BODY_LIMIT * 2) }] });
    const r = await send("POST", "/v1/chat/completions", body);
    expect(r.status, r.text).toBe(413);
    const line = await lineFor(r.requestId, "the 413 line");
    expect(line).not.toContain("request_body_bytes");
    expect(numField(line, "response_body_bytes"), line).toBe(r.received);
  });

  // Every other family writes its line from its own emit site. A request
  // each family refuses on its own (no such model, agent, server, video)
  // reaches that site without needing an upstream of its shape.
  const FAMILIES: Array<{ method: string; path: string; body?: unknown; bodyRead: boolean }> = [
    { method: "POST", path: "/v1/chat/completions", body: "{not json", bodyRead: true },
    { method: "POST", path: "/v1/completions", body: { model: "bytes-nope", prompt: "hi" }, bodyRead: true },
    { method: "POST", path: "/v1/embeddings", body: { model: "bytes-nope", input: "hi" }, bodyRead: true },
    { method: "POST", path: "/v1/rerank", body: { model: "bytes-nope", query: "q", documents: ["a"] }, bodyRead: true },
    { method: "POST", path: "/v1/images/generations", body: { model: "bytes-nope", prompt: "hi" }, bodyRead: true },
    { method: "POST", path: "/v1/audio/speech", body: { model: "bytes-nope", input: "hi", voice: "alloy" }, bodyRead: true },
    { method: "POST", path: "/v1/messages", body: { model: "bytes-nope", max_tokens: 8, messages: [{ role: "user", content: "hi" }] }, bodyRead: true },
    { method: "POST", path: "/v1/messages/count_tokens", body: { model: "bytes-nope", messages: [{ role: "user", content: "hi" }] }, bodyRead: true },
    { method: "POST", path: "/v1/responses", body: { model: "bytes-nope", input: "hi" }, bodyRead: true },
    { method: "POST", path: "/v1/videos", body: { model: "bytes-nope", prompt: "hi" }, bodyRead: true },
    { method: "GET", path: "/v1/videos/bytes-nope/content", bodyRead: false },
    { method: "GET", path: "/v1/batches", bodyRead: false },
    { method: "POST", path: "/mcp", body: { jsonrpc: "2.0", id: 1, method: "tools/list" }, bodyRead: true },
    // An unknown agent is refused before its body is read.
    { method: "POST", path: "/a2a/bytes-nope", body: { jsonrpc: "2.0", id: 1, method: "message/send", params: {} }, bodyRead: false },
  ];

  for (const f of FAMILIES) {
    test(`${f.method} ${f.path} reports its body sizes`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const body = f.body === undefined ? undefined : typeof f.body === "string" ? f.body : JSON.stringify(f.body);
      const r = await send(f.method, f.path, body);
      const line = await lineFor(r.requestId, `the ${f.method} ${f.path} line (status ${r.status}: ${r.text})`);
      if (f.bodyRead) {
        expect(numField(line, "request_body_bytes"), line).toBe(r.sent);
      } else {
        expect(line).not.toContain("request_body_bytes");
      }
      expect(numField(line, "response_body_bytes"), line).toBe(r.received);
    });
  }
});
