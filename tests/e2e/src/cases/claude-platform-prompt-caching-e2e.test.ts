import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// Prompt caching for Claude behind the two platform adapters: Amazon
// Bedrock (InvokeModel when not streaming, ConverseStream when streaming)
// and Vertex AI (rawPredict).
//
// A caller's `cache_control` markers reach the upstream where the caller
// put them — as `cache_control` on the Anthropic body InvokeModel and
// Vertex take, and as a Converse `cachePoint` block right after the marked
// system part, content part, tool result or tool definition. A model's
// `auto_prompt_caching` injects them into a chat completions or responses
// request that carries none. The cache counters the upstream returns come
// back to the caller and land on the usage event.
//
// Reference:
//   - <https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html>
//   - <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
//   - <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/prompt-caching>

const CALLER_PLAINTEXT = "sk-claude-platform-prompt-caching";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "claude-cache-proj";
const LOGSTORE = "claude-cache-store";

const BEDROCK_CLAUDE = "pc-bedrock-claude";
const BEDROCK_CLAUDE_AUTO = "pc-bedrock-claude-auto";
const BEDROCK_OPENAI = "pc-bedrock-openai";
const VERTEX_CLAUDE = "pc-vertex-claude";
const VERTEX_CLAUDE_AUTO = "pc-vertex-claude-auto";
const OPENAI = "pc-openai";
const MODELS = [
  BEDROCK_CLAUDE,
  BEDROCK_CLAUDE_AUTO,
  BEDROCK_OPENAI,
  VERTEX_CLAUDE,
  VERTEX_CLAUDE_AUTO,
  OPENAI,
];

const UNCACHED_IN = 8;
const CACHE_READ = 3144;
const CACHE_WRITE = 100;
const OUT = 2;

const ANTHROPIC_REPLY = {
  id: "msg_cache",
  type: "message",
  role: "assistant",
  model: "claude",
  content: [{ type: "text", text: "ok" }],
  stop_reason: "end_turn",
  usage: {
    input_tokens: UNCACHED_IN,
    output_tokens: OUT,
    cache_read_input_tokens: CACHE_READ,
    cache_creation_input_tokens: CACHE_WRITE,
  },
};
const CONVERSE_REPLY = {
  output: { message: { role: "assistant", content: [{ text: "ok" }] } },
  stopReason: "end_turn",
  usage: { inputTokens: UNCACHED_IN, outputTokens: OUT, totalTokens: UNCACHED_IN + OUT },
  metrics: { latencyMs: 1 },
};
const OPENAI_REPLY = {
  id: "chatcmpl-cache",
  object: "chat.completion",
  created: 0,
  model: "gpt-4o",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: UNCACHED_IN, completion_tokens: OUT, total_tokens: UNCACHED_IN + OUT },
};
const CONVERSE_STREAM = eventStream([
  ["messageStart", { role: "assistant" }],
  ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "ok" } }],
  ["contentBlockStop", { contentBlockIndex: 0 }],
  ["messageStop", { stopReason: "end_turn" }],
  [
    "metadata",
    {
      usage: {
        inputTokens: UNCACHED_IN,
        outputTokens: OUT,
        totalTokens: UNCACHED_IN + OUT + CACHE_READ + CACHE_WRITE,
        cacheReadInputTokens: CACHE_READ,
        cacheWriteInputTokens: CACHE_WRITE,
      },
      metrics: { latencyMs: 1 },
    },
  ],
]);

interface RecordingUpstream {
  baseUrl: string;
  received: Array<{ path: string; body: string }>;
  close(): Promise<void>;
}

/** Answers every platform route the gateway calls here, in its shape. */
async function startUpstream(): Promise<RecordingUpstream> {
  const received: Array<{ path: string; body: string }> = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      received.push({ path, body: Buffer.concat(chunks).toString("utf8") });
      if (path.endsWith("/converse-stream")) {
        res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream" });
        res.end(CONVERSE_STREAM);
        return;
      }
      const reply = path.endsWith("/converse")
        ? CONVERSE_REPLY
        : path.endsWith("/chat/completions")
          ? OPENAI_REPLY
          : ANTHROPIC_REPLY;
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(reply));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

const EPHEMERAL = { type: "ephemeral" };
const HOUR = { type: "ephemeral", ttl: "1h" };
const PNG_B64 = "iVBORw0KGgo=";

/** A `/v1/messages` request with a marker on every block kind that takes
 * one, the way an agent client caches its prefix. */
function markedMessages(model: string, stream: boolean): Record<string, unknown> {
  return {
    model,
    max_tokens: 64,
    stream,
    system: [
      { type: "text", text: "You are a careful assistant." },
      { type: "text", text: "Long reference material.", cache_control: HOUR },
    ],
    tools: [
      {
        name: "lookup",
        description: "Look a value up.",
        input_schema: { type: "object", properties: {} },
        cache_control: EPHEMERAL,
      },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: PNG_B64 },
            cache_control: EPHEMERAL,
          },
          { type: "text", text: "What is the value?" },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "toolu_1", name: "lookup", input: {} }],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_1", content: "42", cache_control: EPHEMERAL },
          { type: "text", text: "Answer briefly." },
        ],
      },
    ],
  };
}

/** A chat completions request with markers on content parts and a tool. */
function markedChat(model: string, stream: boolean): Record<string, unknown> {
  return {
    model,
    stream,
    ...(stream ? { stream_options: { include_usage: true } } : {}),
    messages: [
      {
        role: "system",
        content: [
          { type: "text", text: "You are a careful assistant." },
          { type: "text", text: "Long reference material.", cache_control: HOUR },
        ],
      },
      {
        role: "user",
        content: [
          { type: "text", text: "Earlier conversation.", cache_control: EPHEMERAL },
          { type: "text", text: "What is the value?" },
        ],
      },
    ],
    tools: [
      {
        type: "function",
        function: { name: "lookup", parameters: { type: "object", properties: {} } },
        cache_control: EPHEMERAL,
      },
    ],
  };
}

const plainChat = (model: string, stream: boolean) => ({
  model,
  stream,
  messages: [
    { role: "system", content: "You are a careful assistant." },
    { role: "user", content: "What is the value?" },
  ],
});

/** The usage object of the last SSE frame `pick` finds one in. */
function lastUsage(text: string, pick: (frame: any) => unknown): Record<string, any> {
  let found: Record<string, any> | undefined;
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6).trim();
    if (data === "[DONE]") continue;
    try {
      const usage = pick(JSON.parse(data));
      if (usage && typeof usage === "object") found = usage as Record<string, any>;
    } catch {
      continue;
    }
  }
  if (!found) throw new Error(`no usage in stream:\n${text}`);
  return found;
}

describe("prompt caching for Claude on Amazon Bedrock and Vertex AI", () => {
  let app: SpawnedApp | undefined;
  let upstream: RecordingUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startUpstream();
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "claude-cache-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    const bedrockPk = await seed.createProviderKey({
      display_name: "pc-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-prompt-cache",
        secret_access_key: "sk-prompt-cache",
        region: "us-west-2",
      }),
      api_base: upstream.baseUrl,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "pc-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.prompt-cache",
        project: "proj-e2e",
        region: "us-east5",
      }),
      api_base: upstream.baseUrl,
    });
    const openaiPk = await seed.createProviderKey({
      display_name: "pc-openai-pk",
      provider: "openai",
      adapter: "openai",
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const auto = { enabled: true, ttl: "1h" };
    for (const [display_name, pk, provider, model_name, apc] of [
      [BEDROCK_CLAUDE, bedrockPk, "bedrock", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", undefined],
      [BEDROCK_CLAUDE_AUTO, bedrockPk, "bedrock", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", auto],
      [BEDROCK_OPENAI, bedrockPk, "bedrock", "openai.gpt-oss-120b-1:0", undefined],
      [VERTEX_CLAUDE, vertexPk, "google", "claude-sonnet-4-5@20250929", undefined],
      [VERTEX_CLAUDE_AUTO, vertexPk, "google", "claude-sonnet-4-5@20250929", auto],
      [OPENAI, openaiPk, "openai", "gpt-4o", undefined],
    ] as const) {
      await seed.createModel({
        display_name,
        provider,
        model_name,
        provider_key_id: pk.id,
        ...(apc ? { auto_prompt_caching: apc } : {}),
      });
    }

    // Seeded last: this key authenticating implies the whole seed set.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: MODELS });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  interface Sent {
    status: number;
    text: string;
    requestId: string;
    /** The one upstream request this call made. */
    upstream: { path: string; body: any };
  }

  async function send(path: string, body: unknown): Promise<Sent> {
    const before = upstream!.received.length;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(upstream!.received.length - before).toBe(1);
    const got = upstream!.received.at(-1)!;
    return {
      status: res.status,
      text,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
      upstream: { path: got.path, body: JSON.parse(got.body) },
    };
  }

  async function expectCacheCountersOnUsageEvent(requestId: string): Promise<void> {
    const log = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
    expect(log.get("prompt_tokens")).toBe(String(UNCACHED_IN));
    expect(log.get("cache_read_tokens")).toBe(String(CACHE_READ));
    expect(log.get("cache_creation_tokens")).toBe(String(CACHE_WRITE));
  }

  const skip = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return true;
    }
    return false;
  };

  test("/v1/messages to Claude on Bedrock (InvokeModel) keeps every marker in place", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", markedMessages(BEDROCK_CLAUDE, false));
    expect(sent.upstream.path).toMatch(/\/invoke$/);
    const body = sent.upstream.body;
    expect(body.system).toEqual([
      { type: "text", text: "You are a careful assistant." },
      { type: "text", text: "Long reference material.", cache_control: HOUR },
    ]);
    expect(body.tools[0].cache_control).toEqual(EPHEMERAL);
    expect(body.messages[0].content[0].type).toBe("image");
    expect(body.messages[0].content[0].cache_control).toEqual(EPHEMERAL);
    expect(body.messages[0].content[1].cache_control).toBeUndefined();
    expect(body.messages[2].content[0]).toMatchObject({
      type: "tool_result",
      tool_use_id: "toolu_1",
      cache_control: EPHEMERAL,
    });

    const usage = JSON.parse(sent.text).usage;
    expect(usage.input_tokens).toBe(UNCACHED_IN);
    expect(usage.cache_read_input_tokens).toBe(CACHE_READ);
    expect(usage.cache_creation_input_tokens).toBe(CACHE_WRITE);
    await expectCacheCountersOnUsageEvent(sent.requestId);
  });

  test("streamed /v1/messages to Claude on Bedrock (ConverseStream) sends cache points", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", markedMessages(BEDROCK_CLAUDE, true));
    expect(sent.upstream.path).toMatch(/\/converse-stream$/);
    const body = sent.upstream.body;
    expect(body.system).toEqual([
      { text: "You are a careful assistant." },
      { text: "Long reference material." },
      { cachePoint: { type: "default", ttl: "1h" } },
    ]);
    expect(body.toolConfig.tools[1]).toEqual({ cachePoint: { type: "default" } });
    const first = body.messages[0].content;
    expect(first[0].image).toBeDefined();
    expect(first[1]).toEqual({ cachePoint: { type: "default" } });
    const results = body.messages[2].content;
    expect(results[0].toolResult.toolUseId).toBe("toolu_1");
    expect(results[1]).toEqual({ cachePoint: { type: "default" } });

    const usage = lastUsage(sent.text, (f) => f.usage ?? f.message?.usage);
    expect(usage.input_tokens).toBe(UNCACHED_IN);
    expect(usage.cache_read_input_tokens).toBe(CACHE_READ);
    expect(usage.cache_creation_input_tokens).toBe(CACHE_WRITE);
    await expectCacheCountersOnUsageEvent(sent.requestId);
  });

  test("streamed chat completions to Claude on Bedrock (ConverseStream) sends cache points", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/chat/completions", markedChat(BEDROCK_CLAUDE, true));
    expect(sent.upstream.path).toMatch(/\/converse-stream$/);
    const body = sent.upstream.body;
    expect(body.system).toEqual([
      { text: "You are a careful assistant." },
      { text: "Long reference material." },
      { cachePoint: { type: "default", ttl: "1h" } },
    ]);
    expect(body.messages[0].content).toEqual([
      { text: "Earlier conversation." },
      { cachePoint: { type: "default" } },
      { text: "What is the value?" },
    ]);
    expect(body.toolConfig.tools[1]).toEqual({ cachePoint: { type: "default" } });

    const usage = lastUsage(sent.text, (f) => f.usage);
    expect(usage.prompt_tokens_details?.cached_tokens).toBe(CACHE_READ);
    expect(usage.prompt_tokens).toBe(UNCACHED_IN + CACHE_READ + CACHE_WRITE);
    await expectCacheCountersOnUsageEvent(sent.requestId);
  });

  test("chat completions to Claude on Bedrock (InvokeModel) reports the cache read", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/chat/completions", markedChat(BEDROCK_CLAUDE, false));
    expect(sent.upstream.path).toMatch(/\/invoke$/);
    expect(sent.upstream.body.system[1].cache_control).toEqual(HOUR);
    expect(sent.upstream.body.messages[0].content[0].cache_control).toEqual(EPHEMERAL);
    const usage = JSON.parse(sent.text).usage;
    expect(usage.prompt_tokens_details?.cached_tokens).toBe(CACHE_READ);
  });

  test("/v1/messages to Claude on Vertex AI keeps every marker in place", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", markedMessages(VERTEX_CLAUDE, false));
    expect(sent.upstream.path).toMatch(/:rawPredict$/);
    const body = sent.upstream.body;
    expect(body.system[1].cache_control).toEqual(HOUR);
    expect(body.tools[0].cache_control).toEqual(EPHEMERAL);
    expect(body.messages[0].content[0].cache_control).toEqual(EPHEMERAL);
    expect(body.messages[2].content[0].cache_control).toEqual(EPHEMERAL);

    const usage = JSON.parse(sent.text).usage;
    expect(usage.cache_read_input_tokens).toBe(CACHE_READ);
    expect(usage.cache_creation_input_tokens).toBe(CACHE_WRITE);
    await expectCacheCountersOnUsageEvent(sent.requestId);
  });

  test("an OpenAI model on Bedrock gets no cache point and no marker", async (ctx) => {
    if (skip(ctx)) return;
    for (const [path, body] of [
      ["/v1/chat/completions", markedChat(BEDROCK_OPENAI, true)],
      ["/v1/messages", { ...markedMessages(BEDROCK_OPENAI, true), messages: [
        { role: "user", content: [{ type: "text", text: "hi", cache_control: EPHEMERAL }] },
      ] }],
    ] as const) {
      const sent = await send(path, body);
      expect(sent.upstream.path).toMatch(/\/converse-stream$/);
      const wire = JSON.stringify(sent.upstream.body);
      expect(wire, path).not.toContain("cachePoint");
      expect(wire, path).not.toContain("cache_control");
    }
  });

  test("/v1/messages to an OpenAI-compatible upstream gets the request it always got", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", markedMessages(OPENAI, false));
    expect(sent.upstream.path).toMatch(/\/chat\/completions$/);
    const body = sent.upstream.body;
    expect(JSON.stringify(body)).not.toContain("cache_control");
    expect(body.messages[0]).toEqual({
      role: "system",
      content: "You are a careful assistant.\nLong reference material.",
    });
    expect(body.messages.at(-1)).toEqual({ role: "user", content: "Answer briefly." });
  });

  test("auto_prompt_caching injects breakpoints for Claude on Bedrock and Vertex AI", async (ctx) => {
    if (skip(ctx)) return;
    const invoke = await send("/v1/chat/completions", plainChat(BEDROCK_CLAUDE_AUTO, false));
    expect(invoke.upstream.path).toMatch(/\/invoke$/);
    expect(invoke.upstream.body.system).toEqual([
      { type: "text", text: "You are a careful assistant.", cache_control: HOUR },
    ]);
    expect(invoke.upstream.body.messages.at(-1).content.at(-1).cache_control).toEqual(HOUR);

    const stream = await send("/v1/chat/completions", plainChat(BEDROCK_CLAUDE_AUTO, true));
    expect(stream.upstream.path).toMatch(/\/converse-stream$/);
    const point = { cachePoint: { type: "default", ttl: "1h" } };
    expect(stream.upstream.body.system).toEqual([{ text: "You are a careful assistant." }, point]);
    expect(stream.upstream.body.messages.at(-1).content.at(-1)).toEqual(point);

    const vertex = await send("/v1/chat/completions", plainChat(VERTEX_CLAUDE_AUTO, false));
    expect(vertex.upstream.path).toMatch(/:rawPredict$/);
    expect(vertex.upstream.body.system[0].cache_control).toEqual(HOUR);
    expect(vertex.upstream.body.messages.at(-1).content.at(-1).cache_control).toEqual(HOUR);

    const responses = await send("/v1/responses", {
      model: BEDROCK_CLAUDE_AUTO,
      instructions: "You are a careful assistant.",
      input: "What is the value?",
    });
    expect(responses.upstream.path).toMatch(/\/invoke$/);
    expect(responses.upstream.body.system[0].cache_control).toEqual(HOUR);
    expect(responses.upstream.body.messages.at(-1).content.at(-1).cache_control).toEqual(HOUR);
  });

  test("auto_prompt_caching stands down for a caller's own markers and for /v1/messages", async (ctx) => {
    if (skip(ctx)) return;
    // The caller's single marker is the whole strategy: nothing is added.
    const marked = await send("/v1/chat/completions", markedChat(BEDROCK_CLAUDE_AUTO, true));
    expect(JSON.stringify(marked.upstream.body).match(/cachePoint/g)?.length).toBe(3);

    // `/v1/messages` callers manage their own breakpoints, as on the
    // Anthropic API, where the route is forwarded unchanged.
    const messages = await send("/v1/messages", {
      model: BEDROCK_CLAUDE_AUTO,
      max_tokens: 64,
      system: "You are a careful assistant.",
      messages: [{ role: "user", content: "What is the value?" }],
    });
    expect(JSON.stringify(messages.upstream.body)).not.toContain("cache_control");
  });
});
