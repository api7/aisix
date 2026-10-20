import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStreamException,
  invokeChunk,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// `/v1/messages` and `/v1/messages/count_tokens` for Claude on Amazon
// Bedrock and on Vertex AI go to the platform's own Anthropic Messages
// routes — InvokeModel / InvokeModelWithResponseStream / CountTokens on
// Bedrock, rawPredict / streamRawPredict / the count-tokens model on
// Vertex AI — with the caller's body, and the answer comes back in the
// Anthropic wire it arrived in. What an agent client depends on survives
// the round trip: thinking blocks and their signatures in both directions,
// images inside tool results, the cache counters on `message_start`, the
// upstream's own message id, stop reasons, and Anthropic error types for a
// failure inside the stream.
//
// On the way the body is adapted to what each platform accepts and nothing
// more: the model leaves the body for the URL, the platform's
// `anthropic_version` goes in, only the top-level fields the platform
// takes are kept, `metadata` carries `user_id` only, the `anthropic-beta`
// values the platform knows are sent (in the body on Bedrock, as the header
// on Vertex AI), and media named by URL is downloaded, since neither
// platform reads URLs.
//
// Reference:
//   - <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html>
//   - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html>
//   - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html>
//   - <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude>
//   - <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/count-tokens>
//   - <https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>

const CALLER = "sk-claude-platform-native-messages";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const SLS_PROJECT = "claude-native-proj";
const LOGSTORE = "claude-native-store";

const BEDROCK = "nm-bedrock-claude";
const BEDROCK_THROTTLED = "nm-bedrock-throttled";
const BEDROCK_GUARDED = "nm-bedrock-guarded";
const VERTEX = "nm-vertex-claude";
const BEDROCK_NO_COUNT = "nm-bedrock-no-count";
const ANTHROPIC = "nm-anthropic";
const COUNT_GROUP = "nm-count-group";
const NO_COUNT_ID = "us.anthropic.claude-nocount-4-5-v1:0";
const BEDROCK_MANTLE_ONLY = "nm-bedrock-mantle-only";
const MANTLE_ONLY_ID = "global.anthropic.claude-opus-5-5";
const BEDROCK_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0";
const THROTTLED_ID = "us.anthropic.claude-throttle-4-5-v1:0";
const GUARDED_ID = "us.anthropic.claude-guarded-4-5-v1:0";
const VERTEX_ID = "claude-sonnet-4-5@20250929";
const FORBIDDEN = "PLATFORM-FORBIDDEN-WORD";

const PNG_B64 = "iVBORw0KGgo=";
const UNCACHED_IN = 8;
const CACHE_READ = 3144;
const CACHE_WRITE = 100;

function reply(id: string): Record<string, unknown> {
  return {
    id,
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-5-20250929",
    content: [
      { type: "thinking", thinking: "Let me look.", signature: "sig-reply" },
      { type: "redacted_thinking", data: "opaque-redacted" },
      { type: "text", text: "ok" },
    ],
    stop_reason: "pause_turn",
    stop_sequence: null,
    usage: {
      input_tokens: UNCACHED_IN,
      output_tokens: 2,
      cache_read_input_tokens: CACHE_READ,
      cache_creation_input_tokens: CACHE_WRITE,
      cache_creation: { ephemeral_5m_input_tokens: 40, ephemeral_1h_input_tokens: 60 },
    },
  };
}

function streamEvents(id: string, text: string): Array<Record<string, unknown>> {
  return [
    {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5-20250929",
        content: [],
        stop_reason: null,
        usage: {
          input_tokens: UNCACHED_IN,
          output_tokens: 1,
          cache_read_input_tokens: CACHE_READ,
          cache_creation_input_tokens: CACHE_WRITE,
          cache_creation: { ephemeral_5m_input_tokens: 40, ephemeral_1h_input_tokens: 60 },
        },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Let me look." } },
    { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "sig-stream" } },
    { type: "content_block_stop", index: 0 },
    { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 1, delta: { type: "text_delta", text } },
    { type: "content_block_stop", index: 1 },
    { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 5 } },
    { type: "message_stop" },
  ];
}

const sse = (events: Array<Record<string, unknown>>) =>
  events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join("");

interface Received {
  path: string;
  headers: Record<string, string | string[] | undefined>;
  body: string;
}

interface Upstream {
  baseUrl: string;
  port: number;
  received: Received[];
  close(): Promise<void>;
}

/** Bedrock and Vertex AI Claude routes, plus a media file on `/img.png`. */
async function startUpstream(): Promise<Upstream> {
  const received: Received[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = decodeURIComponent((req.url ?? "/").split("?")[0]);
      if (path === "/img.png") {
        res.writeHead(200, { "content-type": "image/png" });
        res.end(Buffer.from(PNG_B64, "base64"));
        return;
      }
      received.push({ path, headers: req.headers, body: Buffer.concat(chunks).toString("utf8") });
      const json = (body: unknown) => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(body));
      };
      if (path === "/anthropic/v1/messages/count_tokens") {
        if (JSON.parse(received.at(-1)!.body).model.includes("claude-nocount")) {
          res.writeHead(400, { "content-type": "application/json" });
          res.end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "unsupported" } }));
          return;
        }
        return json({ input_tokens: 77 });
      }
      if (path.endsWith("/count-tokens") && (path.includes("claude-nocount") || path.includes("claude-opus-5-5"))) {
        res.writeHead(400, { "content-type": "application/json", "x-amzn-errortype": "ValidationException" });
        res.end(JSON.stringify({ message: "The provided model doesn't support counting tokens." }));
        return;
      }
      if (path.endsWith("/count-tokens")) return json({ inputTokens: 42 });
      if (path === "/v1/messages/count_tokens") return json({ input_tokens: 99 });
      if (path.endsWith("/models/count-tokens:rawPredict")) return json({ input_tokens: 17 });
      if (path.endsWith("/invoke")) return json(reply("msg_bdrk_nonstream"));
      if (path.endsWith(":rawPredict")) return json(reply("msg_vrtx_nonstream"));
      if (path.endsWith(":streamRawPredict")) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end(sse(streamEvents("msg_vrtx_stream", "ok")));
        return;
      }
      if (path.endsWith("/invoke-with-response-stream")) {
        res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream" });
        if (path.includes(THROTTLED_ID)) {
          const events = streamEvents("msg_bdrk_throttled", "partial");
          res.end(
            Buffer.concat([
              ...events.slice(0, 7).map(invokeChunk),
              eventStreamException("throttlingException", { message: "Too many tokens, please wait." }),
            ]),
          );
          return;
        }
        const text = path.includes(GUARDED_ID) ? `the answer is ${FORBIDDEN}` : "ok";
        const events = streamEvents("msg_bdrk_stream", text);
        // Bedrock appends its invocation metrics to `message_stop`.
        events[events.length - 1] = {
          type: "message_stop",
          "amazon-bedrock-invocationMetrics": { inputTokenCount: UNCACHED_IN, outputTokenCount: 5 },
        };
        res.end(Buffer.concat(events.map(invokeChunk)));
        return;
      }
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ message: `no route ${path}` }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    port: addr.port,
    received,
    close: () =>
      new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

/** A Claude Code shaped turn: earlier thinking replayed with its
 * signature, a tool call, and a tool result carrying an image. */
function agentTurn(model: string, stream: boolean, image: Record<string, unknown>) {
  return {
    model,
    max_tokens: 1024,
    stream,
    metadata: { user_id: "user_abc_account_1_session_2", tenant: "internal" },
    stop_sequences: ["</done>"],
    thinking: { type: "enabled", budget_tokens: 512 },
    output_config: { effort: "high" },
    context_management: {
      edits: [{ type: "clear_thinking_20251015", keep: "all" }, { type: "clear_tool_uses_20250919" }],
    },
    gateway_unknown_field: { x: 1 },
    system: [
      { type: "text", text: "You are a coding agent.", cache_control: { type: "ephemeral", ttl: "1h", scope: "global" } },
    ],
    tools: [{ name: "Read", description: "Read a file.", input_schema: { type: "object", properties: {} } }],
    messages: [
      { role: "user", content: [{ type: "text", text: "Look at the screenshot." }] },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should read it.", signature: "sig-history" },
          { type: "redacted_thinking", data: "opaque-history" },
          { type: "tool_use", id: "toolu_1", name: "Read", input: { path: "shot.png" } },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "toolu_1",
            is_error: false,
            content: [{ type: "text", text: "Image read." }, image],
          },
        ],
      },
    ],
  };
}

const BASE64_IMAGE = { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_B64 } };

/** Every `data:` payload of an SSE body, parsed. */
function frames(text: string): any[] {
  return text
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)));
}

describe("Claude on Amazon Bedrock and Vertex AI on their own Messages routes", () => {
  let app: SpawnedApp | undefined;
  let upstream: Upstream | undefined;
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
      extra: {
        upstream: {
          media_fetch: { allowed_hosts: [`localhost:${upstream.port}`] },
          bedrock: { mantle_endpoint: upstream.baseUrl },
        },
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "claude-native-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const bedrockPk = await seed.createProviderKey({
      display_name: "nm-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-native-messages",
        secret_access_key: "sk-native-messages",
        region: "us-west-2",
      }),
      api_base: upstream.baseUrl,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "nm-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.native-messages", project: "proj-e2e", region: "us-east5" }),
      api_base: upstream.baseUrl,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "nm-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    const models: Record<string, { id: string }> = {};
    for (const [display_name, pk, provider, model_name] of [
      [BEDROCK, bedrockPk, "bedrock", BEDROCK_ID],
      [BEDROCK_THROTTLED, bedrockPk, "bedrock", THROTTLED_ID],
      [BEDROCK_GUARDED, bedrockPk, "bedrock", GUARDED_ID],
      [VERTEX, vertexPk, "google", VERTEX_ID],
      [BEDROCK_NO_COUNT, bedrockPk, "bedrock", NO_COUNT_ID],
      [BEDROCK_MANTLE_ONLY, bedrockPk, "bedrock", MANTLE_ONLY_ID],
      [ANTHROPIC, anthropicPk, "anthropic", "claude-sonnet-4-5"],
    ] as const) {
      models[display_name] = await seed.createModel({
        display_name,
        provider,
        model_name,
        provider_key_id: pk.id,
      });
    }
    await seed.createModel({
      display_name: COUNT_GROUP,
      routing: {
        strategy: "failover",
        targets: [{ model: BEDROCK_NO_COUNT }, { model: ANTHROPIC }],
        max_fallbacks: 1,
      },
    });
    const guardrail = await seed.createGuardrail(
      {
        name: "nm-output-keyword",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: FORBIDDEN }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(guardrail.id, models[BEDROCK_GUARDED].id);

    // Seeded last: this key authenticating implies the whole seed set.
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
    await upstream?.close();
    await sls?.close();
  });

  const skip = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return true;
    }
    return false;
  };

  interface Sent {
    status: number;
    text: string;
    requestId: string;
    upstream: Received;
  }

  async function send(path: string, body: unknown, betas?: string): Promise<Sent> {
    const before = upstream!.received.length;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        ...(betas ? { "anthropic-beta": betas } : {}),
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(upstream!.received.length - before, text).toBe(1);
    return {
      status: res.status,
      text,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
      upstream: upstream!.received.at(-1)!,
    };
  }

  const usageRow = (requestId: string) =>
    waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, `usage row for ${requestId}`, 15_000);

  const CLAUDE_CODE_BETAS =
    "claude-code-20250219,interleaved-thinking-2025-05-14,context-1m-2025-08-07,fine-grained-tool-streaming-2025-05-14";

  test("Bedrock InvokeModel gets the caller's body, adapted, and answers natively", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", agentTurn(BEDROCK, false, BASE64_IMAGE), CLAUDE_CODE_BETAS);
    expect(sent.status, sent.text).toBe(200);
    expect(sent.upstream.path).toBe(`/model/${BEDROCK_ID}/invoke`);
    const body = JSON.parse(sent.upstream.body);

    expect(body.anthropic_version).toBe("bedrock-2023-05-31");
    expect(body).not.toHaveProperty("model");
    expect(body).not.toHaveProperty("stream");
    expect(body).not.toHaveProperty("gateway_unknown_field");
    expect(body.metadata).toEqual({ user_id: "user_abc_account_1_session_2" });
    expect(body.stop_sequences).toEqual(["</done>"]);
    expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 512 });
    // Only the Bedrock-known betas, with the ones the body's own features
    // need; the effort beta because Claude 4.5 takes `effort` as a beta.
    expect(body.anthropic_beta).toEqual([
      "claude-code-20250219",
      "context-1m-2025-08-07",
      "context-management-2025-06-27",
      "effort-2025-11-24",
      "fine-grained-tool-streaming-2025-05-14",
      "interleaved-thinking-2025-05-14",
    ]);
    // Bedrock runs only some context-management edits.
    expect(body.context_management).toEqual({ edits: [{ type: "clear_tool_uses_20250919" }] });
    expect(body.system[0].cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    // The replayed history reaches Claude as the client sent it.
    expect(body.messages[1].content[0]).toEqual({
      type: "thinking",
      thinking: "I should read it.",
      signature: "sig-history",
    });
    expect(body.messages[1].content[1]).toEqual({ type: "redacted_thinking", data: "opaque-history" });
    expect(body.messages[2].content[0]).toMatchObject({ type: "tool_result", is_error: false });
    expect(body.messages[2].content[0].content[1]).toEqual(BASE64_IMAGE);

    const out = JSON.parse(sent.text);
    expect(out.id).toBe("msg_bdrk_nonstream");
    expect(out.model).toBe(BEDROCK);
    expect(out.content[0]).toEqual({ type: "thinking", thinking: "Let me look.", signature: "sig-reply" });
    expect(out.content[1]).toEqual({ type: "redacted_thinking", data: "opaque-redacted" });
    expect(out.stop_reason).toBe("pause_turn");
    expect(out.usage.cache_creation).toEqual({ ephemeral_5m_input_tokens: 40, ephemeral_1h_input_tokens: 60 });

    const row = await usageRow(sent.requestId);
    expect(row.get("prompt_tokens")).toBe(String(UNCACHED_IN));
    expect(row.get("cache_read_tokens")).toBe(String(CACHE_READ));
    expect(row.get("cache_creation_tokens")).toBe(String(CACHE_WRITE));
    expect(row.get("provider_request_id")).toBe("msg_bdrk_nonstream");
  });

  test("Bedrock InvokeModelWithResponseStream relays the Anthropic stream natively", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", agentTurn(BEDROCK, true, BASE64_IMAGE), CLAUDE_CODE_BETAS);
    expect(sent.status, sent.text).toBe(200);
    expect(sent.upstream.path).toBe(`/model/${BEDROCK_ID}/invoke-with-response-stream`);
    const body = JSON.parse(sent.upstream.body);
    expect(body).not.toHaveProperty("stream");
    expect(body.anthropic_beta).toEqual([
      "claude-code-20250219",
      "context-1m-2025-08-07",
      "context-management-2025-06-27",
      "effort-2025-11-24",
      "fine-grained-tool-streaming-2025-05-14",
      "interleaved-thinking-2025-05-14",
    ]);
    expect(body.messages[1].content[0].signature).toBe("sig-history");
    expect(body.messages[2].content[0].content[1]).toEqual(BASE64_IMAGE);

    const events = frames(sent.text);
    const start = events.find((e) => e.type === "message_start");
    expect(start.message.id).toBe("msg_bdrk_stream");
    expect(start.message.model).toBe(BEDROCK);
    expect(start.message.usage).toMatchObject({
      input_tokens: UNCACHED_IN,
      cache_read_input_tokens: CACHE_READ,
      cache_creation_input_tokens: CACHE_WRITE,
    });
    expect(events).toContainEqual({
      type: "content_block_delta",
      index: 0,
      delta: { type: "signature_delta", signature: "sig-stream" },
    });
    expect(events.find((e) => e.type === "message_delta").delta.stop_reason).toBe("end_turn");

    const row = await usageRow(sent.requestId);
    expect(row.get("prompt_tokens")).toBe(String(UNCACHED_IN));
    expect(row.get("completion_tokens")).toBe("5");
    expect(row.get("cache_read_tokens")).toBe(String(CACHE_READ));
    expect(row.get("cache_creation_tokens")).toBe(String(CACHE_WRITE));
    expect(row.get("provider_request_id")).toBe("msg_bdrk_stream");
  });

  test("Bedrock receives the betas it runs and none it refuses", async (ctx) => {
    if (skip(ctx)) return;
    const runs = [
      "interleaved-thinking-2025-05-14",
      "output-128k-2025-02-19",
      "structured-outputs-2025-11-13",
      "thinking-token-count-2026-05-13",
      "token-efficient-tools-2025-02-19",
      "tool-examples-2025-10-29",
    ];
    const refused = [
      "dangerous-tool-use-2026-09-03",
      "extended-cache-ttl-2025-04-11",
      "files-api-2025-04-14",
      "prompt-caching-2024-07-31",
      "prompt-caching-scope-2026-01-05",
      "thinking-binding-controls-2026-08-01",
      "thinking-display-updates-2026-08-18",
      "token-counting-2024-11-01",
    ];
    for (const stream of [false, true]) {
      const sent = await send(
        "/v1/messages",
        {
          model: BEDROCK,
          max_tokens: 2000,
          stream,
          thinking: { type: "enabled", budget_tokens: 1024, display: "updates" },
          messages: [{ role: "user", content: "What is 17*23?" }],
        },
        [...refused, ...runs].join(","),
      );
      expect(sent.status, sent.text).toBe(200);
      const body = JSON.parse(sent.upstream.body);
      expect(body.anthropic_beta, `stream=${stream}`).toEqual(runs);
      // Bedrock refuses `display: "updates"`; the default display applies.
      expect(body.thinking, `stream=${stream}`).toEqual({ type: "enabled", budget_tokens: 1024 });
    }

    const summarized = await send("/v1/messages", {
      model: BEDROCK,
      max_tokens: 2000,
      thinking: { type: "enabled", budget_tokens: 1024, display: "summarized" },
      messages: [{ role: "user", content: "What is 17*23?" }],
    });
    expect(summarized.status, summarized.text).toBe(200);
    expect(JSON.parse(summarized.upstream.body).thinking).toEqual({
      type: "enabled",
      budget_tokens: 1024,
      display: "summarized",
    });
  });

  test("an image URL inside a tool result is downloaded for Bedrock and Vertex AI", async (ctx) => {
    if (skip(ctx)) return;
    const image = { type: "image", source: { type: "url", url: `http://localhost:${upstream!.port}/img.png` } };
    for (const model of [BEDROCK, VERTEX]) {
      const sent = await send("/v1/messages", agentTurn(model, false, image));
      expect(sent.status, sent.text).toBe(200);
      const body = JSON.parse(sent.upstream.body);
      expect(body.messages[2].content[0].content[1], model).toEqual(BASE64_IMAGE);
    }
  });

  test("a throttle inside the Bedrock stream reaches the caller as rate_limit_error", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", agentTurn(BEDROCK_THROTTLED, true, BASE64_IMAGE));
    expect(sent.status, sent.text).toBe(200);
    const events = frames(sent.text);
    expect(events.find((e) => e.type === "message_start").message.id).toBe("msg_bdrk_throttled");
    const error = events.at(-1);
    expect(error.type).toBe("error");
    expect(error.error.type).toBe("rate_limit_error");
    const row = await usageRow(sent.requestId);
    expect(row.get("status_code")).toBe("429");
  });

  test("an output guardrail runs on the Bedrock passthrough stream", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", agentTurn(BEDROCK_GUARDED, true, BASE64_IMAGE));
    const events = sent.status === 200 ? frames(sent.text) : [JSON.parse(sent.text)];
    expect(events.some((e) => e.type === "error"), sent.text).toBe(true);
    const row = await usageRow(sent.requestId);
    expect(row.get("guardrail_blocked")).toBe("true");
  });

  test("count_tokens on Claude on Bedrock calls CountTokens", async (ctx) => {
    if (skip(ctx)) return;
    const typed = agentTurn(BEDROCK, false, BASE64_IMAGE) as Record<string, unknown>;
    delete typed.stream;
    const sent = await send("/v1/messages/count_tokens", typed);
    expect(sent.status, sent.text).toBe(200);
    expect(JSON.parse(sent.text)).toEqual({ input_tokens: 42 });
    // The base model: CountTokens takes no inference-profile prefix.
    expect(sent.upstream.path).toBe("/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens");
    const invoke = JSON.parse(
      Buffer.from(JSON.parse(sent.upstream.body).input.invokeModel.body, "base64").toString("utf8"),
    );
    expect(invoke).not.toHaveProperty("model");
    expect(invoke.anthropic_version).toBe("bedrock-2023-05-31");
    expect(invoke.messages[1].content[0].signature).toBe("sig-history");
    // The same adjustments InvokeModel gets: fields, metadata, markers.
    expect(invoke).not.toHaveProperty("gateway_unknown_field");
    expect(invoke.metadata).toEqual({ user_id: "user_abc_account_1_session_2" });
    expect(invoke.system[0].cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    expect(invoke.max_tokens).toBe(1024);

    // A thinking budget above the default max_tokens still makes a valid
    // InvokeModel body; the value does not change the count.
    const budgeted = await send("/v1/messages/count_tokens", {
      model: BEDROCK,
      thinking: { type: "enabled", budget_tokens: 4000 },
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
    });
    expect(budgeted.status, budgeted.text).toBe(200);
    const budgetedBody = JSON.parse(
      Buffer.from(JSON.parse(budgeted.upstream.body).input.invokeModel.body, "base64").toString("utf8"),
    );
    expect(budgetedBody.max_tokens).toBeGreaterThan(4000);

    // Plain-string content counts exactly as its typed form, thinking
    // included.
    const counted = async (content: (text: string) => unknown) => {
      const sent = await send("/v1/messages/count_tokens", {
        model: BEDROCK,
        system: "Be brief.",
        thinking: { type: "enabled", budget_tokens: 2048 },
        messages: [
          { role: "user", content: content("hello") },
          { role: "assistant", content: content("Hi.") },
          { role: "user", content: content("What is 17*23?") },
        ],
      });
      expect(sent.status, sent.text).toBe(200);
      const input = JSON.parse(sent.upstream.body).input;
      expect(input).not.toHaveProperty("converse");
      return JSON.parse(Buffer.from(input.invokeModel.body, "base64").toString("utf8"));
    };
    const plain = await counted((text) => text);
    const blocks = await counted((text) => [{ type: "text", text }]);
    expect(plain.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });
    expect(plain).toEqual(blocks);
  });

  test("a Claude model bedrock-runtime cannot count is counted on Bedrock Mantle", async (ctx) => {
    if (skip(ctx)) return;
    const before = upstream!.received.length;
    const res = await fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: BEDROCK_MANTLE_ONLY,
        system: "Be brief.",
        thinking: { type: "enabled", budget_tokens: 2048 },
        messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(JSON.parse(text)).toEqual({ input_tokens: 77 });
    const [runtime, mantle] = upstream!.received.slice(before);
    expect(runtime.path).toBe("/model/anthropic.claude-opus-5-5/count-tokens");
    expect(mantle.path).toBe("/anthropic/v1/messages/count_tokens");
    expect(mantle.headers["anthropic-version"]).toBe("2023-06-01");
    expect(String(mantle.headers.authorization)).toMatch(
      /^AWS4-HMAC-SHA256 Credential=AKIA-native-messages\/\d{8}\/us-west-2\/bedrock\/aws4_request/,
    );
    expect(JSON.parse(mantle.body)).toEqual({
      model: "anthropic.claude-opus-5-5",
      system: "Be brief.",
      thinking: { type: "enabled", budget_tokens: 2048 },
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
    });
  });

  test("a platform that cannot count a model's tokens hands count_tokens to the group's next target", async (ctx) => {
    if (skip(ctx)) return;
    const before = upstream!.received.length;
    const res = await fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: COUNT_GROUP, messages: [{ role: "user", content: "hello" }] }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(JSON.parse(text)).toEqual({ input_tokens: 99 });
    expect(upstream!.received.slice(before).map((r) => r.path)).toEqual([
      "/model/anthropic.claude-nocount-4-5-v1:0/count-tokens",
      "/anthropic/v1/messages/count_tokens",
      "/v1/messages/count_tokens",
    ]);
  });

  test("Vertex AI rawPredict gets the caller's body, adapted, and answers natively", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send(
      "/v1/messages",
      agentTurn(VERTEX, false, BASE64_IMAGE),
      `${CLAUDE_CODE_BETAS},effort-2025-11-24`,
    );
    expect(sent.status, sent.text).toBe(200);
    expect(sent.upstream.path).toBe(
      `/v1/projects/proj-e2e/locations/us-east5/publishers/anthropic/models/${VERTEX_ID}:rawPredict`,
    );
    expect(sent.upstream.headers["anthropic-beta"]).toBe(
      "context-1m-2025-08-07,context-management-2025-06-27,interleaved-thinking-2025-05-14",
    );
    const body = JSON.parse(sent.upstream.body);
    expect(body.anthropic_version).toBe("vertex-2023-10-16");
    expect(body).not.toHaveProperty("model");
    expect(body).not.toHaveProperty("anthropic_beta");
    expect(body).not.toHaveProperty("gateway_unknown_field");
    expect(body.metadata).toEqual({ user_id: "user_abc_account_1_session_2" });
    expect(body.context_management.edits).toHaveLength(2);
    expect(body.system[0].cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    expect(body.messages[1].content[0].signature).toBe("sig-history");
    expect(body.messages[2].content[0].content[1]).toEqual(BASE64_IMAGE);

    const out = JSON.parse(sent.text);
    expect(out.id).toBe("msg_vrtx_nonstream");
    expect(out.content[0].signature).toBe("sig-reply");
    expect(out.stop_reason).toBe("pause_turn");
  });

  test("Vertex AI streamRawPredict relays the Anthropic stream natively", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages", agentTurn(VERTEX, true, BASE64_IMAGE));
    expect(sent.status, sent.text).toBe(200);
    expect(sent.upstream.path).toMatch(/:streamRawPredict$/);
    expect(JSON.parse(sent.upstream.body).stream).toBe(true);
    const events = frames(sent.text);
    expect(events.find((e) => e.type === "message_start").message.id).toBe("msg_vrtx_stream");
    expect(events).toContainEqual({
      type: "content_block_delta",
      index: 0,
      delta: { type: "signature_delta", signature: "sig-stream" },
    });
    const row = await usageRow(sent.requestId);
    expect(row.get("cache_read_tokens")).toBe(String(CACHE_READ));
    expect(row.get("provider_request_id")).toBe("msg_vrtx_stream");
  });

  test("count_tokens on Claude on Vertex AI calls its count-tokens model", async (ctx) => {
    if (skip(ctx)) return;
    const sent = await send("/v1/messages/count_tokens", {
      model: VERTEX,
      metadata: { user_id: "u-1", tenant: "internal" },
      gateway_unknown_field: 1,
      system: [{ type: "text", text: "s", cache_control: { type: "ephemeral", ttl: "1h", scope: "global" } }],
      messages: [{ role: "user", content: "hello" }],
    });
    expect(sent.status, sent.text).toBe(200);
    expect(JSON.parse(sent.text)).toEqual({ input_tokens: 17 });
    expect(sent.upstream.path).toBe(
      "/v1/projects/proj-e2e/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
    );
    const body = JSON.parse(sent.upstream.body);
    // The count-tokens model refuses a versioned name.
    expect(body.model).toBe("claude-sonnet-4-5");
    expect(body).not.toHaveProperty("gateway_unknown_field");
    expect(body.metadata).toEqual({ user_id: "u-1" });
    expect(body.system[0].cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
  });
});
