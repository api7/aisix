import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  invokeChunk,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a prompt-cache write is reported on the usage event split by cache
// lifetime whenever the upstream reports the split, so a one-hour write can
// be priced at its own (higher) rate. `cache_creation_tokens` stays the
// total; `cache_creation_5m_tokens` / `cache_creation_1h_tokens` are its
// parts, and both are absent when the upstream reported no split.
//
// Every upstream that reports the split is covered, on each client protocol
// that reaches it: Anthropic's `usage.cache_creation` (direct, and Claude on
// Bedrock and Vertex AI through their own Messages routes), on the response
// body and on a stream's `message_start`; and Bedrock Converse's
// `usage.cacheDetails`, non-streamed and on ConverseStream's `metadata`.
//
// Reference:
//   - <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#1-hour-cache-duration>
//   - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html>
//   - <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CacheDetail.html>

const CALLER = "sk-cache-creation-ttl";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const LOGSTORE = "cache-ttl-store";

const UNCACHED_IN = 8;
const CACHE_READ = 900;
const WRITE_5M = 40;
const WRITE_1H = 260;
const CACHE_WRITE = WRITE_5M + WRITE_1H;

const ANTHROPIC = "ttl-anthropic";
const ANTHROPIC_NO_SPLIT = "ttl-anthropic-nosplit";
const ANTHROPIC_PARTIAL = "ttl-anthropic-partial";
const ANTHROPIC_UNCACHED = "ttl-anthropic-uncached";
const BEDROCK_CLAUDE = "ttl-bedrock-claude";
const VERTEX_CLAUDE = "ttl-vertex-claude";
const CONVERSE = "ttl-bedrock-nova";
const CONVERSE_PARTIAL = "ttl-bedrock-nova-partial";

// The model name an upstream sees tells it which split to report.
const NO_SPLIT_MARKER = "nosplit";
const PARTIAL_MARKER = "partial";
const UNCACHED_MARKER = "uncached";

// `split`: the parts add up to the write. `none`: no `cache_creation`.
// `partial`: the parts miss part of the write. `uncached`: nothing was
// written, and the breakdown is all zeros, as Anthropic sends it then.
type Split = "split" | "none" | "partial" | "uncached";

function splitFor(model: unknown): Split {
  const name = String(model);
  if (name.includes(NO_SPLIT_MARKER)) return "none";
  if (name.includes(PARTIAL_MARKER)) return "partial";
  if (name.includes(UNCACHED_MARKER)) return "uncached";
  return "split";
}

function anthropicUsage(split: Split, output: number): Record<string, unknown> {
  const parts: Record<Split, Record<string, unknown>> = {
    split: { cache_creation: { ephemeral_5m_input_tokens: WRITE_5M, ephemeral_1h_input_tokens: WRITE_1H } },
    none: {},
    partial: { cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: WRITE_1H } },
    uncached: { cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 0 } },
  };
  return {
    input_tokens: UNCACHED_IN,
    output_tokens: output,
    cache_read_input_tokens: CACHE_READ,
    cache_creation_input_tokens: split === "uncached" ? 0 : CACHE_WRITE,
    ...parts[split],
  };
}

function anthropicReply(split: Split): Record<string, unknown> {
  return {
    id: "msg_ttl",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-5-20250929",
    content: [{ type: "text", text: "ok" }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: anthropicUsage(split, 2),
  };
}

function anthropicStream(split: Split): Array<Record<string, unknown>> {
  return [
    {
      type: "message_start",
      message: {
        id: "msg_ttl_stream",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5-20250929",
        content: [],
        stop_reason: null,
        usage: anthropicUsage(split, 1),
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 5 } },
    { type: "message_stop" },
  ];
}

const sse = (events: Array<Record<string, unknown>>) =>
  events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join("");

// A partial breakdown does not account for the whole write.
function converseUsage(partial: boolean): Record<string, unknown> {
  return {
    inputTokens: UNCACHED_IN,
    outputTokens: 5,
    totalTokens: UNCACHED_IN + 5,
    cacheReadInputTokens: CACHE_READ,
    cacheWriteInputTokens: CACHE_WRITE,
    cacheDetails: partial
      ? [{ ttl: "1h", inputTokens: WRITE_1H }]
      : [
          { ttl: "5m", inputTokens: WRITE_5M },
          { ttl: "1h", inputTokens: WRITE_1H },
        ],
  };
}

async function startUpstream(): Promise<{ server: Server; baseUrl: string }> {
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = decodeURIComponent((req.url ?? "/").split("?")[0]);
      const raw = Buffer.concat(chunks).toString("utf8");
      const body = raw ? JSON.parse(raw) : {};
      const json = (payload: unknown) => {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(payload));
      };
      const eventStreamBody = (payload: Buffer) => {
        res.writeHead(200, { "content-type": "application/vnd.amazon.eventstream" });
        res.end(payload);
      };
      if (path === "/v1/messages") {
        const split = splitFor(body.model);
        if (body.stream) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(sse(anthropicStream(split)));
          return;
        }
        return json(anthropicReply(split));
      }
      if (path.endsWith("/invoke")) return json(anthropicReply("split"));
      if (path.endsWith("/invoke-with-response-stream")) {
        return eventStreamBody(Buffer.concat(anthropicStream("split").map(invokeChunk)));
      }
      if (path.endsWith(":rawPredict")) return json(anthropicReply("split"));
      if (path.endsWith(":streamRawPredict")) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end(sse(anthropicStream("split")));
        return;
      }
      const partial = path.includes(PARTIAL_MARKER);
      if (path.endsWith("/converse")) {
        return json({
          output: { message: { role: "assistant", content: [{ text: "ok" }] } },
          stopReason: "end_turn",
          usage: converseUsage(partial),
          metrics: { latencyMs: 1 },
        });
      }
      if (path.endsWith("/converse-stream")) {
        return eventStreamBody(
          eventStream([
            ["messageStart", { role: "assistant" }],
            ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "ok" } }],
            ["contentBlockStop", { contentBlockIndex: 0 }],
            ["messageStop", { stopReason: "end_turn" }],
            ["metadata", { usage: converseUsage(partial), metrics: { latencyMs: 1 } }],
          ]),
        );
      }
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ message: `no route ${path}` }));
    });
  });
  // Node closes an idle keep-alive connection after 5s, which is about how
  // long the gateway's pooled connection to this upstream sits idle between
  // the cases that use it; a request sent as the close lands fails that
  // attempt and is retried. Never closing first takes the race away.
  server.keepAliveTimeout = 120_000;
  server.headersTimeout = 125_000;
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { server, baseUrl: `http://127.0.0.1:${addr.port}` };
}

type Protocol = "messages" | "chat" | "responses";

function requestFor(protocol: Protocol, model: string, stream: boolean): [string, Record<string, unknown>] {
  switch (protocol) {
    case "messages":
      return ["/v1/messages", { model, max_tokens: 64, stream, messages: [{ role: "user", content: "hi" }] }];
    case "chat":
      return ["/v1/chat/completions", { model, stream, messages: [{ role: "user", content: "hi" }] }];
    case "responses":
      return ["/v1/responses", { model, stream, input: "hi" }];
  }
}

describe("UsageEvent: prompt-cache writes split by cache lifetime", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const up = await startUpstream();
    upstream = up.server;
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "cache-ttl-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "cache-ttl-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "ttl-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: up.baseUrl,
    });
    const bedrockPk = await seed.createProviderKey({
      display_name: "ttl-bedrock-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({ access_key_id: "AKIA-cache-ttl", secret_access_key: "sk-cache-ttl", region: "us-west-2" }),
      api_base: up.baseUrl,
    });
    const vertexPk = await seed.createProviderKey({
      display_name: "ttl-vertex-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({ access_token: "ya29.cache-ttl", project: "proj-e2e", region: "us-east5" }),
      api_base: up.baseUrl,
    });
    for (const [display_name, pk, provider, model_name] of [
      [ANTHROPIC, anthropicPk, "anthropic", "claude-sonnet-4-5"],
      [ANTHROPIC_NO_SPLIT, anthropicPk, "anthropic", `claude-sonnet-4-5-${NO_SPLIT_MARKER}`],
      [ANTHROPIC_PARTIAL, anthropicPk, "anthropic", `claude-sonnet-4-5-${PARTIAL_MARKER}`],
      [ANTHROPIC_UNCACHED, anthropicPk, "anthropic", `claude-sonnet-4-5-${UNCACHED_MARKER}`],
      [BEDROCK_CLAUDE, bedrockPk, "bedrock", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"],
      [VERTEX_CLAUDE, vertexPk, "google", "claude-sonnet-4-5@20250929"],
      [CONVERSE, bedrockPk, "bedrock", "amazon.nova-lite-v1:0"],
      [CONVERSE_PARTIAL, bedrockPk, "bedrock", `amazon.nova-${PARTIAL_MARKER}-v1:0`],
    ] as const) {
      await seed.createModel({ display_name, provider, model_name, provider_key_id: pk.id });
    }

    // Seeded last: this key authenticating implies the whole seed set.
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
    await sls?.close();
  });

  async function usageRowFor(protocol: Protocol, model: string, stream: boolean): Promise<Map<string, string>> {
    const [path, body] = requestFor(protocol, model, stream);
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId, "x-aisix-request-id").not.toBe("");
    // The attempt that answered: a retried request also has a row for the
    // attempt that failed first, with the same request id and no usage.
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId && l.get("status_code") === "200",
      `usage row for ${requestId}`,
      15_000,
    );
  }

  const split: Array<[string, Protocol, string, boolean]> = [];
  for (const stream of [false, true]) {
    const mode = stream ? "streamed" : "non-streamed";
    split.push(
      [`Anthropic, /v1/messages, ${mode}`, "messages", ANTHROPIC, stream],
      [`Anthropic, /v1/chat/completions, ${mode}`, "chat", ANTHROPIC, stream],
      [`Anthropic, /v1/responses, ${mode}`, "responses", ANTHROPIC, stream],
      [`Claude on Bedrock, /v1/messages, ${mode}`, "messages", BEDROCK_CLAUDE, stream],
      [`Claude on Vertex AI, /v1/messages, ${mode}`, "messages", VERTEX_CLAUDE, stream],
      [`Bedrock Converse, /v1/chat/completions, ${mode}`, "chat", CONVERSE, stream],
    );
  }

  for (const [name, protocol, model, stream] of split) {
    test(`${name} reports the split beside the total`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const row = await usageRowFor(protocol, model, stream);
      expect(row.get("cache_creation_tokens")).toBe(String(CACHE_WRITE));
      expect(row.get("cache_creation_5m_tokens")).toBe(String(WRITE_5M));
      expect(row.get("cache_creation_1h_tokens")).toBe(String(WRITE_1H));
      expect(row.get("cache_read_tokens")).toBe(String(CACHE_READ));
    });
  }

  const unsplit: Array<[string, Protocol, string, boolean]> = [
    ["Anthropic without usage.cache_creation, non-streamed", "messages", ANTHROPIC_NO_SPLIT, false],
    ["Anthropic without usage.cache_creation, streamed", "messages", ANTHROPIC_NO_SPLIT, true],
    ["Anthropic with a usage.cache_creation short of the write, non-streamed", "messages", ANTHROPIC_PARTIAL, false],
    ["Anthropic with a usage.cache_creation short of the write, streamed", "chat", ANTHROPIC_PARTIAL, true],
    ["Bedrock Converse with a partial cacheDetails, non-streamed", "chat", CONVERSE_PARTIAL, false],
    ["Bedrock Converse with a partial cacheDetails, streamed", "chat", CONVERSE_PARTIAL, true],
  ];

  for (const [name, protocol, model, stream] of unsplit) {
    test(`${name} reports the total and no split`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const row = await usageRowFor(protocol, model, stream);
      expect(row.get("cache_creation_tokens")).toBe(String(CACHE_WRITE));
      expect(row.has("cache_creation_5m_tokens")).toBe(false);
      expect(row.has("cache_creation_1h_tokens")).toBe(false);
    });
  }

  // Anthropic reports an all-zero breakdown on a request that wrote nothing
  // to the cache; the parts of a write that did not happen are not reported.
  for (const stream of [false, true]) {
    test(`an uncached Anthropic request, ${stream ? "streamed" : "non-streamed"}, reports no split`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const row = await usageRowFor("messages", ANTHROPIC_UNCACHED, stream);
      expect(row.has("cache_creation_tokens")).toBe(false);
      expect(row.has("cache_creation_5m_tokens")).toBe(false);
      expect(row.has("cache_creation_1h_tokens")).toBe(false);
      expect(row.get("cache_read_tokens")).toBe(String(CACHE_READ));
    });
  }
});
