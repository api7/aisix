import { createHash } from "node:crypto";
import { randomUUID } from "node:crypto";
import { createServer, type Server as HttpServer } from "node:http";
import { createServer as createTcpServer, type Server as TcpServer } from "node:net";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
import { WebSocket as WsClient, WebSocketServer } from "ws";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  eventStreamException,
  expectFailureRowsNamed,
  invokeChunk,
  slsLogsFor,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1921 and #1948. Under a full-content exporter every failure row
// carries the request body and the attempt's own response body; the wording of
// `error_message` and the access log names why an upstream gave no detail, and
// what the caller receives does not change.

const CALLER = "sk-failure-row-caller";
const LIMITED_CALLER = "sk-failure-row-limited";
const MCP_LIMITED_CALLER = "sk-failure-row-mcp-limited";
const MCP_BLOCKED_WORD = "sentinel-mcp-forbidden-word";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const FULL = "failure-row-full";
const META = "failure-row-meta";
const CAPPED = "failure-row-capped";
const CAPPED_MAX_BYTES = 2048;
const RESPONSES_DELTA = `event: response.output_text.delta\r\ndata: ${JSON.stringify({
  type: "response.output_text.delta", item_id: "msg_1", output_index: 0,
  content_index: 0, delta: "partial Responses answer", sequence_number: 2,
})}\r\n\r\n`;
const RESPONSES_FAILED = `event: response.failed\r\ndata: ${JSON.stringify({
  type: "response.failed", sequence_number: 3,
  response: { id: "resp_failed", object: "response", status: "failed", error: null, output: [], custom: { trace: "provider-raw" } },
})}\r\n\r\n`;
const CHAT_STREAM_ERROR = `data: ${JSON.stringify({
  error: { message: "sentinel-in-band", code: 500, metadata: { provider_request: null } },
})}\n\n`;
const IN_BAND_JSON = { error: { message: "provider whole-body error", code: 502, metadata: { ticket: null } } };
const COUNT_IN_BAND_ERROR = '{ "type":"error", "error":{"type":"api_error","message":"provider count failed","details":null} }';
const INVALID_JSON = '{"provider_answer":"readable but cut before the closing quote';
const TIMEOUT_PREFIX = '{"error":{"message":"provider began the error body';
const OUTPUT_EMAIL = "private-output@example.com";
const JOBS_OUTPUT_PREFIX = '{"output":"partial Responses answer';
const JSON_LINES_ERROR = '{"type": "refused", "provider_detail": null}\r\n{"trace": 42}\r\n';
const RESPONSE_MEDIA = "cHJpdmF0ZS1pbWFnZS1wYXlsb2Fk";
const ESCAPED_AUDIO_BODY = String.raw`{"error":{"message":"readable before audio"},"\u0061udio":"${RESPONSE_MEDIA}`;
const MEDIA_FRAME = `event: response.image_generation_call.partial_image\ndata\ndata: ${JSON.stringify({
  type: "response.image_generation_call.partial_image", sequence_number: 1,
  item_id: "img_1", output_index: 0, partial_image_index: 0, partial_image_b64: RESPONSE_MEDIA,
})}\n\n`;
const MEDIA_PREFIX = "c2VudGluZWwtbGFyZ2UtaW1hZ2Uh";
const OVERSIZED_MEDIA_FRAME = `event: response.output_item.done\r\ndata: ${JSON.stringify({
  item: { result: MEDIA_PREFIX.repeat(6_000), type: "image_generation_call", id: "img_large" },
  type: "response.output_item.done", output_index: 1,
})}\r\n\r\n`;
const LONG_DELTA = RESPONSES_DELTA.replace("partial Responses answer", "内容".repeat(600));
const BEDROCK_EXCEPTION = { message: "Bedrock stream throttled", provider_details: { hint: null } };
const ANTHROPIC_START = {
  type: "message_start",
  message: { id: "msg_native", model: "claude", role: "assistant", content: [], usage: { input_tokens: 5, output_tokens: 0 } },
};
const ANTHROPIC_DELTA = { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "partial Bedrock response" } };

const NO_DETAIL = (status: number) =>
  `upstream returned HTTP ${status} with no error detail in the response`;
const READ_FAILURE = (status: number) =>
  new RegExp(`^upstream returned HTTP ${status}: failed to read response body: .+`);

type Row = Map<string, string>;

describe("failure rows carry request and response bodies", () => {
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  const tcpServers: TcpServer[] = [];
  const httpServers: HttpServer[] = [];
  let mcpUpstream: McpUpstream | undefined;
  let realtimeUpstream: WebSocketServer | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;
  let seed: SeedClient;

  const headers = {
    authorization: `Bearer ${CALLER}`,
    "content-type": "application/json",
  };

  async function failingModel(
    name: string,
    opts: Parameters<typeof startOpenAiUpstream>[0],
    { translateResponses = false, streamTimeout }: { translateResponses?: boolean; streamTimeout?: number } = {},
  ): Promise<string> {
    const upstream = await startOpenAiUpstream(opts);
    upstreams.push(upstream);
    const pk = await seed.createProviderKey({
      display_name: `${name}-pk`,
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      ...(translateResponses ? { apis: {} } : {}),
    });
    const model = await seed.createModel({
      display_name: name,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      cooldown: { enabled: false },
      stream_timeout: streamTimeout,
    });
    return model.id;
  }

  /** An upstream answering every path and method the same way. */
  async function fixedUpstream(
    status: number,
    body: string | Buffer,
    responseHeaders: Record<string, string> = { "content-type": "application/json" },
  ): Promise<string> {
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        res.writeHead(status, responseHeaders);
        res.end(body);
      });
    });
    server.keepAliveTimeout = 120_000;
    server.headersTimeout = 125_000;
    httpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  /** An upstream that drops the first connection unanswered, then answers `status`. */
  async function dropsFirstUpstream(status: number, body: string): Promise<string> {
    let dropped = false;
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        if (!dropped) {
          dropped = true;
          req.socket.destroy();
          return;
        }
        res.writeHead(status, { "content-type": "application/json" });
        res.end(body);
      });
    });
    server.keepAliveTimeout = 120_000;
    server.headersTimeout = 125_000;
    httpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  /** An upstream that declares more bytes than it sends, then drops the connection. */
  async function truncatingUpstream(status: string, head: string): Promise<string> {
    const server = createTcpServer((socket) => {
      socket.once("data", () => {
        socket.write(
          `HTTP/1.1 ${status}\r\ncontent-type: application/json\r\n` +
            `content-length: 4096\r\n\r\n${head}`,
          () => socket.destroy(),
        );
      });
      socket.on("error", () => {});
    });
    tcpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  async function modelAt(
    name: string,
    apiBase: string,
    provider: { provider?: string; key_provider?: string; adapter?: string; model_name?: string; api_key?: string } = {},
  ): Promise<string> {
    const { provider: providerName = "openai", key_provider = providerName, adapter, model_name = "gpt-4o-mini", api_key } = provider;
    const pk = await seed.createProviderKey({
      display_name: `${name}-pk`,
      api_key: api_key ?? "sk-mock",
      provider: key_provider,
      ...(adapter ? { adapter } : {}),
      api_base: apiBase,
    });
    const model = await seed.createModel({
      display_name: name,
      provider: providerName,
      model_name,
      provider_key_id: pk.id,
      cooldown: { enabled: false },
    });
    return model.id;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        // The access-log line is an INFO line.
        RUST_LOG: "info",
        SLS_CRED_MOCK_AK_ID: "LTAI_mock_ak",
        SLS_CRED_MOCK_AK_SECRET: "mock_ak_secret",
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    for (const [name, logstore, content_mode] of [
      ["failure-row-full", FULL, "full"],
      ["failure-row-meta", META, "metadata_only"],
    ] as const) {
      await seed.createObservabilityExporter({
        name,
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: "aisix-e2e-obs",
        logstore,
        credential_ref: CREDENTIAL_REF,
        content_mode,
      });
    }
    await seed.createObservabilityExporter({
      name: CAPPED,
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: CAPPED,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
      content_max_bytes: CAPPED_MAX_BYTES,
    });

    for (const status of [400, 401, 403]) {
      await failingModel(`row-${status}`, {
        status,
        errorBody: { error: { message: `sentinel-body-${status}`, type: "upstream_type" } },
      });
    }
    await failingModel("row-first-fails", {
      status: 500,
      errorBody: { error: { message: "sentinel-first-target" } },
    });
    await failingModel("row-second-fails", {
      status: 503,
      errorBody: { error: { message: "sentinel-second-target" } },
    });
    await failingModel("row-ok", {
      nonStreamBody: {
        id: "cmpl-ok",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "fine" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });
    await seed.createModel({
      display_name: "row-all-fail",
      routing: {
        strategy: "failover",
        targets: [{ model: "row-first-fails" }, { model: "row-second-fails" }],
      },
    });
    await seed.createModel({
      display_name: "row-recovers",
      routing: {
        strategy: "failover",
        targets: [{ model: "row-first-fails" }, { model: "row-ok" }],
      },
    });
    await seed.createModel({
      display_name: "row-ensemble-fails",
      ensemble: {
        panel: [{ model: "row-first-fails" }, { model: "row-second-fails" }],
        judge: { model: "row-ok" },
        min_responses: 1,
      },
    });
    const delta = (content: string) =>
      JSON.stringify({
        id: "c",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-mini",
        choices: [{ index: 0, delta: { content }, finish_reason: null }],
      });
    await failingModel("row-stream-cut", {
      streamEvents: [delta("partial-answer"), delta(" more"), delta(" more")],
      disconnectAfterEvents: 1,
      eventDelayMs: 100,
    });
    await failingModel("row-in-band", {
      rawStreamFrames: [CHAT_STREAM_ERROR],
    }, { translateResponses: true });
    await failingModel("row-responses-late", {
      rawStreamFrames: [RESPONSES_DELTA, RESPONSES_FAILED],
      eventDelayMs: 75,
    });
    await failingModel("row-json-error", { nonStreamBody: IN_BAND_JSON });
    await failingModel("row-invalid-json", { rawContentType: "application/json", rawBody: INVALID_JSON });
    await failingModel("row-invalid-escaped-media", { rawContentType: "application/json", rawBody: ESCAPED_AUDIO_BODY });
    await failingModel("row-timeout-after-error-head", {
      status: 503, rawErrorBodyChunks: [TIMEOUT_PREFIX], eventDelayMs: 1_500,
    }, { streamTimeout: 350 });
    await failingModel("row-responses-media", {
      rawStreamFrames: [MEDIA_FRAME, LONG_DELTA, RESPONSES_FAILED],
    });
    await failingModel("row-responses-large-media", {
      rawStreamFrames: [RESPONSES_DELTA, OVERSIZED_MEDIA_FRAME, RESPONSES_FAILED],
    });
    await failingModel("row-responses-octet-sse", {
      rawContentType: "application/octet-stream",
      rawBodyChunks: [RESPONSES_DELTA.slice(0, -1), "\n" + RESPONSES_FAILED],
      eventDelayMs: 25,
    });
    await failingModel("row-responses-rollover", {
      rawStreamFrames: [RESPONSES_DELTA.repeat(900), RESPONSES_FAILED],
    });
    const maskedModel = await failingModel("row-responses-masked", {
      rawStreamFrames: [RESPONSES_DELTA.replace("partial Responses answer", OUTPUT_EMAIL), RESPONSES_FAILED],
    });
    const blockedModel = await failingModel("row-responses-blocked", {
      rawStreamFrames: [RESPONSES_DELTA, RESPONSES_FAILED],
    });
    const bridgedBlockedModel = await failingModel("row-responses-bridge-blocked", {
      streamEvents: [delta("partial Responses answer")],
    }, { translateResponses: true });
    const outputMask = await seed.createGuardrail({
      name: "row-output-mask", kind: "pii", hook_point: "output",
      detectors: [{ type: "email", action: "mask" }],
    }, { attach: false });
    const outputBlock = await seed.createGuardrail({
      name: "row-output-block", kind: "keyword", hook_point: "output",
      patterns: [{ kind: "literal", value: "partial Responses answer" }],
    }, { attach: false });
    await seed.attachGuardrailToModel(outputMask.id, maskedModel);
    await seed.attachGuardrailToModel(outputBlock.id, blockedModel);
    await seed.attachGuardrailToModel(outputBlock.id, bridgedBlockedModel);
    // Configured, and not on the caller's list.
    await failingModel("row-unlisted", { status: 400 });
    await failingModel("row-empty-404", { status: 404, rawErrorBody: "" });
    await failingModel("row-blank-message", {
      status: 404,
      errorBody: { error: { message: "" } },
    });
    await failingModel("row-control-only", {
      status: 500,
      rawErrorBody: "\n\t\r\n",
      errorContentType: "text/plain",
    });
    await failingModel("row-html", {
      status: 502,
      rawErrorBody: "<html>sentinel fronting proxy page</html>",
      errorContentType: "text/html",
    });

    const fragment = '{"error":{"message":"sentinel-cut';
    const truncatedBase = await truncatingUpstream("502 Bad Gateway", fragment);
    await modelAt(
      "row-drops-then-400",
      `${await dropsFirstUpstream(400, JSON.stringify({ error: { message: "sentinel-retried-400" } }))}/v1`,
    );
    await modelAt("row-truncated", `${truncatedBase}/v1`);
    await modelAt("row-truncated-rerank", `${truncatedBase}/v1`, {
      provider: "cohere",
      adapter: "openai",
      model_name: "rerank-v3.5",
    });
    await modelAt("row-truncated-empty", `${await truncatingUpstream("502 Bad Gateway", "")}/v1`);
    const truncatedPk = await seed.createProviderKey({
      display_name: "row-truncated-route-pk",
      api_key: "sk-mock",
      api_base: `${truncatedBase}/v1`,
    });

    const answer400 = JSON.stringify({ error: { message: "sentinel-any-400", type: "upstream_type" } });
    const any400 = await fixedUpstream(400, answer400);
    await modelAt("row-any-400", `${any400}/v1`);
    await modelAt("row-rerank-400", `${any400}/v1`, {
      provider: "cohere",
      adapter: "openai",
      model_name: "rerank-v3.5",
    });
    await modelAt("row-count-400", any400, { provider: "anthropic", model_name: "claude-sonnet-4-5" });
    const invalidCount = await modelAt("row-count-invalid", await fixedUpstream(200, COUNT_IN_BAND_ERROR), {
      provider: "anthropic", model_name: "claude-sonnet-4-5",
    });
    await seed.attachGuardrailToModel(outputBlock.id, invalidCount);
    await modelAt("row-images-unsupported", any400, { key_provider: "anthropic", adapter: "anthropic" });
    const cutJobs = await modelAt("row-jobs-cut", `${await truncatingUpstream("200 OK", JOBS_OUTPUT_PREFIX)}/v1`);
    await seed.attachGuardrailToModel(outputBlock.id, cutJobs);
    const cutJobsError = await modelAt("row-jobs-error-cut", `${await truncatingUpstream("502 Bad Gateway", JOBS_OUTPUT_PREFIX)}/v1`);
    await seed.attachGuardrailToModel(outputBlock.id, cutJobsError);
    const checkedJobs = await modelAt("row-jobs-checked", `${any400}/v1`);
    await seed.attachGuardrailToModel(outputBlock.id, checkedJobs);
    const bedrockProvider = {
      provider: "bedrock",
      adapter: "bedrock",
      api_key: JSON.stringify({
        access_key_id: "AKIA-failure-row",
        secret_access_key: "sk-failure-row",
        region: "us-east-1",
      }),
    };
    await modelAt("row-bedrock-400", await fixedUpstream(
      400,
      JSON.stringify({ message: "sentinel-bedrock-answer" }),
      {
        "content-type": "application/json",
        "x-amzn-errortype": "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/",
      },
    ), {
      ...bedrockProvider,
      model_name: "openai.gpt-6-sol",
    });
    const exception = eventStreamException("throttlingException", BEDROCK_EXCEPTION);
    const eventStreamHeaders = { "content-type": "application/vnd.amazon.eventstream" };
    await modelAt("row-bedrock-converse-stream", await fixedUpstream(200, Buffer.concat([
      eventStream([["messageStart", { role: "assistant" }], ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "partial" } }]]),
      exception,
    ]), eventStreamHeaders), { ...bedrockProvider, model_name: "deepseek.r1-v1:0" });
    await modelAt("row-bedrock-native-stream", await fixedUpstream(200, Buffer.concat([
      invokeChunk(ANTHROPIC_START), invokeChunk(ANTHROPIC_DELTA), exception,
    ]), eventStreamHeaders), { ...bedrockProvider, model_name: "anthropic.claude-haiku-4-5-20251001-v1:0" });
    const textHeaders = { "content-type": "text/plain" };
    await modelAt("row-video-text", `${await fixedUpstream(500, "sentinel video gateway page", textHeaders)}/v1`);
    await modelAt("row-video-empty", `${await fixedUpstream(500, "", textHeaders)}/v1`);
    await modelAt("row-video-json", `${await fixedUpstream(400, JSON.stringify({ error: { code: "bad", message: "sentinel-video" } }))}/v1`);
    await modelAt("row-video-cut", `${truncatedBase}/v1`);
    await modelAt("row-realtime-400", `${any400}/v1`);
    realtimeUpstream = new WebSocketServer({ host: "127.0.0.1", port: 0 });
    await new Promise<void>((resolve) => realtimeUpstream!.once("listening", resolve));
    realtimeUpstream.on("connection", (socket) => {
      socket.send(JSON.stringify({ type: "response.output_text.delta", delta: "partial Responses answer" }));
    });
    const realtimeAddr = realtimeUpstream.address() as { port: number };
    const realtimeBlocked = await modelAt("row-realtime-blocked", `http://127.0.0.1:${realtimeAddr.port}/v1`);
    await seed.attachGuardrailToModel(outputBlock.id, realtimeBlocked);
    const slowBody = {
      id: "cmpl-slow",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-mini",
      choices: [
        { index: 0, message: { role: "assistant", content: "late" }, finish_reason: "stop" },
      ],
    };
    await failingModel("row-slow", { nonStreamBody: slowBody, responseDelayMs: 5_000 });
    await failingModel("row-stream-slow", {
      streamEvents: [delta("cancel-partial"), delta(" more"), delta(" more"), delta(" more")],
      eventDelayMs: 1_000,
    });

    await seed.update("a2a_agents", randomUUID(), {
      name: "row-agent",
      url: await fixedUpstream(500, "sentinel-a2a-answer", textHeaders),
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });

    mcpUpstream = await startMcpUpstream("rowmcp");
    const mcpId = randomUUID();
    await seed.update("mcp_servers", mcpId, {
      name: "rowmcp",
      url: mcpUpstream.url,
      enabled: true,
    });
    const mcpGuard = await seed.createGuardrail(
      {
        name: "row-mcp-guard",
        kind: "keyword",
        patterns: [{ kind: "literal", value: MCP_BLOCKED_WORD }],
      },
      { attach: false },
    );
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: mcpGuard.id,
      scope_type: "mcp_server",
      scope_id: mcpId,
      priority: 100,
    });

    const passthroughUpstream = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "sentinel-passthrough-answer" } },
    });
    upstreams.push(passthroughUpstream);
    await seed.createPassthroughRoute({
      name: "row-passthrough",
      path_prefix: "/row-passthrough",
      target_url: passthroughUpstream.baseUrl,
      provider_key_id: truncatedPk.id,
    });
    await seed.createPassthroughRoute({
      name: "row-passthrough-jsonl",
      path_prefix: "/row-passthrough-jsonl",
      target_url: await fixedUpstream(400, JSON_LINES_ERROR, { "content-type": "application/jsonl; charset=utf-8" }),
      provider_key_id: truncatedPk.id,
    });

    await seed.createApiKey({
      key_hash: hash(MCP_LIMITED_CALLER),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
      mcp_rate_limits: { rowmcp: { rpm: 1 } },
    });
    await seed.createApiKey({
      key_hash: hash(LIMITED_CALLER),
      allowed_models: ["row-ok"],
      rate_limit: { rpm: 1 },
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_routes: ["*"],
      allowed_agents: ["*"],
      mcp_access: { allow: ["*"] },
      allowed_models: [
        "row-truncated-rerank",
        "row-truncated-empty",
        "row-any-400",
        "row-rerank-400",
        "row-count-400",
        "row-count-invalid",
        "row-images-unsupported",
        "row-jobs-cut",
        "row-jobs-error-cut",
        "row-jobs-checked",
        "row-bedrock-400",
        "row-bedrock-converse-stream",
        "row-bedrock-native-stream",
        "row-video-text",
        "row-video-empty",
        "row-video-json",
        "row-video-cut",
        "row-realtime-400",
        "row-realtime-blocked",
        "row-slow",
        "row-stream-slow",
        "row-400",
        "row-drops-then-400",
        "row-401",
        "row-403",
        "row-first-fails",
        "row-second-fails",
        "row-ok",
        "row-all-fail",
        "row-recovers",
        "row-ensemble-fails",
        "row-empty-404",
        "row-blank-message",
        "row-control-only",
        "row-html",
        "row-truncated",
        "row-stream-cut",
        "row-in-band",
        "row-responses-late",
        "row-json-error",
        "row-invalid-json",
        "row-invalid-escaped-media",
        "row-timeout-after-error-head",
        "row-responses-media",
        "row-responses-large-media",
        "row-responses-octet-sse",
        "row-responses-rollover",
        "row-responses-masked",
        "row-responses-blocked",
        "row-responses-bridge-blocked",
      ],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    if (etcdReachable) {
      for (const logstore of [FULL, META, CAPPED]) {
        expect(expectFailureRowsNamed(sls!, logstore)).toBeGreaterThan(0);
      }
    }
    for (const upstream of upstreams) await upstream.close();
    await mcpUpstream?.close();
    for (const server of tcpServers) server.close();
    for (const server of httpServers) server.close();
    if (realtimeUpstream) {
      for (const socket of realtimeUpstream.clients) socket.terminate();
      await new Promise<void>((resolve) => realtimeUpstream!.close(() => resolve()));
    }
    await sls?.close();
  });

  afterEach(() => {
    if (!etcdReachable) return;
    for (const logstore of [FULL, META, CAPPED]) expectFailureRowsNamed(sls!, logstore);
  });

  async function chat(model: string, content: string) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
    });
    return { res, requestId: res.headers.get("x-aisix-request-id")!, body: await res.text() };
  }

  function fullRows(requestId: string): Row[] {
    return slsLogsFor(sls!, FULL).filter((l) => l.get("request_id") === requestId);
  }

  /** The request's final row: a retried request leaves one row per attempt. */
  async function rowOf(requestId: string, what: string): Promise<Row> {
    return waitForSlsLog(
      sls!,
      FULL,
      (l) => l.get("request_id") === requestId && l.get("is_final_attempt") === "true",
      what,
    );
  }

  async function rowsOf(requestId: string, n: number): Promise<Row[]> {
    const deadline = Date.now() + 10_000;
    while (fullRows(requestId).length < n && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
    }
    const rows = fullRows(requestId);
    expect(rows.length, `rows of ${requestId}`).toBeGreaterThanOrEqual(n);
    return rows.sort((a, b) => Number(a.get("attempt_index")) - Number(b.get("attempt_index")));
  }

  const requestText = (row: Row) => row.get("input_messages") ?? row.get("prompt") ?? "";

  test.each([400, 401, 403])(
    "chat upstream %i carries the request and the upstream's own body",
    async (status) => {
      if (!etcdReachable) return;
      const prompt = `chat-prompt-${status}`;
      const { res, requestId } = await chat(`row-${status}`, prompt);
      expect(res.status).toBe(status);
      const row = await rowOf(requestId, `the ${status} row`);
      expect(row.get("status_code")).toBe(String(status));
      expect(requestText(row)).toContain(prompt);
      expect(JSON.parse(row.get("response")!)).toEqual({
        error: { message: `sentinel-body-${status}`, type: "upstream_type" },
      });
    },
  );

  test("a request over content_max_bytes keeps its cut request on the failure row", async () => {
    if (!etcdReachable) return;
    const prompt = "oversized-request-prompt";
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "row-400",
        messages: [{ role: "user", content: prompt }],
        tools: Array.from({ length: 64 }, (_, i) => ({
          type: "function",
          function: {
            name: `oversized_tool_${i}`,
            description: "a tool description long enough to matter ".repeat(8),
            parameters: { type: "object", properties: { value: { type: "string" } } },
          },
        })),
      }),
    });
    const body = await res.text();
    expect(res.status).toBe(400);
    const requestId = res.headers.get("x-aisix-request-id")!;
    const row = await waitForSlsLog(
      sls!,
      CAPPED,
      (l) => l.get("request_id") === requestId,
      "the capped failure row",
    );
    expect(row.get("error_message")).toContain("sentinel-body-400");
    expect(requestText(row)).toContain(prompt);
    expect(row.get("content_truncated") ?? row.get("truncated")).toBe("true");
    expect(body).toContain("sentinel-body-400");
    // The uncapped exporter on the same request has the whole thing.
    const whole = JSON.stringify(Object.fromEntries(await rowOf(requestId, "the uncapped row")));
    expect(whole).toContain("oversized_tool_63");
    expect(JSON.stringify(Object.fromEntries(row)).length).toBeLessThan(whole.length);
  });

  test("a model the key may not use still records what was sent and answered", async () => {
    if (!etcdReachable) return;
    const { res, requestId, body } = await chat("row-unlisted", "forbidden-prompt");
    expect(res.status).toBe(403);
    const row = await rowOf(requestId, "the refusal row");
    expect(requestText(row)).toContain("forbidden-prompt");
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  test.each([
    ["completions", "/v1/completions", { model: "row-count-400", prompt: "completion-prompt" }],
    ["embeddings", "/v1/embeddings", { model: "row-count-400", input: "embeddings-prompt" }],
    ["images", "/v1/images/generations", { model: "row-images-unsupported", prompt: "image-prompt" }],
    ["videos", "/v1/videos", { model: "row-count-400", prompt: "video-prompt" }],
  ] as const)("%s: a 501 records the actual gateway response", async (_what, path, input) => {
    if (!etcdReachable) return;
    const res = await post(path, input);
    const body = await res.text();
    expect(res.status).toBe(501);
    expect(JSON.parse(body).error.type).toBe("not_implemented");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, `the ${path} 501 row`);
    expect(row.get("response")).toBe(body);
  });

  test("Gemini: a refusal records the actual gateway response", async () => {
    if (!etcdReachable) return;
    const res = await post("/v1beta/models/row-unlisted:generateContent", {
      contents: [{ role: "user", parts: [{ text: "forbidden-prompt" }] }],
    });
    const body = await res.text();
    expect(res.status).toBe(403);
    expect(JSON.parse(body).error.status).toBe("PERMISSION_DENIED");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the Gemini refusal row");
    expect(row.get("response")).toBe(body);
  });

  test("every attempt of an all-failed group carries its own body", async () => {
    if (!etcdReachable) return;
    const { requestId } = await chat("row-all-fail", "group-prompt");
    const [first, second] = await rowsOf(requestId, 2);
    for (const row of [first!, second!]) expect(requestText(row)).toContain("group-prompt");
    expect(first!.get("response")).toContain("sentinel-first-target");
    expect(second!.get("response")).toContain("sentinel-second-target");
  });

  test("concurrent failed panel members each export their own body", async () => {
    if (!etcdReachable) return;
    const { requestId } = await chat("row-ensemble-fails", "ensemble-prompt");
    await rowOf(requestId, "the ensemble's terminal row");
    const panel = fullRows(requestId).filter((row) => row.get("attempt_kind") === "panel");
    expect(panel).toHaveLength(2);
    expect(Object.fromEntries(panel.map((row) => [row.get("attempt_model"), JSON.parse(row.get("response")!)]))).toEqual({
      "row-first-fails": { error: { message: "sentinel-first-target" } },
      "row-second-fails": { error: { message: "sentinel-second-target" } },
    });
  });

  test("a failed attempt before a winner carries its own body; the winner keeps its answer", async () => {
    if (!etcdReachable) return;
    const { res, requestId } = await chat("row-recovers", "recover-prompt");
    expect(res.status).toBe(200);
    const [failed, winner] = await rowsOf(requestId, 2);
    expect(requestText(failed!)).toContain("recover-prompt");
    expect(failed!.get("response")).toContain("sentinel-first-target");
    expect(winner!.get("status_code")).toBe("200");
    expect(winner!.get("output_messages") ?? winner!.get("response")).toContain("fine");
    expect(winner!.get("response") ?? "").not.toContain("sentinel-first-target");
  });

  test("a local rate-limit refusal carries the request and the error the caller got", async () => {
    if (!etcdReachable) return;
    const send = () =>
      fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { ...headers, authorization: `Bearer ${LIMITED_CALLER}` },
        body: JSON.stringify({
          model: "row-ok",
          messages: [{ role: "user", content: "limited-prompt" }],
        }),
      });
    let refused: Response | undefined;
    for (let i = 0; i < 5 && !refused; i++) {
      const res = await send();
      if (res.status === 429) refused = res;
      else await res.text();
    }
    expect(refused, "the key's rpm refused a request").toBeDefined();
    const body = await refused!.text();
    const row = await rowOf(refused!.headers.get("x-aisix-request-id")!, "the 429 row");
    expect(row.get("status_code")).toBe("429");
    expect(requestText(row)).toContain("limited-prompt");
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  test("embeddings upstream 400 carries both bodies", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: "row-400", input: "embed-prompt" }),
    });
    await res.text();
    expect(res.status).toBe(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the embeddings row");
    expect(row.get("status_code")).toBe("400");
    expect(row.get("prompt")).toContain("embed-prompt");
    expect(row.get("response")).toContain("sentinel-body-400");
  });

  test("a retried attempt's 400 is on the final row, after the lost connection's row", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: "row-drops-then-400", input: "retried-prompt" }),
    });
    await res.text();
    expect(res.status).toBe(400);
    const requestId = res.headers.get("x-aisix-request-id")!;
    const row = await rowOf(requestId, "the retried embeddings row");
    expect(row.get("status_code")).toBe("400");
    expect(row.get("prompt")).toContain("retried-prompt");
    expect(row.get("response")).toContain("sentinel-retried-400");
    const [lost] = await rowsOf(requestId, 2);
    expect(lost!.get("is_final_attempt")).toBe("false");
    expect(lost!.get("response")).toBe("");
  });

  test("a multipart failure reports text fields and a file checksum, never the file", async () => {
    if (!etcdReachable) return;
    const form = new FormData();
    form.set("model", "row-400");
    form.set("prompt", "transcribe-prompt");
    form.set("file", new Blob(["sentinel-audio-bytes"]), "clip.mp3");
    const res = await fetch(`${app!.proxyUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}` },
      body: form,
    });
    await res.text();
    expect(res.status).toBe(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the transcription row");
    const prompt = JSON.parse(row.get("prompt")!) as Record<string, string>;
    expect(prompt.prompt).toBe("transcribe-prompt");
    expect(prompt.file_sha256).toBe(hash("sentinel-audio-bytes"));
    expect(row.get("prompt")).not.toContain("sentinel-audio-bytes");
    expect(row.get("response")).toContain("sentinel-body-400");
  });

  test("an upload missing its file still reports the fields it sent", async () => {
    if (!etcdReachable) return;
    const form = new FormData();
    form.set("purpose", "upload-without-file-prompt");
    const res = await fetch(`${app!.proxyUrl}/v1/files?model=row-400`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}` },
      body: form,
    });
    expect(res.status).toBe(400);
    await res.text();
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the upload row");
    expect(requestText(row)).toContain("upload-without-file-prompt");
  });

  test("a passthrough request with no body still reports the upstream's failure", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/row-passthrough/anything`, {
      headers: { authorization: `Bearer ${CALLER}` },
    });
    expect(res.status).toBe(500);
    await res.text();
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the bodyless passthrough row");
    // As sent, not the gateway's own envelope quoting it.
    expect(JSON.parse(row.get("response")!)).toEqual({
      error: { message: "sentinel-passthrough-answer" },
    });
  });

  test("a passthrough route's upstream failure carries both bodies", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/row-passthrough/anything`, {
      method: "POST",
      headers,
      body: JSON.stringify({ note: "passthrough-prompt" }),
    });
    await res.text();
    expect(res.status).toBe(500);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the passthrough row");
    expect(row.get("prompt")).toContain("passthrough-prompt");
    expect(row.get("response")).toContain("sentinel-passthrough-answer");
  });

  test("a passthrough JSONL failure records the upstream lines verbatim", async () => {
    if (!etcdReachable) return;
    const res = await post("/row-passthrough-jsonl/anything", {});
    expect(res.status).toBe(400);
    expect(await res.text()).toBe(JSON_LINES_ERROR);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the passthrough JSONL row");
    expect(row.get("response")).toBe(JSON_LINES_ERROR);
  });

  test("count_tokens records an in-band failure with an input-only guardrail path", async () => {
    if (!etcdReachable) return;
    const res = await post("/v1/messages/count_tokens", {
      model: "row-count-invalid", messages: [{ role: "user", content: "count-prompt" }],
    }, { "anthropic-version": "2023-06-01" });
    await res.text();
    expect(res.status).toBe(502);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the failed count_tokens row");
    expect(row.get("response")).toBe(COUNT_IN_BAND_ERROR);
  });

  test.each([
    ["/v1/batches/batch_1", "row-jobs-cut", false],
    ["/v1/fine_tuning/jobs", "row-jobs-cut", false],
    ["/v1/fine_tuning/jobs/job_1", "row-jobs-error-cut", false],
    ["/v1/files/file_1/content", "row-jobs-cut", true],
  ] as const)("%s records a partial response according to its output guardrail path", async (path, model, opaque) => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}${path}?model=${model}`, { headers });
    const body = await res.text();
    expect(res.status).toBe(502);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, `the failed ${path} row`);
    expect(row.get("response")).toBe(opaque ? JOBS_OUTPUT_PREFIX : body);
  });

  test("jobs records the complete upstream error after its output guardrail allows it", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/batches/batch_1?model=row-jobs-checked`, { headers });
    await res.text();
    expect(res.status).toBe(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the checked jobs error row");
    expect(row.get("response")).toBe(ANY_400);
  });

  test("count_tokens and videos, which capture nothing on success, report a failure", async () => {
    if (!etcdReachable) return;
    const counted = await fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: { ...headers, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({
        model: "row-unlisted",
        messages: [{ role: "user", content: "count-prompt" }],
      }),
    });
    const countedBody = await counted.text();
    expect(counted.status).toBe(403);
    const countedRow = await rowOf(counted.headers.get("x-aisix-request-id")!, "the count row");
    expect(requestText(countedRow)).toContain("count-prompt");
    // The Anthropic shape this route answers with, not the OpenAI one.
    expect(JSON.parse(countedRow.get("response")!)).toEqual(JSON.parse(countedBody));
    expect(JSON.parse(countedBody).type).toBe("error");

    const video = await fetch(`${app!.proxyUrl}/v1/videos`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: "row-unlisted", prompt: "video-prompt" }),
    });
    await video.text();
    expect(video.status).toBe(403);
    const videoRow = await rowOf(video.headers.get("x-aisix-request-id")!, "the video row");
    expect(requestText(videoRow)).toContain("video-prompt");
  });

  test.each([
    ["an empty body", "row-empty-404", 404, 404, "upstream returned 404 Not Found"],
    ["a blank message", "row-blank-message", 404, 404, "upstream returned 404 Not Found"],
    ["a body of control characters", "row-control-only", 500, 502, "upstream returned 500 Internal Server Error"],
  ] as const)(
    "%s names the missing detail; the client's answer does not",
    async (_what, model, status, clientStatus, clientMessage) => {
      if (!etcdReachable) return;
      const { res, requestId, body } = await chat(model, "no-detail-prompt");
      expect(res.status).toBe(clientStatus);
      expect((JSON.parse(body) as { error: { message: string } }).error.message).toBe(
        clientMessage,
      );
      expect(body).not.toContain("no error detail");
      const row = await rowOf(requestId, `the ${model} row`);
      expect(row.get("error_message")).toBe(NO_DETAIL(status));
      await waitForLogLine(
        app!,
        (l) => l.includes(requestId) && l.includes(NO_DETAIL(status)),
        `the ${model} access-log line`,
      );
    },
  );

  test("a non-JSON body stays the detail, for the row and the caller", async () => {
    if (!etcdReachable) return;
    const page = "<html>sentinel fronting proxy page</html>";
    const { requestId, body } = await chat("row-html", "html-prompt");
    expect((JSON.parse(body) as { error: { message: string } }).error.message).toBe(page);
    const row = await rowOf(requestId, "the html row");
    expect(row.get("error_message")).toBe(`upstream returned HTTP 502: ${page}`);
    expect(row.get("response")).toBe(page);
  });

  const post = (path: string, body: unknown, extraHeaders: Record<string, string> = {}) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...headers, ...extraHeaders },
      body: JSON.stringify(body),
    });

  const clientMessage = (body: string) =>
    (JSON.parse(body) as { error: { message: string } }).error.message;

  const STATUS_FALLBACK = "upstream returned 502 Bad Gateway";
  const FRAGMENT = '{"error":{"message":"sentinel-cut';
  const cutBatch = ["/v1/batches?model=row-truncated", { input_file_id: "cut-prompt" }] as const;

  // The caller's half is what origin/main answers: these routes differ in
  // whether a fragment of a cut body reaches the caller.
  test.each([
    ["chat", "/v1/chat/completions", { model: "row-truncated", messages: [{ role: "user", content: "cut-prompt" }] }, FRAGMENT],
    ["completions", "/v1/completions", { model: "row-truncated", prompt: "cut-prompt" }, FRAGMENT],
    ["images", "/v1/images/generations", { model: "row-truncated", prompt: "cut-prompt" }, FRAGMENT],
    ["responses", "/v1/responses", { model: "row-truncated", input: "cut-prompt" }, STATUS_FALLBACK],
    ["rerank", "/v1/rerank", { model: "row-truncated-rerank", query: "cut-prompt", documents: ["a"] }, STATUS_FALLBACK],
    ["speech", "/v1/audio/speech", { model: "row-truncated", input: "cut-prompt", voice: "alloy" }, STATUS_FALLBACK],
    ["decisions", "/v1/decisions", { model: "row-truncated", input: "cut-prompt", questions: [{ type: "predicate", name: "q", instructions: "i" }] }, STATUS_FALLBACK],
  ] as const)(
    "%s: a body cut off leaves the caller answer as it was and the row naming the read failure",
    async (_what, path, payload, expected) => {
      if (!etcdReachable) return;
      const res = await post(path, payload);
      const body = await res.text();
      expect(res.status).toBe(502);
      expect(clientMessage(body)).toBe(expected);
      const requestId = res.headers.get("x-aisix-request-id")!;
      const row = await rowOf(requestId, "the truncated row");
      expect(row.get("error_message")).toMatch(READ_FAILURE(502));
      expect(requestText(row)).toContain("cut-prompt");
      // What did arrive is still reported.
      expect(row.get("response")).toContain("sentinel-cut");
      const warn = await waitForLogLine(
        app!,
        (l) =>
          l.includes("failed to read upstream error response body") && l.includes(requestId),
        "the read-failure warning",
      );
      expect(warn).toContain("502");
      await waitForLogLine(
        app!,
        (l) => l.includes(requestId) && l.includes("failed to read response body"),
        "the access-log line",
      );
    },
  );

  test("a body cut off before any byte leaves the caller with the status fallback", async () => {
    if (!etcdReachable) return;
    const { res, body, requestId } = await chat("row-truncated-empty", "cut-empty-prompt");
    expect(res.status).toBe(502);
    expect(clientMessage(body)).toBe(STATUS_FALLBACK);
    const row = await rowOf(requestId, "the empty-cut row");
    expect(row.get("error_message")).toMatch(READ_FAILURE(502));
  });

  test("jobs: a body cut off answers the caller as an unparseable body; the row and the log say why", async () => {
    if (!etcdReachable) return;
    const res = await post(cutBatch[0], cutBatch[1]);
    const body = await res.text();
    expect(res.status).toBe(502);
    expect(clientMessage(body)).toMatch(/^upstream returned an unparseable body: /);
    expect((JSON.parse(body) as { error: { type: string } }).error.type).toBe(
      "upstream_decode_error",
    );
    expect(body).not.toContain("failed to read response body");
    const requestId = res.headers.get("x-aisix-request-id")!;
    const row = await rowOf(requestId, "the jobs row");
    expect(row.get("error_message")).toMatch(READ_FAILURE(502));
    expect(requestText(row)).toContain("cut-prompt");
    expect(row.get("response")).toBe(FRAGMENT);
    await waitForLogLine(
      app!,
      (l) => l.includes(requestId) && l.includes("failed to read response body"),
      "the jobs access-log line",
    );
  });

  test.each([
    ["a non-JSON body", "row-video-text", 502, "sentinel video gateway page", "upstream returned HTTP 500: sentinel video gateway page"],
    ["an empty body", "row-video-empty", 502, "upstream returned 500 Internal Server Error", NO_DETAIL(500)],
    ["a vendor envelope", "row-video-json", 400, "sentinel-video", "upstream returned HTTP 400: sentinel-video"],
  ] as const)(
    "videos: %s reaches the caller as text; the row reports what the upstream said",
    async (_what, model, status, clientText, rowMessage) => {
      if (!etcdReachable) return;
      const res = await post("/v1/videos", { model, prompt: "video-fail-prompt" });
      const body = await res.text();
      expect(res.status).toBe(status);
      expect(clientMessage(body)).toBe(clientText);
      const row = await rowOf(res.headers.get("x-aisix-request-id")!, `the ${model} row`);
      expect(row.get("error_message")).toBe(rowMessage);
      expect(requestText(row)).toContain("video-fail-prompt");
    },
  );

  test("videos: a body cut off answers its status as the other routes do; the row names the read failure", async () => {
    if (!etcdReachable) return;
    const res = await post("/v1/videos", { model: "row-video-cut", prompt: "video-cut-prompt" });
    const body = await res.text();
    expect(res.status).toBe(502);
    expect(clientMessage(body)).toBe(STATUS_FALLBACK);
    expect((JSON.parse(body) as { error: { type: string } }).error.type).toBe("upstream_error");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the cut video row");
    expect(row.get("error_message")).toMatch(READ_FAILURE(502));
    expect(row.get("response")).toContain("sentinel-cut");
  });

  // Every route family: the request body, and the upstream's own answer as it
  // sent it, which the caller's error envelope would not match.
  const ANY_400 = '{"error":{"message":"sentinel-any-400","type":"upstream_type"}}';
  test.each([
    ["completions", "/v1/completions", { model: "row-any-400", prompt: "path-prompt" }, ANY_400],
    ["images", "/v1/images/generations", { model: "row-any-400", prompt: "path-prompt" }, ANY_400],
    ["rerank", "/v1/rerank", { model: "row-rerank-400", query: "path-prompt", documents: ["a"] }, ANY_400],
    ["decisions", "/v1/decisions", { model: "row-any-400", input: "path-prompt", questions: [{ type: "predicate", name: "q", instructions: "i" }] }, ANY_400],
    ["jobs", "/v1/batches?model=row-any-400", { input_file_id: "path-prompt" }, ANY_400],
    ["messages", "/v1/messages", { model: "row-any-400", max_tokens: 5, messages: [{ role: "user", content: "path-prompt" }] }, ANY_400],
    ["count_tokens", "/v1/messages/count_tokens", { model: "row-count-400", messages: [{ role: "user", content: "path-prompt" }] }, ANY_400],
    ["bedrock converse", "/v1/chat/completions", { model: "row-bedrock-400", messages: [{ role: "user", content: "path-prompt" }] }, '{"message":"sentinel-bedrock-answer"}'],
    ["a2a", "/a2a/row-agent", { jsonrpc: "2.0", id: "1", method: "message/send", params: { message: { role: "user", parts: [{ kind: "text", text: "path-prompt" }], messageId: "m-1" } } }, "sentinel-a2a-answer"],
  ] as const)("%s: an upstream failure carries both bodies", async (_what, path, payload, answer) => {
    if (!etcdReachable) return;
    const res = await post(path, payload, { "anthropic-version": "2023-06-01" });
    await res.text();
    expect(res.status).toBeGreaterThanOrEqual(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, `the ${path} row`);
    expect(Number(row.get("status_code"))).toBeGreaterThanOrEqual(400);
    expect(requestText(row)).toContain("path-prompt");
    expect(row.get("response")).toBe(answer);
  });

  test("image edits: an upstream failure carries the text fields, a checksum and the answer", async () => {
    if (!etcdReachable) return;
    const form = new FormData();
    form.set("model", "row-any-400");
    form.set("prompt", "edit-prompt");
    form.set("image", new Blob(["sentinel-image-bytes"]), "in.png");
    const res = await fetch(`${app!.proxyUrl}/v1/images/edits`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}` },
      body: form,
    });
    await res.text();
    expect(res.status).toBe(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the edits row");
    const prompt = JSON.parse(row.get("prompt")!) as Record<string, string>;
    expect(prompt.prompt).toBe("edit-prompt");
    expect(prompt.image_sha256).toBe(hash("sentinel-image-bytes"));
    expect(row.get("response")).toContain("sentinel-any-400");
  });

  const mcpCall = (key: string, text: string) =>
    fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${key}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 3,
        method: "tools/call",
        params: { name: "rowmcp__echo", arguments: { text } },
      }),
    });

  test("mcp: a blocked tool call carries its arguments and the JSON-RPC refusal", async () => {
    if (!etcdReachable) return;
    const res = await mcpCall(CALLER, `say ${MCP_BLOCKED_WORD}`);
    const body = await res.text();
    expect(res.status).toBe(200);
    expect(body).toContain("content policy");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the blocked tool row");
    expect(row.get("guardrail_blocked")).toBe("true");
    expect(requestText(row)).toContain(MCP_BLOCKED_WORD);
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  test("mcp: a rate-limited tool call carries its arguments and the error the caller got", async () => {
    if (!etcdReachable) return;
    let refused: Response | undefined;
    for (let i = 0; i < 5 && !refused; i++) {
      const res = await mcpCall(MCP_LIMITED_CALLER, "limited-tool-prompt");
      if (res.status === 429) refused = res;
      else await res.text();
    }
    expect(refused, "the key's rpm refused a tool call").toBeDefined();
    const body = await refused!.text();
    const row = await rowOf(refused!.headers.get("x-aisix-request-id")!, "the limited tool row");
    expect(row.get("status_code")).toBe("429");
    expect(requestText(row)).toContain("limited-tool-prompt");
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  const openRealtime = (model: string) =>
    new Promise<{ status: number; requestId: string; body: string }>((resolve, reject) => {
      const ws = new WsClient(
        `${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`,
        { headers: { authorization: `Bearer ${CALLER}` } },
      );
      ws.on("unexpected-response", (_req, res) => {
        let body = "";
        res.on("data", (c) => (body += String(c)));
        res.on("end", () =>
          resolve({
            status: res.statusCode ?? 0,
            requestId: String(res.headers["x-aisix-request-id"] ?? ""),
            body,
          }),
        );
      });
      ws.on("open", () => {
        ws.close();
        reject(new Error(`realtime ${model} upgraded`));
      });
      ws.on("error", reject);
    });

  // A WebSocket upgrade has no request body, so the row carries only the answer.
  test("realtime: a refused upgrade reports the error the caller got", async () => {
    if (!etcdReachable) return;
    const { status, requestId, body } = await openRealtime("row-unlisted");
    expect(status).toBe(403);
    const row = await rowOf(requestId, "the refused realtime row");
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  const realtimeFailure = (model: string) =>
    new Promise<{ requestId: string; body: string }>((resolve, reject) => {
      const ws = new WsClient(
        `${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`,
        { headers: { authorization: `Bearer ${CALLER}` } },
      );
      let id = "";
      let body = "";
      ws.on("upgrade", (res) => {
        id = String(res.headers["x-aisix-request-id"] ?? "");
      });
      ws.on("message", (frame) => { body = frame.toString(); });
      ws.on("close", () => resolve({ requestId: id, body }));
      ws.on("error", reject);
    });

  test("realtime: an upstream that refuses the socket has a row naming its status", async () => {
    if (!etcdReachable) return;
    const { requestId } = await realtimeFailure("row-realtime-400");
    const row = await rowOf(requestId, "the failed realtime row");
    expect(row.get("status_code")).toBe("400");
    expect(row.get("error_class")).toBe("upstream_status");
    expect(row.get("response")).toBe(ANY_400);
  });

  test.each([
    ["empty upstream handshake reply", "row-video-empty", "502"],
    ["output guardrail block", "row-realtime-blocked", "400"],
  ] as const)("realtime: %s records the actual gateway response", async (_what, model, status) => {
    if (!etcdReachable) return;
    const { requestId, body } = await realtimeFailure(model);
    expect(JSON.parse(body).type).toBe("error");
    const row = await rowOf(requestId, "the realtime failure row");
    expect(row.get("status_code")).toBe(status);
    expect(row.get("response")).toBe(body);
  });

  test.each([
    ["chat", "/v1/chat/completions", { model: "row-slow", messages: [{ role: "user", content: "cancel-chat-prompt" }] }, "cancel-chat-prompt"],
    ["embeddings", "/v1/embeddings", { model: "row-slow", input: "cancel-embed-prompt" }, "cancel-embed-prompt"],
    ["images", "/v1/images/generations", { model: "row-slow", prompt: "cancel-image-prompt" }, "cancel-image-prompt"],
  ] as const)(
    "%s: a caller who left has a 499 with the request and no invented response",
    async (_what, path, payload, prompt) => {
      if (!etcdReachable) return;
      const abort = new AbortController();
      const sent = fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: abort.signal,
      });
      await new Promise((r) => setTimeout(r, 500));
      abort.abort();
      await sent.catch(() => undefined);
      const row = await waitForSlsLog(
        sls!,
        FULL,
        (l) => l.get("status_code") === "499" && requestText(l).includes(prompt),
        `the ${path} 499 row`,
      );
      expect(row.get("error_class")).toBe("client_disconnected");
      expect(row.get("response") ?? "").toBe("");
    },
  );

  test("a caller who left mid-stream has a 499 with what had been delivered", async () => {
    if (!etcdReachable) return;
    const abort = new AbortController();
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "row-stream-slow",
        stream: true,
        messages: [{ role: "user", content: "cancel-stream-prompt" }],
      }),
      signal: abort.signal,
    });
    const requestId = res.headers.get("x-aisix-request-id")!;
    const reader = res.body!.getReader();
    await reader.read();
    await new Promise((r) => setTimeout(r, 200));
    abort.abort();
    await reader.cancel().catch(() => undefined);
    const row = await rowOf(requestId, "the mid-stream 499 row");
    expect(row.get("status_code")).toBe("499");
    expect(requestText(row)).toContain("cancel-stream-prompt");
    expect(row.get("output_messages") ?? row.get("response")).toContain("cancel-partial");
    expect(row.get("response") ?? "").not.toContain("client_disconnected");
  });

  test("a stream cut off mid-way keeps the request and what had arrived", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "row-stream-cut",
        stream: true,
        messages: [{ role: "user", content: "stream-cut-prompt" }],
      }),
    });
    await res.text().catch(() => "");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the cut stream row");
    expect(Number(row.get("status_code"))).toBeGreaterThanOrEqual(400);
    expect(requestText(row)).toContain("stream-cut-prompt");
    expect(row.get("output_messages") ?? row.get("response")).toContain("partial-answer");
  });

  test("an in-band stream error before any content reports the request and the error", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "row-in-band",
        stream: true,
        messages: [{ role: "user", content: "in-band-prompt" }],
      }),
    });
    await res.text().catch(() => "");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the in-band row");
    expect(Number(row.get("status_code"))).toBeGreaterThanOrEqual(400);
    expect(requestText(row)).toContain("in-band-prompt");
    expect(row.get("response")).toContain("sentinel-in-band");
  });

  test("native Responses exports the SSE from its first delta through the late failure", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await post("/v1/responses", { model: "row-responses-late", input: "test", stream: true });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain(RESPONSES_DELTA);
    expect(text).toContain("event: response.failed");
    expect(text).toContain('"error":null');
    expect(text).toContain('"trace":"provider-raw"');
    const requestId = res.headers.get("x-aisix-request-id")!;
    const row = await rowOf(requestId, "the native Responses failure");
    expect(row.get("status_code")).toBe("502");
    expect(row.get("error_class")).toBe("upstream_in_band");
    expect(row.get("response")).toBe(RESPONSES_DELTA + RESPONSES_FAILED);
    const metadata = await waitForSlsLog(sls!, META, (l) => l.get("request_id") === requestId, "the native Responses metadata");
    expect(metadata.get("response")).toBeUndefined();
  });

  test("translated Responses exports the upstream Chat SSE error before translation", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await post("/v1/responses", { model: "row-in-band", input: "test", stream: true });
    const text = await res.text();
    expect(text).toContain("sentinel-in-band");
    expect(text).not.toContain("provider_request");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the translated Responses failure");
    expect(row.get("error_class")).toBe("upstream_in_band");
    expect(row.get("response")).toBe(CHAT_STREAM_ERROR);
  });

  test.each([
    ["/v1/chat/completions", "row-bedrock-converse-stream", [BEDROCK_EXCEPTION]],
    ["/v1/messages", "row-bedrock-native-stream", [ANTHROPIC_START, ANTHROPIC_DELTA, BEDROCK_EXCEPTION]],
  ] as const)("%s exports the Bedrock SDK's available JSON payloads", async (path, model, expected) => {
    if (!etcdReachable) return;
    const res = await post(path, {
      model, stream: true, max_tokens: 32, messages: [{ role: "user", content: "test" }],
    }, { "anthropic-version": "2023-06-01" });
    expect(res.status).toBe(200);
    expect(await res.text()).toContain(BEDROCK_EXCEPTION.message);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the Bedrock stream exception");
    expect(row.get("status_code")).toBe("429");
    expect(row.get("error_class")).toBe("upstream_in_band");
    expect(row.get("response")!.split("\n").map((line) => JSON.parse(line))).toEqual(expected);
  });

  test("a non-2xx read deadline keeps the consumed prefix", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await post("/v1/responses", { model: "row-timeout-after-error-head", input: "test", stream: true });
    expect(res.status).toBe(502);
    expect(clientMessage(await res.text())).toBe("upstream returned 503 Service Unavailable");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the response body deadline");
    expect(row.get("error_message")).toMatch(READ_FAILURE(503));
    expect(row.get("response")).toBe(TIMEOUT_PREFIX);
  });

  test("a failed SSE export omits image payloads and retains later events", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await post("/v1/responses", { model: "row-responses-media", input: "test", stream: true });
    expect(res.status).toBe(200);
    expect(await res.text()).toContain(RESPONSE_MEDIA);
    const requestId = res.headers.get("x-aisix-request-id")!;
    const row = await rowOf(requestId, "the SSE media failure");
    expect(row.get("response")).toMatch(/^event: response\.image_generation_call\.partial_image\ndata\ndata: \{/);
    expect(row.get("response")).toContain('"partial_image_b64":"[aisix: base64 data uri omitted,');
    expect(row.get("response")).not.toContain(RESPONSE_MEDIA);
    expect(row.get("response")).toContain(LONG_DELTA + RESPONSES_FAILED);
    const capped = await waitForSlsLog(sls!, CAPPED, (l) => l.get("request_id") === requestId, "the capped SSE response");
    expect(capped.get("content_truncated")).toBe("true");
    const cappedResponse = capped.get("response")!;
    expect(Buffer.byteLength(cappedResponse)).toBeLessThanOrEqual(CAPPED_MAX_BYTES);
    const fullResponse = row.get("response")!;
    const mediaFrame = fullResponse.slice(0, fullResponse.indexOf(LONG_DELTA));
    expect(cappedResponse).toBe(mediaFrame + RESPONSES_FAILED);
  });

  test.each([
    ["row-responses-large-media", "text/event-stream", true],
    ["row-responses-octet-sse", "application/octet-stream", false],
    ["row-responses-rollover", "text/event-stream", true],
  ] as const)("failed SSE uses complete frames from %s", async (model, mime, truncated) => {
    if (!etcdReachable) return;
    const res = await post("/v1/responses", { model, input: "test", stream: true });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain(mime);
    const wire = await res.text();
    expect(wire).toContain("event: response.failed");
    expect(wire).toContain('"trace":"provider-raw"');
    if (model === "row-responses-large-media") expect(wire).toContain(MEDIA_PREFIX);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the model's failed SSE");
    expect(row.get("error_class")).toBe("upstream_in_band");
    const response = row.get("response")!;
    expect(response.includes(MEDIA_PREFIX)).toBe(false);
    expect(response.startsWith(RESPONSES_DELTA)).toBe(true);
    expect(response.endsWith(RESPONSES_FAILED)).toBe(true);
    expect(row.get("content_truncated")).toBe(truncated ? "true" : undefined);
    if (model !== "row-responses-rollover") {
      expect(response).toBe(RESPONSES_DELTA + RESPONSES_FAILED);
    } else {
      const capped = await waitForSlsLog(sls!, CAPPED, (l) => l.get("request_id") === row.get("request_id"), "the capped SSE rollover");
      const deltaCount = Math.floor((CAPPED_MAX_BYTES - Buffer.byteLength(RESPONSES_FAILED)) / Buffer.byteLength(RESPONSES_DELTA));
      expect(capped.get("response")).toBe(RESPONSES_DELTA.repeat(deltaCount) + RESPONSES_FAILED);
      expect(capped.get("content_truncated")).toBe("true");
    }
  });

  test("a failed masked stream exports the masked output", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await post("/v1/responses", { model: "row-responses-masked", input: "test", stream: true });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("[EMAIL_REDACTED]");
    expect(text).not.toContain(OUTPUT_EMAIL);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the masked SSE failure");
    expect(row.get("error_class")).toBe("upstream_in_band");
    expect(row.get("response")).toContain("[EMAIL_REDACTED]");
    expect(row.get("response")).not.toContain(OUTPUT_EMAIL);
  });

  test("a malformed upstream JSON response keeps diagnostics before escaped audio", async () => {
    if (!etcdReachable) return;
    const { res, requestId } = await chat("row-invalid-escaped-media", "test");
    expect(res.status).toBe(502);
    const row = await rowOf(requestId, "the malformed JSON failure");
    expect(row.get("response")).toContain("readable before audio");
    expect(row.get("response")).not.toContain(RESPONSE_MEDIA);
  });

  test("a bridged SSE output block records the refusal frames actually sent", async () => {
    if (!etcdReachable) return;
    const res = await post("/v1/responses", { model: "row-responses-bridge-blocked", input: "test", stream: true });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("event: response.failed");
    expect(body).not.toContain("partial Responses answer");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the bridged guardrail failure");
    expect(row.get("guardrail_blocked")).toBe("true");
    expect(row.get("response")).toBe(body);
  });

  test("a blocked stream exports the gateway's actual rejection without the upstream body", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await post("/v1/responses", { model: "row-responses-blocked", input: "test", stream: true });
    expect(res.status).toBe(422);
    const body = await res.text();
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the blocked SSE response");
    expect(row.get("guardrail_blocked")).toBe("true");
    expect(row.get("response")).toBe(body);
    expect(row.get("response")).not.toContain("partial Responses answer");
  });

  test.each([
    ["row-json-error", "upstream_in_band", JSON.stringify(IN_BAND_JSON)],
    ["row-invalid-json", "upstream_decode", INVALID_JSON],
  ])("%s exports the body of the failed 2xx answer", async (model, errorClass, response) => {
    if (!etcdReachable) return;
    const { res, requestId } = await chat(model, "test");
    expect(res.status).toBe(502);
    const row = await rowOf(requestId, `the ${model} failure`);
    expect(row.get("error_class")).toBe(errorClass);
    expect(row.get("response")).toBe(response);
  });

  test("a metadata_only exporter receives none of it", async () => {
    if (!etcdReachable) return;
    const own = await post("/v1/chat/completions", {
      model: "row-400",
      messages: [{ role: "user", content: "metadata-check-prompt" }],
    });
    await own.text();
    const requestId = own.headers.get("x-aisix-request-id");
    await waitForSlsLog(sls!, META, (l) => l.get("request_id") === requestId, "the metadata_only row");
    for (const row of slsLogsFor(sls!, META)) {
      for (const field of ["prompt", "response", "input_messages", "output_messages"]) {
        expect(row.get(field), field).toBeUndefined();
      }
      for (const value of row.values()) expect(value).not.toContain("-prompt");
    }
  });
});
