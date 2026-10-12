import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  startMcpUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type A2aUpstream,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a request the gateway refuses itself files a terminal usage event
// whose `error_message` names the failure with the access log's text — on
// every endpoint, through one shared helper — for the failures whose text
// is entirely the gateway's own. A failure that can quote what the caller
// sent (`invalid_request_error`) keeps the message empty.
//
// Before, those events carried `error_class` alone, so the control plane's
// usage log and the OTLP span showed a semantic router's failed decision,
// an unknown model or a guardrail block with no reason at all.

const CALLER = "sk-gateway-error-message-caller";
const LIMITED = "sk-gateway-error-message-limited";
const LIMITED_A2A = "sk-gateway-error-message-limited-a2a";
const LIMITED_MCP = "sk-gateway-error-message-limited-mcp";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const SLS_PROJECT = "gem-project";
const LOGSTORE = "gem-logstore";
const CREDENTIAL_REF = "gem";
const FORBIDDEN = "gemforbiddenword";

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-gem-stream",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
// A streamed answer the output guardrail refuses once it has read it all.
const LEAKY_STREAM = [
  chatChunk({ role: "assistant" }),
  chatChunk({ content: "here is " }),
  chatChunk({ content: FORBIDDEN }),
  chatChunk({}, "stop"),
  "[DONE]",
];
const STREAM_REFUSAL = "response blocked by content policy (guardrail 'gem-output-block')";

describe("gateway-side failures name themselves on the usage event", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let leaky: OpenAiUpstream | undefined;
  let leakyResponses: OpenAiUpstream | undefined;
  let leakyStream: OpenAiUpstream | undefined;
  let agent: A2aUpstream | undefined;
  let mcp: McpUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream();
    // Answers every route with text the output guardrail refuses.
    leaky = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-leaky",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `here is ${FORBIDDEN}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 3, total_tokens: 6 },
      },
    });

    leakyResponses = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_leaky",
        object: "response",
        status: "completed",
        model: "gpt-4o-mini",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: `here is ${FORBIDDEN}` }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 3, total_tokens: 6 },
      },
    });

    leakyStream = await startOpenAiUpstream({ streamEvents: LEAKY_STREAM });
    agent = await startA2aUpstream({ cardMount: "origin" });
    mcp = await startMcpUpstream("gemmcp");

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "gem-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    const pk = await seed.createProviderKey({
      display_name: "gem-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gem-chat",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const leakyPk = await seed.createProviderKey({
      display_name: "gem-leaky-pk",
      secret: "sk-mock",
      api_base: `${leaky.baseUrl}/v1`,
    });
    const leakyModel = await seed.createModel({
      display_name: "gem-leaky",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: leakyPk.id,
    });

    // A classifier the gateway cannot reach, under a `fail` policy.
    const typesafe = await seed.createProviderKey({
      display_name: "gem-typesafe-pk",
      secret: "ts-mock",
      provider: "typesafe",
      adapter: undefined,
      api_base: "http://127.0.0.1:9",
    });
    await seed.createModel({
      display_name: "gem-semantic",
      semantic: {
        classifier: {
          type: "jev",
          provider_key_id: typesafe.id,
          timeout_ms: 500,
          on_failure: "fail",
        },
        routes: [{ name: "code", target: "gem-chat", description: "programming questions" }],
        default: "gem-chat",
      },
    });

    await seed.createGuardrail({
      name: "gem-input-block",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN }],
    });
    const outputGuard = await seed.createGuardrail(
      {
        name: "gem-output-block",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: FORBIDDEN }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(outputGuard.id, leakyModel.id);
    const leakyResponsesPk = await seed.createProviderKey({
      display_name: "gem-leaky-responses-pk",
      secret: "sk-mock",
      api_base: `${leakyResponses.baseUrl}/v1`,
    });
    const leakyResponsesModel = await seed.createModel({
      display_name: "gem-leaky-responses",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: leakyResponsesPk.id,
    });
    await seed.attachGuardrailToModel(outputGuard.id, leakyResponsesModel.id);

    // Streamed answers, refused after the `200` head: through the chat
    // stream, the /v1/messages translation, the /v1/responses bridge (a
    // provider with no native Responses API) and a passthrough route.
    const leakyStreamPk = await seed.createProviderKey({
      display_name: "gem-leaky-stream-pk",
      secret: "sk-mock",
      api_base: `${leakyStream.baseUrl}/v1`,
    });
    const leakyStreamModel = await seed.createModel({
      display_name: "gem-leaky-stream",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: leakyStreamPk.id,
    });
    await seed.attachGuardrailToModel(outputGuard.id, leakyStreamModel.id);
    const bridgedPk = await seed.createProviderKey({
      display_name: "gem-leaky-bridged-pk",
      secret: "sk-mock",
      api_base: `${leakyStream.baseUrl}/v1`,
      provider: "deepseek",
      adapter: "openai",
    });
    const bridgedModel = await seed.createModel({
      display_name: "gem-leaky-bridged",
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: bridgedPk.id,
    });
    await seed.attachGuardrailToModel(outputGuard.id, bridgedModel.id);
    const streamRoute = await seed.createPassthroughRoute({
      name: "gem-stream-route",
      path_prefix: "/gem-stream-route",
      target_url: `${leakyStream.baseUrl}/v1`,
      provider_key_id: leakyStreamPk.id,
    });
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: outputGuard.id,
      scope_type: "passthrough_route",
      scope_id: streamRoute.id,
      priority: 100,
    });

    await seed.update("a2a_agents", randomUUID(), {
      name: "gem-agent",
      url: agent.url,
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "gemmcp",
      url: mcp.url,
      enabled: true,
    });
    for (const key of [LIMITED_A2A, LIMITED_MCP]) {
      await seed.createApiKey({
        key_hash: hash(key),
        allowed_models: [],
        allowed_agents: ["*"],
        mcp_access: { allow: ["*"] },
        rate_limit: { rpm: 1 },
      });
    }

    await seed.createPassthroughRoute({
      name: "gem-route",
      path_prefix: "/gem-route",
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
    });

    await seed.createApiKey({
      key_hash: hash(LIMITED),
      allowed_models: ["*"],
      rate_limit: { rpm: 1 },
    });
    // Written last: its authenticating implies every row above has landed.
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["*"],
      allowed_routes: ["gem-stream-route"],
      allowed_agents: ["*"],
      mcp_access: { allow: ["*"] },
    });
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
    await leaky?.close();
    await leakyResponses?.close();
    await leakyStream?.close();
    await agent?.close();
    await mcp?.close();
    await sls?.close();
  });

  const ready = () => etcdReachable && !!app && !!sls;

  async function send(
    path: string,
    init: { body?: unknown; form?: FormData; method?: string; key?: string } = {},
  ): Promise<{ status: number; requestId: string }> {
    const headers: Record<string, string> = {
      authorization: `Bearer ${init.key ?? CALLER}`,
    };
    if (path === "/mcp") headers.accept = "application/json, text/event-stream";
    let body: BodyInit | undefined;
    if (init.form) {
      body = init.form;
    } else if (init.body !== undefined) {
      headers["content-type"] = "application/json";
      body = JSON.stringify(init.body);
    }
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: init.method ?? "POST",
      headers,
      body,
    });
    await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  }

  /** The terminal usage event the request filed. */
  async function eventFor(requestId: string, what: string): Promise<Map<string, string>> {
    expect(requestId, `${what}: no x-aisix-request-id`).not.toBe("");
    return waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, what);
  }

  async function expectNamed(
    r: { status: number; requestId: string },
    what: string,
    want: { status: number; errorClass: string; message: string },
  ) {
    expect(r.status, what).toBe(want.status);
    const event = await eventFor(r.requestId, what);
    expect(event.get("error_class"), what).toBe(want.errorClass);
    expect(event.get("error_message"), what).toBe(want.message);
    return event;
  }

  const notFound = (model: string) => ({
    status: 404,
    errorClass: "model_not_found",
    message: `model "${model}" not found`,
  });

  test("chat: semantic decision failure, unknown model, rate limit, input and output guardrail blocks", async (ctx) => {
    if (!ready()) return ctx.skip();
    const chat = (model: string, content = "hello", key?: string) =>
      send("/v1/chat/completions", {
        body: { model, messages: [{ role: "user", content }] },
        key,
      });

    await expectNamed(await chat("gem-semantic", "fix my python"), "semantic", {
      status: 503,
      errorClass: "semantic_decision_failed",
      message: "semantic router 'gem-semantic' could not pick a route (classifier_upstream)",
    });

    await expectNamed(await chat("gem-ghost-chat"), "unknown model", notFound("gem-ghost-chat"));

    expect((await chat("gem-chat", "hello", LIMITED)).status).toBe(200);
    const limited = await chat("gem-chat", "hello", LIMITED);
    expect(limited.status).toBe(429);
    const limitedEvent = await eventFor(limited.requestId, "rate limit");
    expect(limitedEvent.get("error_class")).toBe("rate_limit_exceeded");
    expect(limitedEvent.get("error_message")).toMatch(/limit exceeded/);

    const inputBlock = await chat("gem-chat", `say ${FORBIDDEN}`);
    expect(inputBlock.status).toBe(422);
    const inputEvent = await eventFor(inputBlock.requestId, "input block");
    expect(inputEvent.get("error_class")).toBe("content_filter");
    expect(inputEvent.get("error_message")).toContain("gem-input-block");
    expect(inputEvent.get("error_message")).not.toContain(FORBIDDEN);

    const outputBlock = await chat("gem-leaky");
    expect(outputBlock.status).toBe(422);
    const outputEvent = await eventFor(outputBlock.requestId, "output block");
    expect(outputEvent.get("error_class")).toBe("content_filter");
    expect(outputEvent.get("error_message")).toContain("gem-output-block");
    expect(outputEvent.get("error_message")).not.toContain(FORBIDDEN);
  });

  test("messages and responses: unknown model and an output guardrail block", async (ctx) => {
    if (!ready()) return ctx.skip();
    const messages = (model: string) =>
      send("/v1/messages", {
        body: { model, max_tokens: 64, messages: [{ role: "user", content: "hello" }] },
      });
    await expectNamed(await messages("gem-ghost-messages"), "messages", notFound("gem-ghost-messages"));
    const messagesBlock = await messages("gem-leaky");
    expect(messagesBlock.status).toBe(422);
    const messagesEvent = await eventFor(messagesBlock.requestId, "messages output block");
    expect(messagesEvent.get("error_class")).toBe("content_filter");
    expect(messagesEvent.get("error_message")).toContain("gem-output-block");

    const responses = (model: string) =>
      send("/v1/responses", { body: { model, input: "hello" } });
    await expectNamed(
      await responses("gem-ghost-responses"),
      "responses",
      notFound("gem-ghost-responses"),
    );
    const responsesBlock = await responses("gem-leaky-responses");
    expect(responsesBlock.status).toBe(422);
    const responsesEvent = await eventFor(responsesBlock.requestId, "responses output block");
    expect(responsesEvent.get("error_class")).toBe("content_filter");
    expect(responsesEvent.get("error_message")).toContain("gem-output-block");
  });

  test("every other endpoint names its refusal", async (ctx) => {
    if (!ready()) return ctx.skip();
    const json: Array<[string, string, Record<string, unknown>]> = [
      ["/v1/messages/count_tokens", "gem-ghost-count", { messages: [{ role: "user", content: "hi" }] }],
      ["/v1/embeddings", "gem-ghost-embeddings", { input: "hi" }],
      ["/v1/completions", "gem-ghost-completions", { prompt: "hi" }],
      ["/v1/images/generations", "gem-ghost-images", { prompt: "a cat" }],
      ["/v1/rerank", "gem-ghost-rerank", { query: "q", documents: ["a"] }],
      ["/v1/audio/speech", "gem-ghost-speech", { input: "hi", voice: "alloy" }],
      ["/v1/videos", "gem-ghost-videos", { prompt: "a cat" }],
    ];
    for (const [path, model, body] of json) {
      await expectNamed(await send(path, { body: { model, ...body } }), path, notFound(model));
    }

    const edit = new FormData();
    edit.set("model", "gem-ghost-edits");
    edit.set("prompt", "a hat");
    edit.set("image", new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }), "a.png");
    await expectNamed(await send("/v1/images/edits", { form: edit }), "images edits", notFound("gem-ghost-edits"));

    const audio = new FormData();
    audio.set("model", "gem-ghost-transcriptions");
    audio.set("file", new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" }), "a.wav");
    // Read only inside dispatch, so the event records the model the
    // message names.
    const transcription = await expectNamed(
      await send("/v1/audio/transcriptions", { form: audio }),
      "transcriptions",
      notFound("gem-ghost-transcriptions"),
    );
    expect(transcription.get("requested_model")).toBe("gem-ghost-transcriptions");

    const file = new FormData();
    file.set("model", "gem-ghost-files");
    file.set("purpose", "batch");
    file.set("file", new Blob(["{}\n"], { type: "application/jsonl" }), "b.jsonl");
    const files = await expectNamed(
      await send("/v1/files", { form: file }),
      "files",
      notFound("gem-ghost-files"),
    );
    expect(files.get("requested_model")).toBe("gem-ghost-files");

    // The caller's key grants no passthrough route.
    await expectNamed(await send("/gem-route/chat/completions", { body: {} }), "passthrough", {
      status: 403,
      errorClass: "permission_denied",
      message: 'API key is not allowed to use passthrough route "gem-route"',
    });

    // A plain GET is refused before the model is looked up.
    const realtime = await send("/v1/realtime?model=gem-chat", { method: "GET" });
    expect(realtime.status).toBe(400);
    const realtimeEvent = await eventFor(realtime.requestId, "realtime");
    expect(realtimeEvent.get("error_class")).toBe("websocket_upgrade_required");
    expect(realtimeEvent.get("error_message")).toMatch(/^this endpoint requires a WebSocket upgrade: /);
  });

  test("a failure that can quote the request keeps the message empty", async (ctx) => {
    if (!ready()) return ctx.skip();
    // A semantic router cannot serve embeddings: the refusal is an
    // `invalid_request_error`, a class whose text can carry what the
    // caller sent, so it never reaches the usage event.
    const r = await send("/v1/embeddings", { body: { model: "gem-semantic", input: "hi" } });
    expect(r.status).toBe(400);
    const event = await eventFor(r.requestId, "invalid request");
    expect(event.get("error_class")).toBe("invalid_request_error");
    expect(event.get("error_message")).toBeUndefined();
  });
  const streamed: Array<[string, string, Record<string, unknown>]> = [
    [
      "/v1/chat/completions",
      "chat stream",
      { model: "gem-leaky-stream", stream: true, messages: [{ role: "user", content: "hi" }] },
    ],
    [
      "/v1/messages",
      "messages stream",
      {
        model: "gem-leaky-stream",
        stream: true,
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      },
    ],
    [
      "/v1/responses",
      "bridged responses stream",
      { model: "gem-leaky-bridged", stream: true, input: "hi" },
    ],
    [
      "/gem-stream-route/chat/completions",
      "passthrough stream",
      { model: "gpt-4o-mini", stream: true, messages: [{ role: "user", content: "hi" }] },
    ],
  ];
  for (const [path, what, body] of streamed) {
    test(`a guardrail refusal after the 200 head names itself: ${what}`, async (ctx) => {
      if (!ready()) return ctx.skip();
      const r = await send(path, { body });
      const event = await eventFor(r.requestId, what);
      expect(event.get("guardrail_blocked")).toBe("true");
      expect(event.get("error_class")).toBe("content_filter");
      expect(event.get("error_message")).toBe(STREAM_REFUSAL);
    });
  }

  test("/a2a names a guardrail block and a rate-limit rejection", async (ctx) => {
    if (!ready()) return ctx.skip();
    const a2a = (text: string, key?: string) =>
      send("/a2a/gem-agent", {
        key,
        body: {
          jsonrpc: "2.0",
          id: "1",
          method: "message/send",
          params: {
            message: { role: "user", parts: [{ kind: "text", text }], messageId: "m-1" },
          },
        },
      });
    const a2aBlock = await a2a(`say ${FORBIDDEN}`);
    expect(a2aBlock.status).toBe(422);
    const a2aBlockEvent = await eventFor(a2aBlock.requestId, "a2a block");
    expect(a2aBlockEvent.get("error_class")).toBe("content_filter");
    expect(a2aBlockEvent.get("error_message")).toBe(
      "request blocked by content policy (guardrail 'gem-input-block')",
    );
    await a2a("hello", LIMITED_A2A);
    const a2aLimited = await a2a("hello", LIMITED_A2A);
    expect(a2aLimited.status).toBe(429);
    const a2aLimitedEvent = await eventFor(a2aLimited.requestId, "a2a rate limit");
    expect(a2aLimitedEvent.get("error_class")).toBe("rate_limit_exceeded");
    expect(a2aLimitedEvent.get("error_message")).toMatch(/limit exceeded/);
  });

  test("/mcp names a guardrail block and a rate-limit rejection", async (ctx) => {
    if (!ready()) return ctx.skip();
    const tool = (text: string, key?: string) =>
      send("/mcp", {
        key,
        body: {
          jsonrpc: "2.0",
          id: 3,
          method: "tools/call",
          params: { name: "gemmcp__echo", arguments: { text } },
        },
      });
    const mcpBlock = await tool(`say ${FORBIDDEN}`);
    const mcpBlockEvent = await eventFor(mcpBlock.requestId, "mcp block");
    expect(mcpBlockEvent.get("error_class")).toBe("content_filter");
    expect(mcpBlockEvent.get("error_message")).toBe(
      "tool call blocked by content policy (guardrail 'gem-input-block')",
    );
    await tool("hello", LIMITED_MCP);
    const mcpLimited = await tool("hello", LIMITED_MCP);
    expect(mcpLimited.status).toBe(429);
    const mcpLimitedEvent = await eventFor(mcpLimited.requestId, "mcp rate limit");
    expect(mcpLimitedEvent.get("error_class")).toBe("rate_limit_exceeded");
    expect(mcpLimitedEvent.get("error_message")).toMatch(/limit exceeded/);
  });
});
