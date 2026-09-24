import { createHash } from "node:crypto";
import { WebSocket as WsClient, WebSocketServer, type WebSocket as WsSocket } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `UsageEvent.applied_guardrails` names the guardrails that governed a
// request on every surface that resolves a guardrail chain — refused or not
// (api7/aisix#1030, #543).
//
// The field is how a Logs reader sees WHICH policy governed a row. It was
// written by chat, messages, /mcp and /a2a on every path, but the
// single-attempt family built its failure row without it — so the one row an
// operator is certain to open, a guardrail refusal, said `guardrail_blocked:
// true` while claiming that no guardrail governed the request. `/v1/responses`,
// the passthrough routes, `/v1/realtime` and the jobs surface dropped it on
// the success path as well.
//
// One env-scoped keyword row governs everything here. Each surface addresses
// its own model, so its row is found by `requested_model` — independent of
// the field under test — and every refusal is checked for the block flag as
// well, so a row that proves the field also proves it came from the refusal.

const KEY = "sk-guardrail-applied-set-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "guardrail-applied-set";

const FORBIDDEN = "appliedsetsentinel";
const ROUTE = "gas-tunnel";

/** What the one attached row contributes to every governed request. */
const APPLIED = [{ hook: "input", kind: "keyword" }];

const responsesBody = {
  id: "resp_gas",
  object: "response",
  status: "completed",
  model: "mock-model",
  output: [
    {
      type: "message",
      id: "msg_gas",
      role: "assistant",
      content: [{ type: "output_text", text: "fine" }],
    },
  ],
  usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
};

interface RealtimeUpstream {
  port: number;
  close(): Promise<void>;
}

/** Answers every client frame with a usage-bearing `response.done`. */
async function startRealtimeUpstream(): Promise<RealtimeUpstream> {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.on("message", () => {
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: { usage: { input_tokens: 2, output_tokens: 1 } },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    close: () =>
      new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("applied_guardrails on every guardrail-governed surface", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let realtime: RealtimeUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({ nonStreamBody: responsesBody });
    realtime = await startRealtimeUpstream();

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "gas-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const openaiPk = await seed.createProviderKey({
      display_name: "gas-openai-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const anthropicPk = await seed.createProviderKey({
      display_name: "gas-anthropic-pk",
      secret: "sk-mock",
      api_base: upstream.baseUrl,
      provider: "anthropic",
      adapter: "anthropic",
    });
    const alibabaPk = await seed.createProviderKey({
      display_name: "gas-alibaba-pk",
      secret: "sk-mock",
      api_base: upstream.baseUrl,
      provider: "alibaba",
    });
    const realtimePk = await seed.createProviderKey({
      display_name: "gas-realtime-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${realtime.port}/v1`,
    });
    const model = (
      display_name: string,
      provider_key_id: string,
      provider = "openai",
      model_name = "gpt-4o",
      extra: Record<string, unknown> = {},
    ) =>
      seed.createModel({ display_name, provider, model_name, provider_key_id, ...extra });

    await model("gas-completions", openaiPk.id);
    await model("gas-embeddings", openaiPk.id, "openai", "text-embedding-3-small", {
      kind: "embedding",
    });
    await model("gas-rerank", openaiPk.id);
    await model("gas-image-gen", openaiPk.id, "openai", "dall-e-3");
    await model("gas-image-edit", openaiPk.id, "openai", "dall-e-2");
    await model("gas-speech", openaiPk.id, "openai", "tts-1");
    await model("gas-transcription", openaiPk.id, "openai", "whisper-1");
    await model("gas-video", alibabaPk.id, "alibaba", "wan-mock");
    await model("gas-count-tokens", anthropicPk.id, "anthropic", "claude-3-5-haiku-20241022");
    await model("gas-responses-block", openaiPk.id);
    await model("gas-responses-ok", openaiPk.id);
    await model("gas-batches", openaiPk.id);
    await model("gas-realtime", realtimePk.id, "openai", "gpt-realtime-mock");
    await seed.createPassthroughRoute({
      name: ROUTE,
      path_prefix: "/passthrough/gas",
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: openaiPk.id,
    });
    await seed.createGuardrail({
      name: "gas-keyword",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN }],
    });

    // Caller key LAST: it authenticating implies every row above is live.
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await realtime?.close();
    await sls?.close();
  });

  const postJson = async (path: string, body: unknown, anthropic = false) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: anthropic
        ? {
            "content-type": "application/json",
            "x-api-key": KEY,
            "anthropic-version": "2023-06-01",
          }
        : { "content-type": "application/json", authorization: `Bearer ${KEY}` },
      body: JSON.stringify(body),
    });
    await res.arrayBuffer();
    return res.status;
  };

  const postForm = async (path: string, parts: Record<string, string | Blob>) => {
    const form = new FormData();
    for (const [name, value] of Object.entries(parts)) {
      if (value instanceof Blob) form.set(name, value, "a.bin");
      else form.set(name, value);
    }
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${KEY}` },
      body: form,
    });
    await res.arrayBuffer();
    return res.status;
  };

  const rowWhere = (what: string, pred: (log: Map<string, string>) => boolean) =>
    waitForSlsLog(sls!, LOGSTORE, pred, what);

  const expectApplied = (log: Map<string, string>, what: string) => {
    expect(
      JSON.parse(log.get("applied_guardrails") ?? "[]"),
      `${what}: the row does not name the guardrail that governed it`,
    ).toEqual(APPLIED);
  };

  /** Every refusal: its row is a guardrail block AND names the guardrail. */
  const blocks: Array<{
    surface: string;
    model: string;
    /** Set when the refusal row carries no model: a multipart surface
     *  refuses inside the form parse, before the model is attributed. */
    operation?: string;
    send: () => Promise<number>;
  }> = [
    {
      surface: "/v1/completions",
      model: "gas-completions",
      send: () => postJson("/v1/completions", { model: "gas-completions", prompt: FORBIDDEN }),
    },
    {
      surface: "/v1/embeddings",
      model: "gas-embeddings",
      send: () => postJson("/v1/embeddings", { model: "gas-embeddings", input: FORBIDDEN }),
    },
    {
      surface: "/v1/rerank",
      model: "gas-rerank",
      send: () =>
        postJson("/v1/rerank", { model: "gas-rerank", query: FORBIDDEN, documents: ["a doc"] }),
    },
    {
      surface: "/v1/images/generations",
      model: "gas-image-gen",
      send: () =>
        postJson("/v1/images/generations", { model: "gas-image-gen", prompt: FORBIDDEN }),
    },
    {
      surface: "/v1/images/edits",
      model: "gas-image-edit",
      send: () =>
        postForm("/v1/images/edits", {
          model: "gas-image-edit",
          prompt: FORBIDDEN,
          image: new Blob(["fake-png-bytes"], { type: "image/png" }),
        }),
    },
    {
      surface: "/v1/audio/speech",
      model: "gas-speech",
      send: () =>
        postJson("/v1/audio/speech", { model: "gas-speech", input: FORBIDDEN, voice: "alloy" }),
    },
    {
      surface: "/v1/audio/transcriptions",
      model: "gas-transcription",
      operation: "transcription",
      send: () =>
        postForm("/v1/audio/transcriptions", {
          model: "gas-transcription",
          prompt: FORBIDDEN,
          file: new Blob(["fake-audio-bytes"], { type: "audio/wav" }),
        }),
    },
    {
      surface: "/v1/videos",
      model: "gas-video",
      send: () => postJson("/v1/videos", { model: "gas-video", prompt: FORBIDDEN }),
    },
    {
      surface: "/v1/messages/count_tokens",
      model: "gas-count-tokens",
      send: () =>
        postJson(
          "/v1/messages/count_tokens",
          { model: "gas-count-tokens", messages: [{ role: "user", content: FORBIDDEN }] },
          true,
        ),
    },
    {
      surface: "/v1/responses",
      model: "gas-responses-block",
      send: () => postJson("/v1/responses", { model: "gas-responses-block", input: FORBIDDEN }),
    },
  ];

  for (const { surface, model, operation, send } of blocks) {
    test(`a refusal on ${surface} names the guardrail that refused it`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      expect(await send()).toBe(422);
      const log = await rowWhere(`${surface} refusal row`, (l) =>
        operation === undefined
          ? l.get("requested_model") === model
          : l.get("operation") === operation && l.get("status_code") === "422",
      );
      expect(log.get("status_code")).toBe("422");
      expect(log.get("guardrail_blocked")).toBe("true");
      expectApplied(log, surface);
    });
  }

  test("a refusal on a passthrough route names the guardrail that refused it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const status = await postJson("/passthrough/gas/chat/completions", {
      model: "gpt-4o",
      messages: [{ role: "user", content: FORBIDDEN }],
    });
    expect(status).toBe(422);
    const log = await rowWhere(
      "passthrough refusal row",
      (l) => l.get("passthrough_route_name") === ROUTE && l.get("status_code") === "422",
    );
    expect(log.get("guardrail_blocked")).toBe("true");
    expectApplied(log, "passthrough refusal");
  });

  test("a refusal on /v1/batches names the guardrail that refused it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const status = await postJson("/v1/batches", {
      model: "gas-batches",
      input_file_id: "file-gas-in",
      endpoint: "/v1/chat/completions",
      completion_window: "24h",
      metadata: { note: FORBIDDEN },
    });
    expect(status).toBe(422);
    // The refusal row carries no model — the jobs surface fails before it
    // attributes one — so it is found by its operation instead.
    const log = await rowWhere(
      "batches refusal row",
      (l) => l.get("operation") === "batches" && l.get("status_code") === "422",
    );
    expect(log.get("guardrail_blocked")).toBe("true");
    expectApplied(log, "batches refusal");
  });

  test("an allowed /v1/responses request names the guardrail that screened it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    expect(await postJson("/v1/responses", { model: "gas-responses-ok", input: "hello" })).toBe(
      200,
    );
    const log = await rowWhere("responses row", (l) => l.get("requested_model") === "gas-responses-ok");
    expect(log.get("status_code")).toBe("200");
    expect(log.get("guardrail_blocked")).not.toBe("true");
    expectApplied(log, "/v1/responses");
  });

  test("an allowed passthrough request names the guardrail that screened it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const status = await postJson("/passthrough/gas/responses", {
      model: "gpt-4o",
      input: "hello",
    });
    expect(status).toBe(200);
    const log = await rowWhere(
      "passthrough row",
      (l) => l.get("passthrough_route_name") === ROUTE && l.get("status_code") === "200",
    );
    expectApplied(log, "passthrough");
  });

  test("an allowed /v1/batches request names the guardrail that screened it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const status = await postJson("/v1/batches", {
      model: "gas-batches",
      input_file_id: "file-gas-in",
      endpoint: "/v1/chat/completions",
      completion_window: "24h",
    });
    expect(status).toBe(200);
    const log = await rowWhere("batches row", (l) => l.get("requested_model") === "gas-batches");
    expect(log.get("status_code")).toBe("200");
    expectApplied(log, "/v1/batches");
  });

  test("a /v1/realtime session names the guardrail that screened it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const ws = new WsClient(
      `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=gas-realtime`,
      { headers: { authorization: `Bearer ${KEY}` } },
    );
    await new Promise<void>((resolve, reject) => {
      ws.once("open", () => resolve());
      ws.once("error", reject);
    });
    const done = new Promise<void>((resolve) => ws.once("message", () => resolve()));
    ws.send(JSON.stringify({ type: "session.update", session: { instructions: "hi" } }));
    await done;
    ws.close();
    const log = await rowWhere("realtime row", (l) => l.get("requested_model") === "gas-realtime");
    expectApplied(log, "/v1/realtime");
  });
});
