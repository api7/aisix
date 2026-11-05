import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket } from "ws";
import {
  AdminClient,
  EtcdClient,
  scrapeMetrics,
  SeedClient,
  spawnApp,
  startMockSls,
  sumMetric,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// A route that sends the request itself reads an upstream's failure the way a
// bridge does: chat's answer is the oracle for the caller's envelope, the
// failure row and the access-log line, on every status.

const CALLER = "sk-error-text-caller";
const LOGSTORE = "error-text-rows";
const TEXT = "sentinel upstream failure text";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const openAiEnvelope = (status: number) =>
  JSON.stringify({
    error: {
      message: `${TEXT} ${status}`,
      type: "invalid_request_error",
      code: `sentinel_code_${status}`,
      param: `sentinel_param_${status}`,
    },
  });

describe("an upstream's failure reads the same on every route that sends its own request", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;
  const servers: Server[] = [];
  let cooldownModelIds: Record<string, string> = {};

  async function fixedUpstream(status: number, body: string): Promise<string> {
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        res.writeHead(status, { "content-type": "application/json" });
        res.end(body);
      });
    });
    servers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        RUST_LOG: "info",
        SLS_CRED_MOCK_AK_ID: "LTAI_mock_ak",
        SLS_CRED_MOCK_AK_SECRET: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: LOGSTORE,
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: "mock",
      content_mode: "full",
    });

    const model = async (name: string, provider: string, base: string) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        provider,
        ...(provider === "openai" ? {} : { adapter: "openai" }),
        api_base: base,
      });
      await seed.createModel({
        display_name: name,
        provider,
        model_name: "m",
        provider_key_id: pk.id,
        cooldown: { enabled: false },
      });
      return pk.id;
    };
    const names: string[] = [];
    for (const status of [500, 400]) {
      const base = await fixedUpstream(status, openAiEnvelope(status));
      const pkId = await model(`openai-${status}`, "openai", `${base}/v1`);
      await model(`cohere-${status}`, "cohere", base);
      names.push(`openai-${status}`, `cohere-${status}`);
      await seed.createPassthroughRoute({
        name: `passthrough-${status}`,
        path_prefix: `/passthrough-${status}`,
        target_url: base,
        provider_key_id: pkId,
      });
    }
    const cooledModels: Record<string, string> = {};
    for (const status of [400, 503]) {
      const pk = await seed.createProviderKey({
        display_name: `realtime-cooldown-${status}-pk`,
        secret: "sk-mock",
        provider: "openai",
        api_base: `${await fixedUpstream(status, openAiEnvelope(status))}/v1`,
      });
      const name = `realtime-cooldown-${status}`;
      cooledModels[name] = (
        await seed.createModel({
          display_name: name,
          provider: "openai",
          model_name: "m",
          provider_key_id: pk.id,
          cooldown: { enabled: true },
        })
      ).id;
      names.push(name);
    }
    cooldownModelIds = cooledModels;
    for (const status of [401, 429]) {
      const name = `openai-${status}`;
      await model(name, "openai", `${await fixedUpstream(status, openAiEnvelope(status))}/v1`);
      names.push(name);
    }
    const padded = JSON.stringify({
      error: { message: `${TEXT} padded`, code: "padded_code" },
      padding: "x".repeat(100 * 1024),
    });
    await model("openai-past-the-cap", "openai", `${await fixedUpstream(400, padded)}/v1`);
    names.push("openai-past-the-cap");
    await model(
      "cohere-top-level",
      "cohere",
      await fixedUpstream(400, JSON.stringify({ message: `${TEXT} top-level` })),
    );
    await model(
      "video-envelope",
      "openai",
      `${await fixedUpstream(400, JSON.stringify({ error: { code: "vendor_code", message: `${TEXT} video` } }))}/v1`,
    );
    await model(
      "video-code-only",
      "openai",
      `${await fixedUpstream(400, JSON.stringify({ code: "vendor_code", message: "" }))}/v1`,
    );
    names.push("cohere-top-level", "video-envelope", "video-code-only");
    for (const [shape, body] of [
      ["top-level", { message: `${TEXT} other shape` }],
      ["string", { error: `${TEXT} other shape` }],
      ["detail", { detail: `${TEXT} other shape` }],
      ["detail-object", { detail: { message: `${TEXT} other shape` } }],
      ["blank-envelope", { error: { message: "" }, detail: `${TEXT} other shape` }],
    ] as const) {
      const base = await fixedUpstream(400, JSON.stringify(body));
      await model(`openai-${shape}`, "openai", `${base}/v1`);
      await model(`anthropic-${shape}`, "anthropic", base);
      names.push(`openai-${shape}`, `anthropic-${shape}`);
    }
    for (const [name, status, body] of [
      ["refused-401", 401, openAiEnvelope(401)],
      ["refused-429", 429, openAiEnvelope(429)],
      ["refused-500", 500, openAiEnvelope(500)],
      ["refused-500-empty", 500, ""],
      ["refused-200-body", 200, `${TEXT} not an upgrade`],
      ["refused-200-long", 200, "x".repeat(2048)],
    ] as const) {
      await model(name, "openai", `${await fixedUpstream(status, body)}/v1`);
      names.push(name);
    }
    await model("refused-no-answer", "openai", "http://127.0.0.1:1/v1");
    names.push("refused-no-answer");
    const azurePk = await seed.createProviderKey({
      display_name: "refused-azure-404-pk",
      secret: "sk-mock",
      provider: "azure-openai",
      adapter: "azure-openai",
      api_base: await fixedUpstream(
        404,
        JSON.stringify({ error: { code: "DeploymentNotFound", message: `${TEXT} azure` } }),
      ),
    });
    await seed.createModel({
      display_name: "refused-azure-404",
      provider: "azure-openai",
      model_name: "m",
      provider_key_id: azurePk.id,
      cooldown: { enabled: false },
    });
    names.push("refused-azure-404");
    await seed.createApiKey({ key_hash: sha256(CALLER), allowed_routes: ["*"], allowed_models: names });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const { data } = (await res.json()) as { data?: unknown[] };
      return data?.length === names.length;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    for (const s of servers) s.close();
  });

  const json = (body: unknown): RequestInit => ({
    method: "POST",
    headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const multipart = (fields: Record<string, string>): RequestInit => {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    form.set("file", new Blob([new Uint8Array(16)]), "input.png");
    return { method: "POST", headers: { authorization: `Bearer ${CALLER}` }, body: form };
  };

  async function send(path: string, init: RequestInit) {
    const res = await fetch(`${app!.proxyUrl}${path}`, init);
    return {
      status: res.status,
      body: (await res.json()) as { error: Record<string, unknown> },
      requestId: res.headers.get("x-aisix-request-id")!,
    };
  }

  const finalRow = (requestId: string, what: string) =>
    waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId && l.get("is_final_attempt") === "true",
      what,
    );
  const rowMessage = async (requestId: string, what: string) =>
    (await finalRow(requestId, what)).get("error_message");

  const refusal = (model: string) =>
    new Promise<{ requestId: string; frames: unknown[]; code: number; reason: string }>((resolve, reject) => {
      const ws = new WebSocket(
        `${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`,
        { headers: { authorization: `Bearer ${CALLER}` } },
      );
      let requestId = "";
      const frames: unknown[] = [];
      ws.on("upgrade", (res) => {
        requestId = String(res.headers["x-aisix-request-id"] ?? "");
      });
      ws.on("message", (data) => frames.push(JSON.parse(String(data))));
      ws.on("close", (code, reason) => resolve({ requestId, frames, code, reason: String(reason) }));
      ws.on("error", reject);
    });
  const refusedHandshake = async (model: string) => (await refusal(model)).requestId;

  const DECISION_QUESTIONS = [{ type: "predicate", name: "q", instructions: "i" }];
  const ROUTES: [string, string, (model: string) => RequestInit, "openai" | "cohere"][] = [
    ["decisions", "/v1/decisions", (model) => json({ model, input: "hi", questions: DECISION_QUESTIONS }), "openai"],
    ["rerank", "/v1/rerank", (model) => json({ model, query: "q", documents: ["a"] }), "openai"],
    ["cohere rerank", "/v1/rerank", (model) => json({ model, query: "q", documents: ["a"] }), "cohere"],
    ["speech", "/v1/audio/speech", (model) => json({ model, input: "hi", voice: "alloy" }), "openai"],
    ["transcriptions", "/v1/audio/transcriptions", (model) => multipart({ model }), "openai"],
    ["translations", "/v1/audio/translations", (model) => multipart({ model }), "openai"],
    ["image edits", "/v1/images/edits", (model) => multipart({ model, prompt: "hi" }), "openai"],
    ["native responses", "/v1/responses", (model) => json({ model, input: "hi" }), "openai"],
    ["streaming native responses", "/v1/responses", (model) => json({ model, input: "hi", stream: true }), "openai"],
  ];

  describe.each([500, 400])("on an upstream %i", (upstream) => {
    test.each(ROUTES)("%s answers and records what chat does", async (_what, path, init, provider) => {
      if (!etcdReachable) return;
      const chat = await send(
        "/v1/chat/completions",
        json({ model: `openai-${upstream}`, messages: [{ role: "user", content: "hi" }] }),
      );
      expect(chat.body.error.message).toBe(`${TEXT} ${upstream}`);
      const chatRow = await rowMessage(chat.requestId, "the chat row");
      expect(chatRow).toBe(`upstream returned HTTP ${upstream}: ${TEXT} ${upstream}`);

      const got = await send(path, init(`${provider}-${upstream}`));
      expect(got.status).toBe(chat.status);
      expect(got.body).toEqual(chat.body);
      expect(await rowMessage(got.requestId, "the route's row")).toBe(chatRow);
      const line = await waitForLogLine(
        app!,
        (l) => l.includes(got.requestId) && l.includes(chatRow!),
        "the route's access-log line",
      );
      expect(line).not.toContain("invalid_request_error");
    });

    test("passthrough relays the upstream as sent and records its text", async () => {
      if (!etcdReachable) return;
      const res = await fetch(`${app!.proxyUrl}/passthrough-${upstream}/anything`, json({ a: 1 }));
      expect(res.status).toBe(upstream);
      expect(await res.text()).toBe(openAiEnvelope(upstream));
      expect(await rowMessage(res.headers.get("x-aisix-request-id")!, "the passthrough row")).toBe(
        `upstream returned HTTP ${upstream}: ${TEXT} ${upstream}`,
      );
    });

    test("a realtime handshake records the upstream's text and status class", async () => {
      if (!etcdReachable) return;
      const requestId = await refusedHandshake(`openai-${upstream}`);
      const row = await finalRow(requestId, "the realtime row");
      expect(row.get("error_message")).toBe(`upstream returned HTTP ${upstream}: ${TEXT} ${upstream}`);
      expect(row.get("error_class")).toBe("upstream_status");
    });
  });

  test.each<[number, number, string]>([
    [401, 401, "client_error"],
    [429, 429, "rate_limited"],
    [500, 502, "upstream_error"],
    [400, 400, "client_error"],
  ])("a realtime handshake refused with %i is recorded as %i", async (upstream, recorded, outcome) => {
    if (!etcdReachable) return;
    const model = `openai-${upstream}`;
    const requestId = await refusedHandshake(model);
    const row = await finalRow(requestId, "the realtime row");
    expect(row.get("status_code")).toBe(String(recorded));
    await waitForLogLine(
      app!,
      (l) => l.includes(requestId) && l.includes(`status_code=${recorded}`),
      "the realtime access-log line",
    );
    const metrics = await scrapeMetrics(app!.metricsUrl);
    expect(sumMetric(metrics, "aisix_proxy_requests_total", { endpoint: "/v1/realtime", status: String(recorded) })).toBeGreaterThan(0);
    expect(sumMetric(metrics, "aisix_requests_total", { model, status: String(recorded), outcome })).toBeGreaterThan(0);
  });

  test("a realtime handshake refused with a 5xx is not counted under that 5xx", async () => {
    if (!etcdReachable) return;
    await finalRow(await refusedHandshake("openai-500"), "the realtime row");
    const metrics = await scrapeMetrics(app!.metricsUrl);
    expect(sumMetric(metrics, "aisix_proxy_requests_total", { endpoint: "/v1/realtime", status: "500" })).toBe(0);
  });

  describe("a refused realtime handshake reaches the caller as an HTTP route's error would", () => {
    const fixedText = "failed to connect to the upstream realtime endpoint";
    const answered = (status: number) =>
      `upstream returned HTTP ${status}: ${TEXT} ${status}`;
    test.each<[string, Record<string, string>, number, string, string]>([
      [
        "refused-401",
        { type: "upstream_error", message: `${TEXT} 401`, code: "sentinel_code_401", param: "sentinel_param_401" },
        401,
        "upstream_status",
        answered(401),
      ],
      [
        "refused-429",
        { type: "upstream_error", message: `${TEXT} 429`, code: "sentinel_code_429", param: "sentinel_param_429" },
        429,
        "upstream_status",
        answered(429),
      ],
      ["refused-500", { type: "upstream_error", message: `${TEXT} 500` }, 502, "upstream_status", answered(500)],
      [
        "refused-500-empty",
        { type: "upstream_error", message: "upstream returned 500 Internal Server Error" },
        502,
        "upstream_status",
        "upstream returned HTTP 500 during the realtime handshake",
      ],
      [
        "refused-200-body",
        { type: "upstream_error", message: `${TEXT} not an upgrade` },
        502,
        "upstream_status",
        `upstream returned HTTP 200: ${TEXT} not an upgrade`,
      ],
      [
        "refused-azure-404",
        { type: "upstream_error", message: `${TEXT} azure`, code: "DeploymentNotFound" },
        404,
        "upstream_status",
        `upstream returned HTTP 404: ${TEXT} azure`,
      ],
    ])("%s", async (model, error, status, errorClass, rowText) => {
      if (!etcdReachable) return;
      const got = await refusal(model);
      expect(got.frames).toEqual([{ type: "error", error }]);
      expect(got.code).toBe(1011);
      expect(got.reason).toBe("upstream connect failed");
      const row = await finalRow(got.requestId, "the realtime row");
      expect(row.get("status_code")).toBe(String(status));
      expect(row.get("error_class")).toBe(errorClass);
      expect(row.get("error_message")).toBe(rowText);
    });

    test("a body that is no error is quoted within the usual bound", async () => {
      if (!etcdReachable) return;
      const got = await refusal("refused-200-long");
      const { error } = got.frames[0] as { error: { message: string } };
      expect(error.message).toBe(`${"x".repeat(1024)}…`);
      expect(got.code).toBe(1011);
      expect((await finalRow(got.requestId, "the realtime row")).get("status_code")).toBe("502");
    });

    test("a dial nothing answered keeps the fixed text", async () => {
      if (!etcdReachable) return;
      const got = await refusal("refused-no-answer");
      expect(got.frames).toEqual([{ type: "error", error: { type: "upstream_error", message: fixedText } }]);
      expect(got.code).toBe(1011);
      expect(got.reason).toBe("upstream connect failed");
      const row = await finalRow(got.requestId, "the realtime row");
      expect(row.get("status_code")).toBe("502");
      expect(row.get("error_class")).toBe("transport");
      expect(row.get("error_message")).toMatch(/^transport error: /);
    });
  });

  describe("a refused realtime handshake cools its model down as any upstream status does", () => {
    const status = async (name: string) => {
      const rows = await new AdminClient(app!.adminUrl, app!.adminKey, app!.metricsUrl).listModelStatuses();
      return rows.find((row) => row.id === cooldownModelIds[name])!;
    };

    test("a status on the trigger list cools it down, named for the status", async () => {
      if (!etcdReachable) return;
      await refusedHandshake("realtime-cooldown-503");
      const row = await status("realtime-cooldown-503");
      expect(row.status).toBe("cooldown");
      expect(row.status_reason).toBe("upstream_server_error");
    });

    test("a status off the trigger list does not", async () => {
      if (!etcdReachable) return;
      await refusedHandshake("realtime-cooldown-400");
      expect((await status("realtime-cooldown-400")).status).not.toBe("cooldown");
    });
  });

  test("a top-level message is read out of the body too", async () => {
    if (!etcdReachable) return;
    const got = await send("/v1/rerank", json({ model: "cohere-top-level", query: "q", documents: ["a"] }));
    expect(got.body.error.message).toBe(`${TEXT} top-level`);
    expect(await rowMessage(got.requestId, "the top-level row")).toBe(
      `upstream returned HTTP 400: ${TEXT} top-level`,
    );
  });

  const messages = [{ role: "user", content: "hi" }];
  describe.each(["top-level", "string", "detail", "detail-object", "blank-envelope"])("a body with a %s message", (shape) => {
    test.each([
      ["chat", "/v1/chat/completions", "openai", { messages }],
      ["chat to anthropic", "/v1/chat/completions", "anthropic", { messages }],
      ["completions", "/v1/completions", "openai", { prompt: "hi" }],
      ["embeddings", "/v1/embeddings", "openai", { input: "hi" }],
      ["image generations", "/v1/images/generations", "openai", { prompt: "hi" }],
      ["messages", "/v1/messages", "openai", { messages, max_tokens: 5 }],
      ["native messages", "/v1/messages", "anthropic", { messages, max_tokens: 5 }],
      ["count_tokens", "/v1/messages/count_tokens", "anthropic", { messages }],
      ["bridged responses", "/v1/responses", "anthropic", { input: "hi" }],
      ["native responses", "/v1/responses", "openai", { input: "hi" }],
      ["rerank", "/v1/rerank", "openai", { query: "q", documents: ["a"] }],
      ["speech", "/v1/audio/speech", "openai", { input: "hi", voice: "alloy" }],
      ["videos", "/v1/videos", "openai", { prompt: "hi" }],
      ["decisions", "/v1/decisions", "openai", { input: "hi", questions: DECISION_QUESTIONS }],
    ] as const)("%s reads its text", async (_what, path, provider, payload) => {
      if (!etcdReachable) return;
      const got = await send(path, json({ model: `${provider}-${shape}`, ...payload }));
      expect(got.body.error.message).toBe(`${TEXT} other shape`);
      expect(await rowMessage(got.requestId, "the row")).toBe(
        `upstream returned HTTP 400: ${TEXT} other shape`,
      );
    });
  });

  test("decisions: a body past the error-body cap answers what chat answers", async () => {
    if (!etcdReachable) return;
    const chat = await send(
      "/v1/chat/completions",
      json({ model: "openai-past-the-cap", messages }),
    );
    const got = await send(
      "/v1/decisions",
      json({ model: "openai-past-the-cap", input: "hi", questions: DECISION_QUESTIONS }),
    );
    expect(got.status).toBe(chat.status);
    expect(got.body).toEqual(chat.body);
    expect(await rowMessage(got.requestId, "the capped row")).toBe(
      await rowMessage(chat.requestId, "chat's capped row"),
    );
  });

  test("videos: a vendor's bare code stays in the row", async () => {
    if (!etcdReachable) return;
    const got = await send("/v1/videos", json({ model: "video-code-only", prompt: "hi" }));
    expect(got.body.error.message).toBe("vendor_code");
    expect(await rowMessage(got.requestId, "the code-only row")).toBe(
      "upstream returned HTTP 400: vendor_code",
    );
  });

  test("videos: the row carries the upstream's message; the caller keeps the vendor code", async () => {
    if (!etcdReachable) return;
    const got = await send("/v1/videos", json({ model: "video-envelope", prompt: "hi" }));
    expect(got.body.error.message).toBe(`vendor_code: ${TEXT} video`);
    expect(await rowMessage(got.requestId, "the video row")).toBe(
      `upstream returned HTTP 400: ${TEXT} video`,
    );
  });
});
