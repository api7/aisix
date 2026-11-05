import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket } from "ws";
import {
  AdminClient,
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockSls,
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
    ] as const) {
      const base = await fixedUpstream(400, JSON.stringify(body));
      await model(`openai-${shape}`, "openai", `${base}/v1`);
      await model(`anthropic-${shape}`, "anthropic", base);
      names.push(`openai-${shape}`, `anthropic-${shape}`);
    }
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

  const refusedHandshake = (model: string) =>
    new Promise<string>((resolve, reject) => {
      const ws = new WebSocket(
        `${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`,
        { headers: { authorization: `Bearer ${CALLER}` } },
      );
      let id = "";
      ws.on("upgrade", (res) => {
        id = String(res.headers["x-aisix-request-id"] ?? "");
      });
      ws.on("close", () => resolve(id));
      ws.on("error", reject);
    });

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
  describe.each(["top-level", "string", "detail", "detail-object"])("a body with a %s message", (shape) => {
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
