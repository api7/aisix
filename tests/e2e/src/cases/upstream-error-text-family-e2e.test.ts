import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
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
const HUGE_BODY_BYTES = 256 * 1024 * 1024;
const peakRssBytes = async (pid: number) => {
  const status = await readFile(`/proc/${pid}/status`, "utf8");
  return Number(/^VmHWM:\s+(\d+) kB/m.exec(status)![1]) * 1024;
};

const openAiEnvelope = (status: number) =>
  JSON.stringify({
    error: {
      message: `${TEXT} ${status}`,
      type: "invalid_request_error",
      code: `sentinel_code_${status}`,
      param: `sentinel_param_${status}`,
    },
  });

// What real Azure OpenAI answers: a missing deployment comes with no
// Content-Type, a rejected parameter with one.
const AZURE_NO_DEPLOYMENT_TEXT = "The API deployment for this resource does not exist.";
const AZURE_NO_DEPLOYMENT = JSON.stringify(
  {
    error: { type: "invalid_request_error", code: "DeploymentNotFound", message: AZURE_NO_DEPLOYMENT_TEXT },
  },
  null,
  2,
);
const AZURE_BAD_PARAM_TEXT = "Invalid 'temperature': decimal above maximum value.";
const AZURE_BAD_PARAM = JSON.stringify({
  error: {
    message: AZURE_BAD_PARAM_TEXT,
    type: "invalid_request_error",
    param: "temperature",
    code: "decimal_above_max_value",
  },
});

// What each video provider answered a request for a model it does not have.
const VIDEO_PROVIDER_ANSWERS: [string, string, number, string][] = [
  [
    "real-dashscope",
    "alibaba",
    400,
    JSON.stringify({ code: "InvalidParameter", message: "Model not exist.", request_id: "r-1" }),
  ],
  [
    "real-zhipu",
    "zhipuai",
    400,
    JSON.stringify({ error: { code: "1211", message: "模型不存在，请检查模型代码。" } }),
  ],
  [
    "real-ark",
    "volcengine",
    404,
    JSON.stringify({
      error: {
        code: "InvalidEndpointOrModel.NotFound",
        message: "The model or endpoint qa-video-nope does not exist",
        param: "",
        type: "Not Found",
      },
    }),
  ],
  [
    "real-runway",
    "runwayml",
    400,
    JSON.stringify({ error: "Validation of body failed", issues: [{ path: ["model"] }] }),
  ],
  ["real-openai-empty", "openai", 404, ""],
];

describe("an upstream's failure reads the same on every route that sends its own request", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;
  const servers: Server[] = [];
  let cooldownModelIds: Record<string, string> = {};

  async function fixedUpstream(
    status: number,
    body: string,
    headers: Record<string, string> = {},
    contentType: string | null = "application/json",
  ): Promise<string> {
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        res.writeHead(status, contentType ? { "content-type": contentType, ...headers } : headers);
        res.end(body);
      });
    });
    servers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  // A video task that exists: the submit succeeds, and the follow-up routes
  // answer `failure` either on the poll or, once the task reads as completed,
  // on the content download.
  async function videoTaskUpstream(
    failing: "poll" | "content",
    failure: { status: number; body: string; headers?: Record<string, string>; cutOff?: boolean },
  ): Promise<string> {
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        const fail = () => {
          res.writeHead(failure.status, {
            "content-type": "application/json",
            ...(failure.cutOff ? { "content-length": "4096" } : {}),
            ...failure.headers,
          });
          if (failure.cutOff) res.write(failure.body, () => res.destroy());
          else res.end(failure.body);
        };
        const task = (status: string) => {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({ id: "video_x", object: "video", model: "m", status }));
        };
        if (req.method === "POST") return task("queued");
        if (req.url?.endsWith("/content")) return fail();
        return failing === "poll" ? fail() : task("completed");
      });
    });
    servers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  // A failure body sent in chunks, so nothing but the gateway holds it whole.
  async function hugeBodyUpstream(status: number, bytes: number): Promise<string> {
    const chunk = Buffer.alloc(1024 * 1024, "x");
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", async () => {
        res.writeHead(status, { "content-type": "application/json", "content-length": String(bytes) });
        for (let sent = 0; sent < bytes; sent += chunk.length) {
          if (!res.write(chunk)) await new Promise((r) => res.once("drain", r));
        }
        res.end();
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
    {
      const pk = await seed.createProviderKey({
        display_name: "video-cooldown-retry-after-pk",
        secret: "sk-mock",
        provider: "openai",
        api_base: `${await fixedUpstream(429, openAiEnvelope(429), { "retry-after": "180" })}/v1`,
      });
      const name = "video-cooldown-retry-after";
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
    await model("video-huge-body", "openai", `${await hugeBodyUpstream(400, HUGE_BODY_BYTES)}/v1`);
    names.push("video-huge-body");
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
    const HTML = "<html><body>bad gateway page</body></html>";
    for (const status of [400, 503]) {
      await model(`openai-html-${status}`, "openai", `${await fixedUpstream(status, HTML)}/v1`);
      await model(`openai-empty-${status}`, "openai", `${await fixedUpstream(status, "")}/v1`);
      names.push(`openai-html-${status}`, `openai-empty-${status}`);
    }
    await model(
      "openai-retry-after",
      "openai",
      `${await fixedUpstream(429, openAiEnvelope(429), { "retry-after": "7" })}/v1`,
    );
    names.push("openai-retry-after");
    await model(
      "alibaba-top-level",
      "alibaba",
      await fixedUpstream(
        400,
        JSON.stringify({ code: "InvalidParameter", message: `${TEXT} dashscope`, request_id: "r-1" }),
      ),
    );
    await model(
      "alibaba-top-level-500",
      "alibaba",
      await fixedUpstream(500, JSON.stringify({ code: "InternalError", message: `${TEXT} dashscope` })),
    );
    names.push("alibaba-top-level", "alibaba-top-level-500");
    for (const [name, provider, status, body] of VIDEO_PROVIDER_ANSWERS) {
      await model(name, provider, await fixedUpstream(status, body));
      names.push(name);
    }
    for (const leg of ["poll", "content"] as const) {
      for (const [shape, failure] of [
        ["envelope", { status: 400, body: openAiEnvelope(400) }],
        ["server", { status: 500, body: openAiEnvelope(500) }],
        ["html", { status: 503, body: HTML }],
        ["rate-limited", { status: 429, body: openAiEnvelope(429), headers: { "retry-after": "7" } }],
        ["cut", { status: 500, body: '{"error":{"mess', cutOff: true }],
        ["cut-rate-limited", { status: 429, body: '{"error":{"mess', headers: { "retry-after": "7" }, cutOff: true }],
      ] as const) {
        const name = `video-${leg}-${shape}`;
        await model(name, "openai", `${await videoTaskUpstream(leg, failure)}/v1`);
        names.push(name);
      }
    }
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
      ["refused-200-empty", 200, ""],
      ["refused-302-empty", 302, ""],
      ["refused-200-body", 200, `${TEXT} not an upgrade`],
      ["refused-200-long", 200, "x".repeat(2048)],
    ] as const) {
      await model(name, "openai", `${await fixedUpstream(status, body)}/v1`);
      names.push(name);
    }
    await model("refused-no-answer", "openai", "http://127.0.0.1:1/v1");
    names.push("refused-no-answer");
    for (const [name, status, body, contentType] of [
      [
        "refused-azure-404",
        404,
        JSON.stringify({ error: { code: "DeploymentNotFound", message: `${TEXT} azure` } }),
        "application/json",
      ],
      ["azure-no-deployment", 404, AZURE_NO_DEPLOYMENT, null],
      ["azure-bad-param", 400, AZURE_BAD_PARAM, "application/json"],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        provider: "azure-openai",
        adapter: "azure-openai",
        api_base: await fixedUpstream(status, body, {}, contentType),
      });
      await seed.createModel({
        display_name: name,
        provider: "azure-openai",
        model_name: "m",
        provider_key_id: pk.id,
        cooldown: { enabled: false },
      });
      names.push(name);
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
    ["videos", "/v1/videos", (model) => json({ model, prompt: "hi" }), "openai"],
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
        "upstream returned HTTP 500 with no error detail in the response",
      ],
      [
        "refused-200-empty",
        { type: "upstream_error", message: "upstream returned 200 OK" },
        502,
        "upstream_status",
        "upstream returned HTTP 200 during the realtime handshake",
      ],
      [
        "refused-302-empty",
        { type: "upstream_error", message: "upstream returned 302 Found" },
        502,
        "upstream_status",
        "upstream returned HTTP 302 during the realtime handshake",
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

    test("an empty error body records the text chat records", async () => {
      if (!etcdReachable) return;
      const chat = await send(
        "/v1/chat/completions",
        json({ model: "refused-500-empty", messages: [{ role: "user", content: "hi" }] }),
      );
      const got = await refusal("refused-500-empty");
      expect((got.frames[0] as { error: unknown }).error).toEqual(chat.body.error);
      const realtimeRow = await finalRow(got.requestId, "the realtime row");
      const chatRow = await finalRow(chat.requestId, "the chat row");
      expect(realtimeRow.get("error_message")).toBe(chatRow.get("error_message"));
      expect(JSON.parse(realtimeRow.get("response")!).error.message).toBe(chatRow.get("error_message"));
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

  describe.each([
    ["chat", "/v1/chat/completions", { messages: [{ role: "user", content: "hi" }] }],
    ["chat streamed", "/v1/chat/completions", { messages: [{ role: "user", content: "hi" }], stream: true }],
    ["responses", "/v1/responses", { input: "hi" }],
    ["responses streamed", "/v1/responses", { input: "hi", stream: true }],
  ] as const)("azure 4xx on %s", (_what, path, payload) => {
    test("a missing deployment answers model_not_found", async () => {
      if (!etcdReachable) return;
      const got = await send(path, json({ model: "azure-no-deployment", ...payload }));
      expect(got.status).toBe(404);
      expect(got.body.error).toEqual({
        message: AZURE_NO_DEPLOYMENT_TEXT,
        type: "upstream_error",
        code: "model_not_found",
      });
    });

    test("any other code passes through with its param", async () => {
      if (!etcdReachable) return;
      const got = await send(path, json({ model: "azure-bad-param", ...payload }));
      expect(got.status).toBe(400);
      expect(got.body.error).toEqual({
        message: AZURE_BAD_PARAM_TEXT,
        type: "upstream_error",
        code: "decimal_above_max_value",
        param: "temperature",
      });
    });
  });

  test("azure 4xx on messages keeps the upstream's text under its status-mapped type", async () => {
    if (!etcdReachable) return;
    const got = await send(
      "/v1/messages",
      json({ model: "azure-no-deployment", messages: [{ role: "user", content: "hi" }], max_tokens: 5 }),
    );
    expect(got.status).toBe(404);
    expect(got.body.error).toEqual({ type: "not_found_error", message: AZURE_NO_DEPLOYMENT_TEXT });
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

  describe("videos", () => {
    const chatOn = (model: string) =>
      send("/v1/chat/completions", json({ model, messages: [{ role: "user", content: "hi" }] }));
    const submit = (model: string) => send("/v1/videos", json({ model, prompt: "hi" }));

    test("a vendor's bare code is the caller's code; the row reads as chat's", async () => {
      if (!etcdReachable) return;
      const got = await submit("video-code-only");
      expect(got.status).toBe(400);
      expect(got.body.error).toEqual({
        message: "upstream returned 400 Bad Request",
        type: "upstream_error",
        code: "vendor_code",
      });
      const chat = await chatOn("video-code-only");
      expect(await rowMessage(got.requestId, "the code-only row")).toBe(
        await rowMessage(chat.requestId, "chat's code-only row"),
      );
    });

    test("a vendor code leaves the message alone and reaches the caller as its code", async () => {
      if (!etcdReachable) return;
      const got = await submit("video-envelope");
      expect(got.body.error).toEqual({
        message: `${TEXT} video`,
        type: "upstream_error",
        code: "vendor_code",
      });
      expect(await rowMessage(got.requestId, "the video row")).toBe(
        `upstream returned HTTP 400: ${TEXT} video`,
      );
    });

    test("DashScope's top-level code and message read like an envelope's", async () => {
      if (!etcdReachable) return;
      const got = await submit("alibaba-top-level");
      expect(got.status).toBe(400);
      expect(got.body.error).toEqual({
        message: `${TEXT} dashscope`,
        type: "upstream_error",
        code: "InvalidParameter",
      });
    });

    test("DashScope's 5xx carries its text and no code", async () => {
      if (!etcdReachable) return;
      const got = await submit("alibaba-top-level-500");
      expect(got.status).toBe(502);
      expect(got.body.error).toEqual({ message: `${TEXT} dashscope`, type: "upstream_error" });
    });

    test("DashScope's model-not-found carries its code at the top level", async () => {
      if (!etcdReachable) return;
      const got = await submit("real-dashscope");
      expect(got.status).toBe(400);
      expect(got.body.error).toEqual({
        message: "Model not exist.",
        type: "upstream_error",
        code: "InvalidParameter",
      });
    });

    test.each([
      [
        "Zhipu's model-not-found",
        "real-zhipu",
        400,
        { message: "模型不存在，请检查模型代码。", type: "upstream_error", code: "1211" },
      ],
      [
        "Ark's model-not-found",
        "real-ark",
        404,
        {
          message: "The model or endpoint qa-video-nope does not exist",
          type: "upstream_error",
          code: "InvalidEndpointOrModel.NotFound",
          param: "",
        },
      ],
      [
        "Runway's validation failure",
        "real-runway",
        400,
        { message: "Validation of body failed", type: "upstream_error" },
      ],
      [
        "OpenAI's empty 404",
        "real-openai-empty",
        404,
        { message: "upstream returned 404 Not Found", type: "upstream_error" },
      ],
    ] as const)("%s answers what chat answers", async (_what, model, status, error) => {
      if (!etcdReachable) return;
      const got = await submit(model);
      expect(got.status).toBe(status);
      expect(got.body.error).toEqual(error);
      const chat = await chatOn(model);
      expect(chat.status).toBe(got.status);
      expect(chat.body).toEqual(got.body);
    });

    test.each([
      ["a page that is no JSON on a 4xx", "openai-html-400"],
      ["a page that is no JSON on a 5xx", "openai-html-503"],
      ["an empty body on a 4xx", "openai-empty-400"],
      ["an empty body on a 5xx", "openai-empty-503"],
      ["a body past the error-body cap", "openai-past-the-cap"],
    ])("%s answers what chat answers", async (_what, model) => {
      if (!etcdReachable) return;
      const chat = await chatOn(model);
      const got = await submit(model);
      expect(got.status).toBe(chat.status);
      expect(got.body).toEqual(chat.body);
      expect(got.body.error.message).not.toBe("upstream error");
      expect(await rowMessage(got.requestId, "the video row")).toBe(
        await rowMessage(chat.requestId, "chat's row"),
      );
    });

    test("an upstream's Retry-After reaches the caller as chat's does", async () => {
      if (!etcdReachable) return;
      const chat = await fetch(
        `${app!.proxyUrl}/v1/chat/completions`,
        json({ model: "openai-retry-after", messages: [{ role: "user", content: "hi" }] }),
      );
      const got = await fetch(
        `${app!.proxyUrl}/v1/videos`,
        json({ model: "openai-retry-after", prompt: "hi" }),
      );
      expect(got.status).toBe(429);
      expect(chat.headers.get("retry-after")).toBe("7");
      expect(got.headers.get("retry-after")).toBe("7");
      expect(await got.json()).toEqual(await chat.json());
    });

    test("an upstream's Retry-After sets the model's cooldown", async () => {
      if (!etcdReachable) return;
      const before = Date.now() / 1000;
      const got = await submit("video-cooldown-retry-after");
      expect(got.status).toBe(429);
      const rows = await new AdminClient(app!.adminUrl, app!.adminKey, app!.metricsUrl).listModelStatuses();
      const row = rows.find((r) => r.id === cooldownModelIds["video-cooldown-retry-after"])!;
      expect(row.status).toBe("cooldown");
      const until = row.cooldown_until as unknown as { secs_since_epoch: number };
      expect(until.secs_since_epoch - before).toBeGreaterThan(120);
    });

    test("a body past the error-body cap is not parsed", async () => {
      if (!etcdReachable) return;
      const got = await submit("openai-past-the-cap");
      expect(got.status).toBe(400);
      expect(got.body.error.code).toBeUndefined();
      expect(got.body.error.message).toEqual(expect.stringMatching(/^\{"error":/));
      expect(Buffer.byteLength(String(got.body.error.message))).toBeLessThanOrEqual(1024 + 3);
    });

    test.skipIf(process.platform !== "linux")("a body far past the error-body cap is not held in memory", async () => {
      if (!etcdReachable) return;
      const before = await peakRssBytes(app!.pid);
      const got = await submit("video-huge-body");
      expect(got.status).toBe(400);
      expect(Buffer.byteLength(String(got.body.error.message))).toBeLessThanOrEqual(1024 + 3);
      expect((await peakRssBytes(app!.pid)) - before).toBeLessThan(HUGE_BODY_BYTES / 4);
    });

    describe.each(["poll", "content"] as const)("the %s route of a task that exists", (leg) => {
      const follow = async (model: string) => {
        const submitted = await fetch(`${app!.proxyUrl}/v1/videos`, json({ model, prompt: "hi" }));
        expect(submitted.status).toBe(200);
        const { id } = (await submitted.json()) as { id: string };
        const res = await fetch(
          `${app!.proxyUrl}/v1/videos/${id}${leg === "content" ? "/content" : ""}`,
          { headers: { authorization: `Bearer ${CALLER}` } },
        );
        return {
          status: res.status,
          retryAfter: res.headers.get("retry-after"),
          body: (await res.json()) as { error: Record<string, unknown> },
        };
      };

      test("a 4xx carries the upstream's text, code and param", async () => {
        if (!etcdReachable) return;
        const got = await follow(`video-${leg}-envelope`);
        expect(got.status).toBe(400);
        expect(got.body.error).toEqual({
          message: `${TEXT} 400`,
          type: "upstream_error",
          code: "sentinel_code_400",
          param: "sentinel_param_400",
        });
      });

      test("a 5xx carries the upstream's text alone", async () => {
        if (!etcdReachable) return;
        const got = await follow(`video-${leg}-server`);
        expect(got.status).toBe(502);
        expect(got.body.error).toEqual({ message: `${TEXT} 500`, type: "upstream_error" });
      });

      test("a page that is no JSON is returned as text", async () => {
        if (!etcdReachable) return;
        const got = await follow(`video-${leg}-html`);
        expect(got.status).toBe(502);
        expect(got.body.error).toEqual({
          message: "<html><body>bad gateway page</body></html>",
          type: "upstream_error",
        });
      });

      test("a 429 keeps the upstream's Retry-After", async () => {
        if (!etcdReachable) return;
        const got = await follow(`video-${leg}-rate-limited`);
        expect(got.status).toBe(429);
        expect(got.retryAfter).toBe("7");
      });

      test("a body cut off answers its status, not the fragment", async () => {
        if (!etcdReachable) return;
        const got = await follow(`video-${leg}-cut`);
        expect(got.status).toBe(502);
        expect(got.body.error).toEqual({
          message: "upstream returned 500 Internal Server Error",
          type: "upstream_error",
        });
      });

      test("a 429 cut off keeps the upstream's Retry-After", async () => {
        if (!etcdReachable) return;
        const got = await follow(`video-${leg}-cut-rate-limited`);
        expect(got.status).toBe(429);
        expect(got.retryAfter).toBe("7");
      });
    });
  });
});
