import { createHash } from "node:crypto";
import { createServer as createTcpServer, type Server as TcpServer, type Socket } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket, WebSocketServer } from "ws";
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

// How a realtime session the upstream never let start is answered and recorded:
// a handshake refused with an HTTP status, a dial that ran out of budget, and an
// upgrade that completed only for the upstream to report an error and close,
// which is how OpenAI answers a wrong key.

const CALLER = "sk-realtime-refused-caller";
const FULL = "refused-rows-full";
const META = "refused-rows-meta";
const CONNECT_TIMEOUT_MS = 800;
const READ_BUDGET_MS = 800;
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const envelope = (message: string, extra: Record<string, unknown> = {}) =>
  JSON.stringify({
    error: { message, type: "invalid_request_error", code: "sentinel_code", param: "sentinel_param" },
    ...extra,
  });

// The upstream's refusal, written by hand so head and body can leave separately.
type Script = (socket: Socket) => void | Promise<void>;
const head = (status: string, headers: Record<string, string | number>) =>
  `HTTP/1.1 ${status}\r\n${Object.entries(headers)
    .map(([k, v]) => `${k}: ${v}\r\n`)
    .join("")}\r\n`;

const PADDED = envelope("sentinel split text", { padding: "x".repeat(3900) });
const PAST_THE_CAP = envelope("sentinel past the cap", { padding: "x".repeat(100 * 1024) });
const JSON_TYPE = { "content-type": "application/json" };
const SPLIT = 150;

const REFUSALS: Record<string, Script> = {
  "split-length": async (s) => {
    s.write(head("401 Unauthorized", { ...JSON_TYPE, "content-length": Buffer.byteLength(PADDED) }));
    await sleep(SPLIT);
    s.end(PADDED);
  },
  "split-chunked": async (s) => {
    s.write(head("401 Unauthorized", { ...JSON_TYPE, "transfer-encoding": "chunked" }));
    await sleep(SPLIT);
    const half = Math.floor(PADDED.length / 2);
    s.write(`${half.toString(16)}\r\n${PADDED.slice(0, half)}\r\n`);
    await sleep(SPLIT);
    s.end(`${(PADDED.length - half).toString(16)}\r\n${PADDED.slice(half)}\r\n0\r\n\r\n`);
  },
  "split-to-eof": async (s) => {
    s.write(head("401 Unauthorized", JSON_TYPE));
    await sleep(SPLIT);
    s.end(PADDED);
  },
  "past-the-cap": async (s) => {
    s.write(head("400 Bad Request", { ...JSON_TYPE, "content-length": Buffer.byteLength(PAST_THE_CAP) }));
    await sleep(SPLIT);
    s.end(PAST_THE_CAP);
  },
  stalled: (s) => {
    s.write(`${head("500 Internal Server Error", { ...JSON_TYPE, "content-length": 1000 })}{"error":{"mess`);
  },
  "cut-off": async (s) => {
    s.write(`${head("500 Internal Server Error", { ...JSON_TYPE, "content-length": 1000 })}{"error":{"mess`);
    await sleep(SPLIT);
    s.destroy();
  },
  "cut-off-chunked": async (s) => {
    s.write(`${head("500 Internal Server Error", { ...JSON_TYPE, "transfer-encoding": "chunked" })}23\r\n{"error":{"mess`);
    await sleep(SPLIT);
    s.destroy();
  },
  "stalled-long-budget": (s) => {
    s.write(`${head("500 Internal Server Error", { ...JSON_TYPE, "content-length": 1000 })}{"error":{"mess`);
  },
  "stalled-no-budget": (s) => {
    s.write(`${head("500 Internal Server Error", { ...JSON_TYPE, "content-length": 1000 })}{"error":{"mess`);
  },
  "with-headers": async (s) => {
    const body = envelope("sentinel recorded text");
    s.write(
      head("403 Forbidden", {
        ...JSON_TYPE,
        "content-length": Buffer.byteLength(body),
        "x-request-id": "req-sentinel-refused",
        "x-sentinel-region": "us-sentinel",
        "x-not-listed": "never-recorded",
      }),
    );
    await sleep(SPLIT);
    s.end(body);
  },
  silent: () => {},
};

// The sequences measured against OpenAI's Realtime endpoint.
type Step = { frame: unknown } | { close: [number, string] } | { wait: number } | "drop";
const eventOf = (type: string, code: string, message: string) => ({
  type: "error",
  event_id: "event_sentinel",
  error: { type, code, message, param: null, event_id: null },
});
const WRONG_KEY = eventOf("invalid_request_error", "invalid_api_key", "Incorrect API key provided: sentinel.");
const BAD_MODEL = eventOf("invalid_request_error", "invalid_model", "Model sentinel is not supported.");
const MODEL_NOT_FOUND = eventOf(
  "invalid_request_error",
  "model_not_found",
  "The model `sentinel` does not exist or you do not have access to it.",
);
const RATE_LIMITED = eventOf("rate_limit_error", "rate_limit_exceeded", "Slow down, sentinel.");
const NO_QUOTA = eventOf("insufficient_quota", "insufficient_quota", "Out of quota, sentinel.");
const OTHER = eventOf("server_error", "sentinel_unknown", "Something broke, sentinel.");
const RECOVERABLE = eventOf("invalid_request_error", "invalid_json", "Invalid event, sentinel.");
const CREATED = { type: "session.created", session: { id: "sess_1", model: "m" } };
const SESSIONS: Record<string, Step[]> = {
  "wrong-key": [{ frame: WRONG_KEY }, { wait: 200 }, { close: [3000, "invalid_request_error.invalid_api_key"] }],
  "bad-model": [{ frame: BAD_MODEL }, { wait: 200 }, { close: [4000, "invalid_request_error.invalid_model"] }],
  "model-not-found": [{ frame: MODEL_NOT_FOUND }, { wait: 200 }, { close: [4004, "invalid_request_error.model_not_found"] }],
  "other-invalid-request": [{ frame: RECOVERABLE }, { close: [4000, "invalid_request_error.invalid_json"] }],
  "rate-limited": [{ frame: RATE_LIMITED }, { close: [4000, "rate_limit_error.rate_limit_exceeded"] }],
  "no-quota": [{ frame: NO_QUOTA }, { close: [4000, "insufficient_quota"] }],
  "other-error": [{ frame: OTHER }, { close: [4000, "server_error"] }],
  "error-then-normal-close": [{ frame: BAD_MODEL }, { close: [1000, ""] }],
  "error-then-drop": [{ frame: BAD_MODEL }, "drop"],
  "wrong-key-slow-close": [{ frame: WRONG_KEY }, { wait: 1500 }, { close: [3000, "invalid_request_error.invalid_api_key"] }],
  "no-frames-close": [{ close: [4000, "no frames"] }],
  "recoverable-then-client-close": [{ frame: CREATED }, { frame: RECOVERABLE }],
  "usable-then-error-close": [{ frame: CREATED }, { frame: RECOVERABLE }, { close: [4000, "sentinel_late"] }],
  "error-before-created": [{ frame: RECOVERABLE }, { frame: CREATED }, { close: [1000, ""] }],
  "clean-session": [{ frame: CREATED }, { close: [1000, ""] }],
  "drop-no-frames": ["drop"],
};

// Text and close frames as a server sends them, for an upstream that must answer
// only after it has seen the gateway's own close.
const wsFrame = (opcode: number, payload: Buffer) => {
  const header =
    payload.length < 126
      ? Buffer.from([0x80 | opcode, payload.length])
      : Buffer.from([0x80 | opcode, 126, payload.length >> 8, payload.length & 0xff]);
  return Buffer.concat([header, payload]);
};
const wsText = (value: unknown) => wsFrame(1, Buffer.from(JSON.stringify(value)));
const wsClose = wsFrame(8, Buffer.from([0x03, 0xe8]));

describe("a realtime session the upstream never let start", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;
  const tcpServers: TcpServer[] = [];
  const wsServers: WebSocketServer[] = [];
  const modelIds: Record<string, string> = {};

  async function scriptedUpstream(script: Script): Promise<string> {
    const server = createTcpServer((socket) => {
      socket.on("error", () => {});
      let seen = "";
      socket.on("data", (chunk) => {
        const before = seen;
        seen += chunk.toString();
        if (!before.includes("\r\n\r\n") && seen.includes("\r\n\r\n")) void script(socket);
      });
    });
    tcpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  // Sends `first` on upgrade and `afterClose` once the gateway closes its half.
  async function drainUpstream(first: unknown[], afterClose: unknown[]): Promise<string> {
    const server = createTcpServer((socket) => {
      socket.on("error", () => {});
      let upgraded = false;
      let seen = "";
      socket.on("data", (chunk) => {
        if (upgraded) {
          for (const frame of afterClose) socket.write(wsText(frame));
          socket.end(wsClose);
          return;
        }
        seen += chunk.toString("latin1");
        if (!seen.includes("\r\n\r\n")) return;
        upgraded = true;
        const key = /sec-websocket-key: (.+)\r\n/i.exec(seen)![1];
        const accept = createHash("sha1").update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest("base64");
        socket.write(
          head("101 Switching Protocols", {
            upgrade: "websocket",
            connection: "Upgrade",
            "sec-websocket-accept": accept,
          }),
        );
        for (const frame of first) socket.write(wsText(frame));
      });
    });
    tcpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  async function sessionUpstream(steps: Step[]): Promise<string> {
    const server = new WebSocketServer({ port: 0, host: "127.0.0.1" });
    server.on("headers", (headers) => {
      headers.push("x-request-id: req-sentinel-session", "x-sentinel-region: us-sentinel");
    });
    server.on("connection", (ws) => {
      ws.on("error", () => {});
      void (async () => {
        for (const step of steps) {
          if (step === "drop") ws.terminate();
          else if ("frame" in step) ws.send(JSON.stringify(step.frame));
          else if ("wait" in step) await sleep(step.wait);
          else ws.close(...step.close);
        }
      })();
    });
    wsServers.push(server);
    await new Promise<void>((r) => server.once("listening", () => r()));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    app = await spawnApp({
      logLevel: "info",
      extra: { upstream: { connect_timeout_ms: CONNECT_TIMEOUT_MS } },
      extraEnv: { SLS_CRED_MOCK_AK_ID: "LTAI_mock_ak", SLS_CRED_MOCK_AK_SECRET: "mock_ak_secret" },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [logstore, content_mode] of [
      [FULL, "full"],
      [META, "metadata_only"],
    ] as const) {
      await seed.createObservabilityExporter({
        name: logstore,
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: "aisix-e2e-obs",
        logstore,
        credential_ref: "mock",
        content_mode,
        upstream_response_headers: ["x-sentinel-region"],
      });
    }
    const names: string[] = [];
    const model = async (
      name: string,
      base: string,
      cooldown: Record<string, unknown> = { enabled: false },
      budget: Record<string, number> = { timeout: READ_BUDGET_MS, stream_timeout: READ_BUDGET_MS },
    ) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        provider: "openai",
        api_base: `${base}/v1`,
      });
      modelIds[name] = (
        await seed.createModel({
          display_name: name,
          provider: "openai",
          model_name: "m",
          provider_key_id: pk.id,
          ...budget,
          cooldown,
        })
      ).id;
      names.push(name);
    };
    const budgets: Record<string, Record<string, number>> = {
      "stalled-long-budget": { timeout: 120_000, stream_timeout: 120_000 },
      "stalled-no-budget": { timeout: 0 },
    };
    for (const [name, script] of Object.entries(REFUSALS)) {
      await model(name, await scriptedUpstream(script), { enabled: false }, budgets[name]);
    }
    await model("drained-refusal", await drainUpstream([], [WRONG_KEY]));
    await model("drained-usable", await drainUpstream([RECOVERABLE], [CREATED]));
    for (const [name, steps] of Object.entries(SESSIONS)) await model(name, await sessionUpstream(steps));
    const silentBase = await scriptedUpstream(REFUSALS.silent);
    await model("silent-timeout-gate-only", silentBase, { enabled: true, trigger_on_transport: false });
    await model("silent-timeout-gate-off", silentBase, { enabled: true, trigger_on_timeout: false });
    await model("wrong-key-cooled", await sessionUpstream(SESSIONS["wrong-key"]), { enabled: true });
    await model("bad-model-cooled", await sessionUpstream(SESSIONS["bad-model"]), { enabled: true });
    await model("other-error-no-transport-gate", await sessionUpstream(SESSIONS["other-error"]), {
      enabled: true,
      trigger_on_transport: false,
    });
    await seed.createApiKey({ key_hash: sha256(CALLER), allowed_routes: ["*"], allowed_models: names });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      const { data } = (await res.json()) as { data?: unknown[] };
      return data?.length === names.length;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    for (const s of tcpServers) s.close();
    for (const s of wsServers) s.close();
  });

  type Session = { requestId: string; frames: unknown[]; code: number; reason: string; elapsed: number };
  const session = (
    model: string,
    onFrame?: (ws: WebSocket) => void,
    onOpen?: (ws: WebSocket) => void,
  ): Promise<Session> =>
    new Promise((resolve, reject) => {
      const started = Date.now();
      const ws = new WebSocket(`${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      let requestId = "";
      const frames: unknown[] = [];
      ws.on("upgrade", (res) => {
        requestId = String(res.headers["x-aisix-request-id"] ?? "");
      });
      ws.on("open", () => onOpen?.(ws));
      ws.on("message", (data) => {
        frames.push(JSON.parse(String(data)));
        onFrame?.(ws);
      });
      ws.on("close", (code, reason) =>
        resolve({ requestId, frames, code, reason: String(reason), elapsed: Date.now() - started }),
      );
      ws.on("error", reject);
    });

  const row = (requestId: string, logstore = FULL) =>
    waitForSlsLog(
      sls!,
      logstore,
      (l) => l.get("request_id") === requestId && l.get("is_final_attempt") === "true",
      `the realtime row in ${logstore}`,
    );
  const chat = async (model: string) => {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
    });
    return {
      status: res.status,
      body: (await res.json()) as { error: Record<string, unknown> },
      requestId: res.headers.get("x-aisix-request-id")!,
    };
  };
  const modelStatus = async (name: string) => {
    const rows = await new AdminClient(app!.adminUrl, app!.adminKey, app!.metricsUrl).listModelStatuses();
    return rows.find((r) => r.id === modelIds[name])!;
  };
  const metric = async (name: string, labels: Record<string, string>) =>
    sumMetric(await scrapeMetrics(app!.metricsUrl), name, labels);

  describe("a refused handshake whose body arrives after its head", () => {
    test.each(["split-length", "split-chunked", "split-to-eof"])(
      "%s reaches the caller with the upstream's text, code and param",
      async (model) => {
        if (!etcdReachable) return;
        const got = await session(model);
        expect(got.frames).toEqual([
          {
            type: "error",
            error: {
              type: "upstream_error",
              message: "sentinel split text",
              code: "sentinel_code",
              param: "sentinel_param",
            },
          },
        ]);
        expect(got.code).toBe(1011);
        expect(got.reason).toBe("upstream connect failed");
        expect((await row(got.requestId)).get("error_message")).toBe("upstream returned HTTP 401: sentinel split text");
      },
    );

    test("the frame carries what chat answers on the same upstream", async () => {
      if (!etcdReachable) return;
      const [http, ws] = [await chat("split-length"), await session("split-length")];
      expect((ws.frames[0] as { error: unknown }).error).toEqual(http.body.error);
    });

    test("a body past the bound reads as chat reads it", async () => {
      if (!etcdReachable) return;
      const [http, ws] = [await chat("past-the-cap"), await session("past-the-cap")];
      expect(http.body.error.message).toBe(`${PAST_THE_CAP.slice(0, 1024)}…`);
      expect((ws.frames[0] as { error: unknown }).error).toEqual(http.body.error);
    });

    test("a body that stalls ends at the model's stream budget, with the status fallback", async () => {
      if (!etcdReachable) return;
      const got = await session("stalled");
      expect(got.frames).toEqual([
        { type: "error", error: { type: "upstream_error", message: "upstream returned 500 Internal Server Error" } },
      ]);
      expect(got.code).toBe(1011);
      expect(got.elapsed).toBeGreaterThanOrEqual(READ_BUDGET_MS - 100);
      expect(got.elapsed).toBeLessThan(READ_BUDGET_MS + 4000);
      const failure = await row(got.requestId);
      expect(failure.get("status_code")).toBe("502");
      expect(failure.get("error_class")).toBe("upstream_status");
      expect(failure.get("error_message")).toBe(
        `upstream returned HTTP 500: failed to read response body: timed out after ${READ_BUDGET_MS}ms`,
      );
    });

    test("a body cut off mid-read gets the status fallback and names the failed read", async () => {
      if (!etcdReachable) return;
      const got = await session("cut-off");
      expect(got.frames).toEqual([
        { type: "error", error: { type: "upstream_error", message: "upstream returned 500 Internal Server Error" } },
      ]);
      expect((await row(got.requestId)).get("error_message")).toMatch(
        /^upstream returned HTTP 500: failed to read response body: .+/,
      );
    });
  });

  describe("a client that leaves while the body stalls", () => {
    test.each(["stalled-long-budget", "stalled-no-budget"])("%s ends the read", async (model) => {
      if (!etcdReachable) return;
      const got = await session(model, undefined, (ws) => ws.close(1000));
      const failure = await row(got.requestId);
      expect(failure.get("error_message")).toBe(
        "upstream returned HTTP 500: failed to read response body: the client closed the session",
      );
    });
  });

  describe("the failure row of a refused handshake", () => {
    test("holds what arrived of a body cut short, without its framing", async () => {
      if (!etcdReachable) return;
      expect((await row((await session("cut-off-chunked")).requestId)).get("response")).toBe('{"error":{"mess');
      expect((await row((await session("stalled")).requestId)).get("response")).toBe('{"error":{"mess');
    });

    test("says the upstream was reached and how long it took", async () => {
      if (!etcdReachable) return;
      const failure = await row((await session("with-headers")).requestId);
      expect(Number(failure.get("upstream_latency_ms"))).toBeGreaterThanOrEqual(SPLIT - 50);
    });

    test("records the upstream's response and headers as chat's row does", async () => {
      if (!etcdReachable) return;
      const [http, ws] = [await chat("with-headers"), await session("with-headers")];
      const [chatRow, wsRow] = [await row(http.requestId), await row(ws.requestId)];
      expect(wsRow.get("response")).toBe(envelope("sentinel recorded text"));
      for (const field of ["response", "provider_request_id", "upstream_response_headers"]) {
        expect(wsRow.get(field), field).toBe(chatRow.get(field));
      }
      expect(wsRow.get("provider_request_id")).toBe("req-sentinel-refused");
      expect(JSON.parse(wsRow.get("upstream_response_headers")!)).toEqual({ "x-sentinel-region": "us-sentinel" });
      await waitForLogLine(
        app!,
        (l) => l.includes(ws.requestId) && l.includes("req-sentinel-refused"),
        "the access-log line naming the provider's request id",
      );
    });

    test("leaves the body out of a metadata_only export and keeps its headers", async () => {
      if (!etcdReachable) return;
      const ws = await session("with-headers");
      const meta = await row(ws.requestId, META);
      expect(meta.get("response")).toBeUndefined();
      expect(meta.get("provider_request_id")).toBe("req-sentinel-refused");
      expect(JSON.parse(meta.get("upstream_response_headers")!)).toEqual({ "x-sentinel-region": "us-sentinel" });
    });
  });

  describe("a dial past upstream.connect_timeout_ms", () => {
    test("records a timeout and 504, and the caller keeps the fixed frame", async () => {
      if (!etcdReachable) return;
      const got = await session("silent");
      expect(got.frames).toEqual([
        {
          type: "error",
          error: { type: "upstream_error", message: "failed to connect to the upstream realtime endpoint" },
        },
      ]);
      expect(got.code).toBe(1011);
      expect(got.reason).toBe("upstream connect failed");
      const failure = await row(got.requestId);
      expect(failure.get("status_code")).toBe("504");
      expect(failure.get("error_class")).toBe("timeout");
      expect(failure.get("error_message")).toMatch(/^upstream request timed out after \d+ms: /);
      expect(await metric("aisix_requests_total", { model: "silent", status: "504" })).toBeGreaterThan(0);
    });

    test("cools the model down through the timeout gate", async () => {
      if (!etcdReachable) return;
      await session("silent-timeout-gate-only");
      const cooled = await modelStatus("silent-timeout-gate-only");
      expect(cooled.status).toBe("cooldown");
      expect(cooled.status_reason).toBe("request_timeout");
    });

    test("trigger_on_timeout: false keeps the model available", async () => {
      if (!etcdReachable) return;
      await session("silent-timeout-gate-off");
      expect((await modelStatus("silent-timeout-gate-off")).status).not.toBe("cooldown");
    });
  });

  describe("an upstream that reports an error and closes before the session is usable", () => {
    test.each<[string, typeof WRONG_KEY, number, string, number, string]>([
      ["wrong-key", WRONG_KEY, 401, "client_error", 3000, "invalid_request_error.invalid_api_key"],
      ["bad-model", BAD_MODEL, 404, "client_error", 4000, "invalid_request_error.invalid_model"],
      ["model-not-found", MODEL_NOT_FOUND, 404, "client_error", 4004, "invalid_request_error.model_not_found"],
      ["other-invalid-request", RECOVERABLE, 400, "client_error", 4000, "invalid_request_error.invalid_json"],
      ["rate-limited", RATE_LIMITED, 429, "rate_limited", 4000, "rate_limit_error.rate_limit_exceeded"],
      ["no-quota", NO_QUOTA, 429, "rate_limited", 4000, "insufficient_quota"],
      ["other-error", OTHER, 502, "upstream_error", 4000, "server_error"],
    ])("%s is recorded as a failure and relayed as sent", async (model, event, status, outcome, code, reason) => {
      if (!etcdReachable) return;
      const got = await session(model);
      expect(got.frames).toEqual([event]);
      expect(got.code).toBe(code);
      expect(got.reason).toBe(reason);
      const failure = await row(got.requestId);
      expect(failure.get("status_code")).toBe(String(status));
      expect(failure.get("error_class")).toBe("upstream_in_band");
      expect(failure.get("error_message")).toBe(`upstream reported an in-band error: ${event.error.message}`);
      expect(JSON.parse(failure.get("response")!)).toEqual(event);
      expect(failure.get("provider_request_id")).toBe("req-sentinel-session");
      expect(JSON.parse(failure.get("upstream_response_headers")!)).toEqual({ "x-sentinel-region": "us-sentinel" });
      await waitForLogLine(
        app!,
        (l) => l.includes(got.requestId) && l.includes(`status_code=${status}`) && l.includes(event.error.message),
        "the access-log line",
      );
      expect(
        await metric("aisix_proxy_requests_total", { endpoint: "/v1/realtime", status: String(status) }),
      ).toBeGreaterThan(0);
      expect(await metric("aisix_requests_total", { model, status: String(status), outcome })).toBeGreaterThan(0);
      expect(await metric("aisix_deployment_success_responses_total", { model })).toBe(0);
      expect(await metric("aisix_deployment_failure_responses_total", { model })).toBe(1);
      const meta = await row(got.requestId, META);
      expect(meta.get("status_code")).toBe(String(status));
      expect(meta.get("response")).toBeUndefined();
    });

    test("an error followed by a normal close is a failure too", async () => {
      if (!etcdReachable) return;
      const got = await session("error-then-normal-close");
      expect(got.frames).toEqual([BAD_MODEL]);
      expect(got.code).toBe(1000);
      expect((await row(got.requestId)).get("status_code")).toBe("404");
    });

    test("an error followed by a dropped connection adds no gateway frame", async () => {
      if (!etcdReachable) return;
      const got = await session("error-then-drop");
      expect(got.frames).toEqual([BAD_MODEL]);
      expect(got.code).toBe(1011);
      const failure = await row(got.requestId);
      expect(failure.get("status_code")).toBe("404");
      expect(failure.get("error_class")).toBe("upstream_in_band");
    });

    test("a caller that hangs up on the error event does not turn the failure into a success", async () => {
      if (!etcdReachable) return;
      const got = await session("wrong-key-slow-close", (ws) => ws.close(1000));
      expect(got.frames).toEqual([WRONG_KEY]);
      const failure = await row(got.requestId);
      expect(failure.get("status_code")).toBe("401");
      expect(failure.get("error_class")).toBe("upstream_in_band");
    });

    test("an error that arrives after the caller hung up still counts", async () => {
      if (!etcdReachable) return;
      const got = await session("drained-refusal", undefined, (ws) => ws.close(1000));
      expect((await row(got.requestId)).get("status_code")).toBe("401");
    });

    test("a status on the trigger list cools the model down, named for the status", async () => {
      if (!etcdReachable) return;
      await session("wrong-key-cooled");
      const cooled = await modelStatus("wrong-key-cooled");
      expect(cooled.status).toBe("cooldown");
      expect(cooled.status_reason).toBe("upstream_auth_failure");
    });

    test("a status off the trigger list does not", async () => {
      if (!etcdReachable) return;
      await session("bad-model-cooled");
      expect((await modelStatus("bad-model-cooled")).status).not.toBe("cooldown");
    });

    test("an error with no status follows the transport gate", async () => {
      if (!etcdReachable) return;
      await session("other-error-no-transport-gate");
      expect((await modelStatus("other-error-no-transport-gate")).status).not.toBe("cooldown");
    });
  });

  describe("what stays as it was", () => {
    test("an error event in a usable session is no failure", async () => {
      if (!etcdReachable) return;
      const got = await session("recoverable-then-client-close", (ws) => {
        if (ws.readyState === WebSocket.OPEN) setTimeout(() => ws.close(1000), 200);
      });
      expect(got.frames).toEqual([{ ...CREATED, session: { ...CREATED.session, model: "recoverable-then-client-close" } }, RECOVERABLE]);
      const ended = await row(got.requestId);
      expect(ended.get("status_code")).toBe("200");
      expect(ended.get("error_class")).toBeFalsy();
    });

    test("a usable session that ends on an error close keeps its 200 and the gateway frame", async () => {
      if (!etcdReachable) return;
      const got = await session("usable-then-error-close");
      expect(got.frames).toEqual([
        { ...CREATED, session: { ...CREATED.session, model: "usable-then-error-close" } },
        RECOVERABLE,
        {
          type: "error",
          error: { type: "server_error", message: "upstream websocket closed with code 4000: sentinel_late" },
        },
      ]);
      expect(got.code).toBe(4000);
      const ended = await row(got.requestId);
      expect(ended.get("status_code")).toBe("200");
      expect(ended.get("error_class")).toBeFalsy();
      expect(ended.get("provider_request_id")).toBeUndefined();
    });

    test("an error that a non-error frame then follows leaves the session usable", async () => {
      if (!etcdReachable) return;
      const got = await session("error-before-created");
      expect(got.frames).toHaveLength(2);
      expect((await row(got.requestId)).get("status_code")).toBe("200");
    });

    test("a usable frame that arrives after the caller hung up on an error still counts", async () => {
      if (!etcdReachable) return;
      const got = await session("drained-usable", (ws) => ws.close(1000));
      const ended = await row(got.requestId);
      expect(ended.get("status_code")).toBe("200");
      expect(ended.get("error_class")).toBeFalsy();
    });

    test("a connection dropped before any frame counts as a deployment failure", async () => {
      if (!etcdReachable) return;
      const got = await session("drop-no-frames");
      expect((await row(got.requestId)).get("status_code")).toBe("502");
      expect(await metric("aisix_deployment_success_responses_total", { model: "drop-no-frames" })).toBe(0);
      expect(await metric("aisix_deployment_failure_responses_total", { model: "drop-no-frames" })).toBe(1);
    });

    test("a close with no frames at all keeps its 200 and the gateway frame", async () => {
      if (!etcdReachable) return;
      const got = await session("no-frames-close");
      expect(got.frames).toEqual([
        { type: "error", error: { type: "server_error", message: "upstream websocket closed with code 4000: no frames" } },
      ]);
      expect((await row(got.requestId)).get("status_code")).toBe("200");
      expect(await metric("aisix_deployment_success_responses_total", { model: "no-frames-close" })).toBe(1);
    });

    test("a clean session counts one deployment success", async () => {
      if (!etcdReachable) return;
      await row((await session("clean-session")).requestId);
      expect(await metric("aisix_deployment_success_responses_total", { model: "clean-session" })).toBe(1);
    });
  });
});
