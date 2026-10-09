import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MetricSample,
  type MockSls,
  type OpenAiUpstream,
  type OpenAiUpstreamOptions,
  type SpawnedApp,
} from "../harness/index.js";

// A streamed response's request families — `aisix_proxy_requests_total`,
// `aisix_proxy_failed_requests_total`, `aisix_llm_requests_total` and their
// duration histograms — are recorded when the STREAM ends, under the status
// it ended on: 200 delivered, 499 the caller walked away, 502 the upstream
// broke after the head, 504 a read timeout. The same status the request's
// usage event and access-log line carry. Recording them when the response
// head went out counted every one of those as a 200 success.
//
// The end-of-stream latency histograms carry the same status, and the
// access log's `error_kind` and the usage event's `error_class` name a
// failure with one vocabulary, whatever the response shape.

const CALLER_PLAINTEXT = "sk-stream-terminal-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "mock";
const LOGSTORE = "stream-terminal-events";

const UPSTREAM_MODEL = "gpt-4o-mini";

function chunk(content: string, finish: string | null = null): string {
  return JSON.stringify({
    id: "chatcmpl-terminal",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: UPSTREAM_MODEL,
    choices: [{ index: 0, delta: { content }, finish_reason: finish }],
    ...(finish ? { usage: { prompt_tokens: 3, completion_tokens: 3, total_tokens: 6 } } : {}),
  });
}

const FULL_STREAM = [chunk("one "), chunk("two "), chunk("three", "stop"), "[DONE]"];

/** One model per scenario, each on its own upstream, so no assertion can
 *  pick up another scenario's samples. */
const SCENARIOS: Record<string, OpenAiUpstreamOptions & { model?: Record<string, unknown> }> = {
  // Delivered to the end; slow enough between events that a caller can
  // read the first and hang up with the rest still coming.
  "st-ok": { streamEvents: FULL_STREAM, eventDelayMs: 300 },
  "st-cancel": { streamEvents: FULL_STREAM, eventDelayMs: 500 },
  // Two events, then the upstream drops the connection — after the
  // gateway's 200 head is on the wire.
  "st-break": { streamEvents: FULL_STREAM, eventDelayMs: 100, disconnectAfterEvents: 2 },
  // First event fast, then a gap longer than the model's stream read
  // timeout.
  "st-stall": {
    streamEvents: FULL_STREAM,
    eventDelayMs: 3_000,
    model: { stream_timeout: 400 },
  },
  "ns-timeout": {
    responseDelayMs: 3_000,
    nonStreamBody: { id: "late" },
    model: { timeout: 300 },
  },
  // An upstream that answers 500 before any head, once streamed and once
  // not — the two must be named alike.
  "ns-500": { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
  "st-500": { status: 500, errorBody: { error: { message: "boom", type: "server_error" } } },
  // For the non-streamed cancel: the upstream is still thinking when the
  // caller leaves.
  "ns-slow": { responseDelayMs: 30_000, nonStreamBody: { id: "never" } },
  // The /v1/messages family over the same kinds of stream.
  "msg-ok": { streamEvents: FULL_STREAM, eventDelayMs: 300 },
  "msg-cancel": { streamEvents: FULL_STREAM, eventDelayMs: 500 },
  "msg-break": { streamEvents: FULL_STREAM, eventDelayMs: 100, disconnectAfterEvents: 2 },
};

const CHAT = "/v1/chat/completions";
const MESSAGES = "/v1/messages";

describe("streamed responses record their terminal status", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: Record<string, OpenAiUpstream> = {};

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    for (const [name, opts] of Object.entries(SCENARIOS)) {
      const { model: _model, ...upstreamOpts } = opts;
      upstreams[name] = await startOpenAiUpstream(upstreamOpts);
    }
    sls = await startMockSls();
    app = await spawnApp({
      // The access-log line is `tracing::info!`.
      logLevel: "info",
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-stream-terminal",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    for (const [name, opts] of Object.entries(SCENARIOS)) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstreams[name].baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: UPSTREAM_MODEL,
        provider_key_id: pk.id,
        // A failure must not take the model out of rotation for the next
        // scenario that reuses nothing but its config.
        cooldown: { enabled: false },
        ...(opts.model ?? {}),
      });
    }
    // Seeded last: it authenticating implies everything above is loaded.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await Promise.all(Object.values(upstreams).map((u) => u.close()));
  });

  const ready = () => etcdReachable && app && sls;

  function post(path: string, model: string, stream: boolean, signal?: AbortSignal) {
    const body =
      path === MESSAGES
        ? { model, max_tokens: 64, messages: [{ role: "user", content: "hi" }], stream }
        : { model, messages: [{ role: "user", content: "hi" }], stream };
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    });
  }

  /** Open a stream, read one frame, then hang up with the rest coming. */
  async function abandonAfterFirstFrame(path: string, model: string): Promise<string> {
    const controller = new AbortController();
    const res = await post(path, model, true, controller.signal);
    expect(res.status, "the head must go out — this is the mid-stream ending").toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    const reader = res.body!.getReader();
    const first = await reader.read();
    expect(first.done, "the stream delivered nothing to abandon").toBe(false);
    controller.abort();
    await reader.cancel().catch((err: unknown) => {
      if (!(err instanceof Error) || err.name !== "AbortError") throw err;
    });
    return requestId;
  }

  /** Read a response to its end, tolerating a body the gateway cut short. */
  async function drain(res: Response): Promise<void> {
    await res.arrayBuffer().catch(() => undefined);
  }

  const scrape = () => scrapeMetrics(app!.metricsUrl);

  /** Wait until `name{want}` has grown by `expected` since `before`, then
   *  return a scrape taken after a settle window, so a second recording of
   *  the same request — the bug this suite pins — has had time to land. */
  async function settledAfter(
    before: MetricSample[],
    name: string,
    want: Record<string, string>,
    expected: number,
  ): Promise<MetricSample[]> {
    const deadline = Date.now() + 10_000;
    for (;;) {
      const now = await scrape();
      if (metricDelta(before, now, name, want) >= expected) break;
      if (Date.now() > deadline) {
        throw new Error(`timed out waiting for ${name} ${JSON.stringify(want)} +${expected}`);
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    await new Promise((r) => setTimeout(r, 500));
    return scrape();
  }

  /** The request families of one model's requests, as a status → delta map. */
  function requestDeltas(
    before: MetricSample[],
    after: MetricSample[],
    family: string,
    endpoint: string,
    model: string,
  ): Record<string, number> {
    const out: Record<string, number> = {};
    for (const s of after) {
      if (s.name !== family || s.labels.endpoint !== endpoint || s.labels.model !== model) {
        continue;
      }
      const delta = metricDelta(before, after, family, {
        endpoint,
        model,
        status: s.labels.status,
      });
      if (delta !== 0) out[s.labels.status] = delta;
    }
    return out;
  }

  function expectRequestFamilies(
    before: MetricSample[],
    after: MetricSample[],
    endpoint: string,
    model: string,
    status: number,
  ) {
    const want = { [String(status)]: 1 };
    for (const family of ["aisix_proxy_requests_total", "aisix_llm_requests_total"]) {
      expect(requestDeltas(before, after, family, endpoint, model), family).toEqual(want);
    }
    expect(
      requestDeltas(before, after, "aisix_proxy_failed_requests_total", endpoint, model),
      "aisix_proxy_failed_requests_total",
    ).toEqual(status === 200 ? {} : want);
  }

  /** The request's own end-of-stream latency observations, by status class. */
  function expectLatencyClass(
    before: MetricSample[],
    after: MetricSample[],
    endpoint: string,
    model: string,
    statusClass: string,
  ) {
    for (const series of ["aisix_request_e2e_latency_seconds_count", "aisix_request_ttft_seconds_count"]) {
      const at = (cls: string) =>
        metricDelta(before, after, series, { endpoint, model, side: "downstream", status_class: cls });
      expect(at(statusClass), `${series} status_class=${statusClass}`).toBe(1);
      if (statusClass !== "2xx") {
        expect(at("2xx"), `${series}: a failed stream observed as a success`).toBe(0);
      }
    }
  }

  for (const [endpoint, prefix] of [
    [CHAT, "st"],
    [MESSAGES, "msg"],
  ] as const) {
    describe(endpoint, () => {
      test("a stream delivered to its end records 200, timed over the whole stream", async (ctx) => {
        if (!ready()) return ctx.skip();
        const model = `${prefix}-ok`;
        const before = await scrape();
        const res = await post(endpoint, model, true);
        expect(res.status).toBe(200);
        await drain(res);
        const after = await settledAfter(before, "aisix_proxy_requests_total", { endpoint, model }, 1);
        expectRequestFamilies(before, after, endpoint, model, 200);
        // Four events 300ms apart: a duration taken at the head would be a
        // few milliseconds.
        const seconds = metricDelta(before, after, "aisix_proxy_request_duration_seconds_sum", {
          endpoint,
          model,
        });
        expect(seconds, "the duration must span the stream").toBeGreaterThanOrEqual(0.6);
        expectLatencyClass(before, after, endpoint, model, "2xx");
      });

      test("a stream the caller abandons records 499, once", async (ctx) => {
        if (!ready()) return ctx.skip();
        const model = `${prefix}-cancel`;
        const before = await scrape();
        const requestId = await abandonAfterFirstFrame(endpoint, model);
        const after = await settledAfter(before, "aisix_proxy_requests_total", { endpoint, model }, 1);
        expectRequestFamilies(before, after, endpoint, model, 499);
        expectLatencyClass(before, after, endpoint, model, "4xx");
        const row = await waitForSlsLog(
          sls!,
          LOGSTORE,
          (log) => log.get("request_id") === requestId,
          `the usage row for ${requestId}`,
        );
        expect(row.get("status_code")).toBe("499");
        expect(row.get("error_class")).toBe("client_disconnected");
      });

      test("a stream the upstream breaks after the head records 502", async (ctx) => {
        if (!ready()) return ctx.skip();
        const model = `${prefix}-break`;
        const before = await scrape();
        const res = await post(endpoint, model, true);
        expect(res.status, "the head went out before the upstream broke").toBe(200);
        const requestId = res.headers.get("x-aisix-request-id") ?? "";
        await drain(res);
        const after = await settledAfter(before, "aisix_proxy_requests_total", { endpoint, model }, 1);
        expectRequestFamilies(before, after, endpoint, model, 502);
        expectLatencyClass(before, after, endpoint, model, "5xx");
        const row = await waitForSlsLog(
          sls!,
          LOGSTORE,
          (log) => log.get("request_id") === requestId,
          `the usage row for ${requestId}`,
        );
        expect(row.get("status_code")).toBe("502");
        const line = await waitForLogLine(
          app!,
          (l) => l.includes(`request_id="${requestId}"`) && l.includes("proxy request completed"),
          `the access-log line for ${requestId}`,
        );
        expect(line).toContain("status=502");
        expect(line).toContain(`error_kind="${row.get("error_class")}"`);
      });
    });
  }

  test("a stream that hits its read timeout records 504 / timeout on every record", async (ctx) => {
    if (!ready()) return ctx.skip();
    const model = "st-stall";
    const before = await scrape();
    const res = await post(CHAT, model, true);
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    await drain(res);
    const after = await settledAfter(before, "aisix_proxy_requests_total", { endpoint: CHAT, model }, 1);
    expectRequestFamilies(before, after, CHAT, model, 504);
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `the usage row for ${requestId}`,
    );
    expect(row.get("status_code")).toBe("504");
    expect(row.get("error_class")).toBe("timeout");
    const line = await waitForLogLine(
      app!,
      (l) => l.includes(`request_id="${requestId}"`) && l.includes("proxy request completed"),
      `the access-log line for ${requestId}`,
    );
    expect(line).toContain("status=504");
    expect(line).toContain('error_kind="timeout"');
  });

  test("a non-streamed upstream timeout is 504 / timeout on the line and the row", async (ctx) => {
    if (!ready()) return ctx.skip();
    const res = await post(CHAT, "ns-timeout", false);
    expect(res.status).toBe(504);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    await drain(res);
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `the usage row for ${requestId}`,
    );
    expect(row.get("status_code")).toBe("504");
    expect(row.get("error_class")).toBe("timeout");
    const line = await waitForLogLine(
      app!,
      (l) => l.includes(`request_id="${requestId}"`) && l.includes("proxy request completed"),
      `the access-log line for ${requestId}`,
    );
    expect(line).toContain("status=504");
    expect(line).toContain('error_kind="timeout"');
  });

  test("an upstream 5xx is named alike streamed and not, and the client envelope is unchanged", async (ctx) => {
    if (!ready()) return ctx.skip();
    for (const [model, stream] of [
      ["ns-500", false],
      ["st-500", true],
    ] as const) {
      const res = await post(CHAT, model, stream);
      expect(res.status, model).toBe(502);
      const requestId = res.headers.get("x-aisix-request-id") ?? "";
      const body = (await res.json()) as { error?: { type?: string } };
      // Public API: the envelope keeps its own type.
      expect(body.error?.type, model).toBe("upstream_error");
      const row = await waitForSlsLog(
        sls!,
        LOGSTORE,
        (log) => log.get("request_id") === requestId,
        `the usage row for ${requestId}`,
      );
      expect(row.get("error_class"), model).toBe("upstream_status");
      const line = await waitForLogLine(
        app!,
        (l) => l.includes(`request_id="${requestId}"`) && l.includes("proxy request completed"),
        `the access-log line for ${requestId}`,
      );
      expect(line, model).toContain('error_kind="upstream_status"');
    }
  });

  // What the upstream sees when the caller leaves: the gateway closes the
  // upstream connection rather than holding it open and reading a response
  // nobody will receive.
  test("a caller that leaves mid-stream closes the upstream connection", async (ctx) => {
    if (!ready()) return ctx.skip();
    const upstream = upstreams["st-cancel"];
    const index = upstream.receivedRequests.length;
    await abandonAfterFirstFrame(CHAT, "st-cancel");
    await expect
      .poll(() => upstream.closedByPeer.includes(index), { timeout: 5_000 })
      .toBe(true);
  });

  test("a caller that leaves before a non-streamed answer closes the upstream connection", async (ctx) => {
    if (!ready()) return ctx.skip();
    const upstream = upstreams["ns-slow"];
    const index = upstream.receivedRequests.length;
    const controller = new AbortController();
    const inflight = post(CHAT, "ns-slow", false, controller.signal);
    await expect.poll(() => upstream.receivedRequests.length, { timeout: 5_000 }).toBe(index + 1);
    controller.abort();
    await expect(inflight).rejects.toThrow();
    await expect
      .poll(() => upstream.closedByPeer.includes(index), { timeout: 5_000 })
      .toBe(true);
  });
});
