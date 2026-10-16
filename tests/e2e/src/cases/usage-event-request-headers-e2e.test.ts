import { createHash, randomUUID } from "node:crypto";
import { request as httpRequest } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
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
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// E2E for AISIX-Cloud#1719 as reworked by ticket-analysis#1051: which
// request headers — and which upstream RESPONSE headers — an export
// records is chosen per exporter, by the exporter resource's own
// `request_headers` / `upstream_response_headers` lists.
//
// Read back off real exports from a real `aisix` binary: two Aliyun-SLS
// exporters with different lists (one logstore each) and an OTLP one, so
// each exporter is seen to carry exactly its own names. The exporters run
// in the default metadata-only content mode: the recorded headers are
// attribution, not captured content.

const PLAINTEXT = "sk-usage-request-headers";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const STORE_A = "urh-store-a";
const STORE_B = "urh-store-b";
const MODEL = "urh-model";
const BROKEN_MODEL = "urh-broken";
const FLAKY_MODEL = "urh-flaky";
const ROUTE_PREFIX = "/urh-route";
const KEYED_ROUTE_PREFIX = "/urh-keyed-route";

type Headers = Record<string, string | string[]>;

describe("exporters record the request and upstream response headers they list", () => {
  let upstream: OpenAiUpstream | undefined;
  let broken: OpenAiUpstream | undefined;
  let flaky: OpenAiUpstream | undefined;
  let mcp: McpUpstream | undefined;
  let sls: MockSls | undefined;
  let otlp: MockOtlp | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  // node:http rather than fetch: an array value goes out as separate header
  // lines, which is what a repeated header IS on the wire. fetch would fold
  // it into one line before the gateway ever saw it.
  function send(
    path: string,
    body: unknown,
    headers: Headers,
  ): Promise<{ status: number; requestId: string }> {
    const url = new URL(path, app!.proxyUrl);
    const payload = JSON.stringify(body);
    return new Promise((resolve, reject) => {
      const req = httpRequest(
        url,
        {
          method: "POST",
          headers: {
            authorization: `Bearer ${PLAINTEXT}`,
            "content-type": "application/json",
            "content-length": Buffer.byteLength(payload),
            ...headers,
          },
        },
        (res) => {
          res.resume();
          res.on("end", () =>
            resolve({
              status: res.statusCode ?? 0,
              requestId: String(res.headers["x-aisix-request-id"] ?? ""),
            }),
          );
        },
      );
      req.on("error", reject);
      req.end(payload);
    });
  }

  async function rowFor(
    store: string,
    requestId: string,
    what: string,
    pred: (l: Map<string, string>) => boolean = () => true,
  ): Promise<Map<string, string>> {
    expect(requestId, `${what}: response carries x-aisix-request-id`).toBeTruthy();
    return waitForSlsLog(sls!, store, (l) => l.get("request_id") === requestId && pred(l), what);
  }

  function map(row: Map<string, string>, field: string): Record<string, string> | undefined {
    const raw = row.get(field);
    return raw === undefined ? undefined : JSON.parse(raw);
  }

  const chat = { model: MODEL, messages: [{ role: "user", content: "hi" }] };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    otlp = await startMockOtlp();
    upstream = await startOpenAiUpstream({
      responseHeaders: { "x-upstream-region": "us-east", "x-upstream-secret-ish": "nope" },
    });
    broken = await startOpenAiUpstream({
      status: 500,
      responseHeaders: { "x-upstream-region": "broken-region" },
    });
    // One failed attempt, then a success: each attempt's row carries the
    // headers of its own upstream response.
    flaky = await startOpenAiUpstream({
      scriptedResponses: [
        { status: 500, responseHeaders: { "x-upstream-region": "first-try" } },
        { responseHeaders: { "x-upstream-region": "second-try" } },
      ],
    });
    mcp = await startMcpUpstream("urh");

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const sink = (logstore: string) => ({
      kind: "aliyun_sls",
      endpoint: sls!.url,
      project: "aisix-e2e-obs",
      logstore,
      credential_ref: CREDENTIAL_REF,
    });
    await seed.createObservabilityExporter({
      name: "urh-sls-a",
      enabled: true,
      ...sink(STORE_A),
      // `x-gw-key` is the header-key route's credential slot below: listed
      // here, it must still never be recorded.
      request_headers: ["x-sub-user", "x-department", "x-gw-key"],
      upstream_response_headers: ["x-upstream-region"],
    });
    await seed.createObservabilityExporter({
      name: "urh-sls-b",
      enabled: true,
      ...sink(STORE_B),
      request_headers: ["x-department"],
    });
    await seed.createObservabilityExporter({
      name: "urh-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
      request_headers: ["x-sub-user"],
      upstream_response_headers: ["x-upstream-region"],
    });
    const model = async (name: string, mock: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${mock.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      return pk;
    };
    const pk = await model(MODEL, upstream);
    await model(BROKEN_MODEL, broken);
    await model(FLAKY_MODEL, flaky);
    await seed.createPassthroughRoute({
      name: "urh-route",
      path_prefix: ROUTE_PREFIX,
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
    });
    await seed.createPassthroughRoute({
      name: "urh-keyed-route",
      path_prefix: KEYED_ROUTE_PREFIX,
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
      auth_mode: "header_key",
      auth_header_name: "x-gw-key",
    });
    await seed.update("mcp_servers", randomUUID(), {
      display_name: "urh",
      url: mcp.url,
      enabled: true,
    });
    // Seeded last: it authenticating implies everything above is live.
    await seed.createApiKey({
      key_hash: hash(PLAINTEXT),
      allowed_models: ["*"],
      allowed_routes: ["*"],
      mcp_access: { allow: ["*"] },
    });
    const proxy = new ProxyClient(app.proxyUrl, PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await broken?.close();
    await flaky?.close();
    await mcp?.close();
    await sls?.close();
    await otlp?.close();
  });

  test("each exporter records exactly the headers its own lists name", async (ctx) => {
    if (!etcdReachable || !app || !sls || !otlp) return ctx.skip();
    const res = await send("/v1/chat/completions", chat, {
      "x-sub-user": "alice",
      "x-department": ["eng", "ops"],
      "x-not-listed": "ignored",
    });
    expect(res.status).toBe(200);

    const a = await rowFor(STORE_A, res.requestId, "exporter A row");
    expect(map(a, "request_headers")).toEqual({ "x-sub-user": "alice", "x-department": "eng, ops" });
    expect(map(a, "upstream_response_headers")).toEqual({ "x-upstream-region": "us-east" });

    const b = await rowFor(STORE_B, res.requestId, "exporter B row");
    expect(map(b, "request_headers")).toEqual({ "x-department": "eng, ops" });
    expect(b.get("upstream_response_headers")).toBeUndefined();

    const span = await waitForSpan(
      otlp,
      (s) =>
        s.attributes["aisix.request_id"] === res.requestId &&
        s.attributes["gen_ai.operation.name"] !== undefined,
      "OTLP operation span",
    );
    expect(span.attributes["http.response.header.x-upstream-region"]).toBe("us-east");
    expect(span.attributes["http.response.header.x-upstream-secret-ish"]).toBeUndefined();
    const all = otlp.spans.filter((s) => s.attributes["aisix.request_id"] === res.requestId);
    expect(all.some((s) => s.attributes["http.request.header.x-sub-user"] === "alice")).toBe(true);
    expect(all.some((s) => "http.request.header.x-department" in s.attributes)).toBe(false);
  });

  test("a listed header the request did not send has no key, and none sent means no map", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await send("/v1/chat/completions", chat, { "x-sub-user": "bob" });
    expect(res.status).toBe(200);
    expect(map(await rowFor(STORE_A, res.requestId, "row without x-department"), "request_headers"))
      .toEqual({ "x-sub-user": "bob" });
    expect((await rowFor(STORE_B, res.requestId, "B row")).get("request_headers")).toBeUndefined();
  });

  test("every attempt carries its own upstream response's headers", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await send("/v1/chat/completions", { ...chat, model: FLAKY_MODEL }, {});
    expect(res.status).toBe(200);
    const first = await rowFor(STORE_A, res.requestId, "failed attempt row", (l) => l.get("attempt_index") === "0");
    const second = await rowFor(STORE_A, res.requestId, "winning attempt row", (l) => l.get("attempt_index") === "1");
    expect(map(first, "upstream_response_headers")).toEqual({ "x-upstream-region": "first-try" });
    expect(map(second, "upstream_response_headers")).toEqual({ "x-upstream-region": "second-try" });
  });

  test("/v1/messages and /v1/embeddings record the same maps", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const messages = await send(
      "/v1/messages",
      { model: MODEL, max_tokens: 16, messages: [{ role: "user", content: "hi" }] },
      { "x-sub-user": "carol", "x-department": "sales" },
    );
    expect(messages.status).toBe(200);
    const m = await rowFor(STORE_A, messages.requestId, "messages row");
    expect(map(m, "request_headers")).toEqual({ "x-sub-user": "carol", "x-department": "sales" });
    expect(map(m, "upstream_response_headers")).toEqual({ "x-upstream-region": "us-east" });

    const embeddings = await send("/v1/embeddings", { model: MODEL, input: "hi" }, { "x-sub-user": "dave" });
    const e = await rowFor(STORE_A, embeddings.requestId, "embeddings row");
    expect(map(e, "request_headers")).toEqual({ "x-sub-user": "dave" });
    expect(map(e, "upstream_response_headers")).toEqual({ "x-upstream-region": "us-east" });
  });

  test("a passthrough route records both maps", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await send(`${ROUTE_PREFIX}/chat/completions`, chat, {
      "x-sub-user": "erin",
      "x-department": "legal",
    });
    expect(res.status).toBe(200);
    const row = await rowFor(STORE_A, res.requestId, "passthrough row");
    expect(row.get("passthrough_route_name")).toBe("urh-route");
    expect(map(row, "request_headers")).toEqual({ "x-sub-user": "erin", "x-department": "legal" });
    expect(map(row, "upstream_response_headers")).toEqual({ "x-upstream-region": "us-east" });
  });

  test("a header-key route's credential slot is never recorded, even when an exporter lists it", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await send(`${KEYED_ROUTE_PREFIX}/chat/completions`, chat, {
      authorization: "",
      "x-gw-key": PLAINTEXT,
      "x-sub-user": "heidi",
    });
    expect(res.status).toBe(200);
    const row = await rowFor(STORE_A, res.requestId, "header-key route row");
    expect(map(row, "request_headers")).toEqual({ "x-sub-user": "heidi" });
  });

  test("a request whose upstream fails records both maps on its failure row", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await send("/v1/chat/completions", { ...chat, model: BROKEN_MODEL }, { "x-sub-user": "frank" });
    expect(res.status).toBeGreaterThanOrEqual(500);
    const row = await rowFor(STORE_A, res.requestId, "failure row");
    expect(map(row, "request_headers")).toEqual({ "x-sub-user": "frank" });
    expect(map(row, "upstream_response_headers")).toEqual({ "x-upstream-region": "broken-region" });
  });

  // `/mcp` resolves its caller without the extractor the other families
  // share, so it captures the request headers on its own path.
  test("an MCP tool call records the request map", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const rpc = (id: number, method: string, params: Record<string, unknown>) =>
      send(
        "/mcp",
        { jsonrpc: "2.0", id, method, params },
        { accept: "application/json, text/event-stream", "x-sub-user": "grace" },
      );
    await rpc(1, "initialize", {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "usage-event-request-headers", version: "0.1" },
    });
    const res = await rpc(2, "tools/call", { name: "urh__echo", arguments: { text: "hi" } });
    expect(res.status).toBe(200);
    const row = await waitForSlsLog(
      sls,
      STORE_A,
      (l) => l.get("operation") === "mcp" && l.get("mcp_tool_name") === "echo",
      "mcp row",
    );
    expect(map(row, "request_headers")).toEqual({ "x-sub-user": "grace" });
  });
});

async function waitForSpan(
  otlp: MockOtlp,
  pred: (s: MockOtlp["spans"][number]) => boolean,
  what: string,
  timeoutMs = 15_000,
): Promise<MockOtlp["spans"][number]> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = otlp.spans.find(pred);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`timed out waiting for ${what}`);
}

describe("observability.usage_event.request_headers is retired", () => {
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let app: SpawnedApp | undefined;

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  test("the 1.6.0 key still boots, warns once naming the exporter field, and records nothing", async (ctx) => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return ctx.skip();
    sls = await startMockSls();
    upstream = await startOpenAiUpstream();
    // A list the old validation refused boots too: nothing reads it.
    app = await spawnApp({
      usageEventRequestHeaders: ["x-sub-user", "authorization"],
      logLevel: "warn",
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const warning = await waitForLogLine(
      app,
      (l) => l.includes("observability.usage_event") && l.includes("observability_exporters"),
      "the retired-key warning",
    );
    expect(warning).toMatch(/WARN/i);

    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "urh-retired-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: "urh-retired",
      credential_ref: CREDENTIAL_REF,
    });
    const pk = await seed.createProviderKey({
      display_name: "urh-retired-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: hash(PLAINTEXT), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${PLAINTEXT}`,
        "content-type": "application/json",
        "x-sub-user": "alice",
      },
      body: JSON.stringify(chat()),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    const row = await waitForSlsLog(sls, "urh-retired", (l) => l.get("request_id") === requestId, "row");
    expect(row.get("request_headers")).toBeUndefined();
  }, 60_000);

  function chat() {
    return { model: MODEL, messages: [{ role: "user", content: "hi" }] };
  }
});
