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
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1719: agents sharing one API key send their own
// request headers (a sub-user id, a department), and the operator lists the
// ones every usage event must record, so events exported to an external
// system can be attributed there.
//
// Read back off a real Aliyun-SLS export from a real `aisix` binary — the
// row a consumer actually receives. The exporter runs in the default
// metadata-only content mode: the recorded headers are attribution, not
// captured content.

const PLAINTEXT = "sk-usage-request-headers";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const LOGSTORE = "usage-request-headers";
const MODEL = "urh-model";
const BROKEN_MODEL = "urh-broken";
const ROUTE_PREFIX = "/urh-route";

type Headers = Record<string, string | string[]>;

describe("usage events record operator-selected request headers (AISIX-Cloud#1719)", () => {
  let upstream: OpenAiUpstream | undefined;
  let broken: OpenAiUpstream | undefined;
  let mcp: McpUpstream | undefined;
  let sls: MockSls | undefined;
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

  async function rowFor(requestId: string, what: string): Promise<Map<string, string>> {
    expect(requestId, `${what}: response carries x-aisix-request-id`).toBeTruthy();
    return waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, what);
  }

  function recorded(row: Map<string, string>): Record<string, string> | undefined {
    const raw = row.get("request_headers");
    return raw === undefined ? undefined : JSON.parse(raw);
  }

  const chat = { model: MODEL, messages: [{ role: "user", content: "hi" }] };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream();
    broken = await startOpenAiUpstream({ status: 500 });
    mcp = await startMcpUpstream("urh");

    app = await spawnApp({
      usageEventRequestHeaders: ["x-sub-user", "x-department"],
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "LTAI_mock_ak",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock_ak_secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "urh-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const pk = await seed.createProviderKey({
      display_name: "urh-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const brokenPk = await seed.createProviderKey({
      display_name: "urh-broken-pk",
      secret: "sk-mock",
      api_base: `${broken.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: BROKEN_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: brokenPk.id,
    });
    await seed.createPassthroughRoute({
      name: "urh-route",
      path_prefix: ROUTE_PREFIX,
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
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
    await mcp?.close();
    await sls?.close();
  });

  test("both configured headers are recorded, a repeated one joined in arrival order", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await send("/v1/chat/completions", chat, {
      "x-sub-user": "alice",
      "x-department": ["eng", "ops"],
      "x-not-listed": "ignored",
    });
    expect(res.status).toBe(200);
    const row = await rowFor(res.requestId, "chat row");
    // Exactly these two keys: an unlisted header is never captured.
    expect(recorded(row)).toEqual({ "x-sub-user": "alice", "x-department": "eng, ops" });
  });

  test("a configured header the request did not send has no key", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await send("/v1/chat/completions", chat, {
      "x-sub-user": "bob",
      "x-not-listed": "ignored",
    });
    expect(res.status).toBe(200);
    const row = await rowFor(res.requestId, "chat row without x-department");
    expect(recorded(row)).toEqual({ "x-sub-user": "bob" });

    // With none of them sent, the row carries no map at all rather than
    // an empty one.
    const none = await send("/v1/chat/completions", chat, { "x-not-listed": "ignored" });
    expect(none.status).toBe(200);
    const bare = await rowFor(none.requestId, "chat row without configured headers");
    expect(bare.get("request_headers")).toBeUndefined();
  });

  test("/v1/messages and /v1/embeddings record the same map", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const messages = await send(
      "/v1/messages",
      { model: MODEL, max_tokens: 16, messages: [{ role: "user", content: "hi" }] },
      { "x-sub-user": "carol", "x-department": "sales" },
    );
    expect(messages.status).toBe(200);
    expect(recorded(await rowFor(messages.requestId, "messages row"))).toEqual({
      "x-sub-user": "carol",
      "x-department": "sales",
    });

    const embeddings = await send(
      "/v1/embeddings",
      { model: MODEL, input: "hi" },
      { "x-sub-user": "dave" },
    );
    expect(recorded(await rowFor(embeddings.requestId, "embeddings row"))).toEqual({
      "x-sub-user": "dave",
    });
  });

  test("a passthrough route records the map", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await send(`${ROUTE_PREFIX}/chat/completions`, chat, {
      "x-sub-user": "erin",
      "x-department": "legal",
    });
    expect(res.status).toBe(200);
    const row = await rowFor(res.requestId, "passthrough row");
    expect(row.get("passthrough_route_name")).toBe("urh-route");
    expect(recorded(row)).toEqual({ "x-sub-user": "erin", "x-department": "legal" });
  });

  test("a request whose upstream fails records the map on its failure row", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await send(
      "/v1/chat/completions",
      { ...chat, model: BROKEN_MODEL },
      { "x-sub-user": "frank" },
    );
    expect(res.status).toBeGreaterThanOrEqual(500);
    const row = await rowFor(res.requestId, "failure row");
    expect(recorded(row)).toEqual({ "x-sub-user": "frank" });
  });

  // `/mcp` resolves its caller without the extractor the other families
  // share, so it captures the headers on its own path.
  test("an MCP tool call records the map", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
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
      LOGSTORE,
      (l) => l.get("operation") === "mcp" && l.get("mcp_tool_name") === "echo",
      "mcp row",
    );
    expect(recorded(row)).toEqual({ "x-sub-user": "grace" });
  });
});

describe("observability.usage_event.request_headers startup validation", () => {
  test("a credential header in the list fails startup naming the key", async (ctx) => {
    if (!(await new EtcdClient().ping())) {
      ctx.skip();
      return;
    }
    let caught: unknown;
    try {
      // The env form, which is the only one an env-driven deployment has.
      const app = await spawnApp({
        extraEnv: {
          AISIX_OBSERVABILITY__USAGE_EVENT__REQUEST_HEADERS: "x-sub-user,authorization",
        },
      });
      await app.exit();
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    const msg = (caught as Error).message;
    expect(msg).toContain("exited early with code=1");
    expect(msg).toContain("observability.usage_event.request_headers");
    expect(msg).toContain('"authorization"');
  }, 30_000);
});
