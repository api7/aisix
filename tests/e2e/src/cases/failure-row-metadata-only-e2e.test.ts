import { createHash, randomUUID } from "node:crypto";
import { createServer, type Server as HttpServer } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket as WsClient } from "ws";
import {
  EtcdClient,
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

// Every failure row names its failure in `error_message`, under `metadata_only`
// too, on the routes that hold no typed error as well as those that do. Only an
// exporter at `metadata_only` is configured here, so nothing a full-content
// export would read can be what filled the message in.

const CALLER = "sk-row-meta-caller";
const LOGSTORE = "row-meta";
const CREDENTIAL_REF = "mock";
const SENTINEL = "sentinel-upstream-said-this";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const jsonError = JSON.stringify({ error: { message: SENTINEL } });
const JSON_HEADERS = { "content-type": "application/json" };

describe("failure rows name the failure under metadata_only", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const httpServers: HttpServer[] = [];
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  async function fixedUpstream(
    status: number,
    body: string,
    headers: Record<string, string> = JSON_HEADERS,
  ): Promise<string> {
    const server = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        res.writeHead(status, headers);
        res.end(body);
      });
    });
    httpServers.push(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: LOGSTORE,
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "row-meta-project",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const modelAt = async (name: string, base: string) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: base,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      return pk;
    };

    const chatOk = await startOpenAiUpstream();
    upstreams.push(chatOk);
    await modelAt("meta-chat", `${chatOk.baseUrl}/v1`);

    const failing = await fixedUpstream(500, jsonError);
    await modelAt("meta-jobs-500", `${failing}/v1`);
    await modelAt("meta-realtime-400", `${await fixedUpstream(400, jsonError)}/v1`);
    const failingPk = await modelAt("meta-chat-500", `${failing}/v1`);

    await seed.createPassthroughRoute({
      name: "meta-route-json",
      path_prefix: "/meta-route-json",
      target_url: failing,
      provider_key_id: failingPk.id,
    });
    await seed.createPassthroughRoute({
      name: "meta-route-empty",
      path_prefix: "/meta-route-empty",
      target_url: await fixedUpstream(503, ""),
      provider_key_id: failingPk.id,
    });
    await seed.createPassthroughRoute({
      name: "meta-route-sse",
      path_prefix: "/meta-route-sse",
      target_url: await fixedUpstream(429, `data: ${SENTINEL}\n\n`, {
        "content-type": "text/event-stream",
      }),
      provider_key_id: failingPk.id,
    });

    await seed.update("a2a_agents", randomUUID(), {
      name: "meta-agent",
      url: await fixedUpstream(500, SENTINEL),
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });

    await seed.update("a2a_agents", randomUUID(), {
      name: "meta-agent-stream",
      url: await fixedUpstream(200, "data: not-json\n\n", { "content-type": "text/event-stream" }),
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });

    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["*"],
      allowed_routes: ["meta-route-json", "meta-route-empty", "meta-route-sse"],
      allowed_agents: ["*"],
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
    await sls?.close();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(
      httpServers.map((s) => new Promise<void>((r) => s.close(() => r()))),
    );
  });

  async function post(path: string, body: unknown) {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    return {
      status: res.status,
      text: await res.text(),
      requestId: res.headers.get("x-aisix-request-id") ?? "",
    };
  }

  const rowOf = (requestId: string, what: string) =>
    waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, what);

  test("a validation 400 carries the text the caller was told", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/v1/chat/completions", { model: "meta-chat", messages: [] });
    expect(r.status).toBe(400);
    const row = await rowOf(r.requestId, "the 400 row");
    expect(row.get("error_class")).toBe("invalid_request_error");
    expect(row.get("error_message")).toBe(JSON.parse(r.text).error.message);
  });

  test("an upstream's error on a typed route reads as it does on an attempt", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/v1/chat/completions", {
      model: "meta-chat-500",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(r.status).toBeGreaterThanOrEqual(500);
    const row = await rowOf(r.requestId, "the chat row");
    expect(row.get("error_class")).toBe("upstream_status");
    expect(row.get("error_message")).toBe(`upstream returned HTTP 500: ${SENTINEL}`);
  });

  test("jobs: a relayed upstream failure carries what the upstream said", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/v1/batches?model=meta-jobs-500", { input_file_id: "f" });
    expect(r.status).toBe(500);
    expect(r.text).toBe(jsonError);
    const row = await rowOf(r.requestId, "the jobs row");
    expect(row.get("status_code")).toBe("500");
    expect(row.get("error_class")).toBe("upstream_status");
    expect(row.get("error_message")).toBe(`upstream returned HTTP 500: ${SENTINEL}`);
  });

  test("passthrough: a buffered failure is relayed as sent and named on the row", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/meta-route-json/anything", { x: 1 });
    expect(r.status).toBe(500);
    expect(r.text).toBe(jsonError);
    const row = await rowOf(r.requestId, "the passthrough row");
    expect(row.get("error_class")).toBe("upstream_status");
    expect(row.get("error_message")).toBe(`upstream returned HTTP 500: ${SENTINEL}`);
  });

  test("passthrough: an empty failure says it had no detail", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/meta-route-empty/anything", { x: 1 });
    expect(r.status).toBe(503);
    expect(r.text).toBe("");
    const row = await rowOf(r.requestId, "the empty passthrough row");
    expect(row.get("error_message")).toBe(
      "upstream returned HTTP 503 with no error detail in the response",
    );
  });

  test("passthrough: a failure relayed as a stream reaches the caller untouched", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/meta-route-sse/anything", { x: 1 });
    expect(r.status).toBe(429);
    expect(r.text).toBe(`data: ${SENTINEL}\n\n`);
    const row = await rowOf(r.requestId, "the opaque passthrough row");
    expect(row.get("error_class")).toBe("upstream_status");
    expect(row.get("error_message")).toContain(SENTINEL);
  });

  test("a2a: a failed agent call names the status the caller was told", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/a2a/meta-agent", {
      jsonrpc: "2.0",
      id: "1",
      method: "message/send",
      params: { message: { role: "user", parts: [{ kind: "text", text: "hi" }], messageId: "m-1" } },
    });
    expect(r.status).toBe(502);
    const row = await rowOf(r.requestId, "the a2a row");
    expect(row.get("error_class")).toBe("upstream_status");
    expect(row.get("error_message")).toBe(
      "upstream A2A request failed: upstream returned HTTP 500",
    );
    expect(r.text).toContain(row.get("error_message"));
  });

  test("a2a: a stream the agent breaks after its head names the failure", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const r = await post("/a2a/meta-agent-stream", {
      jsonrpc: "2.0",
      id: "1",
      method: "message/stream",
      params: { message: { role: "user", parts: [{ kind: "text", text: "hi" }], messageId: "m-1" } },
    });
    expect(r.status).toBe(200);
    const row = await rowOf(r.requestId, "the a2a stream row");
    expect(row.get("status_code")).toBe("502");
    expect(row.get("error_class")).toBe("upstream_decode");
    expect(row.get("error_message")).toMatch(/^upstream A2A request failed: malformed JSON-RPC event/);
  });

  test("realtime: an upstream that refuses the socket is named with what it said", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const requestId = await new Promise<string>((resolve, reject) => {
      const ws = new WsClient(
        `${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=meta-realtime-400`,
        { headers: { authorization: `Bearer ${CALLER}` } },
      );
      let id = "";
      ws.on("upgrade", (res) => {
        id = String(res.headers["x-aisix-request-id"] ?? "");
      });
      ws.on("close", () => resolve(id));
      ws.on("error", reject);
    });
    const row = await rowOf(requestId, "the realtime row");
    expect(row.get("status_code")).toBe("502");
    expect(row.get("error_class")).toBe("transport");
    expect(row.get("error_message")).toMatch(
      new RegExp(`^transport error: upstream returned HTTP 400: .*${SENTINEL}`),
    );
  });
});
