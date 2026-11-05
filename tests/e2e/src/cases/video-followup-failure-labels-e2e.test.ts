import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket, WebSocketServer } from "ws";
import {
  EtcdClient,
  metricDelta,
  scrapeMetrics,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// A failure after a follow-up route knows its model is labelled with that
// model, like the same model's successes, so alerts and dashboards keyed on
// `model` see it. Before that point the label stays the `unresolved` sentinel.

const CALLER = "sk-followup-labels-caller";
const REALTIME_DENIED_CALLER = "sk-followup-labels-denied";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const VIDEO_REFUSED = "followup-video-refused";
const VIDEO_PENDING = "followup-video-pending";
const VIDEO_IP_LOCKED = "followup-video-ip-locked";
const REALTIME_MODEL = "followup-realtime";
const REALTIME_NO_PROTOCOL = "followup-realtime-no-protocol";
const REALTIME_WILDCARD = "followup-wild/*";
const REALTIME_SESSION_WILDCARD = "followup-session/*";
const REQUESTS = "aisix_requests_total";

function field(line: string, name: string): string | undefined {
  const m = line.match(new RegExp(`\\b${name}=(?:"([^"]*)"|([^\\s]+))`));
  if (!m) return undefined;
  return m[1] ?? m[2];
}

describe("failure labels on the routes that resolve their model mid-request", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;
  let refusedId = "";
  let pendingId = "";
  let ipLockedId = "";
  const upstreams: OpenAiUpstream[] = [];
  let sessionUpstream: WebSocketServer | undefined;

  const videoId = (modelId: string, alias: string, task: string) =>
    Buffer.from(`${modelId}:${Buffer.from(alias).toString("base64url")}:${task}`).toString("base64url");

  const getVideo = async (id: string, suffix = "") => {
    const res = await fetch(`${app!.proxyUrl}/v1/videos/${id}${suffix}`, {
      headers: { authorization: `Bearer ${CALLER}` },
      redirect: "manual",
    });
    await res.text();
    return { status: res.status, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  };

  const lineOf = (requestId: string) =>
    waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && field(l, "request_id") === requestId,
      `the access-log line of ${requestId}`,
    );

  const refusedHandshake = (model: string) =>
    new Promise<{ status: number; requestId: string }>((resolve, reject) => {
      const ws = new WebSocket(`${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`, {
        headers: { authorization: `Bearer ${REALTIME_DENIED_CALLER}` },
      });
      ws.on("unexpected-response", (_req, res) => {
        res.resume();
        ws.terminate();
        resolve({
          status: res.statusCode ?? 0,
          requestId: String(res.headers["x-aisix-request-id"] ?? ""),
        });
      });
      ws.on("open", () => reject(new Error("the handshake was not refused")));
      ws.on("error", (err) => {
        if (ws.readyState !== WebSocket.CLOSING) reject(err);
      });
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp({ logLevel: "info" });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // How Zhipu refuses a model it does not have.
    const refusing = await startOpenAiUpstream({
      status: 400,
      errorBody: { error: { code: "1211", message: "model does not exist" } },
    });
    const pending = await startOpenAiUpstream({
      nonStreamBody: { id: "task-1", task_status: "PROCESSING" },
    });
    upstreams.push(refusing, pending);

    const seedVideoModel = async (name: string, apiBase: string, extra: Record<string, unknown> = {}) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: apiBase,
        provider: "zhipuai",
      });
      return seed.createModel({
        display_name: name,
        provider: "zhipuai",
        model_name: `${name}-upstream`,
        provider_key_id: pk.id,
        ...extra,
      });
    };
    const refusedModel = await seedVideoModel(VIDEO_REFUSED, refusing.baseUrl);
    const pendingModel = await seedVideoModel(VIDEO_PENDING, pending.baseUrl);
    const ipLockedModel = await seedVideoModel(VIDEO_IP_LOCKED, refusing.baseUrl, {
      allowed_cidrs: ["203.0.113.0/24"],
    });
    ipLockedId = videoId(ipLockedModel.id, VIDEO_IP_LOCKED, "task-1");
    refusedId = videoId(refusedModel.id, VIDEO_REFUSED, "task-1");
    pendingId = videoId(pendingModel.id, VIDEO_PENDING, "task-1");

    const realtimePk = await seed.createProviderKey({
      display_name: `${REALTIME_MODEL}-pk`,
      secret: "sk-mock",
      api_base: refusing.baseUrl,
      provider: "openai",
    });
    await seed.createModel({
      display_name: REALTIME_MODEL,
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: realtimePk.id,
    });

    await seed.createModel({
      display_name: REALTIME_WILDCARD,
      provider: "openai",
      model_name: "*",
      provider_key_id: realtimePk.id,
    });
    const noProtocolPk = await seed.createProviderKey({
      display_name: `${REALTIME_NO_PROTOCOL}-pk`,
      secret: "sk-mock",
      api_base: refusing.baseUrl,
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: REALTIME_NO_PROTOCOL,
      provider: "anthropic",
      model_name: "claude-mock",
      provider_key_id: noProtocolPk.id,
    });

    sessionUpstream = new WebSocketServer({ port: 0, host: "127.0.0.1" });
    await new Promise<void>((r) => sessionUpstream!.once("listening", () => r()));
    const sessionPort = (sessionUpstream.address() as { port: number }).port;
    const sessionPk = await seed.createProviderKey({
      display_name: "followup-session-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${sessionPort}/v1`,
      provider: "openai",
    });
    await seed.createModel({
      display_name: REALTIME_SESSION_WILDCARD,
      provider: "openai",
      model_name: "*",
      provider_key_id: sessionPk.id,
    });

    await seed.createApiKey({ key_hash: sha256(CALLER), allowed_models: ["*"] });
    await seed.createApiKey({
      key_hash: sha256(REALTIME_DENIED_CALLER),
      allowed_models: [VIDEO_REFUSED, REALTIME_NO_PROTOCOL],
    });

    // The keys were seeded last, so a key that authenticates means every row
    // before it has arrived.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${REALTIME_DENIED_CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    sessionUpstream?.close();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test.each([
    ["the poll", ""],
    ["the content route", "/content"],
  ])("%s failing upstream carries the model the id resolved to", async (_name, suffix) => {
    if (!etcdReachable || !app) return;
    const labels = { provider: "zhipuai", model: VIDEO_REFUSED, status: "400", outcome: "client_error" };
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await getVideo(refusedId, suffix);
    expect(status).toBe(400);

    const line = await lineOf(requestId);
    expect(field(line, "requested_model")).toBe(VIDEO_REFUSED);
    expect(field(line, "provider")).toBe("zhipuai");
    expect(field(line, "upstream_model")).toBe(`${VIDEO_REFUSED}-upstream`);

    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, REQUESTS, labels)).toBe(1);
    expect(metricDelta(before, after, REQUESTS, { model: "unresolved" })).toBe(0);
  });

  test("content refused because the job is still running carries the model too", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await getVideo(pendingId, "/content");
    expect(status).toBe(400);

    expect(field(await lineOf(requestId), "requested_model")).toBe(VIDEO_PENDING);
    const after = await scrapeMetrics(app.metricsUrl);
    expect(
      metricDelta(before, after, REQUESTS, { provider: "zhipuai", model: VIDEO_PENDING, status: "400" }),
    ).toBe(1);
  });

  test("a poll the model's IP restriction refuses carries the model", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await getVideo(ipLockedId);
    expect(status).toBe(403);

    expect(field(await lineOf(requestId), "requested_model")).toBe(VIDEO_IP_LOCKED);
    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, REQUESTS, { model: VIDEO_IP_LOCKED, status: "403" })).toBe(1);
    expect(metricDelta(before, after, REQUESTS, { model: "unresolved" })).toBe(0);
  });

  test.each([
    ["an id the gateway never minted", () => "not-a-video-id"],
    ["an id naming no configured model", () => videoId("00000000-0000-4000-8000-000000000000", "gone", "task-1")],
  ])("%s stays unresolved", async (_name, id) => {
    if (!etcdReachable || !app) return;
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await getVideo(id());
    expect(status).toBe(404);

    expect(field(await lineOf(requestId), "requested_model")).toBe("unresolved");
    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, REQUESTS, { model: "unresolved", status: "404" })).toBe(1);
  });

  test("a realtime handshake refused after the model resolved carries that model", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await refusedHandshake(REALTIME_MODEL);
    expect(status).toBe(403);

    expect(field(await lineOf(requestId), "requested_model")).toBe(REALTIME_MODEL);
    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, REQUESTS, { model: REALTIME_MODEL, status: "403" })).toBe(1);
    expect(metricDelta(before, after, REQUESTS, { model: "unresolved" })).toBe(0);
  });

  test("a realtime refusal after the target was selected names the target too", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await refusedHandshake(REALTIME_NO_PROTOCOL);
    expect(status).toBe(400);

    const line = await lineOf(requestId);
    expect(field(line, "requested_model")).toBe(REALTIME_NO_PROTOCOL);
    expect(field(line, "provider")).toBe("anthropic");
    const after = await scrapeMetrics(app.metricsUrl);
    expect(
      metricDelta(before, after, "aisix_proxy_requests_total", {
        provider: "anthropic",
        model: REALTIME_NO_PROTOCOL,
        upstream_model: "claude-mock",
        provider_key_name: `${REALTIME_NO_PROTOCOL}-pk`,
        status: "400",
      }),
    ).toBe(1);
  });

  test("a realtime refusal on a wildcard row labels the row, not the caller's name", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);

    const { status } = await refusedHandshake("followup-wild/minted-by-caller");
    expect(status).toBe(403);

    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, REQUESTS, { model: REALTIME_WILDCARD, status: "403" })).toBe(1);
  });

  // A wildcard row serves any suffix the caller invents, so the suffix must not
  // reach a label through `upstream_model` either (#451).
  const wildcardSeries = async (row: string, status: string) =>
    (await scrapeMetrics(app!.metricsUrl)).filter(
      (s) => s.name === "aisix_proxy_requests_total" && s.labels.model === row && s.labels.status === status,
    );

  const sessionUntilClosed = (model: string, { hangUp }: { hangUp: boolean }) =>
    new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(`${app!.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      ws.on("open", () => {
        if (hangUp) ws.close();
      });
      ws.on("close", () => resolve());
      ws.on("unexpected-response", (_req, res) => reject(new Error(`handshake answered ${res.statusCode}`)));
      ws.on("error", reject);
    });

  test("a wildcard row's refused upstream handshakes share one series", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    // The gateway upgrades the caller first, then closes once the upstream refuses.
    await sessionUntilClosed("followup-wild/suffix-one", { hangUp: false });
    await sessionUntilClosed("followup-wild/suffix-two", { hangUp: false });

    await expect
      .poll(async () => (await wildcardSeries(REALTIME_WILDCARD, "400")).reduce((n, s) => n + s.value, 0))
      .toBe(2);
    const series = await wildcardSeries(REALTIME_WILDCARD, "400");
    expect(series.map((s) => s.labels.upstream_model)).toEqual(["*"]);
  });

  test("a wildcard row's sessions share one series", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    await sessionUntilClosed("followup-session/suffix-one", { hangUp: true });
    await sessionUntilClosed("followup-session/suffix-two", { hangUp: true });

    await expect
      .poll(async () => (await wildcardSeries(REALTIME_SESSION_WILDCARD, "200")).reduce((n, s) => n + s.value, 0))
      .toBe(2);
    const series = await wildcardSeries(REALTIME_SESSION_WILDCARD, "200");
    expect(series.map((s) => s.labels.upstream_model)).toEqual(["*"]);
  });

  test("a realtime handshake naming no configured model stays unresolved", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);

    const { status, requestId } = await refusedHandshake("followup-no-such-model");
    expect(status).toBe(404);

    const line = await lineOf(requestId);
    expect(field(line, "requested_model")).toBeUndefined();
    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, REQUESTS, { model: "unresolved", status: "404" })).toBe(1);
  });
});
