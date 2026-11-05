import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocket } from "ws";
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
const REALTIME_MODEL = "followup-realtime";
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
  const upstreams: OpenAiUpstream[] = [];

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
        resolve({
          status: res.statusCode ?? 0,
          requestId: String(res.headers["x-aisix-request-id"] ?? ""),
        });
      });
      ws.on("open", () => reject(new Error("the handshake was not refused")));
      ws.on("error", reject);
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

    const seedVideoModel = async (name: string, apiBase: string) => {
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
      });
    };
    const refusedModel = await seedVideoModel(VIDEO_REFUSED, refusing.baseUrl);
    const pendingModel = await seedVideoModel(VIDEO_PENDING, pending.baseUrl);
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

    await seed.createApiKey({ key_hash: sha256(CALLER), allowed_models: ["*"] });
    await seed.createApiKey({ key_hash: sha256(REALTIME_DENIED_CALLER), allowed_models: [VIDEO_REFUSED] });

    await waitConfigPropagation(async () => (await getVideo(pendingId)).status === 200);
    await waitConfigPropagation(async () => (await refusedHandshake(REALTIME_MODEL)).status === 403);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
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

    const line = await lineOf(requestId);
    expect(field(line, "requested_model")).toBe(REALTIME_MODEL);
    expect(field(line, "provider")).toBe("openai");
    const after = await scrapeMetrics(app.metricsUrl);
    expect(
      metricDelta(before, after, REQUESTS, { provider: "openai", model: REALTIME_MODEL, status: "403" }),
    ).toBe(1);
    expect(metricDelta(before, after, REQUESTS, { model: "unresolved" })).toBe(0);
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
