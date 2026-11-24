import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, keyHash, metricDelta, ProxyClient, scrapeMetrics, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";
import { funAsrPath, openFunAsr, recognize, runTask, startFunAsrUpstream } from "../harness/fun-asr.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

describe("DashScope Fun-ASR native tasks", () => {
  const caller = "sk-asr-caller";
  const revoked = "sk-asr-revoked";
  let app: SpawnedApp;
  let seed: SeedClient;
  let otlp: MockOtlp;
  let vendor: Awaited<ReturnType<typeof startFunAsrUpstream>>;
  let revokedKeyId: string;
  const ids = new Map<string, string>();

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    vendor = await startFunAsrUpstream();
    otlp = await startMockOtlp();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({ name: "asr-usage", kind: "otlp_http", endpoint: otlp.url });
    for (const [alias, provider] of [["cn", "alibaba-cn"], ["intl", "alibaba"], ["failure", "alibaba"], ["drop", "alibaba"], ["over-budget", "alibaba"]]) {
      const pk = await seed.createProviderKey({
        provider, display_name: alias, api_key: "sk-provider-" + alias, api_base: vendor.base + "/relay/compatible-mode/v1",
      });
      if (alias === "over-budget") {
        const scope = { scope: "provider_key", scope_ref: pk.id, period: "month" };
        await seed.update("budgets", randomUUID(), { ...scope, exhausted: { reason: {
          ...scope, message: "provider key budget exceeded", limit: "0", spent: "0", remaining: "0",
        } } });
      }
      ids.set(alias, (await seed.createModel({
        display_name: alias, model_name: "vendor-" + alias, provider, provider_key_id: pk.id,
      })).id);
    }
    await seed.createGuardrail({ name: "reject-marker", kind: "keyword", enabled: true,
      hook_point: "input", fail_open: false, patterns: [{ kind: "literal", value: "blocked-marker" }] });
    revokedKeyId = (await seed.createApiKey({ key_hash: keyHash(revoked), allowed_models: ["cn"] })).id;
    await seed.createApiKey({ key_hash: keyHash(caller), allowed_models: [...ids.keys()] });
    const probe = new ProxyClient(app.proxyUrl, caller);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await vendor?.close();
    await otlp?.close();
  });

  async function usage(id: string) {
    const carrier = () => otlp.spans.filter(s => s.attributes["gen_ai.response.id"] === id &&
      s.attributes["aisix.attempt_index"] !== undefined);
    await waitConfigPropagation(async () => carrier().length > 0);
    expect(carrier()).toHaveLength(1);
    return carrier()[0];
  }

  test("one connection routes two serial tasks and meters the last cumulative duration on each", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const before = await scrapeMetrics(app.metricsUrl);
      const tasks = [];
      for (const alias of ["cn", "intl"]) {
        const { id, frames, audio } = await recognize(client, alias);
        expect(frames).toEqual([
          { header: { task_id: id, event: "task-started" }, payload: {} },
          { header: { task_id: id, event: "result-generated" }, payload: { output: { sentence: { text: "你好" } }, usage: { duration: 2 } } },
          { header: { task_id: id, event: "result-generated" }, payload: { output: { sentence: { text: "你好世界" } }, usage: { duration: 6 } } },
          { header: { task_id: id, event: "task-finished" }, payload: {} },
        ]);
        expect(vendor.calls.find(call => call.id === id)).toEqual({
          id, path: "/relay" + funAsrPath, auth: "Bearer sk-provider-" + alias, model: "vendor-" + alias, audio: [audio],
        });
        tasks.push({ id, alias });
      }
      const requests = [];
      for (const { id, alias } of tasks) {
        const row = await usage(id);
        expect(row.attributes).toMatchObject({
          "aisix.audio.duration_seconds": 6, "aisix.operation": "transcription", "aisix.requested_model": alias,
          "aisix.model_id": ids.get(alias), "http.response.status_code": 200, "aisix.is_final_attempt": true,
        });
        const req = row.attributes["aisix.request_id"];
        expect(otlp.spans.filter(s => s.kind === 2 && s.attributes["aisix.request_id"] === req)).toHaveLength(1);
        requests.push(req);
      }
      expect(new Set(requests).size).toBe(2);
      expect(metricDelta(before, await scrapeMetrics(app.metricsUrl), "aisix_usage_events_emitted_total",
        { handler: "audio", inbound_protocol: "dashscope" })).toBe(2);
    } finally {
      client.ws.close();
    }
  });

  test("checks credentials and grants again on an upgraded socket", async ctx => {
    if (!app) return ctx.skip();
    await expect(openFunAsr(app.proxyUrl, "unrecognized")).rejects.toThrow(/401/);
    const observer = new ProxyClient(app.proxyUrl, revoked);
    const grants = async (alias: string) => {
      const { body } = await observer.listModels();
      return (body as { data?: { id: string }[] }).data?.some(m => m.id === alias) ?? false;
    };
    for (const change of ["grant", "key"]) {
      await seed.update("api_keys", revokedKeyId, { key_hash: keyHash(revoked), allowed_models: ["cn"] });
      await waitConfigPropagation(() => grants("cn"));
      const client = await openFunAsr(app.proxyUrl, revoked);
      try {
        await recognize(client, "cn");
        if (change === "grant") {
          await seed.update("api_keys", revokedKeyId, { key_hash: keyHash(revoked), allowed_models: ["intl"] });
          await waitConfigPropagation(() => grants("intl"));
        } else {
          await seed.delete("api_keys", revokedKeyId);
          await waitConfigPropagation(async () => (await observer.listModels()).status === 401);
        }
        const id = randomUUID();
        client.ws.send(JSON.stringify(runTask(id, "cn")));
        expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed" });
        expect(vendor.calls.some(call => call.id === id)).toBe(false);
      } finally {
        client.ws.close();
      }
    }
  });

  test("checks the newly selected provider key budget on the second task", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      await recognize(client, "cn");
      const id = randomUUID();
      client.ws.send(JSON.stringify(runTask(id, "over-budget")));
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed", error_code: "budget_exceeded" });
      expect(vendor.calls.some(call => call.id === id)).toBe(false);
    } finally {
      client.ws.close();
    }
  });

  test("rejects a guardrail-blocked command before the upstream handshake", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      const run = runTask(id, "cn");
      client.ws.send(JSON.stringify({ ...run, payload: { ...run.payload, input: { text: "blocked-marker" } } }));
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed" });
      expect(vendor.calls.some(call => call.id === id)).toBe(false);
    } finally {
      client.ws.close();
    }
  });

  test.each([
    ["failure", "upstream_in_band"],
    ["drop", "transport"],
  ])("reports upstream %s on the task that received it", async (alias, errorClass) => {
    if (!app) return;
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const { id, frames } = await recognize(client, alias);
      expect(frames.at(-1).header).toMatchObject({
        task_id: id, event: "task-failed",
        ...(alias === "failure" ? { error_code: "BadRequest", error_message: "bad audio" } : {}),
      });
      expect((await usage(id)).attributes).toMatchObject({
        "aisix.audio.duration_seconds": 6, "aisix.error_class": errorClass, "http.response.status_code": 502,
      });
    } finally {
      client.ws.close();
    }
  });

  test("records partial duration when the client closes an active task", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    const id = randomUUID();
    try {
      client.ws.send(JSON.stringify(runTask(id, "cn")));
      expect((await client.frame()).header.event).toBe("task-started");
      client.ws.send(Buffer.from([0, 128]));
      expect((await client.frame()).header.event).toBe("result-generated");
    } finally {
      client.ws.close();
    }
    expect((await usage(id)).attributes).toMatchObject({
      "aisix.audio.duration_seconds": 2, "aisix.error_class": "client_disconnected", "http.response.status_code": 499,
    });
  });
});
