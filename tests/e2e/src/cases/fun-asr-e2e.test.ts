import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient, EtcdClient, keyHash, metricDelta, ProxyClient, scrapeMetrics, SeedClient, spawnApp,
  startMockSls, waitConfigPropagation, waitForLogLine, waitForSlsLog, type MockSls, type SpawnedApp,
} from "../harness/index.js";
import { funAsrPath, openFunAsr, recognize, runTask, startFunAsrUpstream } from "../harness/fun-asr.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

describe("DashScope Fun-ASR native tasks", () => {
  const caller = "sk-asr-caller";
  const revoked = "sk-asr-revoked";
  let app: SpawnedApp;
  let seed: SeedClient;
  let otlp: MockOtlp;
  let sls: MockSls;
  let vendor: Awaited<ReturnType<typeof startFunAsrUpstream>>;
  let revokedKeyId: string;
  const ids = new Map<string, string>();

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    vendor = await startFunAsrUpstream();
    otlp = await startMockOtlp();
    sls = await startMockSls();
    app = await spawnApp({ logLevel: "info", extraEnv: {
      SLS_CRED_MOCK_AK_ID: "test-key", SLS_CRED_MOCK_AK_SECRET: "test-secret",
    } });
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({ name: "asr-usage", kind: "otlp_http", endpoint: otlp.url });
    for (const mode of ["full", "metadata_only"]) {
      await seed.createObservabilityExporter({
        name: "asr-" + mode, enabled: true, kind: "aliyun_sls", endpoint: sls.url, project: "test",
        logstore: "asr-" + mode, credential_ref: "mock", content_mode: mode, content_max_bytes: 2048,
        upstream_response_headers: ["x-asr-region"],
      });
    }
    for (const alias of [
      "cn", "intl", "failure", "drop", "over-budget", "refuse", "refuse-no-id", "guard-control",
      "guard-output", "tls", "resolve", "idle", "unread-terminal", "slow-finish",
    ]) {
      const provider = alias === "cn" ? "alibaba-cn" : "alibaba";
      const base = alias === "resolve" ? vendor.base.replace("127.0.0.1", "localhost") : vendor.base;
      const path = alias.startsWith("refuse") ? "/" + alias : "/relay";
      const pk = await seed.createProviderKey({
        provider, display_name: alias, api_key: "sk-provider-" + alias, api_base: base + path + "/compatible-mode/v1",
        ...(alias === "resolve" && { resolve_addresses: ["127.0.0.1"] }),
        ...(alias === "tls" && { tls: { verify: false } }),
        ...(alias === "cn" && {
          tls: { verify: true },
          request: {
            default_headers: {
              "x-dashscope-workspace": "${model.name}-${provider_key.name}",
              "x-asr-caller": "${request.api_key.name}", "x-asr-request": "${request.id}",
              "sec-websocket-protocol": "unnegotiated",
            },
            forward_client_headers: ["x-asr-extra", "x-dashscope-workspace", "sec-websocket-extensions"],
          },
        }),
      });
      if (alias === "over-budget") {
        const scope = { scope: "provider_key", scope_ref: pk.id, period: "month" };
        await seed.update("budgets", randomUUID(), { ...scope, exhausted: { reason: {
          ...scope, message: "provider key budget exceeded", limit: "0", spent: "0", remaining: "0",
        } } });
      }
      ids.set(alias, (await seed.createModel({
        display_name: alias, model_name: "vendor-" + alias, provider, provider_key_id: pk.id,
        ...(["idle", "drop", "failure"].includes(alias) && { cooldown: { enabled: true } }),
        ...(["idle", "unread-terminal"].includes(alias) && { stream_timeout: 400 }),
        ...(alias === "slow-finish" && { stream_timeout: 800 }),
      })).id);
    }
    await seed.createGuardrail({ name: "reject-marker", kind: "keyword", enabled: true,
      hook_point: "input", fail_open: false, patterns: [{ kind: "literal", value: "blocked-marker" }] });
    await seed.createGuardrail({ name: "reject-transcript", kind: "keyword", enabled: true,
      hook_point: "output", fail_open: false, patterns: [{ kind: "literal", value: "测试词" }] });
    revokedKeyId = (await seed.createApiKey({ key_hash: keyHash(revoked), allowed_models: ["cn"] })).id;
    await seed.createApiKey({ display_name: "asr-caller", key_hash: keyHash(caller), allowed_models: [...ids.keys()] });
    const probe = new ProxyClient(app.proxyUrl, caller);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await vendor?.close();
    await otlp?.close();
    await sls?.close();
  });

  async function usage(id: string) {
    const carrier = () => otlp.spans.filter(s => s.attributes["gen_ai.response.id"] === id &&
      s.attributes["aisix.attempt_index"] !== undefined);
    await waitConfigPropagation(async () => carrier().length > 0);
    expect(carrier()).toHaveLength(1);
    return carrier()[0];
  }

  const log = (requestId: string, mode = "full") => waitForSlsLog(sls, "asr-" + mode,
    row => row.get("request_id") === requestId && row.get("is_final_attempt") === "true", "ASR task row");

  const modelStatus = async (alias: string) =>
    (await new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl).listModelStatuses())
      .find(row => row.id === ids.get(alias))!.status;

  test("applies provider headers and captures accepted handshake metadata and text by exporter mode", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller, { "x-dashscope-workspace": "caller", "x-asr-extra": "forwarded" });
    try {
      const { id, audio } = await recognize(client, "cn");
      const headers = vendor.handshakes.at(-1)!;
      expect(headers).toMatchObject({ "x-dashscope-workspace": "cn-cn", "x-asr-extra": "forwarded",
        "x-asr-caller": "asr-caller", "x-asr-request": client.requestId, authorization: "Bearer sk-provider-cn" });
      expect(headers["sec-websocket-protocol"]).toBeUndefined();
      expect(headers["sec-websocket-extensions"]).toBeUndefined();
      const row = await log(client.requestId);
      expect(row.get("provider_request_id")).toBe(id);
      expect(JSON.parse(row.get("upstream_response_headers")!)).toEqual({ "x-asr-region": "accepted" });
      expect(JSON.parse(row.get("prompt")!)).toEqual(runTask(id, "cn"));
      expect(row.get("response")).toContain("你好世界");
      expect(row.get("response")).not.toContain(audio.toString("base64"));
      const meta = await log(client.requestId, "metadata_only");
      expect(meta.get("response")).toBeUndefined();
      expect(meta.get("prompt")).toBeUndefined();
    } finally { client.ws.close(); }
  });

  test("keeps the vendor request id, selected headers and refusal body on a refused upgrade", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      client.ws.send(JSON.stringify(runTask(id, "refuse")));
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed" });
      const full = await log(client.requestId);
      expect(full.get("provider_request_id")).toBe("upstream-refused-id");
      expect(JSON.parse(full.get("upstream_response_headers")!)).toEqual({ "x-asr-region": "refused" });
      expect(full.get("response")).toBe(JSON.stringify({ code: "AccessDenied", message: "workspace denied" }));
      expect(JSON.parse(full.get("prompt")!)).toEqual(runTask(id, "refuse"));
      expect((await log(client.requestId, "metadata_only")).get("response")).toBeUndefined();
    } finally { client.ws.close(); }
  });

  test("captures the opening command when model resolution fails before the upstream handshake", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const run = runTask(randomUUID(), "missing-model");
      const before = vendor.handshakes.length;
      client.ws.send(JSON.stringify(run));
      expect((await client.frame()).header).toMatchObject({ task_id: run.header.task_id, event: "task-failed", error_code: "model_not_found" });
      expect(vendor.handshakes).toHaveLength(before);
      expect(JSON.parse((await log(client.requestId)).get("prompt")!)).toEqual(run);
      expect((await log(client.requestId, "metadata_only")).get("prompt")).toBeUndefined();
    } finally { client.ws.close(); }
  });

  test("leaves the provider request id empty when the refused handshake has none", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      client.ws.send(JSON.stringify(runTask(randomUUID(), "refuse-no-id")));
      expect((await client.frame()).header.event).toBe("task-failed");
      expect((await log(client.requestId)).get("provider_request_id")).toBeUndefined();
    } finally { client.ws.close(); }
  });

  test.for(["resolve", "tls"])("refuses the %s override before opening an upstream connection", async (alias, ctx) => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const count = vendor.handshakes.length;
      client.ws.send(JSON.stringify(runTask(randomUUID(), alias)));
      const failure = (await client.frame()).header;
      expect(failure.event).toBe("task-failed");
      expect(failure.error_message).toMatch(/resolve_addresses.*TLS/);
      expect(vendor.handshakes).toHaveLength(count);
      expect((await log(client.requestId)).get("provider_request_id")).toBeUndefined();
    } finally { client.ws.close(); }
  });

  test("filters the decoded transcript and charges its reported duration once", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      client.ws.send(JSON.stringify(runTask(id, "guard-output")));
      expect((await client.frame()).header.event).toBe("task-started");
      client.ws.send(Buffer.from([0, 1]));
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed" });
      const row = await usage(id);
      expect(row.attributes["aisix.audio.duration_seconds"]).toBe(2);
      expect((await log(client.requestId)).get("guardrail_blocked")).toBe("true");
    } finally { client.ws.close(); }
  });

  test("checks decoded control-frame content in both directions", async ctx => {
    if (!app) return ctx.skip();
    for (const model of ["cn", "guard-control"]) {
      const client = await openFunAsr(app.proxyUrl, caller);
      try {
        const id = randomUUID();
        client.ws.send(JSON.stringify(runTask(id, model)));
        if (model === "cn") {
          expect((await client.frame()).header.event).toBe("task-started");
          const finish = { header: { action: "finish-task", task_id: id }, payload: { input: { text: "blocked-marker" } } };
          client.ws.send(JSON.stringify(finish).replace("blocked-marker", "blocked\\u002dmarker"));
        }
        expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed" });
        expect((await log(client.requestId)).get("guardrail_blocked")).toBe("true");
      } finally { client.ws.close(); }
    }
  });

  test("records malformed first frames without emitting usage", async ctx => {
    if (!app) return ctx.skip();
    const before = await scrapeMetrics(app.metricsUrl);
    for (const frame of ["invalid JSON", Buffer.from([0, 1])]) {
      const client = await openFunAsr(app.proxyUrl, caller);
      try {
        client.ws.send(frame);
        expect((await client.frame()).header.event).toBe("task-failed");
        await waitForLogLine(app, line => line.includes(client.requestId) && line.includes("status_code=400"), "invalid ASR control frame");
      } finally { client.ws.close(); }
    }
    const after = await scrapeMetrics(app.metricsUrl);
    expect(metricDelta(before, after, "aisix_proxy_requests_total", { endpoint: funAsrPath, status: "400" })).toBe(2);
    expect(metricDelta(before, after, "aisix_usage_events_emitted_total", { handler: "audio", inbound_protocol: "dashscope" })).toBe(0);
  });

  test("records a client disconnect when its terminal frame cannot be delivered", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      client.ws.send(JSON.stringify(runTask(id, "unread-terminal")));
      expect((await client.frame()).header.event).toBe("task-started");
      client.ws.pause();
      client.ws.send(JSON.stringify({ header: { action: "finish-task", task_id: id }, payload: { input: {} } }));
      expect((await usage(id)).attributes).toMatchObject({
        "http.response.status_code": 499, "aisix.audio.duration_seconds": 6, "aisix.error_class": "client_disconnected",
      });
    } finally { client.ws.terminate(); }
  });

  test("waiting for client audio does not cool down a healthy model", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      client.ws.send(JSON.stringify(runTask(id, "idle")));
      expect((await client.frame()).header.event).toBe("task-started");
      expect((await client.frame()).header.event).toBe("task-failed");
      expect((await usage(id)).attributes["http.response.status_code"]).toBe(400);
      expect(await modelStatus("idle")).not.toBe("cooldown");
    } finally { client.ws.close(); }
  });

  test("applies one stream budget to the remaining results after finish-task", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      client.ws.send(JSON.stringify(runTask(id, "slow-finish")));
      expect((await client.frame()).header.event).toBe("task-started");
      client.ws.send(JSON.stringify({ header: { action: "finish-task", task_id: id }, payload: { input: {} } }));
      expect((await client.frame()).header.event).toBe("result-generated");
      client.ws.ping();
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed", error_message: "upstream request timed out after 800ms" });
      expect((await usage(id)).attributes).toMatchObject({ "http.response.status_code": 504, "aisix.audio.duration_seconds": 6 });
    } finally { client.ws.close(); }
  });

  test.for([false, true])("routes and meters two serial tasks on one connection (incoming traceparent=%s)", async (hasParent, ctx) => {
    if (!app) return ctx.skip();
    const remoteTrace = "4bf92f3577b34da6a3ce929d0e0e4736";
    const remoteParent = "00f067aa0ba902b7";
    const client = await openFunAsr(app.proxyUrl, caller,
      hasParent ? { traceparent: `00-${remoteTrace}-${remoteParent}-01` } : {});
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
      const traces = [];
      const serverIds = [];
      for (const { id, alias } of tasks) {
        const row = await usage(id);
        expect(row.attributes).toMatchObject({
          "aisix.audio.duration_seconds": 6, "aisix.operation": "transcription", "aisix.requested_model": alias,
          "aisix.model_id": ids.get(alias), "http.response.status_code": 200, "aisix.is_final_attempt": true,
        });
        const req = row.attributes["aisix.request_id"];
        const servers = otlp.spans.filter(s => s.kind === 2 && s.attributes["aisix.request_id"] === req);
        expect(servers).toHaveLength(1);
        expect(row.traceId).toMatch(/^[0-9a-f]{32}$/);
        expect(servers[0].traceId).toBe(row.traceId);
        expect(servers[0].parentSpanId).toBe(hasParent ? remoteParent : "");
        traces.push(row.traceId);
        serverIds.push(servers[0].spanId);
        requests.push(req);
      }
      expect(new Set(requests).size).toBe(2);
      expect(new Set(serverIds).size).toBe(2);
      if (hasParent) expect(traces).toEqual([remoteTrace, remoteTrace]);
      else expect(new Set(traces).size).toBe(2);
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
        const before = vendor.handshakes.length;
        client.ws.send(JSON.stringify(runTask(id, "cn")));
        expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed" });
        expect(vendor.handshakes).toHaveLength(before);
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
      const before = vendor.handshakes.length;
      client.ws.send(JSON.stringify(runTask(id, "over-budget")));
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed", error_code: "budget_exceeded" });
      expect(vendor.handshakes).toHaveLength(before);
    } finally {
      client.ws.close();
    }
  });

  test("rejects a guardrail-blocked command before the upstream handshake", async ctx => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const id = randomUUID();
      const before = vendor.handshakes.length;
      const run = runTask(id, "cn");
      client.ws.send(JSON.stringify({ ...run, payload: { ...run.payload, input: { text: "blocked-marker" } } }));
      expect((await client.frame()).header).toMatchObject({ task_id: id, event: "task-failed", error_code: "content_filter" });
      expect(vendor.handshakes).toHaveLength(before);
    } finally {
      client.ws.close();
    }
  });

  test.for([
    ["failure", "upstream_in_band"],
    ["drop", "transport"],
  ])("reports upstream %s on the task that received it", async ([alias, errorClass], ctx) => {
    if (!app) return ctx.skip();
    const client = await openFunAsr(app.proxyUrl, caller);
    try {
      const { id, frames } = await recognize(client, alias);
      expect(frames.at(-1).header).toMatchObject({
        task_id: id, event: "task-failed",
        ...(alias === "failure" ? { error_code: "BadRequest", error_message: "bad audio" } : {}),
      });
      expect((await usage(id)).attributes).toMatchObject({
        "aisix.audio.duration_seconds": 6, "aisix.error_class": errorClass, "http.response.status_code": alias === "failure" ? 400 : 502,
      });
      if (alias === "failure") {
        expect(await modelStatus(alias)).not.toBe("cooldown");
        expect(JSON.parse((await log(client.requestId)).get("response")!)).toEqual(frames.at(-1));
        await waitConfigPropagation(async () => client.ws.readyState === client.ws.CLOSED);
      } else {
        expect(await modelStatus(alias)).toBe("cooldown");
      }
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
