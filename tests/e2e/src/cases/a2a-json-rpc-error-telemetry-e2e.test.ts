import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type A2aUpstream,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// An agent can refuse a call with a JSON-RPC error inside an HTTP 200. The caller
// gets that answer as it was sent; the telemetry records the failure.

const KEY = "sk-a2a-rpc-error";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "a2a-rpc-error";
const SENTINEL = "sentinel-agent-said-this-5c1d";

describe("a JSON-RPC error from an agent inside a 200", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let agent: A2aUpstream | undefined;
  let sls: MockSls | undefined;
  let otlp: MockOtlp | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    agent = await startA2aUpstream({ rpcError: { code: -32004, message: SENTINEL } });
    sls = await startMockSls();
    otlp = await startMockOtlp();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "a2a-rpc-error-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "a2a-rpc-error",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    await seed.createObservabilityExporter({
      name: "a2a-rpc-error-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: "refuser",
      url: agent.url,
      protocol_version: "0.3",
      auth_type: "none",
      enabled: true,
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(KEY).digest("hex"),
      allowed_models: [],
      allowed_agents: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      await res.arrayBuffer();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await agent?.close();
    await sls?.close();
    await otlp?.close();
  });

  test.for([
    ["message/send", "application/json"],
    ["message/stream", "text/event-stream"],
  ] as const)("%s", async ([method, contentType], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await fetch(`${app!.proxyUrl}/a2a/refuser`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
        accept: `${contentType}, application/json`,
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method,
        params: { message: { role: "user", parts: [] } },
      }),
    });
    const body = await res.text();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain(contentType);
    expect(body).toContain(SENTINEL);
    expect(body).toContain("-32004");

    const requestId = res.headers.get("x-aisix-request-id");
    const row = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `row ${requestId}`,
    );
    expect(row.get("status_code")).toBe("200");
    expect(row.get("error_class")).toBe("upstream_in_band");
    expect(row.get("error_message")).toBe("upstream agent returned JSON-RPC error -32004");
    expect(JSON.stringify(Object.fromEntries(row))).not.toContain(SENTINEL);

    const traceId = row.get("trace_id");
    await expect
      .poll(() => otlp!.spans.filter((s) => s.traceId === traceId && s.kind === 3).length, {
        timeout: 10_000,
      })
      .toBeGreaterThan(0);
    const spans = otlp!.spans.filter((s) => s.traceId === traceId);
    const carrier = spans.find((s) => s.attributes["gen_ai.operation.name"] === "invoke_agent")!;
    expect(carrier.kind).toBe(3);
    expect(carrier.statusCode).toBe(2);
    expect(carrier.attributes["error.type"]).toBe("upstream_in_band");
    expect(spans.find((s) => s.kind === 2)?.statusCode ?? 0).toBe(0);
  });
});
