import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  waitConfigPropagation,
  type A2aUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// E2E for AISIX-Cloud#1215: protocol-level observability for the A2A gateway.
//
// The contract pinned here is what an operator can answer AFTER a call is
// over. Before this, an A2A call left "someone reached agent X with method Y"
// and nothing else: no task, no context, no outcome — and it was exported to
// a trace backend encoded as a chat completion, indistinguishable from a model
// inference.
//
// Observed through a real `otlp_http` exporter rather than through the
// gateway's internals, because the span a trace backend receives IS the
// user-visible surface: if the attributes are not on the wire, the feature
// does not exist however well the internals are populated.
const KEY = "sk-a2a-telemetry-e2e";
const sha256 = (value: string) => createHash("sha256").update(value).digest("hex");

/** How long to allow for the exporter's batch to reach the receiver. */
const EXPORT_TIMEOUT_MS = 20_000;

describe("a2a protocol telemetry e2e (AISIX-Cloud#1215)", () => {
  let app: SpawnedApp | undefined;
  let upstream: A2aUpstream | undefined;
  let otlp: MockOtlp | undefined;
  let etcdReachable = false;

  const call = async (agent: string, body: unknown) => {
    const res = await fetch(`${app!.proxyUrl}/a2a/${agent}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    // Drain so a streamed call reaches its end before the assertions run.
    await res.text();
    return res.status;
  };

  /** Wait for a span the predicate accepts, and return it. */
  const awaitSpan = async (
    matches: (span: { name: string; attributes: Record<string, unknown> }) => boolean,
  ) => {
    const deadline = Date.now() + EXPORT_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const found = otlp!.spans.find(matches);
      if (found) return found;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    throw new Error(
      `no matching span exported within ${EXPORT_TIMEOUT_MS}ms; saw: ${JSON.stringify(
        otlp!.spans.map((s) => s.name),
      )}`,
    );
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    otlp = await startMockOtlp();
    upstream = await startA2aUpstream({ cardMount: "origin" });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "a2a-telemetry-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    // Two agents on the SAME upstream, pinned to different wire versions:
    // the only difference between them is the vocabulary their callers use,
    // which is exactly what canonicalisation has to erase.
    for (const [name, version] of [
      ["invoices", "1.0"],
      ["legacy", "0.3"],
    ] as const) {
      await seed.update("a2a_agents", randomUUID(), {
        name,
        url: upstream.url,
        protocol_version: version,
        auth_type: "none",
        enabled: true,
      });
    }
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: [],
      allowed_agents: ["*"],
    });

    await waitConfigPropagation(async () => {
      const status = await call("invoices", {
        jsonrpc: "2.0",
        id: "gate",
        method: "message/send",
        params: { message: { role: "user", parts: [] } },
      });
      return status !== 404 && status !== 401;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await otlp?.close();
  });

  test("a completed call records the task, the context and how it ended", async (ctx) => {
    if (!etcdReachable) return ctx.skip();

    const contextId = `ctx-${randomUUID()}`;
    expect(
      await call("invoices", {
        jsonrpc: "2.0",
        id: 1,
        method: "message/send",
        params: { message: { role: "user", contextId, parts: [] } },
      }),
    ).toBe(200);

    const span = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === contextId);

    // Not a chat completion: an agent invocation, named after the agent.
    expect(span.name).toBe("invoke_agent invoices");
    expect(span.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
    expect(span.attributes["gen_ai.agent.name"]).toBe("invoices");
    // The task the call produced, and the state it ended in.
    expect(span.attributes["aisix.a2a.task_id"]).toBe("task-e2e-1");
    expect(span.attributes["aisix.a2a.task_state"]).toBe("completed");
    expect(span.attributes["aisix.a2a.operation"]).toBe("message/send");
    expect(span.attributes["aisix.a2a.protocol_version"]).toBe("1.0");
  });

  test("both wire vocabularies aggregate under one operation", async (ctx) => {
    if (!etcdReachable) return ctx.skip();

    // The same operation, spelled the way each agent's version spells it.
    const v10Context = `ctx-${randomUUID()}`;
    const v03Context = `ctx-${randomUUID()}`;
    expect(
      await call("invoices", {
        jsonrpc: "2.0",
        id: 2,
        method: "SendMessage",
        params: { message: { role: "user", contextId: v10Context, parts: [] } },
      }),
    ).toBe(200);
    expect(
      await call("legacy", {
        jsonrpc: "2.0",
        id: 3,
        method: "message/send",
        params: { message: { role: "user", contextId: v03Context, parts: [] } },
      }),
    ).toBe(200);

    const v10 = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === v10Context);
    const v03 = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === v03Context);

    // One operation for both — without this every per-operation figure in a
    // mixed-version deployment is silently split in two.
    expect(v10.attributes["aisix.a2a.operation"]).toBe("message/send");
    expect(v03.attributes["aisix.a2a.operation"]).toBe("message/send");
    // ...while the raw method each caller actually sent is still recoverable.
    expect(v10.attributes["aisix.a2a.method"]).toBe("SendMessage");
    expect(v03.attributes["aisix.a2a.method"]).toBe("message/send");
    // And each is attributed to the version its agent was announced as.
    expect(v10.attributes["aisix.a2a.protocol_version"]).toBe("1.0");
    expect(v03.attributes["aisix.a2a.protocol_version"]).toBe("0.3");
  });

  test("a streamed task is recorded with the state its stream ended on", async (ctx) => {
    if (!etcdReachable) return ctx.skip();

    const contextId = `ctx-${randomUUID()}`;
    expect(
      await call("invoices", {
        jsonrpc: "2.0",
        id: 4,
        method: "SendStreamingMessage",
        params: { message: { role: "user", contextId, parts: [] } },
      }),
    ).toBe(200);

    const span = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === contextId);

    expect(span.attributes["aisix.a2a.operation"]).toBe("message/stream");
    expect(span.attributes["aisix.a2a.task_id"]).toBe("task-e2e-stream");
    // The stream walks the task working → completed; a call recorded from the
    // request alone would have stopped at whatever the first event said.
    expect(span.attributes["aisix.a2a.task_state"]).toBe("completed");
  });

  test("an unrecognised method cannot become an unbounded label", async (ctx) => {
    if (!etcdReachable) return ctx.skip();

    // The method is caller-chosen. The raw value stays available for
    // forensics, but the aggregating field must collapse to `unknown`.
    const contextId = `ctx-${randomUUID()}`;
    await call("invoices", {
      jsonrpc: "2.0",
      id: 5,
      method: "vendor/somethingNobodyDefined",
      params: { message: { role: "user", contextId, parts: [] } },
    });

    const span = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === contextId);
    expect(span.attributes["aisix.a2a.operation"]).toBe("unknown");
    expect(span.attributes["aisix.a2a.method"]).toBe("vendor/somethingNobodyDefined");
  });
});
