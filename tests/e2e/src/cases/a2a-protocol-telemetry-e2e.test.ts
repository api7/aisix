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
import { STREAM_ANSWER } from "../harness/upstream-a2a.js";

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
  // Two stubs, each answering in the shape its version actually defines: 1.0
  // wraps the Task in the response's payload oneof, 0.3 puts it flat under
  // `result`. One stub for both would leave the default version's real
  // response shape untested.
  let upstream10: A2aUpstream | undefined;
  let upstream03: A2aUpstream | undefined;
  let otlp: MockOtlp | undefined;
  let otlpFull: MockOtlp | undefined;
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
      )}; unparseable export bodies: ${JSON.stringify(otlp!.parseFailures)}`,
    );
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    otlp = await startMockOtlp();
    otlpFull = await startMockOtlp();
    upstream10 = await startA2aUpstream({
      cardMount: "origin",
      wireShape: "1.0",
      streamAnswer: true,
    });
    upstream03 = await startA2aUpstream({
      cardMount: "origin",
      wireShape: "0.3",
      streamAnswer: true,
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "a2a-telemetry-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    // A second exporter that opts into content. Both receive every event, so
    // the pair is what proves capture is opt-in rather than merely present:
    // the words must reach this one and no other.
    await seed.createObservabilityExporter({
      name: "a2a-telemetry-otlp-full",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlpFull.url,
      content_mode: "full",
    });
    // One agent per wire version, each against the stub that speaks it. Both
    // run the same operations, so what has to come out the same (the canonical
    // operation) and what has to differ (the announced version) are both
    // visible.
    for (const [name, version, agent] of [
      ["invoices", "1.0", upstream10],
      ["legacy", "0.3", upstream03],
    ] as const) {
      await seed.update("a2a_agents", randomUUID(), {
        name,
        url: agent.url,
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

    // The caller key is seeded last, so its arrival means every resource
    // above it has landed in the snapshot too. Gated on an endpoint this
    // suite asserts nothing about, so a defect in A2A dispatch fails its own
    // test by name instead of surfacing as a 60s propagation timeout here.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream10?.close();
    await upstream03?.close();
    await otlp?.close();
    await otlpFull?.close();
  });

  test("a completed call records the task, the context and how it ended", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

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
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

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
    // The two agents return the same task in DIFFERENT envelopes (1.0 wraps
    // it in the response's payload oneof), and both must be read.
    expect(v10.attributes["aisix.a2a.task_id"]).toBe("task-e2e-1");
    expect(v03.attributes["aisix.a2a.task_id"]).toBe("task-e2e-1");
    expect(v10.attributes["aisix.a2a.task_state"]).toBe("completed");
    expect(v03.attributes["aisix.a2a.task_state"]).toBe("completed");
    // ...while the raw method each caller actually sent is still recoverable.
    expect(v10.attributes["aisix.a2a.method"]).toBe("SendMessage");
    expect(v03.attributes["aisix.a2a.method"]).toBe("message/send");
    // And each is attributed to the version its agent was announced as.
    expect(v10.attributes["aisix.a2a.protocol_version"]).toBe("1.0");
    expect(v03.attributes["aisix.a2a.protocol_version"]).toBe("0.3");
  });

  test("a streamed task is recorded with the state its stream ended on", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

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

  test("a streamed call records how it ran, not just how it ended", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

    // The stub paces its three events apart, so a wait for the first one is
    // separable from the total: a stream that is slow to start and one that is
    // slow overall are different problems with different fixes.
    const contextId = `ctx-${randomUUID()}`;
    expect(
      await call("invoices", {
        jsonrpc: "2.0",
        id: 6,
        method: "message/stream",
        params: { message: { role: "user", contextId, parts: [] } },
      }),
    ).toBe(200);

    const span = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === contextId);
    expect(span.attributes["aisix.a2a.stream_event_count"]).toBe(6);
    // The agent's own time to first event. The stub pauses before each of its
    // three events, so a figure that actually stopped at the first one is a
    // fraction of the call; one that quietly measured the whole stream would
    // equal the total. Compared against the total rather than a fixed
    // millisecond bound, so the assertion does not track the stub's pacing.
    const ttfb = span.attributes["aisix.upstream_ttft_ms"] as number;
    const total = span.attributes["aisix.downstream_latency_ms"] as number;
    // The stub pauses equally before each of its three counted events, so an
    // honest time-to-first-event lands near a THIRD of the call. Both bounds
    // matter and each is placed far from that third rather than close to it:
    // a regression that stamped the response head reads near zero and has to
    // clear the floor, one that timed the whole stream reads near `total` and
    // has to clear the ceiling, while ordinary scheduling jitter — which
    // moves the ratio by well under either margin — cannot reach either.
    expect(ttfb).toBeGreaterThan(total / 8);
    expect(ttfb).toBeLessThan((total * 3) / 4);
  });

  test("the a2a metric family slices by agent and operation", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

    // `aisix_proxy_requests_total` already counts these calls, but only by
    // route — it cannot answer "is the invoices agent's stream failing?".
    await call("invoices", {
      jsonrpc: "2.0",
      id: 7,
      method: "message/stream",
      params: { message: { role: "user", parts: [] } },
    });

    const scrape = await fetch(`${app.metricsUrl}/metrics`);
    expect(scrape.status).toBe(200);
    const text = await scrape.text();

    expect(text).toContain(
      'aisix_a2a_requests_total{agent="invoices",operation="message/stream",status="2xx"}',
    );
    expect(text).toContain(
      'aisix_a2a_stream_events_total{agent="invoices",operation="message/stream"}',
    );
    expect(text).toContain('aisix_a2a_task_state_total{agent="invoices",state="completed"}');
    expect(text).toContain("aisix_a2a_ttfb_seconds_bucket");
    // The client-perceived duration series has to cover `/a2a` at all — and
    // cover the calls that FAILED, not only the streams that opened. A
    // streaming-only sample would report the endpoint as having no failures.
    expect(text).toMatch(/aisix_request_e2e_latency_seconds_bucket\{[^}]*endpoint="\/a2a"/);
    await call("does-not-exist-agent", {
      jsonrpc: "2.0",
      id: 8,
      method: "message/send",
      params: { message: { role: "user", parts: [] } },
    });
    await call("invoices", { jsonrpc: "2.0", id: 9, method: "message/send", params: {} });
    const withFailures = await (await fetch(`${app.metricsUrl}/metrics`)).text();
    expect(withFailures).toMatch(
      /aisix_a2a_requests_total\{agent="invoices",operation="message\/send",status="2xx"\}/,
    );
    // The ids that make a task traceable are exactly the ones that must not
    // become label values.
    expect(text).not.toMatch(/aisix_a2a_[a-z_]*\{[^}]*task_id=/);
    expect(text).not.toMatch(/aisix_a2a_[a-z_]*\{[^}]*context_id=/);
  });

  test("the words are metered always and captured only on request", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

    const contextId = `ctx-${randomUUID()}`;
    const prompt = "summarise invoice forty two for the finance team";
    expect(
      await call("invoices", {
        jsonrpc: "2.0",
        id: 10,
        method: "message/send",
        params: {
          message: { role: "user", contextId, parts: [{ kind: "text", text: prompt }] },
        },
      }),
    ).toBe(200);

    // An agent reports no usage of its own, so without the gateway counting
    // them every agent's spend would read as identical and zero.
    const metered = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === contextId);
    expect(metered.attributes["gen_ai.usage.input_tokens"]).toBeGreaterThan(0);

    // The words themselves reach only the exporter that asked for them.
    const deadline = Date.now() + EXPORT_TIMEOUT_MS;
    let captured: (typeof otlpFull.spans)[number] | undefined;
    while (Date.now() < deadline && !captured) {
      captured = otlpFull.spans.find((s) => s.attributes["gen_ai.conversation.id"] === contextId);
      if (!captured) await new Promise((resolve) => setTimeout(resolve, 250));
    }
    expect(captured, "the full-content exporter received the call").toBeDefined();
    expect(captured!.attributes["gen_ai.prompt"]).toContain(prompt);
    expect(captured!.attributes["gen_ai.completion"]).toContain("The invoice is settled.");
    // ...and never the default one, whatever else it carries.
    expect(metered.attributes["gen_ai.prompt"]).toBeUndefined();
    expect(metered.attributes["gen_ai.completion"]).toBeUndefined();
  });

  test("a streamed answer survives the progress notes around it", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

    // The agent delivers its answer as two continued artifact chunks with a
    // progress note between them and a closing statement after — the shape a
    // reference agent produces. An accumulator that treats every statement as
    // a replacement keeps only "Report generated." and loses the report.
    const contextId = `ctx-${randomUUID()}`;
    expect(
      await call("invoices", {
        jsonrpc: "2.0",
        id: 11,
        method: "message/stream",
        params: { message: { role: "user", contextId, parts: [{ kind: "text", text: "report?" }] } },
      }),
    ).toBe(200);

    const deadline = Date.now() + EXPORT_TIMEOUT_MS;
    let captured: (typeof otlpFull.spans)[number] | undefined;
    while (Date.now() < deadline && !captured) {
      captured = otlpFull.spans.find((s) => s.attributes["gen_ai.conversation.id"] === contextId);
      if (!captured) await new Promise((resolve) => setTimeout(resolve, 250));
    }
    expect(captured, "the full-content exporter received the stream").toBeDefined();

    const completion = String(captured!.attributes["gen_ai.completion"]);
    expect(completion).toContain(STREAM_ANSWER.join(""));
    expect(completion).toContain("Report generated.");
    expect(completion).not.toContain("halfway");
    // The answer appears once, not once per chunk that restated it.
    expect(completion.split(STREAM_ANSWER[0]).length - 1).toBe(1);
    // And it is counted, so a long answer is not metered as a short one.
    const metered = await awaitSpan((s) => s.attributes["gen_ai.conversation.id"] === contextId);
    expect(metered.attributes["gen_ai.usage.output_tokens"]).toBeGreaterThan(5);
  });

  test("an unrecognised method cannot become an unbounded label", async (ctx) => {
    if (!etcdReachable || !app || !upstream10 || !upstream03 || !otlp || !otlpFull)
      return ctx.skip();

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
