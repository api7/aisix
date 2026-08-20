import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import {
  startMockOtlp,
  type CapturedSpan,
  type MockOtlp,
} from "../harness/otlp-mock.js";

// E2E (AISIX-Cloud#1279): the OTLP export is a real trace, not a bag of
// disconnected spans. One request — including a failover with a failed
// attempt — exports ONE trace: an HTTP SERVER span for the inbound
// request, a logical GenAI CLIENT span covering every upstream attempt,
// and one CLIENT child per attempt, all sharing a trace id, with
// nanosecond timestamps bracketed parent-over-child.
//
// The trust rules ride along: a valid inbound W3C `traceparent` continues
// the caller's trace (the SERVER span parents under it and carries the
// caller's `tracestate`); anything malformed starts a local trace; an
// inbound sampled=1 flag cannot force export past an exporter's
// `sample_rate: 0`; and the caller's trace context never reaches the
// upstream provider — on the standard pipeline even under a
// `forward_client_headers: ["*"]` glob, and on `/passthrough/*` where
// every unlisted header is otherwise forwarded verbatim.
//
// Delivery integrity: a transiently failing receiver sees the SAME span
// ids again on the retry — ids are frozen at emission time, not minted
// per encode.

const CALLER_PLAINTEXT = "sk-trace-hierarchy-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** A well-formed W3C traceparent the tests reuse. */
const REMOTE_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736";
const REMOTE_PARENT_ID = "00f067aa0ba902b7";
const VALID_TRACEPARENT = `00-${REMOTE_TRACE_ID}-${REMOTE_PARENT_ID}-01`;

/** OTLP SpanKind values (opentelemetry-proto trace.proto). */
const KIND_SERVER = 2;
const KIND_CLIENT = 3;

async function waitForSpans(
  recv: MockOtlp,
  requestId: string,
  count: number,
  timeoutMs = 10_000,
): Promise<CapturedSpan[]> {
  const deadline = Date.now() + timeoutMs;
  const matching = () =>
    recv.spans.filter((s) => s.attributes["aisix.request_id"] === requestId);
  while (Date.now() < deadline) {
    const hits = matching();
    if (hits.length >= count) return hits;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `expected ${count} spans for request_id=${requestId}, saw ${matching().length}`,
  );
}

const nanos = (value: string): bigint => BigInt(value);

/**
 * Cover the exporter pipelines' independent ~1s flush phases before
 * trusting a negative or exact-count assertion: a late batch (or a
 * duplicate one, if emission regressed) can land up to a flush interval
 * plus a retry after the first spans arrive.
 */
const settle = () => new Promise((r) => setTimeout(r, 2_500));

describe("trace hierarchy e2e (AISIX-Cloud#1279)", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  const upstreams: OpenAiUpstream[] = [];
  const receivers: MockOtlp[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
      // The passthrough leg addresses a route, not a model.
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await Promise.all(receivers.map((r) => r.close()));
  });

  async function propagate(): Promise<void> {
    const canary = `sk-canary-trace-${Date.now()}-${Math.random()}`;
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      await res.text();
      return res.status === 200;
    });
  }

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extraPk: Record<string, unknown> = {},
  ): Promise<void> {
    const providerKey = await seed!.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      ...extraPk,
    });
    await seed!.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
    });
  }

  function okUpstreamBody(id: string) {
    return {
      id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "gpt-4o-mini",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "ok" },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
    };
  }

  async function driveChat(
    model: string,
    headers: Record<string, string> = {},
  ): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "trace me" }],
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();
    return requestId!;
  }

  test("a failover request exports one trace: SERVER → logical CLIENT → attempt children", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-hier-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const primary = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "primary down", type: "server_error" } },
    });
    const secondary = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-hier"),
    });
    upstreams.push(primary, secondary);
    await createOpenAiModel("trace-hier-primary", primary);
    await createOpenAiModel("trace-hier-secondary", secondary);
    await seed.createModel({
      display_name: "trace-hier-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "trace-hier-primary" },
          { model: "trace-hier-secondary" },
        ],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    await propagate();

    const requestId = await driveChat("trace-hier-virtual");

    // 2 attempts + logical + server.
    await waitForSpans(otlp, requestId, 4);
    await settle();
    const spans = otlp.spans.filter(
      (s) => s.attributes["aisix.request_id"] === requestId,
    );
    expect(spans).toHaveLength(4);

    // One trace: every span shares one non-zero trace id, all ids distinct.
    const traceIds = new Set(spans.map((s) => s.traceId));
    expect(traceIds.size).toBe(1);
    expect([...traceIds][0]).toMatch(/^[0-9a-f]{32}$/);
    expect([...traceIds][0]).not.toBe("0".repeat(32));
    expect(new Set(spans.map((s) => s.spanId)).size).toBe(4);

    const server = spans.find((s) => s.kind === KIND_SERVER);
    expect(server, "exactly one SERVER span").toBeTruthy();
    // No inbound traceparent → the SERVER span roots the trace.
    expect(server!.parentSpanId).toBe("");

    const clients = spans.filter((s) => s.kind === KIND_CLIENT);
    expect(clients).toHaveLength(3);
    const logical = clients.find((s) => s.parentSpanId === server!.spanId);
    expect(logical, "one CLIENT span parented to SERVER").toBeTruthy();

    const attempts = clients
      .filter((s) => s.parentSpanId === logical!.spanId)
      .sort(
        (a, b) =>
          Number(a.attributes["aisix.attempt_index"]) -
          Number(b.attributes["aisix.attempt_index"]),
      );
    expect(attempts).toHaveLength(2);
    expect(attempts[0].attributes["aisix.attempt_kind"]).toBe("initial");
    expect(attempts[0].attributes["aisix.error_class"]).toBe("upstream_status");
    expect(attempts[1].attributes["aisix.attempt_kind"]).toBe("fallback");
    expect(attempts[1].attributes["gen_ai.usage.input_tokens"]).toBe(3);

    // Nanosecond bracketing: server ⊇ logical ⊇ each attempt, and the
    // failed attempt starts no later than the fallback.
    expect(nanos(server!.startTimeUnixNano)).toBeLessThanOrEqual(
      nanos(logical!.startTimeUnixNano),
    );
    expect(nanos(logical!.endTimeUnixNano)).toBeLessThanOrEqual(
      nanos(server!.endTimeUnixNano),
    );
    for (const attempt of attempts) {
      expect(nanos(logical!.startTimeUnixNano)).toBeLessThanOrEqual(
        nanos(attempt.startTimeUnixNano),
      );
      expect(nanos(attempt.endTimeUnixNano)).toBeLessThanOrEqual(
        nanos(logical!.endTimeUnixNano),
      );
    }
    expect(nanos(attempts[0].startTimeUnixNano)).toBeLessThanOrEqual(
      nanos(attempts[1].startTimeUnixNano),
    );
  });

  test("a valid inbound traceparent parents the SERVER span and continues the caller's trace", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-remote-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-remote"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-remote-direct", upstream);
    await propagate();

    const requestId = await driveChat("trace-remote-direct", {
      traceparent: VALID_TRACEPARENT,
      tracestate: "vendor=x",
    });

    // Direct model, one attempt: server + logical + attempt.
    const spans = await waitForSpans(otlp, requestId, 3);
    for (const span of spans) {
      expect(span.traceId).toBe(REMOTE_TRACE_ID);
    }
    const server = spans.find((s) => s.kind === KIND_SERVER)!;
    expect(server.parentSpanId).toBe(REMOTE_PARENT_ID);
    expect(server.traceState).toBe("vendor=x");
    // Span flags: sampled (0x01) | HAS_IS_REMOTE (0x100) | IS_REMOTE (0x200).
    expect(server.flags & 0x200).toBe(0x200);
  });

  test("a malformed traceparent starts a local trace and discards tracestate", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-badparent-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-bad"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-bad-direct", upstream);
    await propagate();

    for (const bad of [
      // Forbidden version.
      `ff-${REMOTE_TRACE_ID}-${REMOTE_PARENT_ID}-01`,
      // All-zero trace id.
      `00-${"0".repeat(32)}-${REMOTE_PARENT_ID}-01`,
      // Uppercase hex.
      `00-${REMOTE_TRACE_ID.toUpperCase()}-${REMOTE_PARENT_ID}-01`,
    ]) {
      const requestId = await driveChat("trace-bad-direct", {
        traceparent: bad,
        tracestate: "vendor=x",
      });
      const spans = await waitForSpans(otlp, requestId, 3);
      const server = spans.find((s) => s.kind === KIND_SERVER)!;
      expect(server.parentSpanId, `local root for ${bad}`).toBe("");
      expect(spans[0].traceId).not.toBe(REMOTE_TRACE_ID);
      expect(server.traceState).toBe("");
    }
  });

  test("an inbound sampled=1 flag cannot override an exporter's sample_rate: 0", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const zeroRecv = await startMockOtlp();
    const controlRecv = await startMockOtlp();
    receivers.push(zeroRecv, controlRecv);
    await seed.createObservabilityExporter({
      name: "trace-sample-zero",
      enabled: true,
      kind: "otlp_http",
      endpoint: zeroRecv.url,
      sample_rate: 0.0,
    });
    await seed.createObservabilityExporter({
      name: "trace-sample-control",
      enabled: true,
      kind: "otlp_http",
      endpoint: controlRecv.url,
      sample_rate: 1.0,
    });
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-sample"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-sample-direct", upstream);
    await propagate();

    const requestId = await driveChat("trace-sample-direct", {
      traceparent: VALID_TRACEPARENT, // flags 01 = sampled
    });

    // The control exporter proves the export pipeline ran end-to-end AND
    // that the inbound traceparent was accepted (otherwise this test says
    // nothing about the sampled flag) — the SERVER span continues it.
    const control = await waitForSpans(controlRecv, requestId, 3);
    const controlServer = control.find((s) => s.kind === KIND_SERVER)!;
    expect(controlServer.parentSpanId).toBe(REMOTE_PARENT_ID);
    // ...and only after covering the zero exporter's own flush phase is
    // its silence a sampling decision rather than slowness: the caller's
    // sampled=1 did not force its way in.
    await settle();
    expect(
      zeroRecv.spans.filter(
        (s) => s.attributes["aisix.request_id"] === requestId,
      ),
    ).toHaveLength(0);
  });

  test("the caller's trace context never reaches the provider — even under a forward_client_headers glob", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-leak"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-leak-direct", upstream, {
      request: { forward_client_headers: ["*"] },
    });
    await propagate();

    await driveChat("trace-leak-direct", {
      traceparent: VALID_TRACEPARENT,
      tracestate: "vendor=x",
      "x-keep": "forwarded",
    });

    expect(upstream.receivedRequests).toHaveLength(1);
    const seen = upstream.receivedRequests[0].headers;
    // The glob genuinely forwarded the benign header...
    expect(seen["x-keep"]).toBe("forwarded");
    // ...and the trace context stayed behind.
    expect(seen["traceparent"]).toBeUndefined();
    expect(seen["tracestate"]).toBeUndefined();
  });

  test("passthrough strips the caller's trace context unconditionally", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-pt"),
    });
    upstreams.push(upstream);
    const pk = await seed.createProviderKey({
      display_name: "trace-pt-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createPassthroughRoute({
      name: "trace-pt-route",
      path_prefix: "/passthrough/trace-pt",
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
    });
    await propagate();

    const res = await fetch(
      `${app.proxyUrl}/passthrough/trace-pt/v1/chat/completions`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
          traceparent: VALID_TRACEPARENT,
          tracestate: "vendor=x",
          "x-custom-app": "kept",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "tunnel me" }],
        }),
      },
    );
    expect(res.status).toBe(200);
    await res.text();

    expect(upstream.receivedRequests).toHaveLength(1);
    const seen = upstream.receivedRequests[0].headers;
    // Passthrough forwards unlisted headers verbatim...
    expect(seen["x-custom-app"]).toBe("kept");
    // ...which is exactly why the trace context needs the explicit strip.
    expect(seen["traceparent"]).toBeUndefined();
    expect(seen["tracestate"]).toBeUndefined();
  });

  test("a streamed request exports the same three-span hierarchy from the drop-guard emit", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-stream-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const upstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "chatcmpl-trace-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content: "hi" }, finish_reason: null }],
        }),
        JSON.stringify({
          id: "chatcmpl-trace-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
        }),
        "[DONE]",
      ],
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-stream-direct", upstream);
    await propagate();

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "trace-stream-direct",
        messages: [{ role: "user", content: "stream me" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    // Drain — the terminal emit fires from the stream's drop guard.
    await res.text();

    const spans = await waitForSpans(otlp, requestId!, 3);
    expect(spans).toHaveLength(3);
    const server = spans.find((s) => s.kind === KIND_SERVER)!;
    const logical = spans.find(
      (s) => s.kind === KIND_CLIENT && s.parentSpanId === server.spanId,
    )!;
    const attempt = spans.find((s) => s.parentSpanId === logical.spanId)!;
    expect(attempt.attributes["gen_ai.usage.output_tokens"]).toBe(4);
    expect(nanos(attempt.endTimeUnixNano)).toBeLessThanOrEqual(
      nanos(server.endTimeUnixNano),
    );
  });

  test("an all-failed failover exports one SERVER span with the failed attempts under it", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-allfail-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const first = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "down 1", type: "server_error" } },
    });
    const second = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "down 2", type: "server_error" } },
    });
    upstreams.push(first, second);
    await createOpenAiModel("trace-allfail-a", first);
    await createOpenAiModel("trace-allfail-b", second);
    await seed.createModel({
      display_name: "trace-allfail-virtual",
      routing: {
        strategy: "failover",
        targets: [{ model: "trace-allfail-a" }, { model: "trace-allfail-b" }],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    await propagate();

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "trace-allfail-virtual",
        messages: [{ role: "user", content: "everything is down" }],
      }),
    });
    expect(res.status).toBeGreaterThanOrEqual(500);
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    // Exactly SERVER + logical + two failed attempts: no terminal event
    // exists on this shape, so the LAST failed attempt's emission must
    // carry the structural spans — and only once.
    await waitForSpans(otlp, requestId!, 4);
    await settle();
    const spans = otlp.spans.filter(
      (s) => s.attributes["aisix.request_id"] === requestId,
    );
    expect(spans).toHaveLength(4);
    expect(spans.filter((s) => s.kind === KIND_SERVER)).toHaveLength(1);
    const server = spans.find((s) => s.kind === KIND_SERVER)!;
    const logical = spans.find(
      (s) => s.kind === KIND_CLIENT && s.parentSpanId === server.spanId,
    )!;
    const attempts = spans.filter((s) => s.parentSpanId === logical.spanId);
    expect(attempts).toHaveLength(2);
    for (const attempt of attempts) {
      expect(attempt.attributes["aisix.error_class"]).toBe("upstream_status");
    }
  });

  test("/v1/messages and /v1/responses export the same hierarchy shape", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-family-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-family"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-family-direct", upstream);
    await propagate();

    const drive = async (path: string, body: unknown) => {
      const res = await fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });
      expect(res.status, path).toBe(200);
      const requestId = res.headers.get("x-aisix-request-id");
      expect(requestId, path).toBeTruthy();
      await res.text();
      return requestId!;
    };

    for (const [path, body] of [
      [
        "/v1/messages",
        {
          model: "trace-family-direct",
          max_tokens: 32,
          messages: [{ role: "user", content: "trace me" }],
        },
      ],
      ["/v1/responses", { model: "trace-family-direct", input: "trace me" }],
    ] as const) {
      const requestId = await drive(path, body);
      const spans = await waitForSpans(otlp, requestId, 3);
      expect(spans, path).toHaveLength(3);
      const server = spans.find((s) => s.kind === KIND_SERVER)!;
      expect(server, path).toBeTruthy();
      const logical = spans.find(
        (s) => s.kind === KIND_CLIENT && s.parentSpanId === server.spanId,
      )!;
      expect(logical, path).toBeTruthy();
      const attempt = spans.find((s) => s.parentSpanId === logical.spanId)!;
      expect(attempt, path).toBeTruthy();
      expect(new Set(spans.map((s) => s.traceId)).size, path).toBe(1);
    }
  });

  test("an ensemble request exports one SERVER span and no dangling parents", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-ens-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const member = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-ens-member"),
    });
    const judge = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-ens-judge"),
    });
    upstreams.push(member, judge);
    await createOpenAiModel("trace-ens-member-a", member);
    await createOpenAiModel("trace-ens-member-b", member);
    await createOpenAiModel("trace-ens-judge", judge);
    await seed.createModel({
      display_name: "trace-ens-virtual",
      ensemble: {
        panel: [{ model: "trace-ens-member-a" }, { model: "trace-ens-member-b" }],
        judge: { model: "trace-ens-judge" },
        min_responses: 2,
      },
    });
    await propagate();

    const requestId = await driveChat("trace-ens-virtual");

    // Two panel sub-calls + the judge's terminal emission (SERVER +
    // logical carrier) = four spans.
    await waitForSpans(otlp, requestId, 4);
    await settle();
    const spans = otlp.spans.filter(
      (s) => s.attributes["aisix.request_id"] === requestId,
    );
    expect(spans).toHaveLength(4);
    expect(spans.filter((s) => s.kind === KIND_SERVER)).toHaveLength(1);
    // One trace: parent resolution below is meaningful only within it.
    expect(new Set(spans.map((s) => s.traceId)).size).toBe(1);

    // Every parent resolves within the exported trace — the orphan-span
    // regression this test exists to prevent.
    const ids = new Set(spans.map((s) => s.spanId));
    for (const span of spans) {
      if (span.parentSpanId !== "") {
        expect(
          ids.has(span.parentSpanId),
          `dangling parent ${span.parentSpanId} on ${span.name}`,
        ).toBe(true);
      }
    }
    const panel = spans.filter(
      (s) => s.attributes["aisix.attempt_kind"] === "panel",
    );
    expect(panel).toHaveLength(2);
  });

  test("a cache hit exports the SERVER span alone — no fictitious upstream span", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const otlp = await startMockOtlp();
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-cache-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-cache"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-cache-direct", upstream);
    await seed.createCachePolicy({
      name: "trace-cache-policy",
      enabled: true,
      applies_to: "all",
    });
    await propagate();

    const driveOnce = async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "trace-cache-direct",
          messages: [{ role: "user", content: "cache me exactly" }],
        }),
      });
      expect(res.status).toBe(200);
      const outcome = res.headers.get("x-aisix-cache");
      const requestId = res.headers.get("x-aisix-request-id");
      await res.text();
      return { outcome, requestId: requestId! };
    };

    // First call misses and populates; poll until a call reports a hit
    // (the write is asynchronous).
    await driveOnce();
    let hit: { outcome: string | null; requestId: string } | undefined;
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
      const attempt = await driveOnce();
      if (attempt.outcome === "hit") {
        hit = attempt;
        break;
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(hit, "no cache hit within 10s").toBeTruthy();

    await waitForSpans(otlp, hit!.requestId, 1);
    await settle();
    const spans = otlp.spans.filter(
      (s) => s.attributes["aisix.request_id"] === hit!.requestId,
    );
    expect(spans).toHaveLength(1);
    expect(spans[0].kind).toBe(KIND_SERVER);
    expect(spans[0].parentSpanId).toBe("");
  });

  test("a transient receiver failure re-delivers byte-identical span ids", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // First POST → 503 (spans still recorded); retry → 200.
    const otlp = await startMockOtlp({ failFirst: 1 });
    receivers.push(otlp);
    await seed.createObservabilityExporter({
      name: "trace-retry-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    const upstream = await startOpenAiUpstream({
      nonStreamBody: okUpstreamBody("cmpl-trace-retry"),
    });
    upstreams.push(upstream);
    await createOpenAiModel("trace-retry-direct", upstream);
    await propagate();

    const requestId = await driveChat("trace-retry-direct");

    // Wait for the retried delivery: the same spans must arrive twice.
    const deadline = Date.now() + 15_000;
    const byPost = () => {
      const mine = otlp.spans.filter(
        (s) => s.attributes["aisix.request_id"] === requestId,
      );
      const posts = new Map<number, CapturedSpan[]>();
      for (const s of mine) {
        posts.set(s.postIndex, [...(posts.get(s.postIndex) ?? []), s]);
      }
      return posts;
    };
    while (Date.now() < deadline && byPost().size < 2) {
      await new Promise((r) => setTimeout(r, 100));
    }
    const posts = [...byPost().values()];
    expect(posts.length).toBeGreaterThanOrEqual(2);

    const idSet = (spans: CapturedSpan[]) =>
      spans
        .map(
          (s) =>
            `${s.traceId}/${s.spanId}/${s.parentSpanId}/${s.kind}/` +
            `${s.startTimeUnixNano}/${s.endTimeUnixNano}`,
        )
        .sort()
        .join("|");
    // The 503'd delivery and its retry carry byte-identical ids and
    // timestamps — the regenerate-per-encode defect is gone.
    expect(idSet(posts[1])).toBe(idSet(posts[0]));
  });
});
