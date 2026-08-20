import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (#655): the DP emits one UsageEvent per upstream attempt — the
// initial try, each retry, and each fallback — all sharing the request's
// `request_id` (the trace/group key). A Model Group whose primary fails
// and secondary succeeds must therefore emit TWO telemetry records: a
// failed `initial` attempt on the primary and a successful `fallback`
// attempt on the secondary.
//
// Usage telemetry has no cp-api receiver in DP e2e, so we observe the
// emitted field VALUES through the per-env OTLP/HTTP fan-out: register a
// mock OTLP receiver as an `observability_exporter`, drive one failover
// request, and assert two spans carrying the per-attempt attributes
// (`aisix.attempt_index` / `aisix.attempt_kind` / `aisix.attempt_model`
// / `aisix.error_class`) share the same `aisix.request_id`.

const CALLER_PLAINTEXT = "sk-per-attempt-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface OtlpReceiver {
  url: string;
  /** All span attribute maps recorded across every posted batch. */
  spanAttrs: Array<Record<string, string>>;
  /**
   * Same spans, plus the emitted `latency_ms`. The sink derives a span's
   * start from `occurred_at - latency_ms`, so `end - start` recovers the
   * UsageEvent's `latency_ms` exactly (`occurred_at`'s second-level
   * precision cancels out in the subtraction).
   */
  spans: Array<{ attrs: Record<string, string>; latencyMs: number }>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
  const spans: Array<{ attrs: Record<string, string>; latencyMs: number }> = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spanAttrs.push(attrs);
              const latencyMs = Number(
                (BigInt(span.endTimeUnixNano ?? 0) -
                  BigInt(span.startTimeUnixNano ?? 0)) /
                  1_000_000n,
              );
              spans.push({ attrs, latencyMs });
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spanAttrs,
    spans,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function waitForAttempts(
  recv: OtlpReceiver,
  requestId: string,
  count: number,
  timeoutMs = 10_000,
): Promise<Array<Record<string, string>>> {
  const deadline = Date.now() + timeoutMs;
  // Attempt carriers only: since AISIX-Cloud#1279 the export also ships
  // the structural SERVER / logical spans of the trace hierarchy, which
  // share `aisix.request_id` but carry no per-attempt attributes.
  const matching = () =>
    recv.spanAttrs.filter(
      (a) =>
        a["aisix.request_id"] === requestId &&
        a["aisix.attempt_index"] !== undefined,
    );
  while (Date.now() < deadline) {
    const hits = matching();
    if (hits.length >= count) return hits;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `expected ${count} attempt spans for request_id=${requestId}, ` +
      `saw ${matching().length}`,
  );
}

/** Like `waitForAttempts`, but keeps each span's recovered `latency_ms`. */
async function waitForAttemptSpans(
  recv: OtlpReceiver,
  requestId: string,
  count: number,
  timeoutMs = 10_000,
): Promise<Array<{ attrs: Record<string, string>; latencyMs: number }>> {
  const deadline = Date.now() + timeoutMs;
  // Attempt carriers only — see `waitForAttempts`.
  const matching = () =>
    recv.spans.filter(
      (s) =>
        s.attrs["aisix.request_id"] === requestId &&
        s.attrs["aisix.attempt_index"] !== undefined,
    );
  while (Date.now() < deadline) {
    const hits = matching();
    if (hits.length >= count) {
      return hits.sort(
        (a, b) =>
          Number(a.attrs["aisix.attempt_index"]) -
          Number(b.attrs["aisix.attempt_index"]),
      );
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `expected ${count} attempt spans for request_id=${requestId}, ` +
      `saw ${matching().length}`,
  );
}

describe("per-attempt telemetry e2e (#655): one UsageEvent per upstream attempt", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let otlp: OtlpReceiver | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    otlp = await startOtlpReceiver();
    await seed.createObservabilityExporter({
      name: "per-attempt-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await otlp?.close();
  });

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
    });
  }

  test("a failover request emits a failed initial + successful fallback attempt sharing request_id", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const primary = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "primary down", type: "server_error" } },
    });
    const secondary = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-per-attempt-fallback",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "fallback worked" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      },
    });
    upstreams.push(primary, secondary);

    await createOpenAiModel("attempt-primary", primary);
    await createOpenAiModel("attempt-secondary", secondary);
    await seed.createModel({
      display_name: "attempt-virtual",
      routing: {
        strategy: "failover",
        targets: [{ model: "attempt-primary" }, { model: "attempt-secondary" }],
        retries: 0,
        max_fallbacks: 1,
      },
    });

    // Gate on DP-snapshot presence rather than probing the virtual —
    // probing would warm the primary's cooldown (every retryable upstream
    // failure cools the failing direct target) and the measured request
    // would then skip the primary entirely. A throwaway canary key seeded
    // AFTER the virtual authenticates only once the snapshot has caught
    // up past it (watch events apply in revision order).
    const canary = `sk-canary-attempt-${Date.now()}`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "attempt-virtual",
        messages: [{ role: "user", content: "fail over please" }],
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    await res.text();

    // Exactly the primary (initial) then the secondary (fallback).
    expect(primary.receivedRequests.length).toBe(1);
    expect(secondary.receivedRequests.length).toBe(1);

    const attempts = await waitForAttempts(otlp, requestId!, 2);
    attempts.sort(
      (a, b) =>
        Number(a["aisix.attempt_index"]) - Number(b["aisix.attempt_index"]),
    );

    // Attempt 0: failed initial try on the primary.
    expect(attempts[0]["aisix.attempt_index"]).toBe("0");
    expect(attempts[0]["aisix.attempt_kind"]).toBe("initial");
    expect(attempts[0]["aisix.attempt_model"]).toBe("attempt-primary");
    expect(attempts[0]["aisix.error_class"]).toBe("upstream_status");
    expect(attempts[0]["http.response.status_code"]).toBe("502");
    expect(attempts[0]["gen_ai.usage.input_tokens"]).toBe("0");

    // Attempt 1: successful fallback on the secondary with real tokens.
    expect(attempts[1]["aisix.attempt_index"]).toBe("1");
    expect(attempts[1]["aisix.attempt_kind"]).toBe("fallback");
    expect(attempts[1]["aisix.attempt_model"]).toBe("attempt-secondary");
    expect(attempts[1]["aisix.error_class"]).toBeUndefined();
    expect(attempts[1]["http.response.status_code"]).toBe("200");
    expect(attempts[1]["gen_ai.usage.input_tokens"]).toBe("3");
    expect(attempts[1]["gen_ai.usage.output_tokens"]).toBe("4");
  });

  // `latency_ms` is documented (usage.rs) as scoped to ONE attempt, which
  // is what makes the events summable. The winning attempt used to report
  // the whole request instead — it measured from request entry, so it
  // swallowed every preceding failed attempt (plus parsing, guardrails and
  // the retry backoff) and double-counted them against the failed events'
  // own latency.
  //
  // Both tests make the FAILING attempt the slow one: a correctly scoped
  // winner is then far faster than the loser, while a request-scoped
  // winner is necessarily slower (it contains the loser).
  const SLOW_PRIMARY_MS = 700;

  // Non-streaming coverage runs against /v1/responses: the /v1/chat/completions
  // handler already scoped its winner correctly, while the `/v1/responses` and
  // `/v1/messages` handlers passed the request-level elapsed through.
  test("the winning attempt's latency_ms covers that attempt only, not the whole request", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const primary = await startOpenAiUpstream({
      status: 502,
      responseDelayMs: SLOW_PRIMARY_MS,
      errorBody: { error: { message: "slow then down", type: "server_error" } },
    });
    const secondary = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_latency_scope",
        object: "response",
        status: "completed",
        model: "gpt-4o-mini",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "fast winner" }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
      },
    });
    upstreams.push(primary, secondary);

    await createOpenAiModel("latency-scope-primary", primary);
    await createOpenAiModel("latency-scope-secondary", secondary);
    await seed.createModel({
      display_name: "latency-scope-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "latency-scope-primary" },
          { model: "latency-scope-secondary" },
        ],
        retries: 0,
        max_fallbacks: 1,
      },
    });

    const canary = `sk-canary-latency-${Date.now()}`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });

    const metricsBefore = await scrapeMetrics(app.metricsUrl);
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "latency-scope-virtual",
        input: "measure me",
      }),
    });
    expect(res.status).toBe(200);
    // The Responses passthrough labels the response `x-aisix-request-id`.
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    await res.text();

    const spans = await waitForAttemptSpans(otlp, requestId!, 2);
    expect(spans[0].attrs["aisix.attempt_kind"]).toBe("initial");
    expect(spans[1].attrs["aisix.attempt_kind"]).toBe("fallback");

    // The failed initial paid the upstream's delay.
    expect(spans[0].latencyMs).toBeGreaterThanOrEqual(SLOW_PRIMARY_MS - 100);
    // The winner talked to a fast upstream, so its own latency is small.
    // Request-scoped it would be >= the primary's delay instead.
    expect(spans[1].latencyMs).toBeLessThan(SLOW_PRIMARY_MS / 2);
    expect(spans[1].latencyMs).toBeLessThan(spans[0].latencyMs);

    // The same failover, seen through the per-attempt COUNTERS
    // (AISIX-Cloud#1299). Driven here rather than in a fixture of its own
    // because the emit chokepoint is shared with chat and messages, while
    // the requested-model reference each handler hands it is not — so the
    // fallback label is only pinned on this endpoint by driving it.
    const metricsAfter = await scrapeMetrics(app.metricsUrl);
    const delta = (name: string, want: Record<string, string>) =>
      metricDelta(metricsBefore, metricsAfter, name, want);
    expect(
      delta("aisix_deployment_failure_responses_total", {
        model: "latency-scope-primary",
      }),
    ).toBe(1);
    expect(
      delta("aisix_deployment_success_responses_total", {
        model: "latency-scope-secondary",
      }),
    ).toBe(1);
    expect(
      delta("aisix_routing_successful_fallbacks_total", {
        model: "latency-scope-virtual",
        fallback_model: "latency-scope-secondary",
      }),
    ).toBe(1);
  });

  test("a streamed winner's latency_ms is attempt-scoped too (end-of-stream emit)", async (ctx) => {
    if (!etcdReachable || !app || !seed || !otlp) {
      ctx.skip();
      return;
    }

    const primary = await startOpenAiUpstream({
      status: 502,
      responseDelayMs: SLOW_PRIMARY_MS,
      errorBody: { error: { message: "slow then down", type: "server_error" } },
    });
    const secondary = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "chatcmpl-latency-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content: "hi" }, finish_reason: null }],
        }),
        JSON.stringify({
          id: "chatcmpl-latency-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
        }),
        "[DONE]",
      ],
    });
    upstreams.push(primary, secondary);

    await createOpenAiModel("latency-stream-primary", primary);
    await createOpenAiModel("latency-stream-secondary", secondary);
    await seed.createModel({
      display_name: "latency-stream-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "latency-stream-primary" },
          { model: "latency-stream-secondary" },
        ],
        retries: 0,
        max_fallbacks: 1,
      },
    });

    const canary = `sk-canary-latency-stream-${Date.now()}`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });

    const metricsBefore = await scrapeMetrics(app.metricsUrl);

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "latency-stream-virtual",
        messages: [{ role: "user", content: "stream me" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    // Drain: the winner's UsageEvent is emitted at end-of-stream.
    await res.text();

    const spans = await waitForAttemptSpans(otlp, requestId!, 2);
    expect(spans[0].attrs["aisix.attempt_kind"]).toBe("initial");
    expect(spans[0].attrs["http.response.status_code"]).toBe("502");
    expect(spans[1].attrs["aisix.attempt_kind"]).toBe("fallback");
    expect(spans[1].attrs["http.response.status_code"]).toBe("200");

    expect(spans[0].latencyMs).toBeGreaterThanOrEqual(SLOW_PRIMARY_MS - 100);
    expect(spans[1].latencyMs).toBeLessThan(SLOW_PRIMARY_MS / 2);
    expect(spans[1].latencyMs).toBeLessThan(spans[0].latencyMs);


    // The streaming dispatch loop is written separately from the

    // non-streaming one and was rewired separately too, so the counters

    // need their own real-traffic assertion here (AISIX-Cloud#1299).

    const metricsAfter = await scrapeMetrics(app.metricsUrl);

    const delta = (name: string, want: Record<string, string>) =>

      metricDelta(metricsBefore, metricsAfter, name, want);

    expect(

      delta("aisix_deployment_failure_responses_total", {

        model: "latency-stream-primary",

      }),

    ).toBe(1);

    expect(

      delta("aisix_deployment_success_responses_total", {

        model: "latency-stream-secondary",

      }),

    ).toBe(1);

    expect(

      delta("aisix_routing_successful_fallbacks_total", {

        model: "latency-stream-virtual",

        fallback_model: "latency-stream-secondary",

      }),

    ).toBe(1);
  });
});
