import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1011: TTFT and end-to-end request latency must be exposed
// as REAL Prometheus histograms — `_bucket{le=…}` series that
// `histogram_quantile()` can aggregate across DP instances — with a
// dedicated low-cardinality label set (env_id / endpoint / model /
// provider / status_class / streaming). The pre-existing duration series
// render as summaries (client-side quantiles), which cannot be
// re-aggregated; they stay untouched.
//
// Drives a real `aisix` + etcd + mock upstream through non-streaming,
// streaming, and failing requests on /v1/chat/completions and
// /v1/messages, then scrapes the dedicated metrics listener.

const CALLER_PLAINTEXT = "sk-latency-histo-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const E2E_SERIES = "aisix_request_e2e_latency_seconds";
const TTFT_SERIES = "aisix_request_ttft_seconds";

describe("latency histograms e2e: bucketed TTFT + e2e latency (#1011)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let failUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-histo",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "quick reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
      },
    });
    // The mock serves SSE whenever streamEvents is set, so streaming
    // traffic gets its own upstream + model.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-histo","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        '{"id":"strm-histo","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"streamed text"},"finish_reason":null}]}',
        '{"id":"strm-histo","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 20,
    });
    failUpstream = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "mock outage", type: "server_error" } },
    });

    app = await spawnApp();
    const admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "histo-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "histo-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const streamPk = await admin.createProviderKey({
      display_name: "histo-stream-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "histo-stream-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });
    const failPk = await admin.createProviderKey({
      display_name: "histo-fail-pk",
      secret: "sk-mock",
      api_base: `${failUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "histo-fail-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: failPk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["histo-model", "histo-stream-model", "histo-fail-model"],
    });

    await waitConfigPropagation(async () => {
      const r = await chat("histo-model", false);
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
    await failUpstream?.close();
  });

  async function chat(model: string, stream: boolean): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "hello" }],
        ...(stream ? { stream: true } : {}),
      }),
    });
    await res.text();
    return res;
  }

  async function scrape(): Promise<string> {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    return res.text();
  }

  /** Lines of `series` matching every given label pair. */
  function bucketLines(body: string, series: string, labels: Record<string, string>): string[] {
    return body.split("\n").filter(
      (l) =>
        l.startsWith(`${series}_bucket{`) &&
        Object.entries(labels).every(([k, v]) => l.includes(`${k}="${v}"`)),
    );
  }

  test("non-streaming, streaming and failed requests land in le-bucketed series", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    expect((await chat("histo-model", false)).status).toBe(200);
    expect((await chat("histo-stream-model", true)).status).toBe(200);
    expect((await chat("histo-fail-model", false)).status).toBeGreaterThanOrEqual(500);

    // Streaming observation happens at stream completion; poll the scrape.
    let body = "";
    const deadline = Date.now() + 10_000;
    for (;;) {
      body = await scrape();
      const haveAll =
        bucketLines(body, E2E_SERIES, { streaming: "false", status_class: "2xx" }).length > 0 &&
        bucketLines(body, E2E_SERIES, { streaming: "true", status_class: "2xx" }).length > 0 &&
        bucketLines(body, E2E_SERIES, { status_class: "5xx" }).length > 0 &&
        bucketLines(body, TTFT_SERIES, { streaming: "true" }).length > 0;
      if (haveAll || Date.now() > deadline) break;
      await new Promise((r) => setTimeout(r, 100));
    }

    // Real histogram exposition for all three request shapes + TTFT.
    const nonStream = bucketLines(body, E2E_SERIES, {
      streaming: "false",
      status_class: "2xx",
      model: "histo-model",
      endpoint: "/v1/chat/completions",
      provider: "openai",
    });
    expect(nonStream.length, `non-streaming buckets in:\n${body}`).toBeGreaterThan(0);
    expect(nonStream.some((l) => l.includes('le="+Inf"'))).toBe(true);

    const streamed = bucketLines(body, E2E_SERIES, {
      streaming: "true",
      status_class: "2xx",
      model: "histo-stream-model",
    });
    expect(streamed.length, "streaming e2e buckets").toBeGreaterThan(0);

    const failed = bucketLines(body, E2E_SERIES, {
      status_class: "5xx",
      model: "histo-fail-model",
    });
    expect(failed.length, "failed-request buckets").toBeGreaterThan(0);

    const ttft = bucketLines(body, TTFT_SERIES, {
      streaming: "true",
      model: "histo-stream-model",
    });
    expect(ttft.length, "TTFT buckets").toBeGreaterThan(0);

    // histogram_quantile() needs _sum/_count too.
    expect(body).toContain(`${E2E_SERIES}_sum`);
    expect(body).toContain(`${E2E_SERIES}_count`);
    expect(body).toContain(`${TTFT_SERIES}_sum`);

    // env_id label present on every series line (standalone DP → unknown).
    for (const l of [...nonStream, ...streamed, ...failed, ...ttft]) {
      expect(l).toContain('env_id="');
    }
  });

  test("the SLO series carry no per-key/per-user labels; legacy series stay summaries", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const body = await scrape();
    for (const line of body.split("\n")) {
      if (!line.startsWith(E2E_SERIES) && !line.startsWith(TTFT_SERIES)) continue;
      for (const highCard of ["api_key_id", "user_id", "team_id", "provider_key_id", "user_name"]) {
        expect(line, `no ${highCard} on SLO series`).not.toContain(`${highCard}="`);
      }
    }
    // The pre-existing duration series keep their summary exposition —
    // adding buckets to them would multiply their high-cardinality labels.
    expect(body).not.toContain("aisix_llm_request_duration_seconds_bucket");
    expect(body).not.toContain("aisix_proxy_request_duration_seconds_bucket");
    expect(body).toContain("aisix_llm_request_duration_seconds");
  });
});
