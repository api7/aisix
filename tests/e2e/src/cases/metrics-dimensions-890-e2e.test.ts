import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Issue #890: extra metric dimensions —
//  req-1 `stream` label on the request/duration series,
//  req-2 `is_fallback` label + the request counter now also counting
//        FAILED requests (so a success rate is computable),
//  req-3 readable `provider_key_name` + `user_name` alongside the ids,
//  req-4 a dedicated bounded `aisix_llm_tokens_by_client_total{client_type}`.
const CALLER_PLAINTEXT = "sk-dims-890-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PK_NAME = "prom890-pk";
const USER_NAME = "alice-890";
const USER_ID = "member-890";
const NONSTREAM_MODEL = "dims890-nonstream";
const STREAM_MODEL = "dims890-stream";
const ERROR_MODEL = "dims890-error";

describe("metrics dimensions #890 e2e", () => {
  let app: SpawnedApp | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let errorUpstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    nonStreamUpstream = await startOpenAiUpstream({ nonStreamBody: responseBody() });
    streamUpstream = await startOpenAiUpstream({
      streamEvents: streamEvents(),
      // Small per-event delay so the stream is genuinely incremental and
      // TTFT is measurable (> 0).
      eventDelayMs: 2,
    });
    errorUpstream = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "boom" } },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const nsPk = await admin.createProviderKey({
      display_name: PK_NAME,
      secret: "sk-mock",
      api_base: `${nonStreamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: NONSTREAM_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: nsPk.id,
    });

    const stPk = await admin.createProviderKey({
      display_name: `${PK_NAME}-stream`,
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: STREAM_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stPk.id,
    });

    const errPk = await admin.createProviderKey({
      display_name: `${PK_NAME}-error`,
      secret: "sk-mock",
      api_base: `${errorUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: ERROR_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: errPk.id,
    });

    // The api-key payload carries user_id + user_name — the DP stamps the
    // readable name onto metric labels (req-3), proving cp-api's pushed name
    // flows end-to-end. The DP standalone admin API intentionally only
    // accepts key_hash/allowed_models/rate_limit (member/team identity is a
    // cp-api concern), so we write the full ApiKey straight to etcd — the
    // real production path (cp-api → etcd → DP loader).
    await new EtcdClient().put(
      `${app.etcdPrefix}/api_keys/${randomUUID()}`,
      JSON.stringify({
        key_hash: CALLER_KEY_HASH,
        allowed_models: [NONSTREAM_MODEL, STREAM_MODEL, ERROR_MODEL],
        user_id: USER_ID,
        user_name: USER_NAME,
      }),
    );
  });

  afterAll(async () => {
    await app?.exit();
    await nonStreamUpstream?.close();
    await streamUpstream?.close();
    await errorUpstream?.close();
  });

  // req-3 (names) + req-4 (client_type) + req-1 (stream="false" on a
  // non-streaming request) + req-2 (is_fallback label present, off the
  // bucketed histogram).
  test("non-streaming request carries readable names, client_type, stream=false, is_fallback=false", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: NONSTREAM_MODEL,
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status === 200;
    });

    // Raw fetch so we can set a recognised SDK User-Agent — the ProxyClient
    // wrapper hardcodes its headers. The DP normalises this to a bounded
    // `client_type` (req-4); the full UA never becomes a label.
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
        "user-agent": "openai-python/1.30.1",
      },
      body: JSON.stringify({
        model: NONSTREAM_MODEL,
        messages: [{ role: "user", content: "dims" }],
      }),
    });
    expect(res.status).toBe(200);

    const text = await scrape(app);

    // req-3: readable names ride alongside the ids (1:1, cardinality-neutral).
    expect(text).toContain(`provider_key_name="${PK_NAME}"`);
    expect(text).toContain(`user_name="${USER_NAME}"`);
    expect(text).toContain(`user_id="${USER_ID}"`);

    // req-1: a non-streaming request is labelled stream="false" on the E2E
    // latency histogram, so a TTFT-vs-E2E comparison can exclude it.
    expect(text).toMatch(
      /aisix_llm_request_duration_seconds\{[^}]*stream="false"[^}]*\}/,
    );

    // req-2: is_fallback is present on the request counter (no fallback here).
    expect(text).toMatch(
      /aisix_llm_requests_total\{[^}]*is_fallback="false"[^}]*\}/,
    );

    // req-4: dedicated bounded client metric; client_type normalised from UA.
    expect(text).toContain("aisix_llm_tokens_by_client_total");
    expect(text).toMatch(
      /aisix_llm_tokens_by_client_total\{[^}]*client_type="openai-python"[^}]*\}/,
    );

    // is_fallback must NOT leak onto the (bucketed) duration histogram — it
    // is a success-rate dimension, kept off the per-bucket series.
    for (const line of text.split("\n")) {
      if (line.startsWith("aisix_llm_request_duration_seconds")) {
        expect(line).not.toContain("is_fallback=");
      }
    }
  });

  // req-1: a streaming request is labelled stream="true" on the E2E latency
  // histogram, and TTFT (streaming-only by nature) is recorded — so the two
  // can be compared on the same streaming-only sample.
  test('streaming request carries stream="true" and records TTFT', async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: STREAM_MODEL,
        messages: [{ role: "user", content: "ready" }],
        stream: true,
      });
      return probe.status === 200;
    });

    const r = await proxy.chat({
      model: STREAM_MODEL,
      messages: [{ role: "user", content: "stream" }],
      stream: true,
    });
    expect(r.status).toBe(200);

    // TTFT is recorded from the SSE on_complete callback, which may land a
    // beat after the client finishes reading the body — poll briefly.
    let text = "";
    for (let i = 0; i < 60; i++) {
      text = await scrape(app);
      const hasStreamTrue =
        /aisix_llm_request_duration_seconds\{[^}]*stream="true"[^}]*\}/.test(
          text,
        );
      if (hasStreamTrue && text.includes("aisix_llm_time_to_first_token_seconds")) {
        break;
      }
      await new Promise((res) => setTimeout(res, 50));
    }

    expect(text).toMatch(
      /aisix_llm_request_duration_seconds\{[^}]*stream="true"[^}]*\}/,
    );
    expect(text).toContain("aisix_llm_time_to_first_token_seconds");
  });

  // req-2: the request counter previously only counted successes, so a
  // success rate was not computable from it. A FAILED upstream request must
  // now also increment aisix_llm_requests_total (outcome != success) and
  // populate aisix_proxy_failed_requests_total.
  test("a failed upstream request increments the request counter (success rate is computable)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      // Once the model has propagated, dispatch reaches the upstream and the
      // forced 500 surfaces as a 5xx — distinguishes "ready" from the 404
      // model-not-found of an un-propagated snapshot.
      const probe = await proxy.chat({
        model: ERROR_MODEL,
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status >= 500;
    });

    const r = await proxy.chat({
      model: ERROR_MODEL,
      messages: [{ role: "user", content: "fail" }],
    });
    expect(r.status).toBeGreaterThanOrEqual(500);

    const text = await scrape(app);
    // The failure is counted on the rich request counter with a non-success
    // outcome — the denominator now includes failures.
    expect(text).toMatch(
      /aisix_llm_requests_total\{[^}]*outcome="upstream_error"[^}]*\}/,
    );
    // The previously-dead failed-requests counter is now populated too.
    expect(text).toContain("aisix_proxy_failed_requests_total");
  });
});

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

function responseBody() {
  return {
    id: "chatcmpl-dims-890",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "hello" },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 11, completion_tokens: 13, total_tokens: 24 },
  };
}

function streamEvents(): string[] {
  return [
    JSON.stringify({
      id: "chatcmpl-dims-890-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { role: "assistant" } }],
    }),
    JSON.stringify({
      id: "chatcmpl-dims-890-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { content: "hello" } }],
    }),
    JSON.stringify({
      id: "chatcmpl-dims-890-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 },
    }),
    "[DONE]",
  ];
}
