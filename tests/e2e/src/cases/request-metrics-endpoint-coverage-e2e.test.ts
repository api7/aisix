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

// AISIX-Cloud#1234: `aisix_proxy_requests_total` / `aisix_llm_requests_total`
// and their duration histograms were emitted by the chat and messages
// handlers ONLY. Every other endpoint recorded just the legacy
// `aisix_requests_total`, so `/v1/responses` traffic (Codex and friends) was
// missing from every request-count and success-rate query built on the
// detailed families — while still appearing in the legacy one, which made
// the gap look like a query mistake rather than absent instrumentation.
//
// These specs pin the two halves of the fix: the inference endpoints reach
// the LLM families, and the non-inference proxy surfaces reach the proxy
// families WITHOUT being counted as LLM requests.

const CALLER_PLAINTEXT = "sk-request-metrics-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const MODEL = "reqmetrics-gpt";

describe("request metrics endpoint coverage e2e", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: responsesBody() });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: `${MODEL}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("/v1/responses reaches the LLM request families", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const probe = await postResponses(app!, { model: MODEL, input: "ready" });
      return probe.status === 200;
    });

    const { status } = await postResponses(app, { model: MODEL, input: "hi" });
    expect(status).toBe(200);

    const text = await scrape(app);

    // The reported bug: this series did not exist for /v1/responses at all.
    expect(
      seriesFor(text, "aisix_llm_requests_total", "/v1/responses"),
    ).toContainEqual(
      expect.stringContaining(`model="${MODEL}"`),
    );
    expect(
      seriesFor(text, "aisix_llm_requests_total", "/v1/responses").join("\n"),
    ).toContain('status="200"');

    // Same gap on the proxy tier and on both duration histograms.
    expect(
      seriesFor(text, "aisix_proxy_requests_total", "/v1/responses"),
    ).not.toHaveLength(0);
    expect(
      seriesFor(text, "aisix_llm_request_duration_seconds_count", "/v1/responses"),
    ).not.toHaveLength(0);
    expect(
      seriesFor(
        text,
        "aisix_proxy_request_duration_seconds_count",
        "/v1/responses",
      ),
    ).not.toHaveLength(0);

    // The detailed label set is filled in, not left at the defaults — an
    // `upstream_model="unknown"` here would mean the handler emitted the
    // series without threading what it actually called.
    const line = seriesFor(
      text,
      "aisix_llm_requests_total",
      "/v1/responses",
    )[0];
    expect(line).toContain('inbound_protocol="openai"');
    expect(line).toContain('provider="openai"');
    expect(line).toContain('upstream_model="gpt-4o-mini"');
    expect(line).toContain('outcome="success"');
  }, 30_000);

  test("a failed /v1/responses request lands in the same denominator", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const { status } = await postResponses(app, {
      model: "reqmetrics-no-such-model",
      input: "hi",
    });
    expect(status).toBeGreaterThanOrEqual(400);

    const text = await scrape(app);
    const failed = seriesFor(
      text,
      "aisix_llm_requests_total",
      "/v1/responses",
    ).filter((l) => !l.includes('outcome="success"'));
    expect(failed).not.toHaveLength(0);
    // A model that never resolved must not put caller-supplied text into a
    // label (#451) — it collapses to the fixed sentinel.
    expect(failed.join("\n")).toContain('model="unresolved"');
    expect(failed.join("\n")).not.toContain("reqmetrics-no-such-model");

    // Failures also raise the proxy-side failure counter the success-rate
    // query divides by.
    expect(
      seriesFor(text, "aisix_proxy_failed_requests_total", "/v1/responses"),
    ).not.toHaveLength(0);
  }, 30_000);

  test("passthrough is counted as a proxy request but never as an LLM one", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // An unclaimed tunnel path: the 410 tombstone is still counted, and
    // the caller-supplied path segment must never become a label — the
    // same #451 contract the old tunnel's failure path carried.
    const res = await fetch(
      `${app.proxyUrl}/passthrough/reqmetrics-bogus-provider/v1/anything`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ hello: "world" }),
      },
    );
    expect(res.status).toBe(410);

    const text = await scrape(app);
    const endpoint = "/passthrough_route";

    expect(seriesFor(text, "aisix_proxy_requests_total", endpoint)).not.toHaveLength(
      0,
    );
    // The tier split: a tunnelled request reaches no model, so it must stay
    // out of the LLM families or every tokens-per-request average is wrong.
    expect(seriesFor(text, "aisix_llm_requests_total", endpoint)).toHaveLength(0);

    // Cardinality: the bogus provider name must not have become a label.
    expect(text).not.toContain("reqmetrics-bogus-provider");
    expect(
      seriesFor(text, "aisix_proxy_requests_total", endpoint).join("\n"),
    ).toContain('provider="unresolved"');
  }, 30_000);
});

async function postResponses(
  app: SpawnedApp,
  body: unknown,
): Promise<{ status: number }> {
  const res = await fetch(`${app.proxyUrl}/v1/responses`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  await res.text();
  return { status: res.status };
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

/** Every sample line of `metric` carrying `endpoint="<endpoint>"`. */
function seriesFor(
  scrapeText: string,
  metric: string,
  endpoint: string,
): string[] {
  return scrapeText
    .split("\n")
    .filter(
      (line) =>
        line.startsWith(`${metric}{`) &&
        line.includes(`endpoint="${endpoint}"`),
    );
}

function responsesBody() {
  return {
    id: "resp_reqmetrics",
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status: "completed",
    model: "gpt-4o-mini",
    output: [
      {
        id: "msg_reqmetrics",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "hello" }],
      },
    ],
    usage: { input_tokens: 11, output_tokens: 13, total_tokens: 24 },
  };
}
