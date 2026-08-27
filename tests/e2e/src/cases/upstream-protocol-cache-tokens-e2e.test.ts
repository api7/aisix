import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MetricSample,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1403 (`upstream_protocol`) and #1404 (upstream prompt-cache
// token counters), asserted together because they answer the same question
// from two sides and share every label: which protocol served this traffic,
// and how much of its input came out of that upstream's prompt cache.
//
// The two cache-read counters exist because providers report cache hits
// under incompatible accounting rules, and this spec is where that is
// pinned against a real scrape:
//
//   OpenAI shape    `prompt_tokens_details.cached_tokens` is INSIDE
//                   `prompt_tokens` → `aisix_llm_cached_input_tokens_total`,
//                   and `aisix_llm_input_tokens_total` must not grow by it.
//   Anthropic shape `cache_read_input_tokens` / `cache_creation_input_tokens`
//                   sit BESIDE `input_tokens` → the two
//                   `aisix_llm_cache_*_input_tokens_total` counters, and
//                   `aisix_llm_total_tokens_total` folds them in (#1002).
//
// Every assertion is a delta across one request (see `harness/metrics.ts`):
// the suite drives readiness probes through the same app, so absolute
// counter values carry earlier traffic.

const CALLER_PLAINTEXT = "sk-upstream-proto-e2e";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const OPENAI_MODEL = "proto-openai-upstream";
const ANTHROPIC_MODEL = "proto-anthropic-upstream";

/**
 * The families that carry `inbound_protocol` and deliberately do NOT carry
 * `upstream_protocol`, mirroring `upstream_protocol_exempt_families` in
 * `crates/aisix-obs/src/metrics.rs` — where the reason for each one is
 * written at the metric's own definition. Both are counted at a point
 * where no upstream has been selected.
 */
const UPSTREAM_PROTOCOL_EXEMPT = new Set([
  "aisix_proxy_in_flight_requests",
  "aisix_proxy_request_body_limit_rejections_total",
]);

/** Collapse a histogram's `_bucket` / `_sum` / `_count` back to its family. */
function histogramFamily(name: string): string {
  for (const suffix of ["_bucket", "_sum", "_count"]) {
    if (name.endsWith(suffix)) return name.slice(0, -suffix.length);
  }
  return name;
}

// OpenAI-shape usage: `cached_tokens` is part of the 120 prompt tokens.
const OAI_PROMPT_TOKENS = 120;
const OAI_COMPLETION_TOKENS = 7;
const OAI_CACHED_TOKENS = 96;

// Anthropic-shape usage: the two cache counters are additional to the 40
// input tokens, never part of them.
const ANT_INPUT_TOKENS = 40;
const ANT_OUTPUT_TOKENS = 11;
const ANT_CACHE_READ_TOKENS = 800;
const ANT_CACHE_CREATION_TOKENS = 200;

function openAiBody(): unknown {
  return {
    id: "chatcmpl-proto-1",
    object: "chat.completion",
    created: 1,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "cached hello" },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: OAI_PROMPT_TOKENS,
      completion_tokens: OAI_COMPLETION_TOKENS,
      total_tokens: OAI_PROMPT_TOKENS + OAI_COMPLETION_TOKENS,
      prompt_tokens_details: { cached_tokens: OAI_CACHED_TOKENS },
    },
  };
}

function anthropicBody(): unknown {
  return {
    id: "msg_proto_1",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: "cached hello" }],
    model: "claude-3-5-haiku-20241022",
    stop_reason: "end_turn",
    usage: {
      input_tokens: ANT_INPUT_TOKENS,
      output_tokens: ANT_OUTPUT_TOKENS,
      cache_read_input_tokens: ANT_CACHE_READ_TOKENS,
      cache_creation_input_tokens: ANT_CACHE_CREATION_TOKENS,
    },
  };
}

describe("upstream_protocol + prompt-cache token metrics e2e", () => {
  let app: SpawnedApp | undefined;
  let openAiUpstream: OpenAiUpstream | undefined;
  let anthropicUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    openAiUpstream = await startOpenAiUpstream({ nonStreamBody: openAiBody() });
    // The mock is path-agnostic, so it doubles as the Anthropic upstream
    // when fed an Anthropic-shaped body.
    anthropicUpstream = await startOpenAiUpstream({
      nonStreamBody: anthropicBody(),
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const oaiPk = await seed.createProviderKey({
      display_name: "proto-openai-pk",
      provider: "openai",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${openAiUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: OPENAI_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: oaiPk.id,
    });

    // The Anthropic bridge appends `/v1/messages` itself, so this
    // api_base is the bare host — the opposite of the OpenAI convention
    // above.
    const antPk = await seed.createProviderKey({
      display_name: "proto-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: anthropicUpstream.baseUrl,
    });
    await seed.createModel({
      display_name: ANTHROPIC_MODEL,
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: antPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [OPENAI_MODEL, ANTHROPIC_MODEL],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await openAiUpstream?.close();
    await anthropicUpstream?.close();
  });

  const scrape = (): Promise<MetricSample[]> =>
    scrapeMetrics(app!.metricsUrl);

  async function postMessages(model: string): Promise<number> {
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 64,
        messages: [{ role: "user", content: "cached prompt" }],
      }),
    });
    return res.status;
  }

  test("OpenAI in / OpenAI out: cached tokens ride inside the input counter", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: OPENAI_MODEL,
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status === 200;
    });

    const before = await scrape();
    const r = await proxy.chat({
      model: OPENAI_MODEL,
      messages: [{ role: "user", content: "cached prompt" }],
    });
    expect(r.status).toBe(200);
    const after = await scrape();

    const want = {
      endpoint: "/v1/chat/completions",
      inbound_protocol: "openai",
      upstream_protocol: "openai",
      model: OPENAI_MODEL,
    };

    // #1404: the cache detail is reported, and reported ONCE.
    expect(
      metricDelta(before, after, "aisix_llm_cached_input_tokens_total", want),
    ).toBe(OAI_CACHED_TOKENS);
    // The whole point of keeping the OpenAI shape on its own counter: the
    // input total is the upstream's `prompt_tokens`, which ALREADY contains
    // the cached tokens. Adding them again would inflate every cost figure.
    expect(
      metricDelta(before, after, "aisix_llm_input_tokens_total", want),
    ).toBe(OAI_PROMPT_TOKENS);
    expect(
      metricDelta(before, after, "aisix_llm_total_tokens_total", want),
    ).toBe(OAI_PROMPT_TOKENS + OAI_COMPLETION_TOKENS);
    // An OpenAI upstream reports no Anthropic-shape cache counters, so
    // those two series must not move — and, being sparse, need not exist.
    expect(
      metricDelta(
        before,
        after,
        "aisix_llm_cache_read_input_tokens_total",
        want,
      ),
    ).toBe(0);
    expect(
      metricDelta(
        before,
        after,
        "aisix_llm_cache_creation_input_tokens_total",
        want,
      ),
    ).toBe(0);
  });

  test("OpenAI in / Anthropic out: cross-protocol labels, cache tokens beside the input counter", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: ANTHROPIC_MODEL,
        messages: [{ role: "user", content: "ready" }],
      });
      return probe.status === 200;
    });

    const before = await scrape();
    const r = await proxy.chat({
      model: ANTHROPIC_MODEL,
      messages: [{ role: "user", content: "cached prompt" }],
    });
    expect(r.status).toBe(200);
    const after = await scrape();

    // #1403: the caller spoke OpenAI, the upstream speaks Anthropic. This
    // pair is unanswerable from `provider` alone.
    const want = {
      endpoint: "/v1/chat/completions",
      inbound_protocol: "openai",
      upstream_protocol: "anthropic",
      model: ANTHROPIC_MODEL,
    };
    expect(metricDelta(before, after, "aisix_llm_requests_total", want)).toBe(1);

    expect(
      metricDelta(
        before,
        after,
        "aisix_llm_cache_read_input_tokens_total",
        want,
      ),
    ).toBe(ANT_CACHE_READ_TOKENS);
    expect(
      metricDelta(
        before,
        after,
        "aisix_llm_cache_creation_input_tokens_total",
        want,
      ),
    ).toBe(ANT_CACHE_CREATION_TOKENS);
    // Anthropic's `input_tokens` excludes both, so the input counter must
    // report the bare 40 — this is the half that makes a cross-protocol
    // cache ratio need `input + cache_read + cache_creation` as its
    // denominator.
    expect(
      metricDelta(before, after, "aisix_llm_input_tokens_total", want),
    ).toBe(ANT_INPUT_TOKENS);
    // …while the canonical total DOES fold them in (#1002), unchanged by
    // this work.
    expect(
      metricDelta(before, after, "aisix_llm_total_tokens_total", want),
    ).toBe(
      ANT_INPUT_TOKENS +
        ANT_OUTPUT_TOKENS +
        ANT_CACHE_READ_TOKENS +
        ANT_CACHE_CREATION_TOKENS,
    );
    expect(
      metricDelta(before, after, "aisix_llm_cached_input_tokens_total", want),
    ).toBe(0);
  });

  test("Anthropic in / OpenAI out: the reversed cross-protocol pair keeps its cache detail", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => (await postMessages(OPENAI_MODEL)) === 200);

    const before = await scrape();
    expect(await postMessages(OPENAI_MODEL)).toBe(200);
    const after = await scrape();

    const want = {
      endpoint: "/v1/messages",
      inbound_protocol: "anthropic",
      upstream_protocol: "openai",
      model: OPENAI_MODEL,
    };
    expect(metricDelta(before, after, "aisix_llm_requests_total", want)).toBe(1);
    // The `/v1/messages` telemetry struct is Anthropic-shaped, so carrying
    // the bridged upstream's OpenAI-shape cache hit through it is the part
    // that has to be wired explicitly rather than falling out.
    expect(
      metricDelta(before, after, "aisix_llm_cached_input_tokens_total", want),
    ).toBe(OAI_CACHED_TOKENS);
    expect(
      metricDelta(before, after, "aisix_llm_input_tokens_total", want),
    ).toBe(OAI_PROMPT_TOKENS);

    // The usage-event counter answers the same question about the same
    // request and has to land on the same protocol, or an operator
    // grouping the two families together gets one of them back empty. It
    // names the surface by handler rather than route template.
    expect(
      metricDelta(before, after, "aisix_usage_events_emitted_total", {
        handler: "messages",
        inbound_protocol: "anthropic",
        upstream_protocol: "openai",
        model: OPENAI_MODEL,
      }),
    ).toBe(1);
  });

  test("a request that selects no upstream is labelled unknown, not dropped", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const before = await scrape();
    const r = await proxy.chat({
      model: "no-such-model-for-proto-e2e",
      messages: [{ role: "user", content: "nope" }],
    });
    expect(r.status).toBeGreaterThanOrEqual(400);
    const after = await scrape();

    // Same label set as a served request, with the placeholder every other
    // unresolved dimension uses — so a family-wide query still sees this
    // request instead of silently losing it to a missing label.
    expect(
      metricDelta(before, after, "aisix_proxy_requests_total", {
        endpoint: "/v1/chat/completions",
        inbound_protocol: "openai",
        upstream_protocol: "unknown",
      }),
    ).toBe(1);
    // …and the usage-event counter reads `unknown` rather than borrowing
    // the caller's protocol for an upstream that was never chosen.
    expect(
      metricDelta(before, after, "aisix_usage_events_emitted_total", {
        handler: "chat",
        inbound_protocol: "openai",
        upstream_protocol: "unknown",
      }),
    ).toBe(1);
  });

  test("every scraped sample carrying inbound_protocol carries upstream_protocol", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Derived from the scrape rather than a list of family names. The
    // predecessor of this test named eight families by hand and passed
    // while `aisix_usage_events_emitted_total` shipped a whole release
    // candidate without the label: a hand-written list only ever pins the
    // set that existed when someone typed it.
    const withInbound = (await scrape()).filter(
      (s) => s.labels.inbound_protocol,
    );
    expect(
      withInbound.length,
      "the scrape carries no inbound_protocol samples at all — the traffic \
above did not reach the metrics listener, so this test checked nothing",
    ).toBeGreaterThan(0);

    const offenders = withInbound
      .filter((s) => !UPSTREAM_PROTOCOL_EXEMPT.has(histogramFamily(s.name)))
      .filter((s) => !s.labels.upstream_protocol);
    expect(
      [...new Set(offenders.map((s) => s.name))],
      "these families carry inbound_protocol without upstream_protocol, so \
`sum by (upstream_protocol)` drops them out of any join with the rest",
    ).toEqual([]);

    // An exemption that stopped being true is a hole in the check above,
    // so assert the other direction too. Whether each exempt family still
    // EXISTS is the Rust guard's job — it reads the emit sites and so
    // sees families no traffic here would render.
    for (const row of withInbound) {
      if (!UPSTREAM_PROTOCOL_EXEMPT.has(histogramFamily(row.name))) continue;
      expect(
        row.labels.upstream_protocol,
        `${row.name} is exempted from upstream_protocol but now emits it`,
      ).toBeUndefined();
    }
  });
});
