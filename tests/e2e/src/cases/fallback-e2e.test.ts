import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: routing failover. A "virtual" Model carries a Routing block
// with two targets — `fb-bad` (returns 502) and `fb-good` (returns
// 200). The default `failover` strategy starts at the first target;
// when `fb-bad` returns a retryable upstream error, dispatch falls
// back to `fb-good` and the caller never sees the failure.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for
// the request/response shape the caller sees.

const CALLER_PLAINTEXT = "sk-fb-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("fallback e2e: virtual routing fails over from 5xx to next target", () => {
  let app: SpawnedApp | undefined;
  let badUpstream: OpenAiUpstream | undefined;
  let goodUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    badUpstream = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "upstream down", type: "server_error" } },
    });
    goodUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-good",
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
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const badPk = await seed.createProviderKey({
      display_name: "fb-bad-pk",
      secret: "sk-mock",
      api_base: `${badUpstream.baseUrl}/v1`,
    });
    const goodPk = await seed.createProviderKey({
      display_name: "fb-good-pk",
      secret: "sk-mock",
      api_base: `${goodUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "fb-bad",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: badPk.id,
      // This test is specifically about retry-time failover (bad→good
      // within one request). Cooldown (post-PR #268) would mark fb-bad
      // after the readiness probe exercises the failover path, and
      // the subsequent test request would then skip fb-bad — defeating
      // the very contract the test is pinning. Disable cooldown here
      // so the test exercises only its target behavior.
      cooldown: { enabled: false },
    });
    await seed.createModel({
      display_name: "fb-good",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: goodPk.id,
    });
    // Virtual Model: routing-only. The schema is `oneOf` — a model
    // either carries a `routing` block (virtual router — no
    // provider/model_name/provider_key_id) or carries those three
    // (direct upstream — no routing). `failover` is the default
    // strategy; making it explicit keeps the test self-documenting.
    await seed.createModel({
      display_name: "fb-virtual",
      routing: {
        strategy: "failover",
        targets: [{ model: "fb-bad" }, { model: "fb-good" }],
      },
    });
    // Permit the caller for `fb-good` too so the readiness probe can
    // hit a single-target Model directly, instead of probing through
    // `fb-virtual` (which fires the bad→good fallback on every retry
    // and risks off-by-one when an iteration partially succeeds).
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["fb-virtual", "fb-good"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await badUpstream?.close();
    await goodUpstream?.close();
  });

  test("fb-bad returns 502, dispatch falls over to fb-good and caller sees 200", async (ctx) => {
    if (!etcdReachable || !app || !badUpstream || !goodUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Two-stage readiness gate. The previous version probed only
    // `fb-good` and assumed the routing block for `fb-virtual` would
    // be loaded by the time `fb-good` was callable — that's plausible
    // (single etcd watch loop) but unverified by the test itself. The
    // second probe explicitly gates on `fb-virtual` so the test call
    // cannot fire while the routing dispatcher is still missing the
    // virtual model. The probe traffic is folded into the baseline
    // captured below it.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "fb-good",
          messages: [{ role: "user", content: "ready-probe-good" }],
        });
        return probe.choices[0]?.message.content === "fallback worked";
      } catch {
        return false;
      }
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "fb-virtual",
          messages: [{ role: "user", content: "ready-probe-virtual" }],
        });
        return probe.choices[0]?.message.content === "fallback worked";
      } catch {
        return false;
      }
    });

    // The fb-virtual probe stage above DOES exercise the bad→good
    // fallback path, so the bad upstream MUST have received at least
    // one request during probing. Surface this as an explicit
    // assertion (rather than letting it hide inside the baseline)
    // so a regression that silently routes fb-virtual past the bad
    // target — making the entire test trivially pass on the good
    // upstream — is caught here.
    expect(badUpstream.receivedRequests.length).toBeGreaterThanOrEqual(1);

    // Snapshot upstream counts AFTER the probe so the assertions below
    // measure only the effect of the actual test call, not the probe.
    const badBaseline = badUpstream.receivedRequests.length;
    const goodBaseline = goodUpstream.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "fb-virtual",
      messages: [{ role: "user", content: "hello" }],
    });

    // Caller sees the good upstream's response — a regression that
    // surfaces the bad upstream's 502 (or wraps it as gateway 502)
    // would fail here.
    expect(completion.choices[0]?.message.content).toBe("fallback worked");
    expect(completion.choices[0]?.message.role).toBe("assistant");

    // Test call hit each upstream exactly once: bad first (failed try),
    // good second (successful fallback). Baseline isolation rules out
    // counting probe traffic.
    expect(badUpstream.receivedRequests.length - badBaseline).toBe(1);
    expect(goodUpstream.receivedRequests.length - goodBaseline).toBe(1);
  });

  // AISIX-Cloud#1299: an operator compared a 5xx count from the usage log
  // against `sum(increase(aisix_proxy_requests_total{status=~"5.."}[24h]))`
  // and found the metric two orders of magnitude lower. The request family
  // is per-REQUEST and carries the status the caller saw, so a request its
  // fallback rescued is a `status="200"` sample there and the 502 its first
  // target returned appears in NO request-level series at all.
  //
  // That is by design, but until this test the per-ATTEMPT counters which
  // do see it — `aisix_deployment_*`, `aisix_routing_*_fallbacks_total` —
  // had no caller anywhere in the workspace and never appeared in a scrape,
  // so an operator had nothing in Prometheus to reconcile the log against.
  // Every assertion below is on the delta of ONE request, and each of the
  // four families is checked against the same event, which is what makes
  // the test a statement about their relationship rather than about any one
  // counter.
  test("one rescued request: the attempt counters see the 502 the request counter cannot", async (ctx) => {
    if (!etcdReachable || !app || !badUpstream || !goodUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "fb-virtual",
          messages: [{ role: "user", content: "ready-probe-metrics" }],
        });
        return probe.choices[0]?.message.content === "fallback worked";
      } catch {
        return false;
      }
    });

    const before = await scrapeCounters(app.metricsUrl);
    const completion = await client.chat.completions.create({
      model: "fb-virtual",
      messages: [{ role: "user", content: "metrics" }],
    });
    expect(completion.choices[0]?.message.content).toBe("fallback worked");
    const after = await scrapeCounters(app.metricsUrl);
    const delta = (name: string, labels: Record<string, string> = {}) =>
      sum(after, name, labels) - sum(before, name, labels);

    // The attempt that failed and the attempt that served it, each filed
    // under the TARGET it hit — not under the group the caller named.
    expect(delta("aisix_deployment_requests_total", { model: "fb-bad" })).toBe(
      1,
    );
    expect(
      delta("aisix_deployment_failure_responses_total", { model: "fb-bad" }),
    ).toBe(1);
    expect(delta("aisix_deployment_requests_total", { model: "fb-good" })).toBe(
      1,
    );
    expect(
      delta("aisix_deployment_success_responses_total", { model: "fb-good" }),
    ).toBe(1);

    // The move to the second target, labelled by what the caller asked for
    // and where it went.
    expect(
      delta("aisix_routing_successful_fallbacks_total", {
        model: "fb-virtual",
        fallback_model: "fb-good",
      }),
    ).toBe(1);
    expect(delta("aisix_routing_failed_fallbacks_total")).toBe(0);

    // The contrast that provoked the issue: one request-level sample, and
    // it is a 200. A 5xx count taken from this family legitimately does not
    // include the failed attempt above.
    expect(delta("aisix_proxy_requests_total")).toBe(1);
    expect(delta("aisix_proxy_requests_total", { status: "200" })).toBe(1);
    expect(
      after.filter(
        (s) =>
          s.name === "aisix_proxy_requests_total" &&
          /^5\d\d$/.test(s.labels.status ?? ""),
      ).length,
    ).toBe(0);

    // …while the per-attempt usage events — the rows the dashboard log
    // counts — do carry it. This counter is the one an operator should
    // reconcile a 5xx log-row count against.
    expect(
      delta("aisix_usage_events_emitted_total", { status_code: "5xx" }),
    ).toBe(1);
    expect(
      delta("aisix_usage_events_emitted_total", { status_code: "2xx" }),
    ).toBe(1);
  });
});

interface Series {
  name: string;
  labels: Record<string, string>;
  value: number;
}

/** Parse a prometheus text scrape into its counter samples. */
async function scrapeCounters(metricsUrl: string): Promise<Series[]> {
  const res = await fetch(`${metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  const out: Series[] = [];
  for (const line of (await res.text()).split("\n")) {
    const m = /^([a-z_]+)(\{(.*)\})? ([0-9.e+-]+)$/.exec(line.trim());
    if (!m) continue;
    const labels: Record<string, string> = {};
    for (const pair of m[3]?.match(/[a-z_]+="[^"]*"/g) ?? []) {
      const [k, v] = pair.split("=");
      labels[k] = v.slice(1, -1);
    }
    out.push({ name: m[1], labels, value: Number(m[4]) });
  }
  return out;
}

/**
 * Total of every sample of `name` whose labels include `want`. Summing
 * rather than requiring a single series on purpose: these families carry
 * label dimensions this test says nothing about, and pinning the full
 * tuple would make the test fail the next time one is added.
 */
function sum(
  series: Series[],
  name: string,
  want: Record<string, string> = {},
): number {
  return series
    .filter(
      (s) =>
        s.name === name &&
        Object.entries(want).every(([k, v]) => s.labels[k] === v),
    )
    .reduce((acc, s) => acc + s.value, 0);
}
