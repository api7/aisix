import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: weighted routing distribution. A virtual Model carries a
// Routing block with `strategy: "weighted"` and two targets — `wr-a`
// (weight 70) and `wr-b` (weight 30). Per docs `api-admin.md` §4.1
// (direct vs routing two-mode split) and the routing schema enum
// `["round_robin", "weighted", "failover"]`, the gateway is expected
// to dispatch incoming traffic in a 70:30 ratio across the two
// targets.
//
// One contract pinned here:
//
//   - Weighted strategy honours the integer `weight` field per
//     target. After N requests the observed split lands inside a
//     statistically reasonable tolerance window around the declared
//     ratio. A regression that ignored `weight` and round-robined
//     instead would fail (each side would land ~50%, well outside
//     [55, 85] / [15, 45]).
//
// Reference: OpenAI Chat Completions API spec for the shape the
// caller sees (https://platform.openai.com/docs/api-reference/chat).
//
// The 100-request count and the [55, 85] / [15, 45] tolerance are
// chosen so a scheduler that completely ignores weight (e.g.
// round-robins or pins to one target) cannot pass — round-robin
// lands at 50/50, well outside [55, 85] for the heavy side — while
// the legitimate 70/30 path stays comfortably inside. Two
// independent binomial windows: 70±15 over n=100 with σ≈4.58 puts
// the gate at ~3.3σ, P(false positive) ≈ 0.1%. The previous ±10
// gate sat at ~2.2σ (≈2.8%) and tripped roughly once per ~36 CI
// runs — wide enough to be a steady CI flake.

const CALLER_PLAINTEXT = "sk-wr-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TOTAL_REQUESTS = 100;
const HEAVY_WEIGHT = 70;
const LIGHT_WEIGHT = 30;
// Tolerance: weight ±15 absolute on a 100-sample window. See header
// comment for the statistical-power tradeoff vs the previous ±10.
const HEAVY_LO = 55;
const HEAVY_HI = 85;
const LIGHT_LO = 15;
const LIGHT_HI = 45;

describe("weighted routing distribution e2e: 70/30 split lands inside [55,85] / [15,45]", () => {
  let app: SpawnedApp | undefined;
  let upstreamA: OpenAiUpstream | undefined;
  let upstreamB: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstreamA = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-wr-a",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "served by A" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreamB = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-wr-b",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "served by B" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Two distinct ProviderKeys so each Model points at its own
    // upstream — necessary for receivedRequests counts to be
    // attributable per side.
    const pkA = await admin.createProviderKey({
      display_name: "wr-a-pk",
      secret: "sk-mock",
      api_base: `${upstreamA.baseUrl}/v1`,
    });
    const pkB = await admin.createProviderKey({
      display_name: "wr-b-pk",
      secret: "sk-mock",
      api_base: `${upstreamB.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "wr-a",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkA.id,
    });
    await admin.createModel({
      display_name: "wr-b",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkB.id,
    });
    // Virtual Model: routing-only, weighted strategy. Per the schema
    // enum the gateway publishes (round_robin / weighted / failover),
    // `weighted` should honour each target's `weight` integer.
    await admin.createModel({
      display_name: "wr-virtual",
      routing: {
        strategy: "weighted",
        targets: [
          { model: "wr-a", weight: HEAVY_WEIGHT },
          { model: "wr-b", weight: LIGHT_WEIGHT },
        ],
      },
    });
    // Caller is allowed all three Models so the readiness probes can
    // hit the leaves directly without firing the weighted dispatcher.
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["wr-virtual", "wr-a", "wr-b"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstreamA?.close();
    await upstreamB?.close();
  });

  test("100 sequential calls split inside the declared 70/30 tolerance window", async (ctx) => {
    if (!etcdReachable || !app || !upstreamA || !upstreamB) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Two-stage readiness gate: probe each leaf Model directly so
    // both ProviderKey registrations are observed by the proxy
    // before we exercise the virtual router. Probing through
    // `wr-virtual` here would fire the weighted dispatcher and
    // pollute the count baseline.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "wr-a",
          messages: [{ role: "user", content: "ready-probe-a" }],
        });
        return probe.choices[0]?.message.content === "served by A";
      } catch {
        return false;
      }
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "wr-b",
          messages: [{ role: "user", content: "ready-probe-b" }],
        });
        return probe.choices[0]?.message.content === "served by B";
      } catch {
        return false;
      }
    });
    // One probe through the virtual Model so the weighted
    // dispatcher's lazy state (if any — schedulers often build the
    // weight wheel on first dispatch) is constructed before we start
    // counting.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "wr-virtual",
          messages: [{ role: "user", content: "ready-probe-virtual" }],
        });
        const content = probe.choices[0]?.message.content;
        return content === "served by A" || content === "served by B";
      } catch {
        return false;
      }
    });

    // Snapshot upstream counts AFTER probes so the assertion
    // measures only the effect of the weighted-distribution batch.
    const aBaseline = upstreamA.receivedRequests.length;
    const bBaseline = upstreamB.receivedRequests.length;

    for (let i = 0; i < TOTAL_REQUESTS; i++) {
      const completion = await client.chat.completions.create({
        model: "wr-virtual",
        messages: [{ role: "user", content: `req-${i}` }],
      });
      // Sanity: every dispatch lands on one of the two upstreams,
      // returning the canned content from whichever served the call.
      const content = completion.choices[0]?.message.content;
      expect(content === "served by A" || content === "served by B").toBe(true);
    }

    const aDelta = upstreamA.receivedRequests.length - aBaseline;
    const bDelta = upstreamB.receivedRequests.length - bBaseline;

    // Total call accounting: every test request hit exactly one
    // upstream (no double-dispatch, no retries). Without this gate
    // a regression that quietly retried each call against both
    // upstreams could still appear "balanced" by ratio.
    expect(aDelta + bDelta).toBe(TOTAL_REQUESTS);

    // Distribution assertion: heavy side ~70, light side ~30, both
    // inside ±10. A round-robin regression (50/50) fails both gates;
    // a pin-to-one regression (100/0) fails both gates.
    expect(aDelta).toBeGreaterThanOrEqual(HEAVY_LO);
    expect(aDelta).toBeLessThanOrEqual(HEAVY_HI);
    expect(bDelta).toBeGreaterThanOrEqual(LIGHT_LO);
    expect(bDelta).toBeLessThanOrEqual(LIGHT_HI);
    // Per-test timeout lifted to 90s. The default suite timeout
    // (60s, vitest.config.ts) is tight for 100 sequential round-trips
    // when upstream latency drifts above ~50ms/call; 90s leaves
    // headroom without changing the global cap for other cases.
  }, 90_000);
});
