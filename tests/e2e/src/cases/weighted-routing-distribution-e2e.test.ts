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

// E2E: weighted round-robin distribution. A virtual Model carries a
// Routing block with `strategy: "round_robin"` and two targets — `wr-a`
// (weight 70) and `wr-b` (weight 30). round_robin is smooth WEIGHTED
// round-robin (the nginx algorithm; AISIX-Cloud#1206 merged the former
// `weighted` strategy into it), so the gateway must dispatch traffic in
// an exact 70:30 ratio across the two targets.
//
// One contract pinned here:
//
//   - round_robin honours the integer `weight` field per target, and is
//     deterministic: smooth WRR is periodic with period = total weight
//     (100 here), and ANY window of one full period contains each
//     target exactly `weight` times — so 100 sequential requests land
//     at exactly 70/30 regardless of how many warm-up probes ran
//     before the counted batch.
//
// Reference: OpenAI Chat Completions API spec for the shape the
// caller sees (https://platform.openai.com/docs/api-reference/chat).

const CALLER_PLAINTEXT = "sk-wr-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TOTAL_REQUESTS = 100;
const HEAVY_WEIGHT = 70;
const LIGHT_WEIGHT = 30;

describe("weighted round-robin distribution e2e: 70/30 split is exact over one WRR period", () => {
  let app: SpawnedApp | undefined;
  let upstreamA: OpenAiUpstream | undefined;
  let upstreamB: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
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
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Two distinct ProviderKeys so each Model points at its own
    // upstream — necessary for receivedRequests counts to be
    // attributable per side.
    const pkA = await seed.createProviderKey({
      display_name: "wr-a-pk",
      secret: "sk-mock",
      api_base: `${upstreamA.baseUrl}/v1`,
    });
    const pkB = await seed.createProviderKey({
      display_name: "wr-b-pk",
      secret: "sk-mock",
      api_base: `${upstreamB.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "wr-a",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkA.id,
    });
    await seed.createModel({
      display_name: "wr-b",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkB.id,
    });
    // Virtual Model: routing-only, weighted round-robin. round_robin
    // honours each target's `weight` integer exactly (smooth WRR).
    await seed.createModel({
      display_name: "wr-virtual",
      routing: {
        strategy: "round_robin",
        targets: [
          { model: "wr-a", weight: HEAVY_WEIGHT },
          { model: "wr-b", weight: LIGHT_WEIGHT },
        ],
      },
    });
    // Caller is allowed all three Models so the readiness probes can
    // hit the leaves directly without firing the WRR dispatcher.
    await seed.createApiKey({
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
    // `wr-virtual` here would fire the WRR dispatcher and
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
    // One probe through the virtual Model so the WRR dispatcher's
    // lazy state is constructed before we start counting. Periodicity
    // makes the exact assertion below offset-independent.
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

    // Distribution assertion: smooth WRR is exact over one full period
    // (total weight = 100 = TOTAL_REQUESTS), at any starting offset. An
    // equal-rotation regression lands 50/50; a pin-to-one regression
    // lands 100/0; a random-sampling regression flakes around 70 — all
    // fail an exact gate.
    expect(aDelta).toBe(HEAVY_WEIGHT);
    expect(bDelta).toBe(LIGHT_WEIGHT);
    // Per-test timeout lifted to 90s. The default suite timeout
    // (60s, vitest.config.ts) is tight for 100 sequential round-trips
    // when upstream latency drifts above ~50ms/call; 90s leaves
    // headroom without changing the global cap for other cases.
  }, 90_000);
});
