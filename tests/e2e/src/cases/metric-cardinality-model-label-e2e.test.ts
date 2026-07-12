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

// E2E for #911 finding [27]: on a pre-resolution failure (model-not-found)
// the typed proxy endpoints recorded the RAW client-supplied `model` field as
// the Prometheus `model` label. Because that field is caller-controlled free
// text, a caller could mint unbounded metric series — a cardinality DoS — by
// sending many unique unknown model names. The fix collapses any unresolved
// model to a fixed "unresolved" sentinel, the typed-endpoint analogue of
// passthrough's PASSTHROUGH_MODEL_LABEL guard (#451).

const CALLER_PLAINTEXT = "sk-model-card-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Unique unknown model names; none of these is a configured model.
const BOGUS_PREFIX = "cardinality-bomb-model-";
const BOGUS_COUNT = 25;

function chatReply(content: string): unknown {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("metric label cardinality for unresolved model (#911 [27])", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: chatReply("ready") });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = (
      await seed.createProviderKey({
        display_name: "card-pk",
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      })
    ).id;

    // One real model, used only to gate on config propagation.
    await seed.createModel({
      display_name: "card-gate",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk,
    });

    // Wildcard so ANY model name passes the allowed_models authz check and
    // reaches model resolution — where the unknown names fail (model-not-
    // found) and hit the metric-recording error path under test.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    const gate = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await gate.chat.completions.create({
          model: "card-gate",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.choices[0]?.message.content === "ready";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("many unknown model names collapse to a single 'unresolved' label", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Fire many unique unknown model names at a typed endpoint. Each fails
    // resolution (model-not-found) and records the request metric — assert the
    // error status rather than swallowing it, so a regression that started
    // resolving these (and thus recording a real model label) is caught here.
    for (let i = 0; i < BOGUS_COUNT; i++) {
      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: `${BOGUS_PREFIX}${i}`,
          messages: [{ role: "user", content: "x" }],
        }),
      });
      await res.text();
      expect(res.ok).toBe(false);
    }

    const scrape = await fetch(`${app.metricsUrl}/metrics`).then((r) => r.text());
    const requestLines = scrape
      .split("\n")
      .filter((l) => l.startsWith("aisix_requests_total{"));

    // No raw unknown model name may appear in any label.
    const leaked = requestLines.filter((l) => l.includes(BOGUS_PREFIX));
    expect(
      leaked,
      `raw model names leaked into metric labels:\n${leaked.join("\n")}`,
    ).toHaveLength(0);

    // The unresolved requests collapse to the fixed sentinel series.
    const sentinel = requestLines.filter((l) => /model="unresolved"/.test(l));
    expect(sentinel.length).toBeGreaterThanOrEqual(1);
  }, 30_000);
});
