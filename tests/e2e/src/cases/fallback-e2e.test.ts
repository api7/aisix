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

// E2E: routing failover. A "virtual" Model carries a Routing block
// with two targets — `fb-bad` (returns 502) and `fb-good` (returns
// 200). The default `failover` strategy starts at the first target;
// when `fb-bad` returns a retryable upstream error, dispatch falls
// back to `fb-good` and the caller never sees the failure. This
// pins the same contract the in-process
// `routing_failover_retries_to_second_target_when_first_5xxs` covers,
// but end-to-end through a real binary, etcd watch, and OpenAI SDK.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create); the
// Routing schema lives at `crates/aisix-core/src/models/routing.rs`.

const CALLER_PLAINTEXT = "sk-fb-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("fallback e2e: virtual routing fails over from 5xx to next target", () => {
  let app: SpawnedApp | undefined;
  let badUpstream: OpenAiUpstream | undefined;
  let goodUpstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
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
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const badPk = await admin.createProviderKey({
      display_name: "fb-bad-pk",
      secret: "sk-mock",
      api_base: `${badUpstream.baseUrl}/v1`,
    });
    const goodPk = await admin.createProviderKey({
      display_name: "fb-good-pk",
      secret: "sk-mock",
      api_base: `${goodUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "fb-bad",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: badPk.id,
    });
    await admin.createModel({
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
    await admin.createModel({
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
    await admin.createApiKey({
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

    // Probe via `fb-good` directly (single-target, no fallback path)
    // so the readiness check doesn't pre-warm the bad→good route or
    // risk inflating either upstream's baseline. A 200 means
    // Model + ProviderKey + ApiKey are all loaded; routing-block
    // propagation for `fb-virtual` is on the same etcd watch loop, so
    // by the time `fb-good` is callable, `fb-virtual` is too.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "fb-good",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return probe.choices[0]?.message.content === "fallback worked";
      } catch {
        return false;
      }
    });

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
});
