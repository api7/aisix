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

const CALLER_PLAINTEXT = "sk-latency-routing-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function okBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("latency-aware (least_latency) routing e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
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
      ...extra,
    });
  }

  function client(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  test("routes to the fastest target once latencies are learned", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const slow = await startOpenAiUpstream({
      responseDelayMs: 400,
      nonStreamBody: okBody("slow-served"),
    });
    const fast = await startOpenAiUpstream({ nonStreamBody: okBody("fast-served") });
    upstreams.push(slow, fast);

    await createOpenAiModel("lat-slow", slow);
    await createOpenAiModel("lat-fast", fast);
    // Declare the slow target FIRST — least_latency must reorder by observed
    // latency, not honor declaration order.
    await seed.createModel({
      display_name: "lat-virtual",
      routing: {
        strategy: "least_latency",
        targets: [{ model: "lat-slow" }, { model: "lat-fast" }],
      },
    });

    const c = client();
    // Wait for the virtual to dispatch through the DP snapshot. The probe is
    // expected to 404 until the watch loop applies the record, so a failed
    // attempt legitimately retries here. This also seeds cold-start latency
    // samples (unmeasured targets are tried in declaration order first).
    await waitConfigPropagation(async () => {
      try {
        const p = await c.chat.completions.create({
          model: "lat-virtual",
          messages: [{ role: "user", content: "warmup" }],
        });
        return ["slow-served", "fast-served"].includes(
          p.choices[0]?.message.content ?? "",
        );
      } catch {
        return false;
      }
    });
    // Extra warmup so BOTH targets definitely have a latency EWMA (cold-start
    // tries them in declaration order, one per request).
    for (let i = 0; i < 3; i++) {
      await c.chat.completions.create({
        model: "lat-virtual",
        messages: [{ role: "user", content: `warm-${i}` }],
      });
    }

    const slowBaseline = slow.receivedRequests.length;
    const fastBaseline = fast.receivedRequests.length;
    const contents: string[] = [];
    for (let i = 0; i < 5; i++) {
      const r = await c.chat.completions.create({
        model: "lat-virtual",
        messages: [{ role: "user", content: `measure-${i}` }],
      });
      contents.push(r.choices[0]?.message.content ?? "");
    }

    // Steady state: every request now goes to the fast target.
    expect(contents).toEqual([
      "fast-served",
      "fast-served",
      "fast-served",
      "fast-served",
      "fast-served",
    ]);
    expect(fast.receivedRequests.length - fastBaseline).toBe(5);
    expect(slow.receivedRequests.length - slowBaseline).toBe(0);
  });
});
