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

const CALLER_PLAINTEXT = "sk-least-busy-e2e-caller";
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

describe("least-busy (least_busy) routing e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
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
    if (!admin) throw new Error("admin client not initialized");
    const providerKey = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
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

  test("routes away from an in-flight target to the idle one", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Target A responds slowly, so a request to it stays in flight; B is fast.
    const slow = await startOpenAiUpstream({
      responseDelayMs: 800,
      nonStreamBody: okBody("a-served"),
    });
    const fast = await startOpenAiUpstream({ nonStreamBody: okBody("b-served") });
    upstreams.push(slow, fast);

    await createOpenAiModel("busy-a", slow);
    await createOpenAiModel("busy-b", fast);
    await admin.createModel({
      display_name: "busy-virtual",
      routing: {
        strategy: "least_busy",
        targets: [{ model: "busy-a" }, { model: "busy-b" }],
      },
    });

    const c = client();
    // Gate on the virtual dispatching through the DP snapshot. The probe (both
    // targets idle → declaration order) lands on A and completes before we
    // proceed, so its transient in-flight count is released.
    await waitConfigPropagation(async () => {
      try {
        const p = await c.chat.completions.create({
          model: "busy-virtual",
          messages: [{ role: "user", content: "warmup" }],
        });
        return ["a-served", "b-served"].includes(
          p.choices[0]?.message.content ?? "",
        );
      } catch {
        return false;
      }
    });

    const slowBase = slow.receivedRequests.length;
    const fastBase = fast.receivedRequests.length;

    // Fire a request that lands on A (both idle → declaration order) and stays
    // in flight because A is slow. Do NOT await it yet.
    const inflight = c.chat.completions.create({
      model: "busy-virtual",
      messages: [{ role: "user", content: "occupy-a" }],
    });
    // Give it time to reach A and raise A's in-flight count.
    await new Promise((r) => setTimeout(r, 200));

    // A now has 1 in-flight, B has 0 → least_busy routes this request to B.
    const diverted = await c.chat.completions.create({
      model: "busy-virtual",
      messages: [{ role: "user", content: "should-divert-to-b" }],
    });
    expect(diverted.choices[0]?.message.content).toBe("b-served");

    const first = await inflight;
    expect(first.choices[0]?.message.content).toBe("a-served");

    // A took only the occupy request; B took the diverted one.
    expect(slow.receivedRequests.length - slowBase).toBe(1);
    expect(fast.receivedRequests.length - fastBase).toBe(1);
  });
});
