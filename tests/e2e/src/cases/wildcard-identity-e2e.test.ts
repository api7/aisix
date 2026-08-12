import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  awaitWindowHeadroom,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a wildcard alias row (`wid/*`) is ONE configured identity, so its
// gates and telemetry must key on the row, not on whatever concrete
// suffix a caller mints (model-kind audit):
//
//   1. The inline rate_limit bucket is shared across every alias the row
//      serves — previously each distinct suffix opened a fresh
//      full-size bucket, letting any caller multiply the declared cap
//      without bound.
//   2. The Prometheus `model` label for wildcard-served SUCCESS traffic
//      is the row's display_name; caller-minted strings must not mint
//      unbounded series (the #451 cardinality guard, extended to
//      resolvable names).

const CALLER_PLAINTEXT = "sk-wildcard-identity-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function chatBody(content: string) {
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

describe("wildcard alias identity e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  async function chat(model: string): Promise<{ status: number; type?: string }> {
    if (!app) throw new Error("app not ready");
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
    });
    const body = (await res.json()) as { error?: { type?: string } };
    return { status: res.status, type: body.error?.type };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });

    const upstream = await startOpenAiUpstream({ nonStreamBody: chatBody("served-wid") });
    upstreams.push(upstream);
    const pk = await seed.createProviderKey({
      display_name: "wid-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "wid/*",
      provider: "openai",
      model_name: "*",
      provider_key_id: pk.id,
      rate_limit: { rpm: 1 },
    });

    // Readiness via a wildcard-served alias: any suffix must resolve.
    // listModels hides wildcard patterns, so probe with a chat call —
    // 404 until the row propagates. The probe consumes the shared rpm
    // slot, so tests below re-align on a fresh window first.
    await waitConfigPropagation(async () => {
      try {
        const r = await chat("wid/readiness-probe");
        return r.status === 200 || r.status === 429;
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("every caller-minted alias shares the wildcard row's rate-limit bucket, and success metrics label as the row", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    // The readiness probe already consumed a slot of the SHARED bucket
    // (itself evidence of the fix), so align by burning `wid/alpha`
    // until a fresh window admits it — that 200 is alias #1's slot.
    const deadline = Date.now() + 90_000;
    let aligned = false;
    while (Date.now() < deadline) {
      const r = await chat("wid/alpha");
      if (r.status === 200) {
        aligned = true;
        break;
      }
      expect(r.status).toBe(429);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    expect(aligned).toBe(true);

    // A DIFFERENT alias in the same window must land in the same
    // bucket. Pre-fix each suffix opened its own rpm=1 bucket and this
    // passed with 200.
    const second = await chat("wid/beta");
    expect(second.status).toBe(429);
    expect(second.type).toBe("rate_limit_exceeded");

    // Success + throttle series both label as the configured row; the
    // caller-minted suffixes never become label values.
    const metrics = await (await fetch(`${app.metricsUrl}/metrics`)).text();
    expect(metrics).toContain('model="wid/*"');
    expect(metrics).not.toContain('model="wid/alpha"');
    expect(metrics).not.toContain('model="wid/beta"');
  });
});
