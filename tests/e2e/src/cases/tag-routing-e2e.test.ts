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

const CALLER_PLAINTEXT = "sk-tag-routing-e2e-caller";
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

describe("tag/metadata conditional routing e2e", () => {
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
    });
  }

  function client(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  async function askWithTags(tags: string | undefined): Promise<string | null> {
    const opts = tags ? { headers: { "x-aisix-routing-tags": tags } } : undefined;
    const completion = await client().chat.completions.create(
      { model: "tag-router", messages: [{ role: "user", content: "hi" }] },
      opts,
    );
    return completion.choices[0]?.message.content ?? null;
  }


  // Seed a throwaway canary key AFTER the resources under test — watch
  // events apply in revision order, so once this bearer authenticates
  // against /v1/models everything written before it is in the snapshot
  // too. Probing the routed models directly would warm cooldowns and
  // skew the per-target hit counts the assertions rely on.
  async function waitSeedApplied(label: string): Promise<void> {
    const canary = `sk-canary-${label}-${Date.now()}`;
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });
  }

  test("selects the tagged target, defaulting when unmatched or untagged", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const eu = await startOpenAiUpstream({ nonStreamBody: okBody("eu-served") });
    const us = await startOpenAiUpstream({ nonStreamBody: okBody("us-served") });
    const def = await startOpenAiUpstream({ nonStreamBody: okBody("default-served") });
    upstreams.push(eu, us, def);
    await createOpenAiModel("tag-eu", eu);
    await createOpenAiModel("tag-us", us);
    await createOpenAiModel("tag-default", def);
    // failover keeps target selection deterministic: whatever the tag filter
    // leaves, the first survivor is attempted.
    await seed.createModel({
      display_name: "tag-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "tag-eu", tags: ["eu"] },
          { model: "tag-us", tags: ["us"] },
          { model: "tag-default", tags: ["default"] },
        ],
      },
    });

    await waitSeedApplied("tag-router");

    // Matching tags route to their target.
    expect(await askWithTags("eu")).toBe("eu-served");
    expect(await askWithTags("us")).toBe("us-served");
    // An untagged request uses the `default`-tagged target.
    expect(await askWithTags(undefined)).toBe("default-served");
    // A tag matching no target also falls back to default.
    expect(await askWithTags("apac")).toBe("default-served");
    // The routing header is out-of-band and must never reach the upstream body.
    const lastEu = JSON.parse(eu.receivedRequests[eu.receivedRequests.length - 1]!.body);
    expect(lastEu.metadata).toBeUndefined();
  });
});
