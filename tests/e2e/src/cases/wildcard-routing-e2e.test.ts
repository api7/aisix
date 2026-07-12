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

const CALLER_PLAINTEXT = "sk-wildcard-routing-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// A key scoped to a single `vendor-a/*` glob — exercises the wildcard-aware
// `can_access` path end to end.
const SCOPED_PLAINTEXT = "sk-wildcard-routing-e2e-scoped";
const SCOPED_KEY_HASH = createHash("sha256")
  .update(SCOPED_PLAINTEXT)
  .digest("hex");

function okBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "upstream-model",
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

function sentModel(upstream: OpenAiUpstream): string {
  const last = upstream.receivedRequests[upstream.receivedRequests.length - 1];
  return JSON.parse(last!.body).model;
}

describe("wildcard (provider/*) model routing e2e", () => {
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
    await seed.createApiKey({
      key_hash: SCOPED_KEY_HASH,
      allowed_models: ["vendor-a/*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  // A direct model. `modelName` is the upstream `model_name` template — `"*"`
  // makes it a wildcard alias that substitutes the captured segment.
  async function createDirectModel(
    displayName: string,
    modelName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName.replace(/[^a-z0-9]/gi, "-")}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: modelName,
      provider_key_id: providerKey.id,
    });
  }

  function client(plaintext = CALLER_PLAINTEXT): OpenAI {
    return new OpenAI({
      apiKey: plaintext,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
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

  test("resolves provider/* and sends the captured model upstream", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const up = await startOpenAiUpstream({ nonStreamBody: okBody("wild-served") });
    upstreams.push(up);
    await createDirectModel("openai/*", "*", up);

    await waitSeedApplied("wild-openai");

    const baseline = up.receivedRequests.length;
    const completion = await client().chat.completions.create({
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });

    expect(completion.choices[0]?.message.content).toBe("wild-served");
    expect(up.receivedRequests.length - baseline).toBe(1);
    // `openai/*` + template `*` → the captured `gpt-4o` is sent upstream.
    expect(sentModel(up)).toBe("gpt-4o");
  });

  test("an exact model name wins over a matching wildcard", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const wild = await startOpenAiUpstream({ nonStreamBody: okBody("azure-wild") });
    const exact = await startOpenAiUpstream({ nonStreamBody: okBody("azure-exact") });
    upstreams.push(wild, exact);
    await createDirectModel("azure/*", "*", wild);
    await createDirectModel("azure/gpt-4o", "gpt-4o-2024-08-06", exact);

    await waitSeedApplied("wild-azure");

    // Exact name → the concrete model, not the wildcard.
    const exactBaseline = exact.receivedRequests.length;
    const wildBaseline = wild.receivedRequests.length;
    const hitExact = await client().chat.completions.create({
      model: "azure/gpt-4o",
      messages: [{ role: "user", content: "exact" }],
    });
    expect(hitExact.choices[0]?.message.content).toBe("azure-exact");
    expect(exact.receivedRequests.length - exactBaseline).toBe(1);
    expect(wild.receivedRequests.length - wildBaseline).toBe(0);
    expect(sentModel(exact)).toBe("gpt-4o-2024-08-06");

    // A name only the wildcard covers → the wildcard, capture substituted.
    const hitWild = await client().chat.completions.create({
      model: "azure/o3-mini",
      messages: [{ role: "user", content: "wild" }],
    });
    expect(hitWild.choices[0]?.message.content).toBe("azure-wild");
    expect(wild.receivedRequests.length - wildBaseline).toBe(1);
    expect(sentModel(wild)).toBe("o3-mini");
  });

  test("wildcard allowed_models scopes access to matching names", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const allowed = await startOpenAiUpstream({ nonStreamBody: okBody("vendor-a-served") });
    const denied = await startOpenAiUpstream({ nonStreamBody: okBody("vendor-b-served") });
    upstreams.push(allowed, denied);
    await createDirectModel("vendor-a/*", "*", allowed);
    // A concrete model the scoped key must NOT reach — it resolves, so the
    // rejection is authz (403), not not-found (404).
    await createDirectModel("vendor-b/thing", "thing", denied);

    await waitSeedApplied("wild-scoped");

    // In-scope wildcard name → allowed.
    const deniedBaseline = denied.receivedRequests.length;
    const ok = await client(SCOPED_PLAINTEXT).chat.completions.create({
      model: "vendor-a/anything",
      messages: [{ role: "user", content: "allowed" }],
    });
    expect(ok.choices[0]?.message.content).toBe("vendor-a-served");

    // Out-of-scope name → 403 before any upstream dispatch.
    await expect(
      client(SCOPED_PLAINTEXT).chat.completions.create({
        model: "vendor-b/thing",
        messages: [{ role: "user", content: "denied" }],
      }),
    ).rejects.toMatchObject({ status: 403 });
    expect(denied.receivedRequests.length - deniedBaseline).toBe(0);
  });
});
