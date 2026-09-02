import { createHash } from "node:crypto";
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

// E2E: the three optional passthrough operations answer 501 when the
// provider's adapter does not implement them.
//
// Each is produced by a `Bridge` default implementation, and until #1093
// the route decided the status by searching the resulting error's MESSAGE
// for a phrase. Nothing in the tree pinned the link, so rewording that
// sentence would have turned every one of these 501s into a 500 silently.
// These cases are that link, asserted from outside: a real gateway, a real
// adapter that lacks the capability, and the status the caller gets.
//
// The Anthropic adapter is the one that keeps all three defaults. It is
// reached here through the provider key, which is what selects the bridge —
// the model's own `provider` only gates which routes will dispatch at all,
// which is why the image case names `openai` there and still lands on the
// Anthropic bridge.

const CALLER_PLAINTEXT = "sk-unsupported-capability-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

/** Model whose bridge implements chat and nothing else. */
const CHAT_ONLY = "capability-gap-model";
/** Same bridge, but declared `provider: openai` so `/v1/images/generations` dispatches. */
const CHAT_ONLY_IMAGES = "capability-gap-images";

describe("unsupported capability e2e: an adapter that lacks the operation answers 501", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    // Never contacted — the refusal happens before any request is built.
    // Seeded anyway so the configuration is an ordinary one and the 501
    // cannot be an artefact of an unreachable upstream.
    upstream = await startOpenAiUpstream({ nonStreamBody: { ok: true } });

    const pk = await seed.createProviderKey({
      display_name: "capability-gap-pk",
      secret: "sk-anthropic-mock",
      api_base: upstream.baseUrl,
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: CHAT_ONLY,
      provider: "anthropic",
      model_name: "claude-capability-mock",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: CHAT_ONLY_IMAGES,
      provider: "openai",
      model_name: "claude-capability-mock",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: HEADERS.authorization },
      });
      if (res.status !== 200) {
        await res.text();
        return false;
      }
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      const ids = (body.data ?? []).map((m) => m.id);
      return ids.includes(CHAT_ONLY) && ids.includes(CHAT_ONLY_IMAGES);
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const cases: Array<{ route: string; model: string; body: unknown; capability: string }> = [
    {
      route: "/v1/completions",
      model: CHAT_ONLY,
      body: { prompt: "say ok" },
      capability: "text completions",
    },
    {
      route: "/v1/embeddings",
      model: CHAT_ONLY,
      body: { input: "say ok" },
      capability: "embeddings",
    },
    {
      route: "/v1/images/generations",
      model: CHAT_ONLY_IMAGES,
      body: { prompt: "a cardboard city" },
      capability: "image generation",
    },
  ];

  for (const c of cases) {
    test(
      `${c.route} answers 501 when the adapter does not implement it`,
      async (ctx) => {
        if (!etcdReachable || !app || !upstream) return void ctx.skip();

        const baseline = upstream.receivedRequests.length;
        const res = await fetch(`${app.proxyUrl}${c.route}`, {
          method: "POST",
          headers: HEADERS,
          body: JSON.stringify({ model: c.model, ...(c.body as object) }),
        });

        expect(res.status).toBe(501);
        const body = (await res.json()) as {
          error?: { message?: string; type?: string };
        };
        expect(body.error?.type).toBe("not_implemented");
        // The message names the capability, which is what tells an operator
        // to point the alias at a different provider.
        expect(body.error?.message).toBe(
          `this provider does not support ${c.capability}`,
        );

        // No provider was contacted, so there is nothing to bill and
        // nothing to retry.
        expect(upstream.receivedRequests.slice(baseline)).toHaveLength(0);
      },
      60_000,
    );
  }
});
