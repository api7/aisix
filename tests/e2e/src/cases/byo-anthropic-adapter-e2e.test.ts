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

// E2E: a `provider: "byo"` model whose ProviderKey uses `adapter: anthropic`
// fronts a self-hosted / proxied Anthropic endpoint. Both Anthropic-native
// routes must treat it exactly like the catalog vendor:
//
//   - `/v1/messages` forwards the caller's body VERBATIM. The gateway used
//     to branch on the vendor id, so this model went through the
//     cross-provider bridge, which re-encodes the body from its normalized
//     form and drops caller-owned fields — `cache_control` among them,
//     which changes both prompt-cache behavior and what the upstream bills
//     (a 1h cache write is 2x the base input rate, 5m is 1.25x).
//   - `/v1/messages/count_tokens` serves it. The same vendor-id gate
//     rejected it outright with a 400, while its sibling `/v1/messages`
//     happily served the very same model.
//
// The converse also has to hold, and a first-party model on the same mock
// upstream is seeded here as its control: a request rewrite that exists
// because the upstream is NOT Anthropic must key on the vendor id and
// leave the catalog vendor alone. That is what the billing-attribution
// cases at the bottom of this file pin, in both directions.

const CALLER_PLAINTEXT = "sk-byo-anthropic-e2e";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const UPSTREAM_MODEL_ID = "claude-sonnet-4-5";
const MODEL_ALIAS = "byo-claude-e2e";
// A model on the SAME mock upstream reached through the `anthropic`
// catalog vendor, so the two differ in nothing but the vendor id.
const FIRST_PARTY_ALIAS = "first-party-claude-e2e";
const BILLING_LINE =
  "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli; cch=7f3a91;";

const anthropicHeaders = {
  "content-type": "application/json",
  "x-api-key": CALLER_PLAINTEXT,
  "anthropic-version": "2023-06-01",
};

describe("byo + anthropic adapter e2e: the Anthropic-native routes key on the adapter", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      // Serves both routes: the count_tokens body is what
      // `/v1/messages/count_tokens` returns; `/v1/messages` only needs a
      // 200 here since the assertions are on the REQUEST the DP sent.
      nonStreamBody: { input_tokens: 42 },
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "byo-anthropic-pk",
      secret: "sk-byo-upstream",
      api_base: upstream.baseUrl,
      // The combination under test: a vendor id that is NOT "anthropic",
      // with the Anthropic protocol declared on the adapter.
      provider: "byo",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: MODEL_ALIAS,
      provider: "byo",
      model_name: UPSTREAM_MODEL_ID,
      provider_key_id: pk.id,
    });
    const firstPartyPk = await seed.createProviderKey({
      display_name: "first-party-anthropic-pk",
      secret: "sk-first-party-upstream",
      api_base: upstream.baseUrl,
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: FIRST_PARTY_ALIAS,
      provider: "anthropic",
      model_name: UPSTREAM_MODEL_ID,
      provider_key_id: firstPartyPk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL_ALIAS, FIRST_PARTY_ALIAS],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("/v1/messages forwards the caller's cache_control unchanged", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const send = () =>
      fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: anthropicHeaders,
        body: JSON.stringify({
          model: MODEL_ALIAS,
          max_tokens: 64,
          system: [
            {
              type: "text",
              text: "long shared preamble",
              cache_control: { type: "ephemeral", ttl: "5m" },
            },
          ],
          messages: [{ role: "user", content: "hello" }],
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        return (await send()).ok;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await send();
    expect(res.status).toBe(200);

    const req = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(req).toBeDefined();

    const sent = JSON.parse(req!.body) as {
      model?: string;
      system?: Array<{ cache_control?: unknown }>;
    };
    // Alias rewritten to the upstream id, everything else verbatim.
    expect(sent.model).toBe(UPSTREAM_MODEL_ID);
    expect(sent.system?.[0]?.cache_control).toEqual({
      type: "ephemeral",
      ttl: "5m",
    });
    // Anthropic auth shape, not Bearer.
    expect(req!.headers["x-api-key"]).toBe("sk-byo-upstream");
    expect(req!.headers["authorization"]).toBeUndefined();
  });

  test("/v1/messages/count_tokens serves the model instead of rejecting it", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const baseline = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: anthropicHeaders,
      body: JSON.stringify({
        model: MODEL_ALIAS,
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as { input_tokens?: unknown };
    expect(body.input_tokens).toBe(42);

    const req = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages/count_tokens");
    expect(req).toBeDefined();
    expect((JSON.parse(req!.body) as { model?: string }).model).toBe(
      UPSTREAM_MODEL_ID,
    );
  });
  // The billing-attribution line Anthropic-native clients prepend to the
  // system prompt is metadata for Anthropic's own API. It carries a
  // per-request-varying segment and sits at the very front of the prompt,
  // so an upstream that cannot read it still pays for it: every turn
  // presents a different prefix and misses that provider's prompt cache.
  // The vendor id is what decides, not the protocol — this key speaks the
  // Anthropic wire and is still not Anthropic.
  test("/v1/messages drops the billing-attribution block for a third-party Anthropic upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const baseline = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: anthropicHeaders,
      body: JSON.stringify({
        model: MODEL_ALIAS,
        max_tokens: 64,
        system: [
          { type: "text", text: BILLING_LINE },
          {
            type: "text",
            text: "long shared preamble",
            cache_control: { type: "ephemeral", ttl: "5m" },
          },
        ],
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    expect(res.status).toBe(200);

    const req = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(req).toBeDefined();
    expect(req!.body).not.toContain("x-anthropic-billing-header");

    const sent = JSON.parse(req!.body) as {
      system?: Array<{ type?: string; text?: string; cache_control?: unknown }>;
    };
    // Everything else survives the rewrite, cache_control included — the
    // block that stays is what the upstream's prompt cache keys on.
    expect(sent.system).toEqual([
      {
        type: "text",
        text: "long shared preamble",
        cache_control: { type: "ephemeral", ttl: "5m" },
      },
    ]);
  });

  test("/v1/messages/count_tokens drops it too, so the count matches what would be sent", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const baseline = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: anthropicHeaders,
      body: JSON.stringify({
        model: MODEL_ALIAS,
        system: [
          { type: "text", text: BILLING_LINE },
          { type: "text", text: "long shared preamble" },
        ],
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    expect(res.status).toBe(200);

    const req = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages/count_tokens");
    expect(req).toBeDefined();
    expect(req!.body).not.toContain("x-anthropic-billing-header");
    const sent = JSON.parse(req!.body) as { system?: Array<{ text?: string }> };
    expect(sent.system).toEqual([
      { type: "text", text: "long shared preamble" },
    ]);
  });

  test("/v1/messages keeps the billing-attribution block for the anthropic vendor", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const system = [
      { type: "text", text: BILLING_LINE },
      {
        type: "text",
        text: "long shared preamble",
        cache_control: { type: "ephemeral", ttl: "5m" },
      },
    ];

    const send = () =>
      fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: anthropicHeaders,
        body: JSON.stringify({
          model: FIRST_PARTY_ALIAS,
          max_tokens: 64,
          system,
          messages: [{ role: "user", content: "hello" }],
        }),
      });

    await waitConfigPropagation(async () => {
      try {
        return (await send()).ok;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await send();
    expect(res.status).toBe(200);

    const req = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages");
    expect(req).toBeDefined();

    // Anthropic's own API is the one consumer of this line, so it goes
    // through untouched — the strip must key on the vendor id and nothing
    // else.
    const sent = JSON.parse(req!.body) as { system?: unknown };
    expect(sent.system).toEqual(system);
  });
});
