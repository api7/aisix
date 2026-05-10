import { createHash } from "node:crypto";
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

// E2E: cache fingerprint MUST distinguish requests that differ only
// in OpenAI-shape "extras" — `tools`, `tool_choice`, `response_format`,
// `seed`, `stop`, `presence_penalty`, `frequency_penalty`. These
// fields materially change the upstream's behavior; collapsing them
// into the same cache entry would serve a stale answer that ignores
// the caller's actual constraint (e.g. answering as JSON when the
// caller asked for plain text, or returning a non-deterministic
// completion when the caller pinned a `seed`).
//
// Two contracts pinned here:
//
//   1. Identical prompt + same extras → cache HIT, upstream not
//      re-hit. Sanity check that the cache is actually working
//      and doesn't bypass on any present extras.
//
//   2. Identical prompt + DIFFERENT extras → cache MISS each time,
//      upstream re-hit. Catches a regression where the cache key
//      hashed only the prompt and ignored extras — a silent
//      cross-contamination of cached answers.
//
// Reference:
//   - cache key fingerprint contents documented in
//     `docs/api-proxy.md` §4.2 (PR #191)
//   - cache key implementation: cache_policies CRUD on the admin
//     side (issue #201 covers the missing doc)

const CALLER_PLAINTEXT = "sk-cache-fp-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Shared prompt across all probes — only the extras change between
// calls. If the cache fingerprint were broken (only hashed the
// prompt), every call after the first would falsely hit.
const SHARED_PROMPT = "fingerprint-collision-prompt";

// Tool definition used in (3). Concrete schema is irrelevant; only
// its presence vs absence matters for the cache key.
const TOOL_DEF = {
  type: "function" as const,
  function: {
    name: "get_weather",
    description: "Get the weather for a location",
    parameters: {
      type: "object",
      properties: { location: { type: "string" } },
      required: ["location"],
    },
  },
};

describe("cache fingerprint collision e2e: extras change → distinct cache key", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "cache-fp-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cache-fp-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["cache-fp-model"],
    });
    // CachePolicy with no TTL override (uses backend default, plenty
    // long for the test) and applies_to:all so every request runs
    // through the cache layer.
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-fp-policy",
      enabled: true,
      applies_to: "all",
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "identical extras hit; tools / response_format / seed each force a miss",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const reqHeaders = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };

      // Probe with a distinct prompt so it doesn't pollute the
      // fingerprints under test, AND wait until the gateway emits
      // `x-aisix-cache: miss` so we know the CachePolicy is loaded
      // (without it the gateway emits `disabled`). Mirrors the
      // pattern PR #200 audit added.
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: reqHeaders,
            body: JSON.stringify({
              model: "cache-fp-model",
              messages: [{ role: "user", content: "ready-probe" }],
            }),
          });
          await r.text();
          return (
            r.status === 200 &&
            r.headers.get("x-aisix-cache") === "miss"
          );
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;

      // (1) First call with prompt only — cache miss, upstream hit.
      const r1 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-fp-model",
          messages: [{ role: "user", content: SHARED_PROMPT }],
        }),
      });
      expect(r1.status).toBe(200);
      expect(r1.headers.get("x-aisix-cache")).toBe("miss");
      await r1.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      // (2) Same call again — cache hit, upstream NOT re-hit.
      // Sanity gate: confirms the cache is actually working before
      // we move on to extras-induced misses.
      const r2 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-fp-model",
          messages: [{ role: "user", content: SHARED_PROMPT }],
        }),
      });
      expect(r2.status).toBe(200);
      expect(r2.headers.get("x-aisix-cache")).toBe("hit");
      await r2.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      // (3) Same prompt + add `tools` — must MISS (extras differ).
      // A regression that ignored `tools` in the cache key would
      // serve r1's plain-text completion to a caller asking for
      // tool invocation — silently wrong answer.
      const r3 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-fp-model",
          messages: [{ role: "user", content: SHARED_PROMPT }],
          tools: [TOOL_DEF],
        }),
      });
      expect(r3.status).toBe(200);
      expect(r3.headers.get("x-aisix-cache")).toBe("miss");
      await r3.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 2);

      // (4) Same prompt + different `response_format` — must MISS.
      // A regression that ignored `response_format` would return
      // r1's plain text when the caller asked for json_object —
      // breaking every JSON-mode caller.
      const r4 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-fp-model",
          messages: [{ role: "user", content: SHARED_PROMPT }],
          response_format: { type: "json_object" },
        }),
      });
      expect(r4.status).toBe(200);
      expect(r4.headers.get("x-aisix-cache")).toBe("miss");
      await r4.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 3);

      // (5) Same prompt + different `seed` — must MISS. A
      // regression that ignored `seed` would return the same
      // completion to every seed value, defeating reproducibility.
      const r5 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-fp-model",
          messages: [{ role: "user", content: SHARED_PROMPT }],
          seed: 42,
        }),
      });
      expect(r5.status).toBe(200);
      expect(r5.headers.get("x-aisix-cache")).toBe("miss");
      await r5.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 4);

      // (6) Repeat (5) verbatim — must HIT. Confirms each
      // extras-distinct fingerprint is itself stable, not just
      // non-colliding. A regression where the extras hashing was
      // non-deterministic would fail here.
      const r6 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-fp-model",
          messages: [{ role: "user", content: SHARED_PROMPT }],
          seed: 42,
        }),
      });
      expect(r6.status).toBe(200);
      expect(r6.headers.get("x-aisix-cache")).toBe("hit");
      await r6.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 4);
    },
    60_000,
  );
});
