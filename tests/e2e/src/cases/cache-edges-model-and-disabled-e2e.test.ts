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

// E2E: cache edge cases not covered by cache-policy-e2e (#127 / #151
// C4.2 + C4.4):
//
//   1. Cache fingerprint includes the resolved model. Same prompt
//      against two distinct Models must NOT collide. The existing
//      fingerprint-collision case (PR #218) pins distinctness across
//      OpenAI-shape "extras" (`tools` / `seed` / `response_format`),
//      but never compared two different Models — a regression that
//      hashed only the prompt would have served a Model-A response
//      to a Model-B caller.
//
//   2. A scope covered ONLY by a `CachePolicy { enabled: false }`
//      sees no caching: every call re-hits the upstream and the
//      `x-aisix-cache` response header is absent. Per
//      `docs/api-proxy.md` §3 the header values are `hit` or
//      `miss` only — when no enabled policy applies (whether
//      because none exists or because the matching one is
//      disabled), the header is omitted entirely. A regression
//      that selected a disabled policy as the active one (and
//      then cached anyway) would either still emit the header or
//      still skip upstream on the second identical call; either
//      surfaces here.
//
// References:
//   - cache fingerprint contents: `docs/api-proxy.md` §4.2 (PR #191)
//   - CachePolicy schema: `crates/aisix-core/src/models/cache_policy.rs`
//     (doc section tracked in #201)

const CALLER_PLAINTEXT = "sk-cache-edges-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const SHARED_PROMPT = "cache-edges-shared-prompt";

describe("cache edges: model in fingerprint + enabled:false bypass", () => {
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

    // Two distinct Models pointing at the SAME mock upstream — only
    // the gateway-side `display_name` differs (and the upstream
    // `model_name` value the proxy rewrites to). If the cache
    // fingerprint correctly includes the model identifier, requests
    // against the two Models must be cached independently.
    const pk = await admin.createProviderKey({
      display_name: "cache-edges-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cache-edges-A",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createModel({
      display_name: "cache-edges-B",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    // A third Model used only for the "policy disabled" probe so it
    // doesn't share the cache scope with A/B above.
    await admin.createModel({
      display_name: "cache-edges-disabled",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "cache-edges-A",
        "cache-edges-B",
        "cache-edges-disabled",
      ],
    });
    // Two enabled policies scoped narrowly to A and B respectively
    // so test (1) has caching ON for those Models, plus one
    // disabled policy scoped to the third Model. Crucially the
    // enabled policies must NOT use `applies_to:"all"` — if they
    // did, the third Model would also match an enabled rule and
    // test (2) would observe caching despite the disabled policy.
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-edges-policy-a",
      enabled: true,
      applies_to: "model:cache-edges-A",
    });
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-edges-policy-b",
      enabled: true,
      applies_to: "model:cache-edges-B",
    });
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-edges-disabled-policy",
      enabled: false,
      applies_to: "model:cache-edges-disabled",
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "(1) same prompt against two Models → cache fingerprint distinct",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const reqHeaders = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };

      // Readiness probe — wait until the enabled policy is loaded
      // (gateway emits `miss`, not `disabled`, on a fresh fingerprint).
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: reqHeaders,
            body: JSON.stringify({
              model: "cache-edges-A",
              messages: [{ role: "user", content: "ready-probe-1" }],
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

      // Model A — fresh fingerprint, miss + upstream hit.
      const a1 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-edges-A",
          messages: [{ role: "user", content: SHARED_PROMPT }],
        }),
      });
      expect(a1.headers.get("x-aisix-cache")).toBe("miss");
      await a1.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      // Model A again — same fingerprint, hit + upstream NOT re-hit.
      // Sanity gate that the policy is actually caching.
      const a2 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-edges-A",
          messages: [{ role: "user", content: SHARED_PROMPT }],
        }),
      });
      expect(a2.headers.get("x-aisix-cache")).toBe("hit");
      await a2.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      // Model B with the SAME prompt — must MISS. A regression that
      // hashed only prompt would (incorrectly) return A's cached
      // answer to a B caller. Note the upstream `model_name`
      // strings differ (gpt-4o-mini vs gpt-4o), so even if a
      // hypothetical fingerprint hashed the upstream model id
      // instead of the gateway display_name, this still surfaces
      // a model-mismatch regression.
      const b1 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-edges-B",
          messages: [{ role: "user", content: SHARED_PROMPT }],
        }),
      });
      expect(b1.headers.get("x-aisix-cache")).toBe("miss");
      await b1.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 2);

      // Model B repeat — hit (each Model's fingerprint is itself
      // stable). Catches a regression where the hash is non-
      // deterministic per-Model.
      const b2 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "cache-edges-B",
          messages: [{ role: "user", content: SHARED_PROMPT }],
        }),
      });
      expect(b2.headers.get("x-aisix-cache")).toBe("hit");
      await b2.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 2);
    },
    60_000,
  );

  test(
    "(2) scope covered only by enabled:false policy → no caching, no x-aisix-cache header",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const reqHeaders = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };

      // Readiness probe — wait until the gateway can dispatch
      // through this Model AND the disabled-policy state is in
      // effect (no x-aisix-cache header). Asserting header-absent
      // confirms the snapshot does not surface an enabled policy
      // matching this Model.
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: reqHeaders,
            body: JSON.stringify({
              model: "cache-edges-disabled",
              messages: [{ role: "user", content: "ready-probe-2" }],
            }),
          });
          await r.text();
          return (
            r.status === 200 &&
            r.headers.get("x-aisix-cache") === null
          );
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;
      const body = JSON.stringify({
        model: "cache-edges-disabled",
        messages: [{ role: "user", content: SHARED_PROMPT }],
      });

      // Fire two identical calls; both must omit the cache header
      // AND both must re-hit upstream. A regression that picked a
      // disabled policy as if it were enabled would either emit
      // the header on hit or skip upstream on the second call.
      const r1 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body,
      });
      expect(r1.headers.get("x-aisix-cache")).toBeNull();
      await r1.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      const r2 = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body,
      });
      expect(r2.headers.get("x-aisix-cache")).toBeNull();
      await r2.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 2);
    },
    60_000,
  );
});
