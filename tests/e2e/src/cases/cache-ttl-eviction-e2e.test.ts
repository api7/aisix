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

// E2E: cache TTL expiry.
//
// CachePolicy carries a `ttl_seconds` field
// (documented in `docs/api-admin.md` §4.5 cache_policies CRUD).
// When set, an entry stored under that policy must be returned
// from cache only while it is younger than `ttl_seconds`. After
// expiry the gateway must miss and re-dispatch upstream.
//
// One contract pinned here:
//
//   - With `ttl_seconds: 2`, an entry stored at t=0 returns
//     `x-aisix-cache: hit` at t=1 and `x-aisix-cache: miss` at
//     t=4 (the upstream is re-hit on the post-expiry call).
//
// Why this matters: TTL is the only mechanism the operator has
// for time-bounded staleness — a regression that ignored
// `ttl_seconds` (always fell back to the cache backend's global
// TTL) would silently let stale answers serve indefinitely.
//
// Reference: `docs/api-admin.md` §4.5 (CachePolicy schema) and
// `docs/api-proxy.md` §3 (`x-aisix-cache` response header).

const CALLER_PLAINTEXT = "sk-cache-ttl-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TTL_SECONDS = 2;
// Wall-clock buffer above TTL_SECONDS to keep the post-expiry probe
// stable across slow CI runners. Total in-test wait = ~TTL + buffer.
const POST_EXPIRY_WAIT_MS = (TTL_SECONDS + 2) * 1000;

describe("cache TTL eviction e2e: entry expires after ttl_seconds", () => {
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
      display_name: "cache-ttl-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cache-ttl-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["cache-ttl-model"],
    });
    // CachePolicy with a short TTL so the test can wait it out
    // synchronously instead of mocking the clock. CachePolicy is not
    // exposed on the typed AdminClient yet — use the JSON helper.
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-ttl-policy",
      enabled: true,
      applies_to: "all",
      ttl_seconds: TTL_SECONDS,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "entry served from cache before TTL, re-dispatched after",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      // Snapshot propagation — drive the proxy path itself with a
      // distinct prompt so the readiness probe doesn't pollute the
      // cache fingerprint we're about to test against.
      const probeBody = JSON.stringify({
        model: "cache-ttl-model",
        messages: [{ role: "user", content: "ready-probe" }],
      });
      const reqHeaders = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: reqHeaders,
            body: probeBody,
          });
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      // Baseline-isolate the readiness probe so the upstream-call-count
      // assertions below measure only the actual test calls.
      const baseline = upstream.receivedRequests.length;

      const cacheBody = JSON.stringify({
        model: "cache-ttl-model",
        messages: [{ role: "user", content: "cache-ttl-prompt" }],
      });

      // (1) First call — cache miss, upstream hit.
      const first = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: cacheBody,
      });
      expect(first.status).toBe(200);
      expect(first.headers.get("x-aisix-cache")).toBe("miss");
      await first.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      // (2) Second call within TTL — cache hit, upstream NOT re-hit.
      const second = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: cacheBody,
      });
      expect(second.status).toBe(200);
      expect(second.headers.get("x-aisix-cache")).toBe("hit");
      await second.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 1);

      // (3) Wait past TTL.
      await new Promise((r) => setTimeout(r, POST_EXPIRY_WAIT_MS));

      // (4) Third call after TTL — cache miss, upstream re-hit. This
      // is the contract being pinned; a regression that ignored
      // `ttl_seconds` (e.g. relied on the moka backend's global TTL
      // configured to a much larger value) would still return `hit`
      // here and never re-dispatch.
      const third = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: cacheBody,
      });
      expect(third.status).toBe(200);
      expect(third.headers.get("x-aisix-cache")).toBe("miss");
      await third.text();
      expect(upstream.receivedRequests.length).toBe(baseline + 2);
    },
    // Per-test timeout: TTL wait + headroom for the four round-trips
    // and snapshot propagation. Default 60s would also work but is
    // tighter; set explicitly so a slow runner can't time out before
    // the post-expiry probe.
    POST_EXPIRY_WAIT_MS + 30_000,
  );
});
