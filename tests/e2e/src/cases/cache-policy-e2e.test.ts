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

// E2E: prompt-response cache policy, identical-request short-circuit.
// With a `CachePolicy{applies_to: "all", enabled: true}` in scope, two
// identical chat completions must result in the upstream being hit only
// once — the second response is served from cache. The unit-level
// `cache_hit_short_circuits_upstream_and_sets_header` test covers the
// in-process path; this case proves the wire-level contract end-to-end
// (real binary, real etcd watch propagation, real client SDK).
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create); the
// gateway's CachePolicy schema lives in
// `crates/aisix-core/src/models/cache_policy.rs`.

const CALLER_PLAINTEXT = "sk-cache-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("cache policy e2e: identical request hits cache", () => {
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
      display_name: "cache-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cache-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["cache-e2e"],
    });
    // CachePolicy is not exposed on AdminClient yet — call the JSON
    // helper directly. Schema mirrors `aisix-core::CachePolicy`:
    // name + enabled + applies_to is enough to enable the policy.
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-e2e-policy",
      enabled: true,
      applies_to: "all",
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("second identical request is served from cache, upstream not re-hit", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });

    // Wait for the snapshot to carry the Model + ProviderKey + ApiKey
    // + CachePolicy. The probe uses a distinct message so it doesn't
    // pollute the cache fingerprint we're about to test against.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "cache-e2e",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return true;
      } catch {
        return false;
      }
    });

    // Baseline includes the probe (and any retries during propagation).
    const baseline = upstream.receivedRequests.length;

    // First call with a fresh fingerprint — cache miss, upstream hit.
    const first = await client.chat.completions.create({
      model: "cache-e2e",
      messages: [{ role: "user", content: "cached prompt" }],
    });
    expect(upstream.receivedRequests.length).toBe(baseline + 1);

    // Second identical call — cache hit, upstream NOT re-hit.
    const second = await client.chat.completions.create({
      model: "cache-e2e",
      messages: [{ role: "user", content: "cached prompt" }],
    });
    expect(upstream.receivedRequests.length).toBe(baseline + 1);

    // Cache must replay the original response, not synthesize a new
    // one — caller can't distinguish cache hit from upstream re-call.
    expect(second.choices[0]?.message.content).toBe(
      first.choices[0]?.message.content,
    );
    expect(second.choices[0]?.message.role).toBe("assistant");

    // Caller-observable cache signaling — the gateway emits an
    // `x-aisix-cache: hit|miss|disabled` header that cp-api and the
    // dashboard's /logs view depend on. The OpenAI SDK swallows
    // response headers, so drop down to fetch directly to verify.
    // Use a fresh prompt so this header check doesn't reuse the cache
    // entry built up by the SDK calls above.
    const headerCheckHeaders = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const headerCheckBody = JSON.stringify({
      model: "cache-e2e",
      messages: [{ role: "user", content: "header-check-prompt" }],
    });
    const missResp = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: headerCheckHeaders,
      body: headerCheckBody,
    });
    expect(missResp.status).toBe(200);
    expect(missResp.headers.get("x-aisix-cache")).toBe("miss");
    await missResp.text();

    const hitResp = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: headerCheckHeaders,
      body: headerCheckBody,
    });
    expect(hitResp.status).toBe(200);
    expect(hitResp.headers.get("x-aisix-cache")).toBe("hit");
    await hitResp.text();
  });
});
