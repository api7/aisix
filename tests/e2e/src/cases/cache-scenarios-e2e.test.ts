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

// E2E: cache scenarios beyond the identical-prompt happy path.
// The existing cache-policy-e2e covers "identical request → hit";
// this file fills the opposing user journey:
//
//   - Different prompt → MISS — proves the cache fingerprint isn't
//     a trivial always-hit (a regression that fingerprinted on
//     anything constant would silently turn every request into a
//     hit, returning stale answers to different questions).
//
// (The "policy enabled=false → bypass" case is tracked as a
// separate test pending a product fix; see follow-up issue.)
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for
// the request/response shape; the gateway's `x-aisix-cache:
// hit|miss|disabled` response header is its own published contract
// (depended on by cp-api / dashboard `/logs`).

const CALLER_PLAINTEXT = "sk-cache-scen-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("cache scenarios e2e: different prompt → miss", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  // Each test owns its own upstream + Model so request-count assertions
  // are isolated across cases.
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

  test("different prompt → MISS (cache fingerprint reflects the request)", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream();
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "cache-diff-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cache-diff",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-diff-policy",
      enabled: true,
      applies_to: "all",
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });

    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "cache-diff",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return true;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    // Send three DIFFERENT prompts back-to-back. Each should be a
    // cache miss because the fingerprint is supposed to include the
    // prompt content. A regression that hashed on (model, caller)
    // alone would turn the second + third call into hits and the
    // upstream would only see one request — caller would receive
    // the FIRST prompt's answer for the next two prompts.
    const prompts = ["question one", "question two", "question three"];
    for (const prompt of prompts) {
      await client.chat.completions.create({
        model: "cache-diff",
        messages: [{ role: "user", content: prompt }],
      });
    }
    expect(upstream.receivedRequests.length - baseline).toBe(prompts.length);

    // Caller-observable header signal: each MISS must be marked as
    // such. A stale-cache regression that hit on every prompt would
    // emit `hit` here while still using the wrong cached answer.
    // Pin upstream count too — a regression that served from cache
    // but reported `miss` in the header would slip past a header-
    // only check.
    const headerCheckBaseline = upstream.receivedRequests.length;
    const headerCheckHeaders = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };
    const headerPrompts = ["header-prompt-A", "header-prompt-B"];
    for (const prompt of headerPrompts) {
      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: headerCheckHeaders,
        body: JSON.stringify({
          model: "cache-diff",
          messages: [{ role: "user", content: prompt }],
        }),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get("x-aisix-cache")).toBe("miss");
      await res.text();
    }
    expect(upstream.receivedRequests.length - headerCheckBaseline).toBe(
      headerPrompts.length,
    );
  });

});
