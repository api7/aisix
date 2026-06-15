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

// E2E: a non-streaming cache HIT must still run output guardrails (#448).
// Pre-fix, cache hits returned the stored body without any output check,
// so a response cached before an output guardrail existed (or under a key
// without one) could be replayed past a guardrail that should block it.
//
// The mock upstream's canned reply is "mock reply"; we attach an output
// guardrail blocking the literal "reply" AFTER the response is cached,
// then re-issue the identical (cached) request and require it to be
// blocked rather than served from cache.

const CALLER = "sk-cache-gr-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");
const CACHED_PROMPT = "cache-and-guard-me";

describe("cache hit runs output guardrails (#448)", () => {
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
      display_name: "cache-gr-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "cache-gr",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({ key_hash: HASH, allowed_models: ["cache-gr"] });
    await admin.json("POST", "/admin/v1/cache_policies", {
      name: "cache-gr-policy",
      enabled: true,
      applies_to: "all",
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const chat = (content: string) =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER}` },
      body: JSON.stringify({ model: "cache-gr", messages: [{ role: "user", content }] }),
    });

  test("a response cached before the guardrail is blocked on the cache hit", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Wait until model+key+pk+cache are live (clean prompt → 200 + cache miss).
    await waitConfigPropagation(async () => (await chat("ready-probe")).ok);

    // 1) Cache the response BEFORE any output guardrail exists.
    const first = await chat(CACHED_PROMPT);
    expect(first.status, "first request should succeed and populate the cache").toBe(200);
    expect(first.headers.get("x-aisix-cache")).toBe("miss");

    // 2) Attach an output guardrail blocking the canned reply text.
    await admin!.json("POST", "/admin/v1/guardrails", {
      name: "cache-gr-output-keyword",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: "reply" }],
    });

    // Gate on guardrail propagation: a FRESH prompt (cache miss) returns
    // the canned "mock reply", which the output guardrail must now block.
    await waitConfigPropagation(async () => (await chat(`probe-${Math.random()}`)).status === 422);

    // 3) Re-issue the cached request: it is a cache hit, and must now be
    //    blocked by the output guardrail rather than replayed from cache.
    const hit = await chat(CACHED_PROMPT);
    expect(
      hit.status,
      "cache hit must run output guardrails and block the stored reply",
    ).toBe(422);
    const body = await hit.json();
    expect(JSON.stringify(body)).toContain("content_filter");
  });
});
