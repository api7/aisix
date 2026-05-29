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

// E2E: cache-counter pass-through on /v1/chat/completions usage (#542).
//
// Pre-fix the DP's response renderer copied only the canonical
// {prompt,completion,total}_tokens triplet and DROPPED the cache
// counters the upstream reported, so customers lost all prompt-cache
// visibility (cost transparency + billing-audit + cache tuning broken).
//
// The fix is the ecosystem-proven HYBRID:
//   - OpenAI upstream → emit `usage.prompt_tokens_details.cached_tokens`
//     (OpenAI-canonical nested shape) from the upstream's nested field
//   - DeepSeek upstream → BOTH normalize the native top-level
//     `prompt_cache_hit_tokens` into `prompt_tokens_details.cached_tokens`
//     AND pass the native `prompt_cache_hit_tokens` /
//     `prompt_cache_miss_tokens` through verbatim
//
// We assert against the raw response JSON (not the typed OpenAI SDK
// object) because the DeepSeek-native fields aren't on the SDK's typed
// surface — the wire body is what billing pipelines consume.
//
// References:
// - OpenAI usage object: https://platform.openai.com/docs/api-reference/chat/object
// - DeepSeek usage extension: https://api-docs.deepseek.com
// - Issue: api7/AISIX-Cloud#542 (+ #465)

const CALLER_PLAINTEXT = "sk-cache-counters-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

async function chatUsage(
  proxyUrl: string,
  model: string,
): Promise<Record<string, unknown>> {
  const res = await fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: "hello" }],
    }),
  });
  expect(res.status).toBe(200);
  const body = (await res.json()) as { usage?: Record<string, unknown> };
  return body.usage ?? {};
}

describe("usage cache-counter passthrough on /v1/chat/completions (#542)", () => {
  let app: SpawnedApp | undefined;
  let openaiUpstream: OpenAiUpstream | undefined;
  let deepseekUpstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // OpenAI-shape upstream: cache hit nested under prompt_tokens_details.
    openaiUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-oai-cache",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: {
          prompt_tokens: 1000,
          completion_tokens: 50,
          total_tokens: 1050,
          prompt_tokens_details: { cached_tokens: 800 },
        },
      },
    });

    // DeepSeek-shape upstream: cache counters at the top level of usage.
    deepseekUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-ds-cache",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "deepseek-chat",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: {
          prompt_tokens: 1000,
          completion_tokens: 50,
          total_tokens: 1050,
          prompt_cache_hit_tokens: 768,
          prompt_cache_miss_tokens: 232,
        },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Post-#302 Phase A: cp-api writes `provider` + `adapter` on every
    // PK row; the snapshot's two-tier dispatch needs both. DeepSeek
    // dispatches through the OpenAI-compat family bridge (`adapter:
    // "openai"`) — the same bridge whose usage parser this PR fixes.
    const oaiPk = await admin.createProviderKey({
      display_name: "cache-oai-pk",
      secret: "sk-mock",
      api_base: `${openaiUpstream.baseUrl}/v1`,
      provider: "openai",
      adapter: "openai",
    });
    await admin.createModel({
      display_name: "cache-oai",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: oaiPk.id,
    });

    const dsPk = await admin.createProviderKey({
      display_name: "cache-ds-pk",
      secret: "sk-mock",
      api_base: `${deepseekUpstream.baseUrl}/v1`,
      provider: "deepseek",
      adapter: "openai",
    });
    await admin.createModel({
      display_name: "cache-ds",
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: dsPk.id,
    });

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["cache-oai", "cache-ds"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await openaiUpstream?.close();
    await deepseekUpstream?.close();
  });

  test("OpenAI upstream → usage.prompt_tokens_details.cached_tokens reaches the client (#542)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const u = await chatUsage(app!.proxyUrl, "cache-oai");
        return typeof u.prompt_tokens === "number";
      } catch {
        return false;
      }
    });

    const usage = await chatUsage(app.proxyUrl, "cache-oai");
    // Pre-#542 the whole nested object was dropped.
    expect(usage.prompt_tokens_details, JSON.stringify(usage)).toBeDefined();
    expect(
      (usage.prompt_tokens_details as Record<string, unknown>).cached_tokens,
    ).toBe(800);
    // Canonical triplet still byte-for-byte.
    expect(usage.prompt_tokens).toBe(1000);
    expect(usage.total_tokens).toBe(1050);
  });

  test("DeepSeek upstream → native prompt_cache_hit_tokens AND normalized prompt_tokens_details both present (#542 hybrid)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const u = await chatUsage(app!.proxyUrl, "cache-ds");
        return typeof u.prompt_tokens === "number";
      } catch {
        return false;
      }
    });

    const usage = await chatUsage(app.proxyUrl, "cache-ds");
    // (a) native fields passed through verbatim — what a DeepSeek-aware
    //     client / billing reconciler reads. Pre-#542 these were dropped.
    expect(usage.prompt_cache_hit_tokens, JSON.stringify(usage)).toBe(768);
    expect(usage.prompt_cache_miss_tokens).toBe(232);
    // (b) ALSO normalized into the OpenAI-canonical nested shape so a
    //     standard OpenAI SDK client reading prompt_tokens_details works
    //     across providers.
    expect(usage.prompt_tokens_details, JSON.stringify(usage)).toBeDefined();
    expect(
      (usage.prompt_tokens_details as Record<string, unknown>).cached_tokens,
    ).toBe(768);
  });

  test("no cache fields emitted when upstream reports none (#542 — absent, not empty)", async (ctx) => {
    if (!etcdReachable || !app || !openaiUpstream) {
      ctx.skip();
      return;
    }
    // Reuse the OpenAI model but point a fresh upstream with NO cache
    // details, to confirm the renderer omits the nested object entirely
    // rather than emitting an empty `{}` (OpenAI SDK clients branch on
    // presence).
    const plainUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-plain",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 7, completion_tokens: 2, total_tokens: 9 },
      },
    });
    const plainApp = await spawnApp();
    try {
      const plainAdmin = new AdminClient(plainApp.adminUrl, plainApp.adminKey);
      const pk = await plainAdmin.createProviderKey({
        display_name: "cache-plain-pk",
        secret: "sk-mock",
        api_base: `${plainUpstream.baseUrl}/v1`,
        provider: "openai",
        adapter: "openai",
      });
      await plainAdmin.createModel({
        display_name: "cache-plain",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await plainAdmin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["cache-plain"],
      });

      await waitConfigPropagation(async () => {
        try {
          const u = await chatUsage(plainApp.proxyUrl, "cache-plain");
          return typeof u.prompt_tokens === "number";
        } catch {
          return false;
        }
      });

      const usage = await chatUsage(plainApp.proxyUrl, "cache-plain");
      expect(usage.prompt_tokens).toBe(7);
      expect(usage.prompt_tokens_details).toBeUndefined();
      expect(usage.prompt_cache_hit_tokens).toBeUndefined();
      expect(usage.completion_tokens_details).toBeUndefined();
    } finally {
      await plainApp.exit();
      await plainUpstream.close();
    }
  });

  test("streaming DeepSeek → terminal chunk carries native + normalized cache counters (#542 audit MEDIUM-7)", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    // DeepSeek streaming puts the usage block (incl. native cache
    // counters) on the terminal chunk. The harness wraps each
    // streamEvent as `data: <event>\n\n`; the DP parses via the
    // OpenAI-compat stream path → render_chunk applies the same
    // usage policy as the non-streaming path.
    const streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "cmpl-ds-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "deepseek-chat",
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        }),
        JSON.stringify({
          id: "cmpl-ds-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "deepseek-chat",
          choices: [{ index: 0, delta: { content: "hi" }, finish_reason: null }],
        }),
        // Terminal chunk: finish_reason set + usage with DeepSeek
        // native cache counters.
        JSON.stringify({
          id: "cmpl-ds-stream",
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: "deepseek-chat",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: {
            prompt_tokens: 1000,
            completion_tokens: 50,
            total_tokens: 1050,
            prompt_cache_hit_tokens: 640,
            prompt_cache_miss_tokens: 360,
          },
        }),
        "[DONE]",
      ],
    });
    const streamApp = await spawnApp();
    try {
      const streamAdmin = new AdminClient(streamApp.adminUrl, streamApp.adminKey);
      const pk = await streamAdmin.createProviderKey({
        display_name: "cache-ds-stream-pk",
        secret: "sk-mock",
        api_base: `${streamUpstream.baseUrl}/v1`,
        provider: "deepseek",
        adapter: "openai",
      });
      await streamAdmin.createModel({
        display_name: "cache-ds-stream",
        provider: "deepseek",
        model_name: "deepseek-chat",
        provider_key_id: pk.id,
      });
      await streamAdmin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["cache-ds-stream"],
      });

      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${streamApp.proxyUrl}/v1/chat/completions`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              authorization: `Bearer ${CALLER_PLAINTEXT}`,
            },
            body: JSON.stringify({
              model: "cache-ds-stream",
              stream: true,
              messages: [{ role: "user", content: "probe" }],
            }),
          });
          return r.ok;
        } catch {
          return false;
        }
      });

      const res = await fetch(`${streamApp.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
        },
        body: JSON.stringify({
          model: "cache-ds-stream",
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      expect(res.status).toBe(200);

      // Find the SSE chunk that carries the usage block and inspect it.
      const sse = await res.text();
      const usageChunk = sse
        .split("\n")
        .filter((l) => l.startsWith("data: ") && l.includes("\"usage\""))
        .map((l) => JSON.parse(l.slice("data: ".length)))
        .find((c) => c.usage);
      expect(usageChunk, `no usage chunk in stream:\n${sse}`).toBeDefined();
      const usage = usageChunk.usage as Record<string, unknown>;
      // Native passthrough on the streaming path
      expect(usage.prompt_cache_hit_tokens).toBe(640);
      expect(usage.prompt_cache_miss_tokens).toBe(360);
      // Normalized OpenAI-canonical shape on the streaming path
      expect(
        (usage.prompt_tokens_details as Record<string, unknown>).cached_tokens,
      ).toBe(640);
    } finally {
      await streamApp.exit();
      await streamUpstream.close();
    }
  });
});
