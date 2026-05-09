import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// v3 self-hosted CP wire (§9A.7B.4): the snapshot stores SHA-256 of
// the plaintext bearer, never the plaintext itself. The DP hashes
// incoming `Bearer <plaintext>` and looks the key up by hash.
// Keep this helper inline so the test independently re-derives the
// hash the same way `aisix_core::ApiKey::hash_bearer` does on the
// Rust side — divergence between the two is the bug we want to
// catch.
const CALLER_PLAINTEXT = "sk-smoke-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("smoke: admin write → proxy read", () => {
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
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a Model + ApiKey written via Admin API are visible to /v1/models", async (ctx) => {
    if (!etcdReachable || !app || !admin || !upstream) {
      ctx.skip();
      return;
    }

    // Phase B Model shape: ProviderKey carries the upstream secret +
    // optional api_base override; Model references it by id.
    const pk = await admin.createProviderKey({
      display_name: "smoke-openai",
      secret: "sk-mock",
      // The OpenAI bridge appends `/chat/completions`, so the api_base
      // already needs the `/v1` segment to land on `/v1/chat/completions`.
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "smoke-gpt",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["smoke-gpt"],
    });

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    // Poll /v1/models until the freshly-written model is visible — the
    // supervisor's watch pipeline normally catches up within ~50ms but
    // CI runners occasionally lag behind a fixed sleep.
    await waitConfigPropagation(async () => {
      const res = await proxy.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "smoke-gpt");
    });

    const { status, body } = await proxy.listModels();
    expect(status).toBe(200);
    expect(body).toMatchObject({
      object: "list",
      data: expect.arrayContaining([expect.objectContaining({ id: "smoke-gpt" })]),
    });
  });

  test("a chat completion forwards to the mock upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    // The Model + ProviderKey writes from the previous test propagate
    // independently. /v1/models confirms the Model row but not the
    // referenced ProviderKey — poll the chat path itself until the
    // dispatcher stops returning `unknown provider_key_id`, the only
    // signal that captures the snapshot's complete state.
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: "smoke-gpt",
        messages: [{ role: "user", content: "ping" }],
      });
      if (probe.status === 200) return true;
      const msg = JSON.stringify(probe.body);
      return !msg.includes("unknown provider_key_id");
    });

    // Baseline-isolate the readiness probe so the assertion below
    // measures only the test call's effect on the upstream.
    const baseline = upstream.receivedRequests.length;
    const { status, body } = await proxy.chat({
      model: "smoke-gpt",
      messages: [{ role: "user", content: "hello" }],
    });

    if (status !== 200) {
      throw new Error(
        `chat returned ${status}: ${JSON.stringify(body)}\n  upstream paths: ${JSON.stringify(upstream.receivedRequests.map((r) => r.path))}`,
      );
    }
    expect(body).toMatchObject({
      object: "chat.completion",
      choices: expect.arrayContaining([
        expect.objectContaining({
          message: expect.objectContaining({ role: "assistant" }),
        }),
      ]),
    });

    // Test call hit the upstream exactly once at the OpenAI Chat
    // Completions path. `some()` would let a regression that double-
    // fires (or short-circuits and leaks through a stray route)
    // silently pass.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
  });
});
