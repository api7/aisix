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

    await admin.createModel({
      name: "smoke-gpt",
      model: "openai/gpt-4o-mini",
      // The OpenAI bridge appends `/chat/completions`, so the api_base
      // already needs the `/v1` segment to land on `/v1/chat/completions`.
      provider_config: { api_key: "sk-mock", api_base: `${upstream.baseUrl}/v1` },
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["smoke-gpt"],
    });

    await waitConfigPropagation();

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
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

    const seen = upstream.receivedRequests.some((r) =>
      r.path.startsWith("/v1/chat/completions"),
    );
    if (!seen) {
      throw new Error(
        `upstream did not receive /v1/chat/completions; saw paths: ${JSON.stringify(upstream.receivedRequests.map((r) => `${r.method} ${r.path}`))}`,
      );
    }
  });
});
