import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: when an upstream provider returns 429 with a `Retry-After`
// header, the gateway must forward BOTH the 429 status AND the
// `Retry-After` value to the client, so SDKs back off on the provider's
// actual cool-down instead of a default (#144). Before the fix the
// gateway forwarded the 429 but dropped the header.

const CALLER_PLAINTEXT = "sk-retry-after-passthrough";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("upstream 429 Retry-After passthrough e2e", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Every upstream reply is a 429 carrying `Retry-After: 30`. The
    // readiness probe below uses the proxy's model list (not the
    // upstream), so only the chat call actually hits this 429.
    upstream = await startOpenAiUpstream({
      status: 429,
      responseHeaders: { "retry-after": "30" },
      errorBody: { error: { message: "slow down", type: "rate_limit_error" } },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "retry-after-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "retry-after-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["retry-after-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("forwards the upstream Retry-After header on a 429 passthrough", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "retry-after-model");
    });

    // maxRetries=0 so the SDK surfaces the first 429 instead of
    // silently retrying around it.
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "retry-after-model",
        messages: [{ role: "user", content: "hi" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(429);
    // The load-bearing assertion: the upstream's Retry-After must reach
    // the client verbatim. OpenAI Node SDK 4.x lowercases header lookups.
    expect(caught.headers?.["retry-after"]).toBe("30");
  });
});
