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

// E2E: a customer-fixable upstream-config error must surface to the
// client as a 400, not a 500 (#367). The scenario is the issue's own
// example — a family-adapter vendor (openrouter) admitted without an
// api_base: the family bridge refuses to fall back to api.openai.com
// and errors before dispatch. A 5xx would tell SDKs to retry and
// monitoring to alert on a server fault, when the fix is on the
// operator's side (populate api_base on the ProviderKey).

const CALLER_PLAINTEXT = "sk-invalid-config-400";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("invalid upstream config maps to 400 e2e", () => {
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

    // openrouter resolves via the openai adapter family. With no
    // api_base the family bridge refuses to fall back to
    // api.openai.com — a customer-fixable misconfig. (No api_base set.)
    const pk = await admin.createProviderKey({
      display_name: "invalid-config-pk",
      provider: "openrouter",
      adapter: "openai",
      secret: "sk-mock",
    });
    await admin.createModel({
      display_name: "invalid-config-model",
      provider: "openrouter",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["invalid-config-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("family-adapter PK without api_base surfaces as a 400, not a 500", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "invalid-config-model");
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "invalid-config-model",
        messages: [{ role: "user", content: "hi" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    // The load-bearing assertion: customer-fixable config is a 4xx.
    expect(caught.status).toBe(400);
    expect((caught.error as { type?: string } | undefined)?.type).toBe(
      "invalid_request_error",
    );
  });
});
