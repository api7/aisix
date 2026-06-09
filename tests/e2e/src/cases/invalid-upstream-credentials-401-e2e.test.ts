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

// E2E: a customer-fixable upstream *credential* error must surface to
// the client as a 401 authentication_error, not a 400 or 500
// (#367 follow-up). This is the credential sibling of
// invalid-upstream-config-400-e2e: there the misconfig is request/
// routing shape (no api_base) → 400; here the ProviderKey has a valid
// api_base but a secret that can't be a valid Authorization header
// value (a newline-injected key), so the bridge can't build the auth
// header and errors before dispatch. Auth-material problems map to 401
// (auth-material, not a request-shape 400), so SDKs that branch
// on 401 to refresh credentials classify it right. (An empty secret is
// the same error class but is rejected earlier by the admin schema's
// min-length check, so this drives the post-admit header-bytes guard.)

const CALLER_PLAINTEXT = "sk-invalid-cred-401";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("invalid upstream credentials maps to 401 e2e", () => {
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

    // Valid api_base (routing shape is fine) but a secret that can't be
    // an Authorization header value (embedded newline) — the openai
    // bridge's api_key() guard rejects this before dispatch as a
    // credential problem.
    const pk = await admin.createProviderKey({
      display_name: "invalid-cred-pk",
      secret: "sk-live\n-injected",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "invalid-cred-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["invalid-cred-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("PK with an unusable secret surfaces as a 401 authentication_error", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "invalid-cred-model");
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "invalid-cred-model",
        messages: [{ role: "user", content: "hi" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    // The load-bearing assertion: an unusable credential is a 401
    // authentication_error, not a 400 or 500.
    expect(caught.status).toBe(401);
    expect((caught.error as { type?: string } | undefined)?.type).toBe(
      "authentication_error",
    );
  });
});
