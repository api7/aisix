import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: ApiKey.allowed_models enforcement. The caller's key permits
// only `am-allowed`; calling `am-forbidden` (a model that exists in
// the snapshot but is NOT in the key's allowed list) must return 403
// without ever touching the upstream. The unit-level
// `forbidden_model_returns_403` covers the in-process path; this
// case proves the wire contract holds with a real binary, etcd
// watch, and OpenAI SDK surfacing the 403 as APIError.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create); the
// ApiKey schema lives at `crates/aisix-core/src/models/apikey.rs`.

const CALLER_PLAINTEXT = "sk-am-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("allowed_models e2e: 403 on disallowed model, upstream untouched", () => {
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

    // Single ProviderKey + two Models. They share the upstream so
    // upstream.receivedRequests is a single source of truth — if the
    // forbidden call leaks through, it will register here.
    const pk = await admin.createProviderKey({
      display_name: "am-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "am-allowed",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createModel({
      display_name: "am-forbidden",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Caller permitted ONLY for am-allowed. am-forbidden exists in
    // the snapshot — the 403 must come from authz, not from "model
    // not found".
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["am-allowed"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("allowed model passes; disallowed model gets 403 and upstream stays untouched on the blocked call", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Allowed call doubles as readiness probe. A 200 means the
    // Model + ProviderKey + ApiKey have all propagated.
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "am-allowed",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const upstreamHitsBeforeBlock = upstream.receivedRequests.length;

    // Disallowed model: SDK surfaces 403 as APIError.
    await expect(
      client.chat.completions.create({
        model: "am-forbidden",
        messages: [{ role: "user", content: "should be blocked" }],
      }),
    ).rejects.toBeInstanceOf(APIError);
    await expect(
      client.chat.completions.create({
        model: "am-forbidden",
        messages: [{ role: "user", content: "should be blocked" }],
      }),
    ).rejects.toMatchObject({ status: 403 });

    // The forbidden call never reaches the upstream. A regression
    // that authorized post-dispatch (or skipped the allowed_models
    // check) would have inflated the count.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBeforeBlock);
  });
});
