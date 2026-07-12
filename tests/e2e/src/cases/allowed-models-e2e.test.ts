import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: ApiKey.allowed_models enforcement. The caller's key permits
// only `am-allowed`; calling `am-forbidden` (a model that exists in
// the snapshot but is NOT in the key's allowed list) must return 403
// without ever touching the upstream. The OpenAI SDK surfaces the
// 403 as APIError.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) and
// OpenAI error-envelope spec
// (https://platform.openai.com/docs/guides/error-codes/api-errors).

const CALLER_PLAINTEXT = "sk-am-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("allowed_models e2e: 403 on disallowed model, upstream untouched", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Single ProviderKey + two Models. They share the upstream so
    // upstream.receivedRequests is a single source of truth — if the
    // forbidden call leaks through, it will register here.
    const pk = await seed.createProviderKey({
      display_name: "am-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "am-allowed",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "am-forbidden",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Caller permitted ONLY for am-allowed. am-forbidden exists in
    // the snapshot — the 403 must come from authz, not from "model
    // not found".
    await seed.createApiKey({
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

    // Disallowed model: SDK surfaces 403 as APIError. Status alone
    // is not enough — a regression that 403'd via a different path
    // (e.g. the caller key looked malformed) would still match
    // `status: 403` but reach a confusing user-visible error. Pin
    // the OpenAI-shape error envelope so the caller's error handling
    // can distinguish authz-block from "no such model".
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "am-forbidden",
        messages: [{ role: "user", content: "should be blocked" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(403);
    // OpenAI error-envelope spec: `error.type` is a non-empty string
    // identifying the error class. The exact value is the gateway's
    // public error vocabulary; this assertion guards against an empty
    // or missing field that would break clients doing
    // type-discrimination on the error.
    expect(caught.error).toBeDefined();
    const errType = (caught.error as { type?: unknown } | undefined)?.type;
    expect(typeof errType).toBe("string");
    expect((errType as string).length).toBeGreaterThan(0);
    // The error message must indicate this is an *authorization*
    // failure — not "model not found" (the model `am-forbidden`
    // does exist in the snapshot; the caller just isn't allowed).
    const errMsg = (caught.error as { message?: unknown } | undefined)
      ?.message;
    expect(typeof errMsg).toBe("string");
    expect((errMsg as string).toLowerCase()).not.toContain("not found");

    // The forbidden call never reaches the upstream. A regression
    // that authorized post-dispatch (or skipped the allowed_models
    // check) would have inflated the count.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBeforeBlock);
  });
});
