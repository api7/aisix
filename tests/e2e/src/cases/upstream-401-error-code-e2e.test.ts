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

// E2E: upstream 401 bad-key error envelope must preserve `error.code`
// even when the JSON body is labelled with a non-`application/json`
// Content-Type (#543).
//
// Pre-fix, the DP gated upstream error-body parsing on Content-Type.
// OpenAI's 401 `invalid_api_key` path (and edge/proxy layers fronting
// some upstreams) return the JSON error body with a non-JSON
// Content-Type, so the DP skipped the parse — dumping the raw body
// into `message` and emitting an EMPTY `error.code`. Customer SDKs
// that branch on `error.code === "invalid_api_key"` to decide whether
// to refresh credentials couldn't classify the error.
//
// This drives a real /v1/chat/completions through the DP against a
// mock upstream that returns the canonical OpenAI 401 body with a
// `text/plain` Content-Type, and asserts the client-visible envelope
// still carries `code: "invalid_api_key"`.
//
// References:
// - OpenAI error envelope: https://platform.openai.com/docs/guides/error-codes/api-errors
// - Issue: api7/AISIX-Cloud#543

const CALLER_PLAINTEXT = "sk-401-code-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const OPENAI_401_BODY = {
  error: {
    message:
      "Incorrect API key provided: sk-inval***c66a. You can find your API key at https://platform.openai.com/account/api-keys.",
    type: "invalid_request_error",
    code: "invalid_api_key",
    param: null,
  },
};

describe("upstream 401 error.code preserved across Content-Type (#543)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // Upstream returns the canonical OpenAI 401 JSON body but labels it
    // `text/plain` — the exact shape #543 was surfaced on.
    upstream = await startOpenAiUpstream({
      status: 401,
      errorBody: OPENAI_401_BODY,
      errorContentType: "text/plain",
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "err401-pk",
      secret: "sk-mock-bad",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "err401-gpt",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["err401-gpt"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("401 with non-JSON content-type still surfaces error.code (#543)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Propagation probe: the model must be live (any non-5xx-from-DP
    // response means config propagated; the upstream always 401s).
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${CALLER_PLAINTEXT}`,
          },
          body: JSON.stringify({
            model: "err401-gpt",
            messages: [{ role: "user", content: "probe" }],
          }),
        });
        // 401 propagated correctly is what we want; 404 = model not yet live.
        return r.status === 401;
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "err401-gpt",
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    // DP forwards the upstream 401 status.
    expect(res.status).toBe(401);

    const body = (await res.json()) as {
      error?: { message?: string; type?: string; code?: string };
    };
    expect(body.error, JSON.stringify(body)).toBeDefined();
    // The fix: `code` is preserved (pre-#543 it was empty/absent).
    expect(body.error!.code).toBe("invalid_api_key");
    // `message` is the clean upstream message, NOT the raw
    // JSON-stringified body.
    expect(body.error!.message).toContain("Incorrect API key provided");
    expect(body.error!.message).not.toContain('"error"');
  });
});
