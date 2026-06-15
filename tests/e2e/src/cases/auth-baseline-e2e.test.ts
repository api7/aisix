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

// E2E: caller-authentication baseline. The four user journeys
// pinned here are the gateway's entry contract — every other
// e2e test assumes the caller is authenticated, but nothing
// previously verified what happens when the auth handshake fails.
//
// Pinned scenarios (all caller-side, all expecting an OpenAI-shape
// error envelope without the upstream ever being touched):
//
//   1. No `Authorization` header at all
//   2. `Authorization` header present but malformed (not `Bearer …`)
//   3. `Bearer <plaintext>` where the plaintext is not a registered
//      ApiKey hash
//   4. The admin API rejects a wrong admin bearer
//
// Reference:
// - OpenAI Chat Completions API spec
//   <https://platform.openai.com/docs/api-reference/chat/create>
// - OpenAI authentication doc
//   <https://platform.openai.com/docs/api-reference/authentication>
// - RFC 6750 §3 Bearer token error format
//   <https://datatracker.ietf.org/doc/html/rfc6750#section-3>

const VALID_PLAINTEXT = "sk-auth-baseline-valid";
const VALID_KEY_HASH = createHash("sha256")
  .update(VALID_PLAINTEXT)
  .digest("hex");
const UNKNOWN_PLAINTEXT = "sk-auth-baseline-unregistered";

describe("auth baseline e2e: missing/malformed/unknown bearer all fail closed", () => {
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

    // A single Model + ProviderKey + ApiKey configured. The valid
    // ApiKey is what makes the negative cases meaningful: failing
    // auth must happen even though a valid key exists; failing auth
    // must NOT degrade to "any-key-works".
    const pk = await admin.createProviderKey({
      display_name: "auth-baseline-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "auth-baseline",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: VALID_KEY_HASH,
      allowed_models: ["auth-baseline"],
    });

    // Confirm config has propagated by exercising the happy path
    // with the valid bearer once. If this never goes 200, the
    // negative tests below would race against propagation and
    // produce 401-from-no-snapshot rather than 401-from-bad-auth.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${VALID_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "auth-baseline",
          messages: [{ role: "user", content: "ready-probe" }],
        }),
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("missing Authorization header → 401, upstream untouched", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const upstreamHitsBefore = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "auth-baseline",
        messages: [{ role: "user", content: "no auth header" }],
      }),
    });

    expect(res.status).toBe(401);
    const body = (await res.json()) as { error?: { type?: unknown; message?: unknown } };
    // OpenAI error envelope: `error.type` is a non-empty string,
    // `error.message` is human-readable. Don't pin specific values
    // (the gateway's exact vocabulary is its own contract); just
    // verify the SDK-required shape is honoured.
    expect(typeof body.error?.type).toBe("string");
    expect((body.error?.type as string).length).toBeGreaterThan(0);
    expect(typeof body.error?.message).toBe("string");
    expect((body.error?.message as string).length).toBeGreaterThan(0);

    // Hard contract: an unauthenticated request must never reach
    // the upstream. A regression that authenticated post-dispatch
    // (or that 401'd only after burning an upstream call) would
    // inflate this counter.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });

  test("malformed Authorization (not 'Bearer …') → 401, upstream untouched", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const upstreamHitsBefore = upstream.receivedRequests.length;
    // Three malformed shapes a real client might emit by accident:
    // raw token (no scheme), wrong scheme (Basic), and Bearer with
    // empty token. Per RFC 6750, only `Bearer <token>` is valid for
    // OpenAI-compatible APIs.
    const malformed = [
      VALID_PLAINTEXT, // raw, no scheme
      `Basic ${Buffer.from(`${VALID_PLAINTEXT}:`).toString("base64")}`, // wrong scheme
      "Bearer ", // empty token
    ];

    for (const auth of malformed) {
      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: auth,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "auth-baseline",
          messages: [{ role: "user", content: "malformed auth" }],
        }),
      });
      expect(res.status, `auth=${JSON.stringify(auth)}`).toBe(401);
      await res.text();
    }

    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });

  test("Bearer with unregistered token → 401 (distinct from 403 disallowed-model)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const upstreamHitsBefore = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${UNKNOWN_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        // Targets a registered model — this rules out "401 because
        // we couldn't even resolve the model" and pins that the
        // 401 originates from the auth layer specifically.
        model: "auth-baseline",
        messages: [{ role: "user", content: "unknown caller key" }],
      }),
    });

    // 401 (not 403) is the correct status: 403 is reserved for
    // "authenticated but not authorized for this resource", which
    // is what allowed-models-e2e covers. A regression that 403'd
    // here would be visible to clients as "your key works but you
    // can't use this model" — misleading and would mask key-rotation
    // mistakes during incident response.
    expect(res.status).toBe(401);
    const body = (await res.json()) as {
      error?: { type?: unknown; message?: unknown };
    };
    expect(typeof body.error?.type).toBe("string");
    // Discriminate auth-layer 401 from a snapshot-not-loaded 401 or
    // a model-resolution 401. Same envelope-discrimination pattern
    // allowed-models-e2e uses. The error message must NOT read like
    // "model not found" — that would mean the test passed because
    // the gateway couldn't even resolve the model, not because the
    // auth layer rejected the unregistered token.
    expect(typeof body.error?.message).toBe("string");
    expect((body.error?.message as string).toLowerCase()).not.toContain(
      "not found",
    );

    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });

  test("admin API rejects wrong admin bearer → 401", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await fetch(`${app.adminUrl}/admin/v1/models`, {
      method: "GET",
      headers: { authorization: "Bearer not-the-admin-key" },
    });
    expect(res.status).toBe(401);
    await res.text();

    // No bearer at all — same outcome.
    const noAuthRes = await fetch(`${app.adminUrl}/admin/v1/models`, {
      method: "GET",
    });
    expect(noAuthRes.status).toBe(401);
    await noAuthRes.text();
  });
});
