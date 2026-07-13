import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: API key lifecycle enforcement (#933). Pins the contract that a
// key which EXISTS in the snapshot is still rejected when it is
// expired or administratively disabled — distinct from the
// unknown-token 401 covered by auth-baseline-e2e.
//
// Pinned scenarios:
//
//   1. `expires_at` in the past → 401 `api_key_expired`, upstream untouched
//   2. `expires_at` in the future → authenticates normally
//   3. `disabled: true` → 401 `api_key_disabled`, upstream untouched
//   4. live transition: disable an in-use key → 401 after propagation,
//      re-enable → 200 again (key preserved, no re-issue)
//   5. a key crosses its expiry deadline while the gateway is running →
//      401 without any config change (expiry is evaluated per request,
//      not at load time)
//   6. the Anthropic surface (`/v1/messages`) rejects a disabled key
//      with the Anthropic-shaped 401 envelope
//   7. rotate: the old plaintext stops working, the returned new
//      plaintext works, and the resource id is unchanged
//
// Reference: OpenAI authentication doc
// <https://platform.openai.com/docs/api-reference/authentication>,
// RFC 6750 §3 <https://datatracker.ietf.org/doc/html/rfc6750#section-3>.

const MODEL = "apikey-lifecycle";

function sha256(plaintext: string): string {
  return createHash("sha256").update(plaintext).digest("hex");
}

describe("api key lifecycle e2e: expired/disabled keys fail closed, rotate swaps the secret", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "apikey-lifecycle-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  function chat(plaintext: string, content: string): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${plaintext}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "user", content }],
      }),
    });
  }

  /** Seed a key and wait until the gateway's snapshot can see it.
   * Lifecycle-rejected keys never return 200, so propagation is
   * confirmed by the response being the expected lifecycle 401
   * (`error.code`) rather than the unknown-token 401 (no code). */
  async function seedKey(
    plaintext: string,
    extra: Record<string, unknown>,
    propagated: (res: Response) => Promise<boolean>,
  ): Promise<{ id: string }> {
    const created = await seed!.createApiKey({
      key_hash: sha256(plaintext),
      allowed_models: [MODEL],
      ...extra,
    });
    await waitConfigPropagation(async () => propagated(await chat(plaintext, "probe")));
    return { id: created.id };
  }

  const isLifecycle401 = (code: string) => async (res: Response) => {
    const body = (await res.json()) as { error?: { code?: unknown } };
    return res.status === 401 && body.error?.code === code;
  };

  const is200 = async (res: Response) => {
    await res.text();
    return res.status === 200;
  };

  test("expired key → 401 api_key_expired, upstream untouched", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-expired";
    await seedKey(
      plaintext,
      { expires_at: "2020-01-01T00:00:00Z" },
      isLifecycle401("api_key_expired"),
    );

    const upstreamHitsBefore = upstream.receivedRequests.length;
    const res = await chat(plaintext, "expired key");
    expect(res.status).toBe(401);
    const body = (await res.json()) as {
      error?: { type?: unknown; code?: unknown; message?: unknown };
    };
    expect(body.error?.type).toBe("invalid_api_key");
    expect(body.error?.code).toBe("api_key_expired");
    // The message names the reason: the caller holds the secret
    // already, so telling them it expired (vs a generic invalid-key
    // message) leaks nothing and is actionable.
    expect(String(body.error?.message).toLowerCase()).toContain("expired");
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });

  test("future expires_at authenticates normally", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-future";
    await seedKey(plaintext, { expires_at: "2099-01-01T00:00:00Z" }, is200);

    const res = await chat(plaintext, "future expiry");
    expect(res.status).toBe(200);
    await res.text();
  });

  test("disabled key → 401 api_key_disabled, upstream untouched", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-disabled";
    await seedKey(plaintext, { disabled: true }, isLifecycle401("api_key_disabled"));

    const upstreamHitsBefore = upstream.receivedRequests.length;
    const res = await chat(plaintext, "disabled key");
    expect(res.status).toBe(401);
    const body = (await res.json()) as {
      error?: { type?: unknown; code?: unknown };
    };
    expect(body.error?.type).toBe("invalid_api_key");
    expect(body.error?.code).toBe("api_key_disabled");
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });

  test("disable then re-enable an in-use key without re-issuing it", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-toggle";
    const keyHash = sha256(plaintext);
    const { id } = await seedKey(plaintext, {}, is200);

    // Disable: same key_hash, flipped flag — the credential the
    // caller holds is unchanged, only its status moves.
    await seed.update("api_keys", id, {
      key_hash: keyHash,
      allowed_models: [MODEL],
      disabled: true,
    });
    await waitConfigPropagation(async () =>
      isLifecycle401("api_key_disabled")(await chat(plaintext, "post-disable probe")),
    );

    // Re-enable: the original plaintext works again — proving disable
    // is reversible and did not rotate or delete the credential.
    await seed.update("api_keys", id, {
      key_hash: keyHash,
      allowed_models: [MODEL],
    });
    await waitConfigPropagation(async () => is200(await chat(plaintext, "post-enable probe")));
  });

  test("key crossing its expiry deadline at runtime starts failing without any config change", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-deadline";
    // Expires well after propagation completes (seedKey asserts a 200
    // first), but soon enough for the test to observe the flip. The
    // runway must outlast worst-case propagation on shared-etcd CI
    // runners — if propagation eats it anyway, the "no config change"
    // premise is void, so skip rather than fail unrecoverably.
    const runwayMs = 15_000;
    const expiresAt = new Date(Date.now() + runwayMs).toISOString();
    await seedKey(plaintext, { expires_at: expiresAt }, is200);
    if (Date.now() >= Date.parse(expiresAt)) {
      ctx.skip();
      return;
    }

    // No admin writes from here on: the ONLY thing that changes is the
    // wall clock. Poll until the gateway starts rejecting.
    await waitConfigPropagation(
      async () => isLifecycle401("api_key_expired")(await chat(plaintext, "deadline probe")),
      runwayMs + 15_000,
    );
  }, 60_000);

  test("Anthropic surface rejects a disabled key with the Anthropic envelope", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-disabled-anthropic";
    await seedKey(plaintext, { disabled: true }, isLifecycle401("api_key_disabled"));

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "x-api-key": plaintext,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 16,
        messages: [{ role: "user", content: "disabled key via anthropic" }],
      }),
    });

    expect(res.status).toBe(401);
    const body = (await res.json()) as {
      type?: unknown;
      error?: { type?: unknown; message?: unknown };
    };
    // Anthropic error envelope: top-level discriminator + canonical
    // per-status error type, per
    // <https://docs.anthropic.com/en/api/errors>.
    expect(body.type).toBe("error");
    expect(body.error?.type).toBe("authentication_error");
  });

  test("rotate swaps the secret in place: old plaintext dies, new one works, id unchanged", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-rotate";
    const { id } = await seedKey(plaintext, {}, is200);

    // Rotate through the canonical `api_keys` route; the disabled-key
    // rotation below keeps the former `apikeys` spelling so both stay
    // exercised end-to-end.
    const rotated = await admin.json<{
      entry: { id: string };
      plaintext: string;
    }>("POST", `/admin/v1/api_keys/${id}/rotate`);
    expect(rotated.entry.id).toBe(id);
    expect(rotated.plaintext).not.toBe(plaintext);

    // New secret authenticates…
    await waitConfigPropagation(async () => is200(await chat(rotated.plaintext, "rotated probe")));
    // …and the old one is invalid (unknown-token 401), so a leaked
    // pre-rotation secret is worthless immediately after rotation.
    const res = await chat(plaintext, "old secret after rotate");
    expect(res.status).toBe(401);
    await res.text();
  });

  test("rotate preserves lifecycle fields: a disabled key's new secret is still disabled", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const plaintext = "sk-lifecycle-rotate-disabled";
    const expiresAt = "2099-01-01T00:00:00Z";
    const { id } = await seedKey(
      plaintext,
      { disabled: true, expires_at: expiresAt },
      isLifecycle401("api_key_disabled"),
    );

    const rotated = await admin!.json<{
      entry: { id: string; value: { disabled?: boolean; expires_at?: string } };
      plaintext: string;
    }>("POST", `/admin/v1/apikeys/${id}/rotate`);

    // Rotation swaps ONLY the secret — a regression that rebuilt the
    // entry from a partial body would silently re-enable a disabled
    // key under its fresh plaintext.
    expect(rotated.entry.value.disabled).toBe(true);
    expect(Date.parse(rotated.entry.value.expires_at ?? "")).toBe(Date.parse(expiresAt));
    await waitConfigPropagation(async () =>
      isLifecycle401("api_key_disabled")(await chat(rotated.plaintext, "rotated but disabled")),
    );
  });
});
