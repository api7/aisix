import { createHash } from "node:crypto";
import { setTimeout as sleep } from "node:timers/promises";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockIdp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockIdp,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: inbound OIDC/JWT authentication (AISIX-Cloud#1080, #1081).
//
// The environment trusts a mock identity provider (`oidc_providers`
// row); API keys carry `jwt_subject` bindings. Pinned journeys:
//
//   1. A valid agent JWT authenticates and runs as its bound key —
//      the key's allowed_models and rate_limit apply unchanged.
//   2. Expired / tampered / wrong-audience / cross-issuer / exp-less
//      tokens are all rejected before the upstream is touched, with
//      stable `error.code` values.
//   3. required_scopes and bound_claims reject entitlement gaps with
//      403, distinct from the 401 credential failures.
//   4. A valid token whose identity has no bound key is rejected
//      (`jwt_identity_unmapped`) — never an anonymous pass.
//   5. IdP key rotation is picked up without a gateway restart.
//   6. Deleting the trust provider fails closed.
//   7. `/v1/messages` renders the Anthropic error envelope.
//   8. Auth decisions surface on `aisix_auth_decisions_total`.
//   9. Without any trust provider, a JWT-shaped bearer falls through
//      to the API-key path (and dotted custom keys keep working even
//      with a provider configured).
//
// References:
// - RFC 7519 (JWT) §4.1 registered claims
//   <https://datatracker.ietf.org/doc/html/rfc7519#section-4.1>
// - RFC 7517 (JWK) <https://datatracker.ietf.org/doc/html/rfc7517>
// - OIDC Discovery 1.0 §4 <https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderConfig>

const MODEL = "jwt-auth-model";
const OTHER_MODEL = "jwt-auth-other-model";

function chatBody(model = MODEL): string {
  return JSON.stringify({
    model,
    messages: [{ role: "user", content: "jwt auth probe" }],
  });
}

async function chat(
  app: SpawnedApp,
  token: string,
  model = MODEL,
): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: chatBody(model),
  });
}

async function errorCode(res: Response): Promise<string | undefined> {
  const body = (await res.json()) as { error?: { code?: string } };
  return body.error?.code;
}

describe("jwt auth e2e: OIDC trust providers + jwt_subject key binding", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let idp: MockIdp | undefined;
  let idp2: MockIdp | undefined;
  let seed: SeedClient | undefined;
  let providerId: string | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    idp = await startMockIdp();
    idp2 = await startMockIdp();
    app = await spawnApp({});
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "jwt-auth-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    for (const name of [MODEL, OTHER_MODEL]) {
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }

    // The trusted provider, pinned to a direct jwks_uri. Requires the
    // `ai.access` scope and a department bound claim.
    const provider = await seed.createOidcProvider({
      name: "mock-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
      required_scopes: ["ai.access"],
      bound_claims: { department: "ai-lab" },
    });
    providerId = provider.id;

    // A second provider resolved via OIDC discovery (no jwks_uri).
    await seed.createOidcProvider({
      name: "mock-idp-discovery",
      issuer: idp2.url,
      audiences: ["aisix-gateway"],
    });

    // agent-1 → a key allowed MODEL only, rpm-limited to 2 so the
    // rate-limit inheritance is observable. Bound to the mock-idp
    // provider: the pair (jwt_provider, jwt_subject) is what a token
    // resolves against.
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-jwt-agent-1").digest("hex"),
      allowed_models: [MODEL],
      jwt_subject: "agent-1",
      jwt_provider: "mock-idp",
      rate_limit: { rpm: 2 },
    });
    // agent-2 under mock-idp, unrestricted models.
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-jwt-agent-2").digest("hex"),
      allowed_models: ["*"],
      jwt_subject: "agent-2",
      jwt_provider: "mock-idp",
    });
    // A DISTINCT key with the SAME subject "agent-2" but bound to the
    // discovery provider — proves subjects are namespaced by provider
    // (audit H1): a token from mock-idp-discovery resolves here, not to
    // the mock-idp agent-2 key above.
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-jwt-agent-2-disc").digest("hex"),
      allowed_models: ["*"],
      jwt_subject: "agent-2",
      jwt_provider: "mock-idp-discovery",
    });
    // agent-disabled → a disabled key under mock-idp.
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-jwt-agent-3").digest("hex"),
      allowed_models: ["*"],
      jwt_subject: "agent-disabled",
      jwt_provider: "mock-idp",
      disabled: true,
    });

    await waitConfigPropagation(async () => {
      const res = await chat(app!, idp!.sign(validClaims()));
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await idp?.close();
    await idp2?.close();
  });

  function validClaims(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return agentClaims(idp!.url, {
      scope: "openid ai.access",
      department: "ai-lab",
      ...overrides,
    });
  }

  function skipUnlessUp(ctx: { skip: () => void }): boolean {
    if (!etcdReachable || !app || !upstream || !idp || !idp2) {
      ctx.skip();
      return true;
    }
    return false;
  }

  test("valid agent JWT authenticates and runs as its bound key", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(app!, idp!.sign(validClaims()));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { choices?: unknown[] };
    expect(Array.isArray(body.choices)).toBe(true);
  });

  test("the bound key's allowed_models applies: other model → 403", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(app!, idp!.sign(validClaims()), OTHER_MODEL);
    expect(res.status).toBe(403);
    await res.text();
  });

  test("the bound key's rate_limit applies: burst past rpm → 429", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // rpm=2 on agent-1's key. Earlier tests in this file consumed the
    // window unpredictably, so just burst until a 429 shows up — the
    // contract is that JWT-authenticated traffic hits the bound key's
    // limiter at all (an unbound identity would never 429).
    let saw429 = false;
    for (let i = 0; i < 6 && !saw429; i++) {
      const res = await chat(app!, idp!.sign(validClaims()));
      saw429 = res.status === 429;
      await res.text();
    }
    expect(saw429).toBe(true);
  });

  test("expired token → 401 jwt_expired, upstream untouched", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const hits = upstream!.receivedRequests.length;
    const res = await chat(
      app!,
      idp!.sign(validClaims({ exp: Math.floor(Date.now() / 1000) - 3600 })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_expired");
    expect(upstream!.receivedRequests.length).toBe(hits);
  });

  test("token without exp → 401 (default deny), upstream untouched", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const hits = upstream!.receivedRequests.length;
    const claims = validClaims();
    delete claims.exp;
    const res = await chat(app!, idp!.sign(claims));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
    expect(upstream!.receivedRequests.length).toBe(hits);
  });

  test("tampered payload → 401 jwt_invalid, upstream untouched", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const hits = upstream!.receivedRequests.length;
    const token = idp!.sign(validClaims());
    const parts = token.split(".");
    parts[1] = Buffer.from(
      JSON.stringify(validClaims({ sub: "agent-2" })),
    ).toString("base64url");
    const res = await chat(app!, parts.join("."));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
    expect(upstream!.receivedRequests.length).toBe(hits);
  });

  test("wrong audience → 401 jwt_invalid", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(app!, idp!.sign(validClaims({ aud: "someone-else" })));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("token from one issuer signed by another's key → 401", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // idp2 signs a token claiming idp1 as its issuer: the row matched
    // is idp1's, whose JWKS cannot verify idp2's signature. Trust must
    // never cross providers.
    const res = await chat(app!, idp2!.sign(validClaims()));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("unknown issuer → 401 jwt_invalid (issuer allow-list)", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(
      app!,
      idp!.sign(validClaims({ iss: "https://unregistered.example.com" })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("missing required scope → 403 jwt_claims_rejected", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(app!, idp!.sign(validClaims({ scope: "openid" })));
    expect(res.status).toBe(403);
    expect(await errorCode(res)).toBe("jwt_claims_rejected");
  });

  test("bound claim mismatch or absence → 403 jwt_claims_rejected", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const wrong = await chat(app!, idp!.sign(validClaims({ department: "finance" })));
    expect(wrong.status).toBe(403);
    expect(await errorCode(wrong)).toBe("jwt_claims_rejected");

    const claims = validClaims();
    delete claims.department;
    const missing = await chat(app!, idp!.sign(claims));
    expect(missing.status).toBe(403);
    expect(await errorCode(missing)).toBe("jwt_claims_rejected");
  });

  test("valid token with unbound identity → 401 jwt_identity_unmapped", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(app!, idp!.sign(validClaims({ sub: "agent-nobody" })));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_identity_unmapped");
  });

  test("identity bound to a disabled key → 401 api_key_disabled", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(app!, idp!.sign(validClaims({ sub: "agent-disabled" })));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("api_key_disabled");
  });

  test("discovery-resolved provider authenticates (no jwks_uri configured)", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Signed by idp2, issuer idp2 → resolves the mock-idp-discovery
    // provider (no jwks_uri, endpoint found via OIDC discovery) → maps
    // to the agent-2 key bound to that provider.
    const res = await chat(
      app!,
      idp2!.sign(agentClaims(idp2!.url, { sub: "agent-2" })),
    );
    expect(res.status).toBe(200);
    await res.text();
  });

  test("cross-provider impersonation is blocked: same subject, other provider → 401", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // A token from the discovery IdP asserting sub "agent-1" — a subject
    // bound ONLY to the mock-idp provider. Even though the token is
    // genuinely signed by a trusted provider, the identity must not
    // resolve to the mock-idp agent-1 key (audit H1). agent-1 has no
    // binding under mock-idp-discovery, so it is unmapped.
    const res = await chat(
      app!,
      idp2!.sign(agentClaims(idp2!.url, { sub: "agent-1" })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_identity_unmapped");
  });

  test("no fallback: a JWT-shaped bearer that is also a valid API key is not retried as a key", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Seed a key whose PLAINTEXT is a real JWT from the trusted IdP but
    // whose claims fail (wrong audience). Presented as a bearer it would
    // authenticate on the key path — but the JWT path owns it and its
    // failure is final, with no fall-through to the key lookup.
    const plaintext = idp!.sign(validClaims({ aud: "someone-else" }));
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(plaintext).digest("hex"),
      allowed_models: [MODEL],
    });
    // Give the key time to propagate, then confirm it is still rejected
    // as a JWT (never accepted as the key it also is).
    await sleep(2500);
    const res = await chat(app!, plaintext);
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("algorithm confusion: an HS256 token is rejected (asymmetric-only allowlist)", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Forge the JOSE header alg to HS256 — the classic attempt to make
    // the public JWKS modulus act as an HMAC secret. The signature is
    // still the RS256 one, but the alg is outside ALLOWED_ALGS, so it is
    // rejected before any key is tried.
    const res = await chat(
      app!,
      idp!.sign(validClaims(), { header: { alg: "HS256" } }),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("a token with no kid verifies against a single-key JWKS", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Omitting kid is legal; against a one-key set the sole signature
    // key is the candidate. This exercises the no-kid fall-through path.
    const res = await chat(app!, idp!.sign(validClaims(), { omitKid: true }));
    expect(res.status).toBe(200);
    await res.text();
  });

  test("the JWKS is cached: a burst of requests does not refetch per request", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Warm the cache, record the fetch count, then burst. One inbound
    // JWT must never be one outbound JWKS fetch (audit H2) — the count
    // must not grow by the number of requests.
    await waitFor200(() => chat(app!, idp!.sign(validClaims({ sub: "agent-2" }))));
    const before = idp!.jwksFetches;
    for (let i = 0; i < 8; i++) {
      const res = await chat(app!, idp!.sign(validClaims({ sub: "agent-2" })));
      await res.text();
    }
    // Allow a small slack for a TTL-boundary refresh, but nothing near
    // one-per-request.
    expect(idp!.jwksFetches - before).toBeLessThanOrEqual(1);
  });

  test("IdP key rotation is picked up without a restart", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Ensure the current JWKS is cached, then rotate.
    const before = await chat(app!, idp!.sign(validClaims()));
    // May be 200 or 429 (rate-limit test shares the key) — either way
    // the JWKS is warm now.
    await before.text();
    const oldKid = idp!.currentKid;
    idp!.rotate();

    // Sit out the unknown-kid refresh rate limit, then a token under
    // the new kid must verify — the gateway refetches the JWKS on the
    // unknown kid instead of waiting for TTL expiry or a restart.
    await sleep(1100);
    const rotated = await waitFor200(() =>
      chat(app!, idp!.sign(agentClaims(idp!.url, {
        sub: "agent-2",
        scope: "ai.access",
        department: "ai-lab",
      }))),
    );
    expect(rotated).toBe(true);

    // A token still signed by the retired key no longer verifies.
    await sleep(1100);
    const stale = await chat(
      app!,
      idp!.sign(validClaims({ sub: "agent-2" }), { signWithKid: oldKid }),
    );
    expect(stale.status).toBe(401);
    await stale.text();
  });

  test("/v1/messages renders the Anthropic error envelope on a JWT failure", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${idp!.sign(
          validClaims({ exp: Math.floor(Date.now() / 1000) - 60 }),
        )}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 16,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(401);
    const body = (await res.json()) as {
      type?: string;
      error?: { type?: string };
    };
    expect(body.type).toBe("error");
    expect(body.error?.type).toBe("authentication_error");
  });

  test("auth decisions surface on aisix_auth_decisions_total", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("aisix_auth_decisions_total");
    expect(text).toMatch(/aisix_auth_decisions_total\{[^}]*method="jwt"[^}]*result="allowed"[^}]*\}/);
    expect(text).toMatch(/aisix_auth_decisions_total\{[^}]*reason="jwt_expired"[^}]*\}/);
  });

  test("a custom dotted API key still authenticates while JWT auth is on", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Dots alone must not route a bearer down the JWT path — only a
    // real JOSE header does. A custom-imported key shaped `a.b.c`
    // keeps working even with trust providers configured.
    const plaintext = "my.imported.key";
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(plaintext).digest("hex"),
      allowed_models: [MODEL],
    });
    await waitConfigPropagation(async () => {
      const res = await chat(app!, plaintext);
      await res.text();
      return res.status === 200;
    });
  });

  test("deleting the trust provider fails closed", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    await seed!.delete("oidc_providers", providerId!);
    // Poll until the deletion propagates: the same valid token flips
    // from authenticating to rejected (its issuer is no longer
    // trusted; the discovery provider still is, so JWT auth stays on
    // and this is the allow-list shrinking, not the feature turning
    // off).
    await waitConfigPropagation(async () => {
      const res = await chat(app!, idp!.sign(validClaims({ sub: "agent-2" })));
      await res.text();
      return res.status === 401;
    });
  });
});

async function waitFor200(request: () => Promise<Response>): Promise<boolean> {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const res = await request();
    await res.text();
    if (res.status === 200) return true;
    await sleep(300);
  }
  return false;
}

describe("jwt auth e2e: no trust provider configured", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let idp: MockIdp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    idp = await startMockIdp();
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "jwt-off-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-plain").digest("hex"),
      allowed_models: [MODEL],
    });
    await waitConfigPropagation(async () => {
      const res = await chat(app!, "sk-plain");
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await idp?.close();
  });

  test("a JWT-shaped bearer falls through to the key path and 401s without any JWKS fetch", async (ctx) => {
    if (!etcdReachable || !app || !idp) {
      ctx.skip();
      return;
    }

    const res = await chat(app!, idp!.sign(agentClaims(idp!.url)));
    expect(res.status).toBe(401);
    await res.text();
    // The token never entered the JWT path: the mock IdP was never
    // consulted. This pins "JWT auth is off unless a provider row
    // exists" — no accidental outbound fetches on unmanaged
    // deployments.
    expect(idp!.jwksFetches).toBe(0);
  });
});
