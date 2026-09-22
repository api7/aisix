import {
  createHash,
  createPublicKey,
  randomUUID,
  type JsonWebKey as CryptoJsonWebKey,
} from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  ProxyClient,
  SeedClient,
  signHs,
  spawnApp,
  startMockIdp,
  startOpenAiUpstream,
  startRestUpstream,
  waitConfigPropagation,
  type MockIdp,
  type OpenAiUpstream,
  type RestUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: shared-secret (HMAC) inbound JWT authentication.
//
// A trust provider that carries `hmac_secret` verifies HS256/384/512
// tokens against that secret instead of fetching a JWKS. Tokens are
// minted out of band — there is no identity provider to stand up — so
// `issuer` and `audiences` become optional on the provider, and a
// provider that declares no issuer is reached by trial rather than by
// the token's `iss`.
//
// Pinned journeys:
//
//   1. An issuer-less shared-secret provider accepts a token carrying
//      neither `iss` nor `aud`, and binds it to an API key through the
//      identity claim.
//   2. A shared-secret provider that DOES pin `issuer` / `audiences`
//      enforces both, exactly as a JWKS provider does.
//   3. The wrong secret, and an expired token, are both rejected.
//   4. Algorithm confusion is refused in both directions: an RS256
//      token never reaches a shared-secret provider, and an HS token —
//      including one signed with a JWKS provider's own published public
//      key — never reaches a JWKS provider.
//   5. `iss` binds: a token naming a JWKS provider is refused there and
//      is NOT retried against a shared-secret provider that would have
//      accepted it.
//   6. Trial order: with two issuer-less providers, a token that
//      verifies only against the second is accepted.
//   7. A provider row that could not verify anything is rejected at
//      load, and the valid rows keep serving.
//   8. The secret reaches neither `/status/config` nor the process log.
//   9. The same verification governs a non-`/v1` surface (a passthrough
//      route), because authentication is one shared path.
//
// References:
// - RFC 7515 (JWS) §3 <https://datatracker.ietf.org/doc/html/rfc7515>
// - RFC 7518 (JWA) §3.2, HMAC key length
//   <https://datatracker.ietf.org/doc/html/rfc7518#section-3.2>

const MODEL = "hmac-jwt-model";
const ROUTE_PREFIX = "/hmac-passthrough";

// Distinct 32+ byte secrets. `SECRET_UNKNOWN` is configured on no
// provider — it is the "wrong secret" a forged token is signed with.
const SECRET_FIRST = "first-provider-shared-secret-000001";
const SECRET_SECOND = "second-provider-shared-secret-00002";
const SECRET_PINNED = "pinned-provider-shared-secret-00003";
const SECRET_UNKNOWN = "not-configured-anywhere-secret-00004";

const PINNED_ISSUER = "https://pinned.hmac.test";
const PINNED_AUDIENCE = "aisix-pinned";

function hmacClaims(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    sub: "agent-first",
    exp: Math.floor(Date.now() / 1000) + 3600,
    ...extra,
  };
}

async function chat(app: SpawnedApp, token: string): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "hmac jwt probe" }],
    }),
  });
}

async function errorCode(res: Response): Promise<string | undefined> {
  const body = (await res.json()) as { error?: { code?: string } };
  return body.error?.code;
}

/**
 * The mock identity provider's published signing key, rendered as the
 * SPKI PEM an algorithm-confusion attack would use as an HMAC secret:
 * the verifier is asked to treat public material as a shared key.
 */
async function publishedPublicKeyPem(idp: MockIdp): Promise<string> {
  const jwks = (await (await fetch(idp.jwksUrl)).json()) as {
    keys: CryptoJsonWebKey[];
  };
  return createPublicKey({ key: jwks.keys[0], format: "jwk" })
    .export({ type: "spki", format: "pem" })
    .toString();
}

describe("jwt auth e2e: shared-secret (HMAC) trust providers", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let rest: RestUpstream | undefined;
  let idp: MockIdp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let jwksPublicKeyPem = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    rest = await startRestUpstream();
    idp = await startMockIdp();
    jwksPublicKeyPem = await publishedPublicKeyPem(idp);
    app = await spawnApp({});
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "hmac-jwt-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    // Two issuer-less shared-secret providers. Trial order is by name,
    // so `hmac-first` is tried before `hmac-second`.
    await seed.createOidcProvider({
      name: "hmac-first",
      hmac_secret: SECRET_FIRST,
    });
    await seed.createOidcProvider({
      name: "hmac-second",
      hmac_secret: SECRET_SECOND,
    });
    // A shared-secret provider that DOES pin both claims.
    await seed.createOidcProvider({
      name: "hmac-pinned",
      issuer: PINNED_ISSUER,
      audiences: [PINNED_AUDIENCE],
      hmac_secret: SECRET_PINNED,
    });
    // A JWKS provider, so both modes coexist in one environment.
    await seed.createOidcProvider({
      name: "jwks-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });

    // One key per provider, all naming the same subject: the binding is
    // `(jwt_provider, jwt_subject)`, so which provider verified the
    // token decides which key the request runs as.
    for (const [provider, subject] of [
      ["hmac-first", "agent-first"],
      ["hmac-second", "agent-second"],
      ["hmac-pinned", "agent-pinned"],
      ["jwks-idp", "agent-jwks"],
    ]) {
      await seed.createApiKey({
        key_hash: createHash("sha256").update(`sk-${provider}`).digest("hex"),
        allowed_models: ["*"],
        allowed_routes: ["*"],
        jwt_subject: subject,
        jwt_provider: provider,
      });
    }

    // A passthrough route behind the same authentication chokepoint.
    const routePk = await seed.createProviderKey({
      display_name: "hmac-route-pk",
      api_key: rest.token,
      api_base: rest.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "hmac-route",
      path_prefix: ROUTE_PREFIX,
      target_url: rest.baseUrl,
      auth_mode: "gateway_key",
      credential_mode: "inject",
      provider_key_id: routePk.id,
    });

    const readyKey = "sk-hmac-ready-probe";
    await seed.createApiKey({
      key_hash: createHash("sha256").update(readyKey).digest("hex"),
      allowed_models: [],
    });
    const proxy = new ProxyClient(app.proxyUrl, readyKey);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await rest?.close();
    await idp?.close();
  });

  function skipUnlessUp(ctx: { skip: () => void }): boolean {
    if (!etcdReachable || !app || !upstream || !rest || !idp) {
      ctx.skip();
      return true;
    }
    return false;
  }

  test("issuer-less provider accepts a token with neither iss nor aud and binds it to a key", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    for (const alg of ["HS256", "HS384", "HS512"] as const) {
      const token = signHs(SECRET_FIRST, hmacClaims(), { alg });
      const res = await chat(app!, token);
      expect(res.status, `${alg}: ${await res.clone().text()}`).toBe(200);
      const body = (await res.json()) as { choices?: unknown[] };
      expect(Array.isArray(body.choices)).toBe(true);
    }
  });

  test("a token whose identity claim binds no key is rejected, never an anonymous pass", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Verifies against `hmac-first`, but no key under that provider
    // claims the subject. The pair is what binds, so this proves the
    // 200 above came from the binding and not from the signature alone.
    const res = await chat(app!, signHs(SECRET_FIRST, hmacClaims({ sub: "nobody" })));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_identity_unmapped");
  });

  test("a pinned issuer and audience are enforced", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const pinned = (extra: Record<string, unknown> = {}) =>
      signHs(
        SECRET_PINNED,
        hmacClaims({
          sub: "agent-pinned",
          iss: PINNED_ISSUER,
          aud: PINNED_AUDIENCE,
          ...extra,
        }),
      );

    expect((await chat(app!, pinned())).status).toBe(200);

    const wrongAud = await chat(app!, pinned({ aud: "someone-else" }));
    expect(wrongAud.status).toBe(401);
    expect(await errorCode(wrongAud)).toBe("jwt_invalid");

    // A mismatched `iss` selects no provider at all, and the issuer-less
    // providers hold different secrets, so it is refused too.
    const wrongIss = await chat(app!, pinned({ iss: "https://elsewhere.test" }));
    expect(wrongIss.status).toBe(401);
    expect(await errorCode(wrongIss)).toBe("jwt_invalid");
  });

  test("the wrong secret is rejected, upstream untouched", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const hits = upstream!.receivedRequests.length;
    const res = await chat(app!, signHs(SECRET_UNKNOWN, hmacClaims()));
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
    expect(upstream!.receivedRequests.length).toBe(hits);
  });

  test("an expired shared-secret token is rejected", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const res = await chat(
      app!,
      signHs(SECRET_FIRST, hmacClaims({ exp: Math.floor(Date.now() / 1000) - 3600 })),
    );
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_expired");
  });

  test("an RS256 token never reaches a shared-secret provider", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // `iss` selects `hmac-pinned`; its family is the HMAC one, so the
    // asymmetric signature is refused rather than verified.
    const token = idp!.sign(
      agentClaims(PINNED_ISSUER, { aud: PINNED_AUDIENCE, sub: "agent-pinned" }),
    );
    const res = await chat(app!, token);
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("an HS token signed with a JWKS provider's published public key is refused", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // The textbook algorithm-confusion attack: the verifier is offered
    // public material as if it were a shared secret. The provider's
    // family is fixed by its row, so the token is refused on `alg`.
    const token = signHs(
      jwksPublicKeyPem,
      agentClaims(idp!.url, { sub: "agent-jwks" }),
    );
    const res = await chat(app!, token);
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("an iss naming a JWKS provider is not retried against a shared-secret provider", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Signed with a secret `hmac-first` would accept — but the token
    // names the JWKS provider, and a named provider is the only one
    // consulted. Falling through here is exactly the confusion the
    // selection rule exists to prevent.
    const token = signHs(
      SECRET_FIRST,
      hmacClaims({ iss: idp!.url, aud: "aisix-gateway" }),
    );
    const res = await chat(app!, token);
    expect(res.status).toBe(401);
    expect(await errorCode(res)).toBe("jwt_invalid");
  });

  test("a token verifying only against the second issuer-less provider is accepted", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // `hmac-first` is tried first and fails on the signature; the trial
    // continues, and `hmac-second`'s own key binding is what the
    // request runs as.
    const res = await chat(app!, signHs(SECRET_SECOND, hmacClaims({ sub: "agent-second" })));
    expect(res.status, await res.clone().text()).toBe(200);

    // The subject is namespaced by the provider that vouched for it: the
    // same subject under the first provider's secret binds nothing.
    const crossed = await chat(app!, signHs(SECRET_FIRST, hmacClaims({ sub: "agent-second" })));
    expect(crossed.status).toBe(401);
    expect(await errorCode(crossed)).toBe("jwt_identity_unmapped");
  });

  test("a shared-secret token authenticates a passthrough route too", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const token = signHs(SECRET_FIRST, hmacClaims());
    const res = await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}/items/42`, {
      method: "GET",
      headers: { authorization: `Bearer ${token}` },
    });
    expect(res.status, await res.clone().text()).toBe(200);
    await res.text();

    const denied = await fetch(`${app!.proxyUrl}${ROUTE_PREFIX}/items/42`, {
      method: "GET",
      headers: { authorization: `Bearer ${signHs(SECRET_UNKNOWN, hmacClaims())}` },
    });
    expect(denied.status).toBe(401);
    await denied.text();
  });

  test("a provider that could not verify anything is rejected at load, and the rest keep serving", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    const etcd = new EtcdClient();
    // Each shape the semantic row validator refuses. All four pass the
    // JSON Schema; none can verify a token, so none may enter the
    // snapshot claiming to.
    const bad: Array<[string, Record<string, unknown>]> = [
      [
        "hmac-with-jwks-uri",
        {
          name: "bad-hmac-jwks-uri",
          hmac_secret: SECRET_FIRST,
          jwks_uri: "https://sso.invalid/jwks",
        },
      ],
      ["jwks-without-issuer", { name: "bad-no-issuer", audiences: ["aisix"] }],
      ["jwks-without-audiences", { name: "bad-no-audiences", issuer: "https://sso.invalid" }],
      ["short-secret", { name: "bad-short-secret", hmac_secret: "too-short" }],
    ];
    const ids = new Map<string, string>();
    for (const [label, doc] of bad) {
      const id = randomUUID();
      ids.set(label, id);
      await etcd.put(`${app!.etcdPrefix}/oidc_providers/${id}`, JSON.stringify(doc));
    }

    type StatusConfig = {
      rejected: Array<{
        resource_kind: string;
        resource_id: string;
        last_error_kind: string;
        last_error: string;
      }>;
    };
    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.metricsUrl}/status/config`);
      cfg = (await res.json()) as StatusConfig;
      return [...ids.values()].every((id) =>
        cfg!.rejected.some((r) => r.resource_id === id),
      );
    });

    for (const [label, id] of ids) {
      const rej = cfg!.rejected.find((r) => r.resource_id === id);
      expect(rej, `${label}: ${JSON.stringify(cfg!.rejected)}`).toBeDefined();
      expect(rej!.resource_kind).toBe("oidc_providers");
      expect(rej!.last_error_kind).toBe("schema_failed");
      // The reason names the rule, so an operator can act on it.
      expect(rej!.last_error.length).toBeGreaterThan(0);
    }

    // The valid providers kept serving throughout.
    const still = await chat(app!, signHs(SECRET_FIRST, hmacClaims()));
    expect(still.status, await still.clone().text()).toBe(200);

    for (const id of ids.values()) {
      await etcd.delete(`${app!.etcdPrefix}/oidc_providers/${id}`);
    }
  });

  test("no configured secret reaches /status/config or the process log", async (ctx) => {
    if (skipUnlessUp(ctx)) return;

    // Drive both an accepted and a refused token first, so every log
    // line the verification path can write has been written.
    await (await chat(app!, signHs(SECRET_FIRST, hmacClaims()))).text();
    await (await chat(app!, signHs(SECRET_UNKNOWN, hmacClaims()))).text();

    const status = await (await fetch(`${app!.metricsUrl}/status/config`)).text();
    const log = app!.output();
    for (const secret of [SECRET_FIRST, SECRET_SECOND, SECRET_PINNED]) {
      expect(status.includes(secret), "secret leaked into /status/config").toBe(false);
      expect(log.includes(secret), "secret leaked into the process log").toBe(false);
    }
  });
});
