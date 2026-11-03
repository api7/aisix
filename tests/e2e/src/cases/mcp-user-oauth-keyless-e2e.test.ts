import { createHash, randomUUID } from "node:crypto";

import { afterAll, beforeAll, describe, expect, test, vi } from "vitest";

import {
  Browser,
  ClientAuthState,
  connectMcpClient,
  consentForm,
  EtcdClient,
  OAUTH_CLIENT_REDIRECT as REDIRECT,
  OAuthMcpUpstream,
  SeedClient,
  signInMcpClient,
  spawnApp,
  startMockControlPlane,
  startMockOidcProvider,
  startPrefixStrippingProxy,
  waitConfigPropagation,
  whoami,
  type MockControlPlane,
  type MockOidcProvider,
  type SpawnedApp,
  waitForLogLine,
} from "../harness/index.js";

// E2E: an MCP client configured with nothing but a URL. The browser pass signs
// the user in at the environment's identity provider and at the upstream, and
// the client receives a session token of the gateway's own. No step carries a
// gateway credential.

const hash = (key: string) => createHash("sha256").update(key).digest("hex");

const SERVER_MAIN = "11111111-1111-4111-8111-111111111111";
const SERVER_OTHER = "22222222-2222-4222-8222-222222222222";
const SERVER_SHORT = "33333333-3333-4333-8333-333333333333";

const VERIFIER = "client-verifier-of-sufficient-length-0123456789";
const CHALLENGE = createHash("sha256").update(VERIFIER).digest("base64url");

const initialize = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-11-25",
    capabilities: {},
    clientInfo: { name: "mcp-user-oauth-keyless-e2e", version: "0.1" },
  },
};
const mcpPost = (url: string, headers: Record<string, string> = {}, body: unknown = initialize) =>
  fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream", ...headers },
    body: JSON.stringify(body),
  });
const bearer = (token: string) => ({ authorization: `Bearer ${token}` });

const authorizeUrl = (serverUrl: string, params: Record<string, string> = {}) => {
  const url = new URL(`${serverUrl}/oauth/authorize`);
  url.search = new URLSearchParams({
    response_type: "code",
    client_id: "client",
    redirect_uri: REDIRECT,
    code_challenge: CHALLENGE,
    code_challenge_method: "S256",
    state: "client-state",
    ...params,
  }).toString();
  return url;
};

interface Tokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  scope: string;
}

const tokenRequest = (serverUrl: string, form: Record<string, string>) =>
  fetch(`${serverUrl}/oauth/token`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(form),
  });

const oauthError = async (res: Response) => ((await res.json()) as { error: string }).error;
const errorCode = async (res: Response) => ((await res.json()) as { error: { code: string } }).error.code;

/** A whole browser pass and the code exchange, as a client without an SDK runs them. */
async function signInByHand(serverUrl: string, browser = new Browser()): Promise<Tokens> {
  const stop = await browser.walk(authorizeUrl(serverUrl));
  if (stop.kind !== "landed") throw new Error(`stopped: ${stop.status} ${stop.text}`);
  expect(stop.url.searchParams.get("state")).toBe("client-state");
  const res = await tokenRequest(serverUrl, {
    grant_type: "authorization_code",
    code: stop.url.searchParams.get("code")!,
    client_id: "client",
    redirect_uri: REDIRECT,
    code_verifier: VERIFIER,
  });
  expect(res.status).toBe(200);
  return (await res.json()) as Tokens;
}

async function seedEnvironment(
  seed: SeedClient,
  base: string,
  provider: MockOidcProvider,
  upstream: OAuthMcpUpstream,
  short: OAuthMcpUpstream,
): Promise<{ providerId: string; mappingId: string }> {
  const providerRow = await seed.createOidcProvider({
    name: "corp",
    issuer: provider.url,
    // Deliberately not the login client: its audience is checked separately.
    audiences: ["some-api"],
    bound_claims: { realm: "staff" },
    login: { client_id: provider.clientId, client_secret: provider.clientSecret, scopes: ["profile"] },
  });
  const shared = await seed.createApiKey({
    key_hash: hash(`sk-shared-${randomUUID()}`),
    allowed_models: [],
    mcp_access: { allow: ["*"] },
  });
  const noGrant = await seed.createApiKey({
    key_hash: hash(`sk-no-grant-${randomUUID()}`),
    allowed_models: [],
    mcp_access: { allow: [] },
  });
  const mapping = await seed.createClaimMapping({
    name: "staff",
    jwt_provider: "corp",
    priority: 100,
    match: [{ claim: "org.team", op: "exact", values: ["engineering"] }],
    resolve: { api_key_id: shared.id },
  });
  await seed.createClaimMapping({
    name: "no-mcp",
    jwt_provider: "corp",
    priority: 200,
    match: [{ claim: "org.team", op: "exact", values: ["no-mcp"] }],
    resolve: { api_key_id: noGrant.id },
  });
  // Last: etcd events apply in order, and a server the gateway knows before its
  // login provider is a relay-mode server, whose authorize writes to the store.
  await seed.update("mcp_auth_settings", randomUUID(), { resource_url: `${base}/mcp` });
  await seed.update("mcp_servers", SERVER_MAIN, {
    name: "main",
    type: "mcp",
    url: upstream.mcpUrl,
    auth_type: "oauth2_authorization_code",
    // A session token must not get through by being named here.
    forward_client_headers: ["authorization"],
  });
  await seed.update("mcp_servers", SERVER_OTHER, {
    name: "other",
    type: "mcp",
    url: upstream.mcpUrl,
    auth_type: "oauth2_authorization_code",
  });
  await seed.update("mcp_servers", SERVER_SHORT, {
    name: "short",
    type: "mcp",
    url: short.mcpUrl,
    auth_type: "oauth2_authorization_code",
  });
  return { providerId: providerRow.id, mappingId: mapping.id };
}

const claimsOf = (sub: string, team = "engineering") => ({
  sub,
  realm: "staff",
  org: { team },
  email: `${sub}@example.com`,
});

async function awaitInteractive(serverUrl: string): Promise<void> {
  await waitConfigPropagation(async () => {
    const res = await fetch(authorizeUrl(serverUrl), { redirect: "manual" });
    const location = res.headers.get("location");
    return res.status === 302 && location !== null && location.includes("/authorize") && location.includes("nonce=");
  });
}

/** The whole of what Cancel owes the client, from a page the walk stopped at. */
async function cancelsToClient(browser: Browser, page: string, cp: MockControlPlane): Promise<void> {
  const before = cp.storeWrites.length;
  const cancelled = await browser.submitConsent(page, "cancel");
  expect(cancelled.status).toBe(303);
  const location = new URL(cancelled.headers.get("location")!);
  expect(`${location.origin}${location.pathname}`).toBe(REDIRECT);
  expect(location.searchParams.get("error")).toBe("access_denied");
  expect(location.searchParams.get("state")).toBe("client-state");
  expect(location.searchParams.get("code")).toBeNull();
  expect(browser.cookies.size).toBe(0);
  expect(cp.storeWrites.slice(before).filter((w) => w.includes("mcp_oauth_code"))).toEqual([]);

  // The cookie went with the click, so neither button can be replayed.
  for (const decision of ["cancel", "allow"] as const) {
    const replay = await browser.submitConsent(page, decision);
    expect(replay.status, decision).toBe(400);
    expect(replay.headers.get("location"), decision).toBeNull();
  }
  expect(cp.storeWrites.slice(before).filter((w) => w.includes("mcp_oauth_code"))).toEqual([]);
}

describe("keyless sign-in on managed gateways", () => {
  let cp: MockControlPlane;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;
  let short: OAuthMcpUpstream;
  const apps: SpawnedApp[] = [];
  let etcdReachable = false;

  const url = (i: number, server = "main") => `${apps[i].proxyUrl}/mcp/${server}`;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    provider = await startMockOidcProvider({ clientSecret: "login-secret" });
    upstream = await new OAuthMcpUpstream().start();
    short = await new OAuthMcpUpstream({ expiresIn: 30 }).start();
    for (let i = 0; i < 2; i++) {
      const app = await spawnApp({
        extra: { managed: cp.managed },
        awaitProxyListener: false,
      });
      apps.push(app);
      const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);
      await seedEnvironment(seed, app.proxyUrl, provider, upstream, short);
      await awaitInteractive(url(i));
    }
  }, 120_000);

  afterAll(async () => {
    for (const app of apps) await app.exit();
    await upstream?.close();
    await short?.close();
    await provider?.close();
    await cp?.close();
  });

  test("the bare URL gets a challenge for this server's own sign-in", async () => {
    if (!etcdReachable) return;
    const res = await mcpPost(url(0));
    expect(res.status).toBe(401);
    expect(res.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${apps[0].proxyUrl}/.well-known/oauth-protected-resource/mcp/main"`,
    );
    const bad = await mcpPost(url(0), bearer("not-a-known-key"));
    expect(bad.status).toBe(401);
    expect(bad.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${apps[0].proxyUrl}/.well-known/oauth-protected-resource/mcp/main"`,
    );
  });

  test("a client's backend preflight gets the same redirect and leaves nothing behind", async () => {
    if (!etcdReachable) return;
    // The first sign-in anywhere in an environment creates its key, once.
    await fetch(authorizeUrl(url(0)), { redirect: "manual" });
    expect(cp.storeWrites.filter((w) => !w.startsWith("PUT mcp_oauth_session_key/"))).toEqual([]);
    const key = [...cp.store].filter(([id]) => id.startsWith("mcp_oauth_session_key/"));
    expect(key).toHaveLength(1);
    expect(key[0][1].ttlSeconds).toBe(365 * 24 * 3600);
    expect(key[0][1].labels).toEqual({});

    const before = cp.storeWrites.length;
    for (let i = 0; i < 2; i++) {
      const res = await fetch(authorizeUrl(url(0)), { redirect: "manual" });
      expect(res.status).toBe(302);
      const location = new URL(res.headers.get("location")!);
      expect(`${location.origin}${location.pathname}`).toBe(`${provider.url}/authorize`);
      expect(location.searchParams.get("client_id")).toBe(provider.clientId);
      expect(location.searchParams.get("redirect_uri")).toBe(`${apps[0].proxyUrl}/mcp-oauth/login/callback`);
      expect(location.searchParams.get("scope")).toBe("openid profile");
      expect(location.searchParams.get("code_challenge_method")).toBe("S256");
      expect(location.searchParams.get("code_challenge")).not.toBe(CHALLENGE);
      expect(res.headers.getSetCookie().some((c) => c.startsWith("mcp_oauth_flow_"))).toBe(true);
    }
    expect(cp.storeWrites.slice(before)).toEqual([]);
  });

  test("the SDK connects with no gateway credential at any step, and the upstream sees only its own token", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    upstream.user = "up-alice";
    const browser = new Browser();
    const { state, client } = await signInMcpClient(url(0), {}, new ClientAuthState(), browser);
    expect(await whoami(client)).toBe("up-alice");

    const session = state.saved!.access_token;
    expect(session).toMatch(/^aisix_mcp_at_/);
    expect(session).not.toContain("alice");
    expect(state.saved!.refresh_token).toMatch(/^aisix_mcp_rt_/);
    expect(state.saved!.expires_in).toBe(3600);
    expect(state.saved!.token_type.toLowerCase()).toBe("bearer");

    const carried = upstream.mcpAuthorizations.filter((a) => a !== undefined);
    expect(carried.length).toBeGreaterThan(0);
    expect(carried.every((a) => a.startsWith("Bearer at-up-alice-"))).toBe(true);
    expect(upstream.mcpAuthorizations.join(" ")).not.toContain("aisix_mcp_");

    // What the provider and the upstream were asked, in one browser pass.
    const exchange = provider.tokenRequests.at(-1)!;
    expect(exchange.client_secret).toBe("login-secret");
    expect(exchange.redirect_uri).toBe(`${apps[0].proxyUrl}/mcp-oauth/login/callback`);
    expect(upstream.authorizeRequests.at(-1)!.code_challenge).not.toBe(
      createHash("sha256").update(state.verifier).digest("base64url"),
    );

    // The consent page says what is being allowed, and how it is protected.
    expect(browser.consentPages).toHaveLength(1);
    expect(browser.consentPages[0]).toContain(">main<");
    expect(browser.consentPages[0]).toContain(">alice<");
    expect(browser.consentPages[0]).toContain(new URL(REDIRECT).origin);

    // The token travels with who it is and no more than that.
    const flowCookie = browser.setCookies.find((c) => c.startsWith("mcp_oauth_flow_"))!;
    expect(flowCookie).toContain("HttpOnly");
    expect(flowCookie).toContain("SameSite=Lax");
    expect(flowCookie).toContain("Max-Age=600");
    expect(flowCookie).not.toContain("Secure");
    expect(browser.cookies.size).toBe(0);
    await client.close();
  });

  test("the consent page is framed by nobody and cached by nobody", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    const browser = new Browser();
    const toProvider = await browser.request(authorizeUrl(url(0)).toString());
    const toCallback = await browser.request(toProvider.headers.get("location")!);
    const consent = await browser.request(toCallback.headers.get("location")!);
    expect(consent.status).toBe(200);
    expect(consent.headers.get("x-frame-options")).toBe("DENY");
    expect(consent.headers.get("cache-control")).toBe("no-store");
    expect(consent.headers.get("referrer-policy")).toBe("no-referrer");
    expect(consent.headers.get("content-security-policy")).toBeNull();
  });

  test("identities keep separate upstream sign-ins, and a returning user still sees consent", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("dana");
    upstream.user = "up-dana";
    const dana = await signInByHand(url(0));
    provider.claims = claimsOf("erin");
    upstream.user = "up-erin";
    const erin = await signInByHand(url(0));

    const danaClient = await connectMcpClient(url(0), bearer(dana.access_token));
    const erinClient = await connectMcpClient(url(0), bearer(erin.access_token));
    expect(await whoami(danaClient.client)).toBe("up-dana");
    expect(await whoami(erinClient.client)).toBe("up-erin");
    await danaClient.client.close();
    await erinClient.client.close();

    const tokens = [...cp.store].filter(([id, r]) => id.startsWith("mcp_oauth_token/") && r.labels.mcp_server_id === SERVER_MAIN);
    expect(tokens.map(([, r]) => r.labels.subject)).toEqual(
      expect.arrayContaining(["jwt:corp:dana", "jwt:corp:erin"]),
    );

    // Dana again: the provider and consent, not the upstream.
    provider.claims = claimsOf("dana");
    upstream.user = "somebody-else-entirely";
    const seen = upstream.authorizeRequests.length;
    const browser = new Browser();
    const again = await signInByHand(url(0), browser);
    expect(upstream.authorizeRequests.length).toBe(seen);
    expect(browser.consentPages).toHaveLength(1);
    const client = await connectMcpClient(url(0), bearer(again.access_token));
    expect(await whoami(client.client)).toBe("up-dana");
    await client.client.close();
  });

  test("an identity with no access gets a page saying so, and no code", async () => {
    if (!etcdReachable) return;
    const cases: [string, Record<string, unknown>, RegExp][] = [
      ["unmapped", claimsOf("frank", "sales"), /not mapped/],
      ["ACL-denied", claimsOf("grace", "no-mcp"), /not allowed to use this MCP server/],
      ["bound claim", { ...claimsOf("heidi"), realm: "contractors" }, /sign-in requirements/],
    ];
    for (const [name, claims, reason] of cases) {
      provider.claims = claims;
      const seen = upstream.authorizeRequests.length;
      const stop = await new Browser().walk(authorizeUrl(url(0)));
      expect(stop.kind, name).toBe("page");
      if (stop.kind !== "page") continue;
      expect(stop.status, name).toBe(403);
      expect(stop.text, name).toMatch(reason);
      expect(stop.text, name).not.toContain("<form");
      expect(upstream.authorizeRequests.length, name).toBe(seen);
      const sub = claims.sub as string;
      expect(stop.text, name).toContain(`<dd id="identity">${sub}</dd>`);
      expect(stop.text, name).toMatch(/contact your administrator with this identity/i);
      expect(stop.text, name).toMatch(/sign out of the identity provider/i);
      await waitForLogLine(apps[0], (l) => l.includes("refused a sign-in") && l.includes(sub), `${name} refusal log`);
    }
  });

  test("a refusal page escapes the identity it names", async () => {
    if (!etcdReachable) return;
    const hostile = `<img src=x onerror="alert('1')">&`;
    provider.claims = claimsOf(hostile, "sales");
    const stop = await new Browser().walk(authorizeUrl(url(0)));
    if (stop.kind !== "page") throw new Error("expected a refusal page");
    expect(stop.status).toBe(403);
    expect(stop.text).toContain(`&lt;img src=x onerror=&quot;alert(&#39;1&#39;)&quot;&gt;&amp;`);
    expect(stop.text).not.toContain("<img");
    await waitForLogLine(
      apps[0],
      (l) => l.includes("refused a sign-in") && l.includes("<img src=x onerror="),
      "the refusal log naming the identity",
    );
  });

  test("an ID token that does not check out gets nowhere", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    const refused = async () => {
      const stop = await new Browser().walk(authorizeUrl(url(0)));
      expect(stop.kind).toBe("page");
      if (stop.kind === "page") expect(stop.status).toBe(403);
    };
    provider.audience = "some-api";
    await refused();
    provider.audience = undefined;
    provider.nonce = "another-flow";
    await refused();
    provider.nonce = undefined;
    provider.signWithForeignKey = true;
    await refused();
    provider.signWithForeignKey = false;
    provider.grantedScope = "openid";
    // Nothing is required of the scope here, so this one passes.
    const passed = await new Browser().walk(authorizeUrl(url(0)));
    expect(passed.kind).toBe("landed");
    provider.grantedScope = undefined;
  });

  test("refresh rotates, a replay is refused without harming the chain, and kinds do not mix", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("ivan");
    upstream.user = "up-ivan";
    const first = await signInByHand(url(0));
    const refresh = (token: string, extra: Record<string, string> = {}) =>
      tokenRequest(url(0), { grant_type: "refresh_token", refresh_token: token, client_id: "client", ...extra });

    const rotated = await refresh(first.refresh_token);
    expect(rotated.status).toBe(200);
    const second = (await rotated.json()) as Tokens;
    expect(second.refresh_token).not.toBe(first.refresh_token);
    expect(second.access_token).not.toBe(first.access_token);
    expect(second.expires_in).toBe(3600);
    expect(typeof second.scope).toBe("string");

    const used = [...cp.store].filter(([id]) => id.startsWith("mcp_oauth_refresh_used/"));
    expect(used).toHaveLength(1);
    expect(used[0][1].labels).toEqual({ mcp_server_id: SERVER_MAIN });
    expect(used[0][1].ttlSeconds).toBeGreaterThan(13 * 24 * 3600);
    expect(used[0][1].ttlSeconds).toBeLessThanOrEqual(14 * 24 * 3600);

    const replay = await refresh(first.refresh_token);
    expect(replay.status).toBe(400);
    expect(await oauthError(replay)).toBe("invalid_grant");

    // Refused was only the replay: the newest pair works.
    const third = await refresh(second.refresh_token);
    expect(third.status).toBe(200);
    const tokens = (await third.json()) as Tokens;
    expect((await mcpPost(url(0), bearer(tokens.access_token))).status).toBe(200);

    // Each kind is refused where the other belongs.
    const asRefresh = await refresh(tokens.access_token);
    expect(asRefresh.status).toBe(400);
    expect(await oauthError(asRefresh)).toBe("invalid_grant");
    const asAccess = await mcpPost(url(0), bearer(tokens.refresh_token));
    expect(asAccess.status).toBe(401);

    // Named for another server's address.
    const wrongTarget = await refresh(tokens.refresh_token, { resource: url(0, "other") });
    expect(wrongTarget.status).toBe(400);
    expect(await oauthError(wrongTarget)).toBe("invalid_target");
    const rightTarget = await refresh(tokens.refresh_token, { resource: url(0) });
    expect(rightTarget.status).toBe(200);
  });

  test("a code is bound to its client, redirect URI, verifier and resource, and works once", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("judy");
    upstream.user = "up-judy";
    const landed = async () => {
      const stop = await new Browser().walk(authorizeUrl(url(0)));
      if (stop.kind !== "landed") throw new Error("no code");
      return stop.url.searchParams.get("code")!;
    };
    const redeem = (code: string, overrides: Record<string, string> = {}) =>
      tokenRequest(url(0), {
        grant_type: "authorization_code",
        code,
        client_id: "client",
        redirect_uri: REDIRECT,
        code_verifier: VERIFIER,
        ...overrides,
      });
    for (const overrides of [
      { code_verifier: "not-the-verifier-of-this-flow-0123456789" },
      { redirect_uri: "http://127.0.0.1:1/other" },
      { client_id: "other" },
    ] as Record<string, string>[]) {
      const res = await redeem(await landed(), overrides);
      expect(res.status, JSON.stringify(overrides)).toBe(400);
      expect(await oauthError(res), JSON.stringify(overrides)).toBe("invalid_grant");
    }
    const wrongResource = await redeem(await landed(), { resource: url(0, "other") });
    expect(wrongResource.status).toBe(400);
    expect(await oauthError(wrongResource)).toBe("invalid_target");

    const code = await landed();
    expect((await redeem(code, { resource: url(0) })).status).toBe(200);
    const again = await redeem(code);
    expect(again.status).toBe(400);
    expect(await oauthError(again)).toBe("invalid_grant");
  });

  test("authorize refuses a resource that is not this server", async () => {
    if (!etcdReachable) return;
    const res = await fetch(authorizeUrl(url(0), { resource: url(0, "other") }), { redirect: "manual" });
    const location = new URL(res.headers.get("location")!);
    expect(location.origin).toBe(new URL(REDIRECT).origin);
    expect(location.searchParams.get("error")).toBe("invalid_target");
    expect((await fetch(authorizeUrl(url(0), { resource: url(0) }), { redirect: "manual" })).status).toBe(302);
    expect((await fetch(authorizeUrl(url(0), { scope: "anything at all" }), { redirect: "manual" })).status).toBe(302);
  });

  test("a session is for one server and one surface", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("kim");
    upstream.user = "up-kim";
    const tokens = await signInByHand(url(0));
    expect((await mcpPost(url(0), bearer(tokens.access_token))).status).toBe(200);
    // The same person is connected to the other server too, so only the
    // token's own binding can keep it out of there.
    const forOther = await signInByHand(url(0, "other"));
    expect((await mcpPost(url(0, "other"), bearer(forOther.access_token))).status).toBe(200);

    const onOther = await mcpPost(url(0, "other"), bearer(tokens.access_token));
    expect(onOther.status).toBe(401);
    expect(onOther.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${apps[0].proxyUrl}/.well-known/oauth-protected-resource/mcp/other"`,
    );
    const onAggregate = await mcpPost(`${apps[0].proxyUrl}/mcp`, bearer(tokens.access_token));
    expect(onAggregate.status).toBe(401);
    const onChat = await fetch(`${apps[0].proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", ...bearer(tokens.access_token) },
      body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    });
    expect(onChat.status).toBe(401);
    // Garbage with the right prefix falls through to nothing else.
    const forged = await mcpPost(url(0), bearer("aisix_mcp_at_AAAA"));
    expect(forged.status).toBe(401);
    expect(forged.headers.get("www-authenticate")).toContain("resource_metadata");
  });

  test("a session beats a key on the same request, and is attributed like a JWT caller", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("liam");
    upstream.user = "up-liam";
    const tokens = await signInByHand(url(0));
    const session = await connectMcpClient(url(0), {
      ...bearer(tokens.access_token),
      "x-aisix-api-key": "sk-this-key-does-not-exist",
    });
    expect(await whoami(session.client)).toBe("up-liam");
    await session.client.close();
    expect(upstream.mcpAuthorizations.join(" ")).not.toContain("aisix_mcp_");
    await vi.waitFor(
      () => expect(cp.telemetryBodies.some((b) => b.includes('"jwt_subject":"liam"'))).toBe(true),
      { timeout: 30_000, interval: 500 },
    );
  });

  test("the consent POST is bound to the flow's cookie and its CSRF value", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    const browser = new Browser();
    const stop = await browser.walk(authorizeUrl(url(0)), { clickAllow: false });
    if (stop.kind !== "page") throw new Error("expected the consent page");
    // A reload shows the same page, spending nothing at the provider.
    const exchanges = provider.tokenRequests.length;
    const reloaded = await browser.request(stop.url);
    expect(reloaded.status).toBe(200);
    expect(await reloaded.text()).toContain("Allow");
    expect(provider.tokenRequests.length).toBe(exchanges);

    const form = consentForm(stop.text);
    const post = (who: Browser | undefined, fields: Record<string, string>) =>
      (who?.request.bind(who) ?? fetch)(form.action, {
        method: "POST",
        redirect: "manual",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams(fields).toString(),
      });

    const noCookie = await post(undefined, form.fields);
    expect(noCookie.status).toBe(400);
    expect(noCookie.headers.get("location")).toBeNull();
    const wrongCsrf = await post(browser, { ...form.fields, csrf: "0".repeat(64) });
    expect(wrongCsrf.status).toBe(403);
    expect(wrongCsrf.headers.get("location")).toBeNull();
    const noCsrf = await post(browser, { flow: form.fields.flow, decision: "allow" });
    expect(noCsrf.status).toBe(403);
    const cancelWithoutCsrf = await post(browser, { flow: form.fields.flow, decision: "cancel" });
    expect(cancelWithoutCsrf.status).toBe(403);
    expect(cancelWithoutCsrf.headers.get("location")).toBeNull();
    const cancelWithoutCookie = await post(undefined, { ...form.fields, decision: "cancel" });
    expect(cancelWithoutCookie.status).toBe(400);
    expect(cancelWithoutCookie.headers.get("location")).toBeNull();
    const undecidedForms: Record<string, string>[] = [{ decision: "maybe" }, { decision: "" }, {}];
    for (const decision of undecidedForms) {
      const undecided = await post(browser, { ...form.fields, ...decision });
      expect(undecided.status, JSON.stringify(decision)).toBe(400);
      expect(undecided.headers.get("location"), JSON.stringify(decision)).toBeNull();
    }

    // Refusals consumed nothing: the real click still lands.
    const allowed = await post(browser, { ...form.fields, decision: "allow" });
    expect(allowed.status).toBe(303);
    const location = new URL(allowed.headers.get("location")!);
    expect(location.searchParams.get("code")).not.toBeNull();
    expect(location.searchParams.get("state")).toBe("client-state");
    expect(cp.storeWrites.some((w) => w.startsWith("PUT mcp_oauth_code/"))).toBe(true);

    // The cookie is gone with the click, so it cannot mint a second code.
    expect(browser.cookies.size).toBe(0);
    expect((await post(browser, { ...form.fields, decision: "allow" })).status).toBe(400);
  });

  test("Cancel sends the client access_denied and its state, and leaves no code", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    const browser = new Browser();
    const stop = await browser.walk(authorizeUrl(url(0)), { clickAllow: false });
    if (stop.kind !== "page") throw new Error("expected the consent page");
    expect(stop.text).toContain(">Cancel</button>");
    await cancelsToClient(browser, stop.text, cp);
  });

  test("a consent POST for a flow still at the provider is refused", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    const browser = new Browser();
    const toProvider = await browser.request(authorizeUrl(url(0)).toString());
    const handle = new URL(toProvider.headers.get("location")!).searchParams.get("state")!;
    const early = await browser.request(`${apps[0].proxyUrl}/mcp-oauth/consent`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ flow: handle, csrf: "x" }).toString(),
    });
    expect(early.status).toBe(403);
    expect(early.headers.get("location")).toBeNull();
  });

  test("an upstream credential that has expired is refreshed; a refused refresh re-runs the upstream step", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("mona");
    short.user = "up-mona";
    const first = await signInByHand(url(0, "short"));
    const client = await connectMcpClient(url(0, "short"), bearer(first.access_token));
    expect(await whoami(client.client)).toBe("up-mona");
    await client.client.close();

    // Still alive once refreshed: straight to consent.
    const before = short.authorizeRequests.length;
    const grants = short.refreshGrants().length;
    await signInByHand(url(0, "short"));
    expect(short.authorizeRequests.length).toBe(before);
    expect(short.refreshGrants().length).toBeGreaterThan(grants);

    // The upstream has revoked it: signing in must not finish with a dead one.
    short.refreshAnswer = { status: 400, error: "invalid_grant" };
    short.user = "up-mona-again";
    const browser = new Browser();
    const stop = await browser.walk(authorizeUrl(url(0, "short")), { clickAllow: false });
    short.refreshAnswer = "ok";
    expect(stop.kind).toBe("page");
    expect(short.authorizeRequests.length).toBe(before + 1);
    expect(browser.consentPages).toHaveLength(1);
    const allowed = await browser.allow(browser.consentPages[0]);
    const code = new URL(allowed.headers.get("location")!).searchParams.get("code")!;
    const tokens = (await (
      await tokenRequest(url(0, "short"), {
        grant_type: "authorization_code",
        code,
        client_id: "client",
        redirect_uri: REDIRECT,
        code_verifier: VERIFIER,
      })
    ).json()) as Tokens;
    const again = await connectMcpClient(url(0, "short"), bearer(tokens.access_token));
    expect(await whoami(again.client)).toBe("up-mona-again");
    await again.client.close();
  });

  test("an SDK client whose upstream credential is gone is sent back to the browser, not refreshed in a loop", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("pat");
    short.user = "up-pat";
    const first = await signInMcpClient(url(0, "short"), {});
    expect(await whoami(first.client)).toBe("up-pat");
    await first.client.close();

    short.refreshAnswer = { status: 400, error: "invalid_grant" };
    const refused = await mcpPost(url(0, "short"), bearer(first.state.saved!.access_token));
    short.refreshAnswer = "ok";
    expect(refused.status).toBe(401);

    short.user = "up-pat-again";
    const opens = first.state.browserOpens.length;
    const again = await signInMcpClient(url(0, "short"), {}, first.state);
    expect(first.state.browserOpens.length).toBe(opens + 1);
    expect(await whoami(again.client)).toBe("up-pat-again");
    await again.client.close();
  });

  test("a refused authorization request does not send the browser to an address nobody consented to", async () => {
    if (!etcdReachable) return;
    const res = await fetch(authorizeUrl(url(0), { redirect_uri: "https://evil.example/", response_type: "token" }), {
      redirect: "manual",
    });
    expect(res.status).toBe(400);
    expect(res.headers.get("location")).toBeNull();
    expect(await oauthError(res)).toBe("unsupported_response_type");
  });

  test("two gateways accept each other's sessions and share one key", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("nina");
    upstream.user = "up-nina";
    const onFirst = await signInByHand(url(0));
    // The same provider and upstream behind the other gateway's own address.
    const onSecond = await connectMcpClient(url(1), bearer(onFirst.access_token));
    expect(await whoami(onSecond.client)).toBe("up-nina");
    await onSecond.client.close();

    const refreshed = await tokenRequest(url(1), {
      grant_type: "refresh_token",
      refresh_token: onFirst.refresh_token,
    });
    expect(refreshed.status).toBe(200);
    const replayedElsewhere = await tokenRequest(url(0), {
      grant_type: "refresh_token",
      refresh_token: onFirst.refresh_token,
    });
    expect(await oauthError(replayedElsewhere)).toBe("invalid_grant");

    expect([...cp.store.keys()].filter((id) => id.startsWith("mcp_oauth_session_key/"))).toHaveLength(1);
    expect(cp.storeWrites.filter((w) => w.startsWith("PUT mcp_oauth_session_key/"))).toHaveLength(1);
  });

  test("requests are served from the held key, with no store round trip and through an outage", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("omar");
    upstream.user = "up-omar";
    const tokens = await signInByHand(url(0));
    const client = await connectMcpClient(url(0), bearer(tokens.access_token));
    expect(await whoami(client.client)).toBe("up-omar");

    const storeCalls = () => cp.calls.filter((c) => c.startsWith("/dp/store/")).length;
    const before = storeCalls();
    for (let i = 0; i < 3; i++) expect(await whoami(client.client)).toBe("up-omar");
    expect(storeCalls()).toBe(before);

    cp.storeStatusOverride = 503;
    try {
      expect(await whoami(client.client)).toBe("up-omar");
      const refresh = await tokenRequest(url(0), { grant_type: "refresh_token", refresh_token: tokens.refresh_token });
      expect(refresh.status).toBe(503);
      expect(await oauthError(refresh)).toBe("temporarily_unavailable");
    } finally {
      cp.storeStatusOverride = undefined;
    }
    await client.client.close();
  });
});

describe("keyless sign-in behind a front gateway that strips a path prefix", () => {
  const PREFIX = "/team-a/gw";
  let etcdReachable = false;
  let app: SpawnedApp;
  let front: Awaited<ReturnType<typeof startPrefixStrippingProxy>>;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;
  let base: string;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    provider = await startMockOidcProvider();
    upstream = await new OAuthMcpUpstream().start();
    app = await spawnApp();
    front = await startPrefixStrippingProxy(PREFIX, new URL(app.proxyUrl));
    base = `${front.url}${PREFIX}`;
    await seedEnvironment(new SeedClient(etcd, app.etcdPrefix), base, provider, upstream, upstream);
    await awaitInteractive(`${app.proxyUrl}/mcp/main`);
  }, 60_000);

  afterAll(async () => {
    await front?.close();
    await app?.exit();
    await upstream?.close();
    await provider?.close();
  });

  test("the whole flow runs through the metadata the prefix can route, naming no resource", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    upstream.user = "up-alice";
    const metadata = (await (await fetch(`${base}/mcp/main/.well-known/openid-configuration`)).json()) as {
      authorization_endpoint: string;
      token_endpoint: string;
      registration_endpoint: string;
      grant_types_supported: string[];
      token_endpoint_auth_methods_supported: string[];
    };
    expect(metadata.authorization_endpoint).toBe(`${base}/mcp/main/oauth/authorize`);
    expect(metadata.grant_types_supported).toContain("refresh_token");
    expect(metadata.token_endpoint_auth_methods_supported).toEqual(["none"]);
    const registered = (await (
      await fetch(metadata.registration_endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ redirect_uris: [REDIRECT] }),
      })
    ).json()) as { client_id: string; token_endpoint_auth_method: string };
    expect(registered.token_endpoint_auth_method).toBe("none");

    const authorize = new URL(metadata.authorization_endpoint);
    authorize.search = new URLSearchParams({
      response_type: "code",
      client_id: registered.client_id,
      redirect_uri: REDIRECT,
      code_challenge: CHALLENGE,
      code_challenge_method: "S256",
    }).toString();
    const browser = new Browser();
    const stop = await browser.walk(authorize);
    if (stop.kind !== "landed") throw new Error(`stopped: ${stop.status} ${stop.text}`);

    expect(provider.authorizeRequests.at(-1)!.redirect_uri).toBe(`${base}/mcp-oauth/login/callback`);
    expect(upstream.authorizeRequests.at(-1)!.redirect_uri).toBe(`${base}/mcp-oauth/callback`);
    const cookie = browser.setCookies.find((c) => c.startsWith("mcp_oauth_flow_"))!;
    expect(cookie).toContain(`Path=${PREFIX}`);
    expect(consentForm(browser.consentPages[0]).action).toBe(`${base}/mcp-oauth/consent`);
    expect(front.seen).toContain(`${PREFIX}/mcp-oauth/login/callback`);
    expect(front.seen).toContain(`${PREFIX}/mcp-oauth/callback`);
    expect(front.seen).toContain(`${PREFIX}/mcp-oauth/consent`);

    const tokens = (await (
      await tokenRequest(`${base}/mcp/main`, {
        grant_type: "authorization_code",
        code: stop.url.searchParams.get("code")!,
        client_id: registered.client_id,
        redirect_uri: REDIRECT,
        code_verifier: VERIFIER,
      })
    ).json()) as Tokens;
    const client = await connectMcpClient(`${base}/mcp/main`, bearer(tokens.access_token));
    expect(await whoami(client.client)).toBe("up-alice");
    await client.client.close();

    const refused = await mcpPost(`${base}/mcp/main`);
    expect(refused.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${base}/.well-known/oauth-protected-resource/mcp/main"`,
    );
  });
});

describe("keyless sessions follow the configuration on the next request", () => {
  let etcdReachable = false;
  let app: SpawnedApp;
  let seed: SeedClient;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;
  let providerId: string;
  let mappingId: string;
  const url = (server = "main") => `${app.proxyUrl}/mcp/${server}`;
  const providerRow = () => ({
    name: "corp",
    issuer: provider.url,
    audiences: ["some-api"],
    bound_claims: { realm: "staff" },
    login: { client_id: provider.clientId, scopes: ["profile"] },
  });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    provider = await startMockOidcProvider();
    upstream = await new OAuthMcpUpstream().start();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    ({ providerId, mappingId } = await seedEnvironment(seed, app.proxyUrl, provider, upstream, upstream));
    await awaitInteractive(url());
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await provider?.close();
  });

  /** Etcd events apply in order, so a row written last proves the ones before it landed. */
  const settled = async () => {
    const sentinel = `sk-sentinel-${randomUUID()}`;
    await seed.createApiKey({ key_hash: hash(sentinel), allowed_models: [], mcp_access: { allow: [] } });
    await waitConfigPropagation(
      async () => (await fetch(`${app.proxyUrl}/v1/models`, { headers: bearer(sentinel) })).status === 200,
    );
  };

  const refusedSoon = async (token: string) => {
    await settled();
    const res = await mcpPost(url(), bearer(token));
    expect(res.status).toBe(401);
    expect(res.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${app.proxyUrl}/.well-known/oauth-protected-resource/mcp/main"`,
    );
    return res;
  };

  const listTools = async (token: string) => {
    const res = await mcpPost(url(), bearer(token), { jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(res.status).toBe(200);
    return ((await res.json()) as { result: { tools: { name: string }[] } }).result.tools.map((t) => t.name);
  };

  const opened = async (token: string) => {
    const res = await mcpPost(url(), bearer(token));
    expect(res.status).toBe(200);
    expect(((await res.json()) as { result?: unknown }).result).toBeDefined();
  };

  /** A key of its own and the mapping onto it, so a test can change the key alone. */
  const ownKey = async (team: string, mcpAccess: Record<string, unknown> = { allow: ["*"] }) => {
    const doc = { key_hash: hash(`sk-${team}-${randomUUID()}`), allowed_models: [], mcp_access: mcpAccess };
    const key = await seed.createApiKey(doc);
    await seed.createClaimMapping({
      name: team,
      jwt_provider: "corp",
      priority: 50,
      match: [{ claim: "org.team", op: "exact", values: [team] }],
      resolve: { api_key_id: key.id },
    });
    await settled();
    return { id: key.id, doc };
  };

  test("a session token beside a key is not forwarded by the servers that forward Authorization", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    upstream.user = "up-alice";
    const tokens = await signInByHand(url());
    const relaying = await new OAuthMcpUpstream().start();
    try {
      await seed.update("mcp_servers", randomUUID(), {
        name: "relaying",
        type: "mcp",
        url: relaying.mcpUrl,
        auth_type: "oauth_delegate",
      });
      await seed.update("mcp_servers", randomUUID(), {
        name: "forwarding",
        type: "mcp",
        url: relaying.mcpUrl,
        auth_type: "none",
        forward_client_headers: ["authorization"],
      });
      const KEY = "sk-keyless-beside-session";
      await seed.createApiKey({ key_hash: hash(KEY), allowed_models: [], mcp_access: { allow: ["*"] } });
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app.proxyUrl}/v1/models`, { headers: bearer(KEY) });
        return res.status === 200;
      });
      await waitConfigPropagation(async () => {
        await mcpPost(`${app.proxyUrl}/mcp`, { ...bearer(tokens.access_token), "x-aisix-api-key": KEY }, {
          jsonrpc: "2.0",
          id: 2,
          method: "tools/list",
        });
        return relaying.requestsSeen.length > 0;
      });
      for (const server of ["relaying", "forwarding"]) {
        await mcpPost(url(server), { ...bearer("sk-unused"), "x-aisix-api-key": KEY });
      }
      expect(relaying.mcpAuthorizations.join(" ")).not.toContain("aisix_mcp_");
    } finally {
      await relaying.close();
    }
  });

  test("a disabled provider refuses its sessions, and a removed mapping does too", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("alice");
    upstream.user = "up-alice";
    const tokens = await signInByHand(url());
    expect((await mcpPost(url(), bearer(tokens.access_token))).status).toBe(200);

    await seed.update("oidc_providers", providerId, { ...providerRow(), enabled: false });
    await refusedSoon(tokens.access_token);
    const refresh = await tokenRequest(url(), { grant_type: "refresh_token", refresh_token: tokens.refresh_token });
    // With the only login provider disabled the environment is in relay mode,
    // where the token endpoint wants a gateway credential.
    expect(refresh.status).toBe(401);
    expect(await oauthError(refresh)).toBe("invalid_client");

    await seed.update("oidc_providers", providerId, providerRow());
    await settled();
    expect((await mcpPost(url(), bearer(tokens.access_token))).status).toBe(200);

    await seed.delete("claim_mappings", mappingId);
    expect(await errorCode(await refusedSoon(tokens.access_token))).toBe("jwt_identity_unmapped");
    const afterMapping = await tokenRequest(url(), {
      grant_type: "refresh_token",
      refresh_token: tokens.refresh_token,
    });
    expect(afterMapping.status).toBe(400);
    expect(await oauthError(afterMapping)).toBe("invalid_grant");
  });

  test("a mapping added after sign-in applies to a claim nothing referred to at sign-in", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("nina", "snapshot");
    upstream.user = "up-nina";
    await ownKey("snapshot");
    const tokens = await signInByHand(url());
    expect(await listTools(tokens.access_token)).toContain("whoami");

    const narrow = await seed.createApiKey({
      key_hash: hash(`sk-narrow-${randomUUID()}`),
      allowed_models: [],
      mcp_access: { allow: [] },
    });
    const added = await seed.createClaimMapping({
      name: "by-email",
      jwt_provider: "corp",
      priority: 10,
      match: [{ claim: "email", op: "exact", values: ["nina@example.com"] }],
      resolve: { api_key_id: narrow.id },
    });
    await settled();
    expect(await listTools(tokens.access_token)).toEqual([]);

    await seed.delete("claim_mappings", added.id);
    await settled();
    expect(await listTools(tokens.access_token)).toContain("whoami");
  });

  test("a claim about the token itself decides nothing, at sign-in as later", async () => {
    if (!etcdReachable) return;
    await seed.createClaimMapping({
      name: "jti-rule",
      jwt_provider: "corp",
      priority: 10,
      match: [{ claim: "jti", op: "exact", values: ["token-id"] }],
      resolve: { api_key_id: (await ownKey("by-jti")).id },
    });
    await settled();
    provider.claims = { ...claimsOf("uma", "nobody"), jti: "token-id" };
    const stop = await new Browser().walk(authorizeUrl(url()));
    expect(stop.kind).toBe("page");
    if (stop.kind === "page") expect(stop.text).toMatch(/not mapped/);
  });

  test("an ID token too large for the sign-in's cookie ends on a page saying so", async () => {
    if (!etcdReachable) return;
    await ownKey("large");
    provider.claims = { ...claimsOf("vera", "large"), groups: Array.from({ length: 200 }, () => randomUUID()) };
    const stop = await new Browser().walk(authorizeUrl(url()));
    expect(stop.kind).toBe("page");
    if (stop.kind === "page") {
      expect(stop.status).toBe(400);
      expect(stop.text).toMatch(/too large/);
    }
  });

  /** Sign in as a person of their own, on a key of their own: earlier tests remove the shared mapping. */
  const signedInOnOwnKey = async (name: string) => {
    await ownKey(name);
    provider.claims = claimsOf(name, name);
    upstream.user = `up-${name}`;
    return signInByHand(url());
  };
  const refreshGrant = (refreshToken: string) =>
    tokenRequest(url(), { grant_type: "refresh_token", refresh_token: refreshToken });
  const setProvider = async (changes: Record<string, unknown>) => {
    await seed.update("oidc_providers", providerId, { ...providerRow(), ...changes });
    await settled();
  };
  const elsewhere = () => `${provider.url}/elsewhere`;

  test("a provider edit that keeps the issuer leaves a session alone, whichever way the issuer is spelled", async () => {
    if (!etcdReachable) return;
    let tokens = await signedInOnOwnKey("gina");
    const stillWorks = async () => {
      await opened(tokens.access_token);
      const refreshed = await refreshGrant(tokens.refresh_token);
      expect(refreshed.status).toBe(200);
      tokens = (await refreshed.json()) as Tokens;
      await opened(tokens.access_token);
    };

    try {
      await setProvider({ jwks_uri: `${provider.url}/jwks` });
      await stillWorks();
      await setProvider({ issuer: `${provider.url}/` });
      await stillWorks();
    } finally {
      await setProvider({});
    }
    await stillWorks();
  });

  test("a session is refused once its provider means another identity provider, and nothing of it is spent", async () => {
    if (!etcdReachable) return;
    const tokens = await signedInOnOwnKey("ines");
    await opened(tokens.access_token);

    const other = await startMockOidcProvider();
    try {
      await seed.update("oidc_providers", providerId, {
        ...providerRow(),
        issuer: other.url,
        login: { client_id: other.clientId, scopes: ["profile"] },
      });
      await refusedSoon(tokens.access_token);
      const metrics = await (await fetch(`${app.metricsUrl}/metrics`)).text();
      expect(metrics).toMatch(
        /aisix_auth_decisions_total\{[^}]*method="mcp_oauth_session"[^}]*reason="session_issuer_changed"[^}]*\}/,
      );
      const refused = await refreshGrant(tokens.refresh_token);
      expect(refused.status).toBe(400);
      expect(await oauthError(refused)).toBe("invalid_grant");

      // The same identity value at the new provider is someone else: their own
      // upstream sign-in, their own token, and the old session stays out.
      other.claims = claimsOf("ines", "ines");
      upstream.user = "up-ines-at-the-new-idp";
      const seen = upstream.authorizeRequests.length;
      const theirs = await signInByHand(url());
      expect(upstream.authorizeRequests.length).toBe(seen + 1);
      const theirClient = await connectMcpClient(url(), bearer(theirs.access_token));
      expect(await whoami(theirClient.client)).toBe("up-ines-at-the-new-idp");
      await theirClient.client.close();
      await refusedSoon(tokens.access_token);
    } finally {
      await setProvider({});
      await other.close();
    }

    // The refused refresh rotated nothing, so the token is still good, and it
    // still reaches the upstream as the person who signed in with it.
    await opened(tokens.access_token);
    const ownClient = await connectMcpClient(url(), bearer(tokens.access_token));
    expect(await whoami(ownClient.client)).toBe("up-ines");
    await ownClient.client.close();
    expect((await refreshGrant(tokens.refresh_token)).status).toBe(200);
  });

  test("a code issued before the issuer changed is not redeemed after it", async () => {
    if (!etcdReachable) return;
    await ownKey("hugo");
    provider.claims = claimsOf("hugo", "hugo");
    upstream.user = "up-hugo";
    const stop = await new Browser().walk(authorizeUrl(url()));
    if (stop.kind !== "landed") throw new Error(`stopped: ${stop.status} ${stop.text}`);

    try {
      await setProvider({ issuer: elsewhere() });
      const res = await tokenRequest(url(), {
        grant_type: "authorization_code",
        code: stop.url.searchParams.get("code")!,
        client_id: "client",
        redirect_uri: REDIRECT,
        code_verifier: VERIFIER,
      });
      expect(res.status).toBe(400);
      expect(await oauthError(res)).toBe("invalid_grant");
    } finally {
      await setProvider({});
    }
  });

  test("an upstream sign-in finishing after the issuer changed stores no credential under the provider's name", async () => {
    if (!etcdReachable) return;
    await ownKey("jonas");
    provider.claims = claimsOf("jonas", "jonas");
    upstream.user = "up-jonas";
    const browser = new Browser();
    const atUpstream = await browser.walk(authorizeUrl(url()), { landing: upstream.url });
    if (atUpstream.kind !== "landed") throw new Error(`stopped: ${atUpstream.status} ${atUpstream.text}`);

    try {
      await setProvider({ issuer: elsewhere() });
      const refused = await browser.walk(atUpstream.url);
      if (refused.kind !== "landed") throw new Error(`stopped: ${refused.status} ${refused.text}`);
      expect(refused.url.searchParams.get("error")).toBe("access_denied");
      expect(refused.url.searchParams.get("code")).toBeNull();
      const metrics = await (await fetch(`${app.metricsUrl}/metrics`)).text();
      expect(metrics).toMatch(
        /aisix_auth_decisions_total\{[^}]*method="mcp_oauth_login"[^}]*reason="session_issuer_changed"[^}]*\}/,
      );
    } finally {
      await setProvider({});
    }

    // Had the credential been kept, this pass would not need the upstream.
    const seen = upstream.authorizeRequests.length;
    await signInByHand(url());
    expect(upstream.authorizeRequests.length).toBe(seen + 1);
  });

  test("a disabled or expired key answers as it does to a JWT, and refresh is refused until it is back", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("omar", "lifecycle");
    upstream.user = "up-omar";
    const { id, doc } = await ownKey("lifecycle");
    const tokens = await signInByHand(url());
    expect((await mcpPost(url(), bearer(tokens.access_token))).status).toBe(200);

    await seed.update("api_keys", id, { ...doc, disabled: true });
    expect(await errorCode(await refusedSoon(tokens.access_token))).toBe("api_key_disabled");
    const refreshed = await tokenRequest(url(), { grant_type: "refresh_token", refresh_token: tokens.refresh_token });
    expect(refreshed.status).toBe(400);
    expect(await oauthError(refreshed)).toBe("invalid_grant");

    await seed.update("api_keys", id, { ...doc, expires_at: "2020-01-01T00:00:00Z" });
    expect(await errorCode(await refusedSoon(tokens.access_token))).toBe("api_key_expired");

    await seed.update("api_keys", id, doc);
    await settled();
    expect((await mcpPost(url(), bearer(tokens.access_token))).status).toBe(200);
  });

  test("a bound claim the session no longer meets is the policy refusal a JWT gets", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("quinn", "bound");
    upstream.user = "up-quinn";
    await ownKey("bound");
    const tokens = await signInByHand(url());

    await seed.update("oidc_providers", providerId, { ...providerRow(), bound_claims: { realm: "elsewhere" } });
    await settled();
    const refused = await mcpPost(url(), bearer(tokens.access_token));
    expect(refused.status).toBe(403);
    expect(refused.headers.get("www-authenticate")).toBeNull();
    expect(await errorCode(refused)).toBe("jwt_claims_rejected");
    const refreshed = await tokenRequest(url(), { grant_type: "refresh_token", refresh_token: tokens.refresh_token });
    expect(await oauthError(refreshed)).toBe("invalid_grant");

    await seed.update("oidc_providers", providerId, providerRow());
    await settled();
    expect((await mcpPost(url(), bearer(tokens.access_token))).status).toBe(200);
  });

  test("losing access to the server hides its tools; getting it back needs no new sign-in", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("rosa", "acl-flip");
    upstream.user = "up-rosa";
    const { id, doc } = await ownKey("acl-flip");
    const tokens = await signInByHand(url());
    const logins = provider.authorizeRequests.length;
    expect(await listTools(tokens.access_token)).toContain("whoami");
    const seenBeforeDenial = upstream.tokensSeenFor("up-rosa").length;

    await seed.update("api_keys", id, { ...doc, mcp_access: { allow: [] } });
    await settled();
    await opened(tokens.access_token);
    expect(await listTools(tokens.access_token)).toEqual([]);
    const called = await mcpPost(url(), bearer(tokens.access_token), {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "whoami", arguments: {} },
    });
    expect(called.status).toBe(200);
    expect(((await called.json()) as { error: { code: number } }).error.code).toBe(-32602);
    expect(upstream.tokensSeenFor("up-rosa").length).toBe(seenBeforeDenial);

    const refreshed = await tokenRequest(url(), { grant_type: "refresh_token", refresh_token: tokens.refresh_token });
    expect(refreshed.status).toBe(200);
    const renewed = (await refreshed.json()) as Tokens;

    await seed.update("api_keys", id, doc);
    await settled();
    const client = await connectMcpClient(url(), bearer(renewed.access_token));
    expect(await whoami(client.client)).toBe("up-rosa");
    await client.client.close();
    expect(upstream.tokensSeenFor("up-rosa").length).toBeGreaterThan(seenBeforeDenial);
    expect(provider.authorizeRequests.length).toBe(logins);
  });

  test("without access and without an upstream credential, MCP stays open and refresh sends it to the browser", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("sam", "acl-gone");
    upstream.user = "up-sam";
    const { id, doc } = await ownKey("acl-gone");
    const tokens = await signInByHand(url());

    upstream.rejectEveryToken = true;
    upstream.refreshAnswer = { status: 400, error: "invalid_grant" };
    const gone = await mcpPost(url(), bearer(tokens.access_token), { jsonrpc: "2.0", id: 2, method: "tools/list" });
    upstream.rejectEveryToken = false;
    upstream.refreshAnswer = "ok";
    expect(gone.status).toBe(401);
    expect(await errorCode(gone)).toBe("mcp_oauth_authorization_required");

    await seed.update("api_keys", id, { ...doc, mcp_access: { allow: [] } });
    await settled();
    await opened(tokens.access_token);
    expect(await listTools(tokens.access_token)).toEqual([]);
    const refreshed = await tokenRequest(url(), { grant_type: "refresh_token", refresh_token: tokens.refresh_token });
    expect(refreshed.status).toBe(400);
    expect(await oauthError(refreshed)).toBe("invalid_grant");
  });

  test("required scopes are requested, and refused when the token response did not grant them", async () => {
    if (!etcdReachable) return;
    await seed.update("oidc_providers", providerId, { ...providerRow(), required_scopes: ["mcp:tools"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(authorizeUrl(url()), { redirect: "manual" });
      return (res.headers.get("location") ?? "").includes("mcp%3Atools");
    });
    await seed.createClaimMapping({
      name: "staff-again",
      jwt_provider: "corp",
      priority: 50,
      match: [{ claim: "org.team", op: "exact", values: ["engineering"] }],
      resolve: { api_key_id: (await seed.createApiKey({
        key_hash: hash(`sk-again-${randomUUID()}`),
        allowed_models: [],
        mcp_access: { allow: ["*"] },
      })).id },
    });
    provider.claims = claimsOf("alice");
    await waitConfigPropagation(async () => {
      const stop = await new Browser().walk(authorizeUrl(url()));
      return stop.kind === "landed";
    });
    const location = new URL((await fetch(authorizeUrl(url()), { redirect: "manual" })).headers.get("location")!);
    expect(location.searchParams.get("scope")).toBe("openid profile mcp:tools");

    provider.grantedScope = "openid profile";
    const stop = await new Browser().walk(authorizeUrl(url()));
    expect(stop.kind).toBe("page");
    if (stop.kind === "page") {
      expect(stop.status).toBe(403);
      expect(stop.text).toMatch(/did not grant/);
    }
    // RFC 6749 §5.1: no `scope` in the response means what was asked for.
    provider.grantedScope = null;
    expect((await new Browser().walk(authorizeUrl(url()))).kind).toBe("landed");
    provider.grantedScope = undefined;
    await seed.update("oidc_providers", providerId, providerRow());
  });

  test("a second provider with login, or login on one that cannot use it, is a 500 at authorize", async () => {
    if (!etcdReachable) return;
    const second = await seed.createOidcProvider({
      name: "second",
      issuer: "https://second.example",
      audiences: ["a"],
      login: { client_id: "x" },
    });
    const misconfigured = async () => {
      const res = await fetch(authorizeUrl(url()), { redirect: "manual" });
      return res.status === 500 && ((await res.json()) as { error: { code: string } }).error.code === "mcp_oauth_misconfigured";
    };
    await waitConfigPropagation(misconfigured);
    await seed.delete("oidc_providers", second.id);
    await waitConfigPropagation(async () => (await fetch(authorizeUrl(url()), { redirect: "manual" })).status === 302);

    await seed.update("oidc_providers", providerId, {
      name: "corp",
      hmac_secret: "x".repeat(32),
      login: { client_id: provider.clientId },
    });
    await waitConfigPropagation(misconfigured);
  });
});

describe("an operator's own OAuth addresses under the default SSRF settings", () => {
  let etcdReachable = false;
  let app: SpawnedApp;
  let seed: SeedClient;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;
  let elsewhere: OAuthMcpUpstream;
  const started: OAuthMcpUpstream[] = [];

  /** A running upstream, an entry for it on the gateway, and that entry's address. */
  const server = async (
    name: string,
    options: ConstructorParameters<typeof OAuthMcpUpstream>[0],
    fields: (mock: OAuthMcpUpstream) => Record<string, unknown> = () => ({}),
    gateway: SpawnedApp = app,
    into: SeedClient = seed,
  ) => {
    const mock = await new OAuthMcpUpstream(options).start();
    started.push(mock);
    await into.update("mcp_servers", randomUUID(), {
      name,
      type: "mcp",
      url: mock.mcpUrl,
      auth_type: "oauth2_authorization_code",
      ...fields(mock),
    });
    const at = `${gateway.proxyUrl}/mcp/${name}`;
    await awaitInteractive(at);
    return { mock, at };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    provider = await startMockOidcProvider();
    upstream = await new OAuthMcpUpstream().start();
    elsewhere = await new OAuthMcpUpstream({ staticClient: { id: "pre-registered", secret: "s" } }).start();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seedEnvironment(seed, app.proxyUrl, provider, upstream, upstream);
    await awaitInteractive(`${app.proxyUrl}/mcp/main`);
    provider.claims = claimsOf("tara");
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(started.map((mock) => mock.close()));
    await elsewhere?.close();
    await upstream?.close();
    await provider?.close();
  });

  test("a provider and a server on private addresses, with what the server publishes on its own address", async () => {
    if (!etcdReachable) return;
    upstream.user = "up-tara";
    const tokens = await signInByHand(`${app.proxyUrl}/mcp/main`);
    const client = await connectMcpClient(`${app.proxyUrl}/mcp/main`, bearer(tokens.access_token));
    expect(await whoami(client.client)).toBe("up-tara");
    await client.client.close();
    expect(provider.tokenRequests.length).toBeGreaterThan(0);
    expect(upstream.requestsSeen).toContain("GET /.well-known/oauth-authorization-server");
    expect(upstream.registrations.length).toBeGreaterThan(0);
  });

  test("a configured token and registration URL on another private address", async () => {
    if (!etcdReachable) return;
    const toLocalhost = (url: string) => url.replace("127.0.0.1", "localhost");
    const { mock, at } = await server("configured", { discovery: false }, (m) => ({
      authorization_url: `${m.url}/authorize`,
      token_url: `${toLocalhost(m.url)}/token`,
      registration_url: `${toLocalhost(m.url)}/register`,
    }));
    mock.user = "up-configured";
    const tokens = await signInByHand(at);
    const client = await connectMcpClient(at, bearer(tokens.access_token));
    expect(await whoami(client.client)).toBe("up-configured");
    await client.client.close();
    expect(mock.registrations).toHaveLength(1);
    expect(mock.tokenRequests).toHaveLength(1);
  });

  test("a redirect from a configured token URL is not followed", async () => {
    if (!etcdReachable) return;
    const { mock, at } = await server(
      "redirecting",
      { tokenRedirect: `${elsewhere.url}/token`, staticClient: { id: "pre-registered", secret: "s" } },
      (m) => ({ client_id: "pre-registered", secret: "s", authorization_url: `${m.url}/authorize`, token_url: `${m.url}/token` }),
    );
    const grants = elsewhere.tokenRequests.length;
    const stop = await new Browser().walk(authorizeUrl(at));
    expect(stop.kind).toBe("page");
    expect(mock.tokenRequestQueries).toHaveLength(1);
    expect(elsewhere.tokenRequests.length).toBe(grants);
  });

  test.each([
    ["authorization server metadata", { advertise: { authorizationServer: "ELSEWHERE" } }, {}],
    ["registration endpoint", { advertise: { registrationEndpoint: "ELSEWHERE/register" } }, {}],
    [
      "token endpoint",
      { advertise: { tokenEndpoint: "ELSEWHERE/token" }, staticClient: { id: "pre-registered", secret: "s" } },
      { client_id: "pre-registered", secret: "s" },
    ],
  ])("a discovered %s on another private address is refused", async (name, options, fields) => {
    if (!etcdReachable) return;
    const advertise = Object.fromEntries(
      Object.entries(options.advertise).map(([k, v]) => [k, v.replace("ELSEWHERE", elsewhere.url)]),
    );
    const seen = elsewhere.requestsSeen.length;
    const { mock, at } = await server(
      `cross-${name.split(" ")[0]}`,
      { ...options, advertise },
      () => fields,
    );
    const stop = await new Browser().walk(authorizeUrl(at));
    expect(stop.kind).toBe("page");
    if (stop.kind === "page") expect(stop.status).toBe(500);
    expect(elsewhere.requestsSeen.length).toBe(seen);
    // What the server publishes on its own address was read to get this far.
    expect(mock.requestsSeen).toContain("GET /.well-known/oauth-protected-resource/mcp");
  });

  test("allowed_hosts still opens a discovered address on another authority", async () => {
    if (!etcdReachable) return;
    const open = await spawnApp({ extra: { outbound: { ssrf_protection: { allowed_hosts: ["127.0.0.1"] } } } });
    try {
      const openSeed = new SeedClient(new EtcdClient(), open.etcdPrefix);
      await seedEnvironment(openSeed, open.proxyUrl, provider, upstream, upstream);
      const registrar = await new OAuthMcpUpstream().start();
      started.push(registrar);
      const { at } = await server(
        "cross-allowed",
        { advertise: { registrationEndpoint: `${registrar.url}/register` } },
        () => ({}),
        open,
        openSeed,
      );
      await new Browser().walk(authorizeUrl(at));
      expect(registrar.registrations).toHaveLength(1);
    } finally {
      await open.exit();
    }
  });
});

describe("whose stored credential it is, across two gateways", () => {
  let etcdReachable = false;
  let cp: MockControlPlane;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;
  const apps: SpawnedApp[] = [];
  const seeds: SeedClient[] = [];
  const providerIds: string[] = [];
  const url = (i: number) => `${apps[i].proxyUrl}/mcp/main`;
  const providerRow = () => ({
    name: "corp",
    issuer: provider.url,
    audiences: ["some-api"],
    bound_claims: { realm: "staff" },
    login: { client_id: provider.clientId, scopes: ["profile"] },
  });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    provider = await startMockOidcProvider();
    upstream = await new OAuthMcpUpstream().start();
    for (let i = 0; i < 2; i++) {
      const app = await spawnApp({ extra: { managed: cp.managed }, awaitProxyListener: false });
      apps.push(app);
      const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);
      seeds.push(seed);
      providerIds.push((await seedEnvironment(seed, app.proxyUrl, provider, upstream, upstream)).providerId);
      await awaitInteractive(url(i));
    }
  }, 120_000);

  afterAll(async () => {
    for (const app of apps) await app.exit();
    await upstream?.close();
    await provider?.close();
    await cp?.close();
  });

  /** Both gateways, with the row written last proving the ones before it landed. */
  const setProvider = async (changes: Record<string, unknown>) => {
    for (let i = 0; i < 2; i++) {
      await seeds[i].update("oidc_providers", providerIds[i], { ...providerRow(), ...changes });
      const sentinel = `sk-sentinel-${randomUUID()}`;
      await seeds[i].createApiKey({ key_hash: hash(sentinel), allowed_models: [], mcp_access: { allow: [] } });
      await waitConfigPropagation(
        async () => (await fetch(`${apps[i].proxyUrl}/v1/models`, { headers: bearer(sentinel) })).status === 200,
      );
    }
  };

  const storedFor = (subject: string) =>
    [...cp.store].filter(([id, r]) => id.startsWith("mcp_oauth_token/") && r.labels.subject === subject);
  /** What the control plane does on an administrator's disconnect or a provider change. */
  const purge = (subject: string) => {
    for (const [id] of storedFor(subject)) cp.store.delete(id);
  };
  const upstreamUser = async (i: number, session: string) => {
    const { client } = await connectMcpClient(url(i), bearer(session));
    const user = await whoami(client);
    await client.close();
    return user;
  };
  const refresh = (i: number, refreshToken: string) =>
    tokenRequest(url(i), { grant_type: "refresh_token", refresh_token: refreshToken });

  test("after the identity claim changes, a new value equal to someone's old one is its own person", async () => {
    if (!etcdReachable) return;
    await setProvider({ identity_claim: "preferred_username" });
    try {
      provider.claims = { ...claimsOf("sub-ann"), preferred_username: "ann", given_name: "Annabel" };
      upstream.user = "up-ann";
      const ann = await signInByHand(url(0));
      expect(await upstreamUser(0, ann.access_token)).toBe("up-ann");
      expect(await upstreamUser(1, ann.access_token)).toBe("up-ann");

      await setProvider({ identity_claim: "given_name" });
      purge("jwt:corp:ann");
      for (const i of [0, 1]) expect((await mcpPost(url(i), bearer(ann.access_token))).status).toBe(401);

      provider.claims = { ...claimsOf("sub-zed"), preferred_username: "zed", given_name: "ann" };
      upstream.user = "up-zed";
      const seen = upstream.authorizeRequests.length;
      const browser = new Browser();
      const zed = await signInByHand(url(1), browser);
      expect(upstream.authorizeRequests.length).toBe(seen + 1);
      expect(browser.consentPages[0]).toContain(">ann<");

      // Including where the first person's token is still held.
      expect(await upstreamUser(1, zed.access_token)).toBe("up-zed");
      expect(await upstreamUser(0, zed.access_token)).toBe("up-zed");

      // The label the control plane purges by does not name the claim.
      expect(storedFor("jwt:corp:ann")).toHaveLength(1);
    } finally {
      await setProvider({});
    }
  });

  test("the same value under another claim is not the same identity, purged or not", async () => {
    if (!etcdReachable) return;
    await setProvider({ identity_claim: "preferred_username" });
    try {
      provider.claims = { ...claimsOf("sub-otto"), preferred_username: "otto", given_name: "Bruno" };
      upstream.user = "up-otto";
      const otto = await signInByHand(url(0));
      expect(await upstreamUser(0, otto.access_token)).toBe("up-otto");

      await setProvider({ identity_claim: "given_name" });
      provider.claims = { ...claimsOf("sub-bruno"), preferred_username: "bruno", given_name: "otto" };
      upstream.user = "up-bruno";
      const seen = upstream.authorizeRequests.length;
      const bruno = await signInByHand(url(0));
      expect(upstream.authorizeRequests.length).toBe(seen + 1);
      expect(await upstreamUser(0, bruno.access_token)).toBe("up-bruno");
    } finally {
      await setProvider({});
    }
  });

  test("a claim holding the same value keeps the session but needs a new upstream sign-in", async () => {
    if (!etcdReachable) return;
    provider.claims = { ...claimsOf("rita"), uid: "rita" };
    upstream.user = "up-rita";
    const rita = await signInByHand(url(0));
    expect(await upstreamUser(0, rita.access_token)).toBe("up-rita");

    await setProvider({ identity_claim: "uid" });
    try {
      const challenged = await mcpPost(url(0), bearer(rita.access_token));
      expect(challenged.status).toBe(401);
      expect(await errorCode(challenged)).toBe("mcp_oauth_authorization_required");
      const refused = await refresh(0, rita.refresh_token);
      expect(refused.status).toBe(400);
      expect(await oauthError(refused)).toBe("invalid_grant");

      upstream.user = "up-rita-again";
      const seen = upstream.authorizeRequests.length;
      const again = await signInByHand(url(0));
      expect(upstream.authorizeRequests.length).toBe(seen + 1);
      expect(await upstreamUser(0, again.access_token)).toBe("up-rita-again");
      expect(await upstreamUser(1, again.access_token)).toBe("up-rita-again");
    } finally {
      await setProvider({});
    }
  });

  test("naming the default identity claim changes nothing for someone signed in without it", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("bea");
    upstream.user = "up-bea";
    const bea = await signInByHand(url(0));
    expect(await upstreamUser(0, bea.access_token)).toBe("up-bea");

    await setProvider({ identity_claim: "sub" });
    try {
      upstream.user = "somebody-else-entirely";
      const seen = upstream.authorizeRequests.length;
      expect(await upstreamUser(0, bea.access_token)).toBe("up-bea");
      expect(await upstreamUser(1, bea.access_token)).toBe("up-bea");
      const refreshed = await refresh(1, bea.refresh_token);
      expect(refreshed.status).toBe(200);
      const again = await signInByHand(url(1));
      expect(upstream.authorizeRequests.length).toBe(seen);
      expect(await upstreamUser(1, again.access_token)).toBe("up-bea");
    } finally {
      await setProvider({});
    }
  });

  test("a sign-in and a refresh ask the store, while requests elsewhere keep their cached window", async () => {
    if (!etcdReachable) return;
    provider.claims = claimsOf("cleo");
    upstream.user = "up-cleo";
    const cleo = await signInByHand(url(0));
    expect(await upstreamUser(0, cleo.access_token)).toBe("up-cleo");
    expect(await upstreamUser(1, cleo.access_token)).toBe("up-cleo");

    purge("jwt:corp:cleo");
    expect((await mcpPost(url(0), bearer(cleo.access_token))).status).toBe(200);
    expect((await mcpPost(url(1), bearer(cleo.access_token))).status).toBe(200);

    const refused = await refresh(0, cleo.refresh_token);
    expect(refused.status).toBe(400);
    expect(await oauthError(refused)).toBe("invalid_grant");
    // That gateway has noticed, and the other has not.
    const challenged = await mcpPost(url(0), bearer(cleo.access_token));
    expect(challenged.status).toBe(401);
    expect(await errorCode(challenged)).toBe("mcp_oauth_authorization_required");
    expect((await mcpPost(url(1), bearer(cleo.access_token))).status).toBe(200);

    // A new sign-in where the old token is still held goes to the upstream.
    upstream.user = "up-cleo-again";
    const seen = upstream.authorizeRequests.length;
    const again = await signInByHand(url(1));
    expect(upstream.authorizeRequests.length).toBe(seen + 1);
    expect(storedFor("jwt:corp:cleo")).toHaveLength(1);
    expect(await upstreamUser(1, again.access_token)).toBe("up-cleo-again");
    expect(await upstreamUser(0, again.access_token)).toBe("up-cleo-again");
  });

  test("an upstream sign-in finishing after the identity claim changed stores no credential", async () => {
    if (!etcdReachable) return;
    await setProvider({ identity_claim: "preferred_username" });
    try {
      provider.claims = { ...claimsOf("sub-dora"), preferred_username: "dora", given_name: "Dorothy" };
      upstream.user = "up-dora";
      const browser = new Browser();
      const atUpstream = await browser.walk(authorizeUrl(url(0)), { landing: upstream.url });
      if (atUpstream.kind !== "landed") throw new Error(`stopped: ${atUpstream.status} ${atUpstream.text}`);

      await setProvider({ identity_claim: "given_name" });
      const refused = await browser.walk(atUpstream.url);
      if (refused.kind !== "landed") throw new Error(`stopped: ${refused.status} ${refused.text}`);
      expect(refused.url.searchParams.get("error")).toBe("access_denied");
      expect(refused.url.searchParams.get("code")).toBeNull();
      expect(storedFor("jwt:corp:dora")).toEqual([]);
      const metrics = await (await fetch(`${apps[0].metricsUrl}/metrics`)).text();
      expect(metrics).toMatch(
        /aisix_auth_decisions_total\{[^}]*method="mcp_oauth_login"[^}]*reason="session_identity_changed"[^}]*\}/,
      );

      // Had the credential been kept, this other person would be handed it.
      provider.claims = { ...claimsOf("sub-eve"), preferred_username: "eve", given_name: "dora" };
      upstream.user = "up-eve";
      const seen = upstream.authorizeRequests.length;
      const eve = await signInByHand(url(0));
      expect(upstream.authorizeRequests.length).toBe(seen + 1);
      expect(await upstreamUser(0, eve.access_token)).toBe("up-eve");
    } finally {
      await setProvider({});
    }
  });
});

describe("relay mode, with no provider offering a login", () => {
  let cp: MockControlPlane;
  let app: SpawnedApp;
  let upstream: OAuthMcpUpstream;
  let etcdReachable = false;
  const ALICE = "sk-relay-alice";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    upstream = await new OAuthMcpUpstream().start();
    app = await spawnApp({
      extra: { managed: cp.managed },
      awaitProxyListener: false,
    });
    const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);
    await seed.update("mcp_servers", SERVER_MAIN, {
      name: "main",
      type: "mcp",
      url: upstream.mcpUrl,
      auth_type: "oauth2_authorization_code",
    });
    await seed.update("mcp_auth_settings", randomUUID(), { resource_url: `${app.proxyUrl}/mcp` });
    await seed.createApiKey({
      key_hash: hash(ALICE),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
      user_id: "alice",
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app.proxyUrl}/v1/models`, { headers: bearer(ALICE) }).catch(() => undefined);
      return res?.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await cp?.close();
  });

  test("Cancel sends the client access_denied and its state, and leaves no code", async () => {
    if (!etcdReachable) return;
    const browser = new Browser();
    const stop = await browser.walk(authorizeUrl(`${app.proxyUrl}/mcp/main`), { clickAllow: false });
    if (stop.kind !== "page") throw new Error("expected the consent page");
    expect(stop.text).toContain(">Cancel</button>");
    await cancelsToClient(browser, stop.text, cp);
  });

  test("the client's credential still decides, through the consent page, with no flow record", async () => {
    if (!etcdReachable) return;
    const url = `${app.proxyUrl}/mcp/main`;
    upstream.user = "up-alice";
    const browser = new Browser();
    const headers = { "x-aisix-api-key": ALICE };
    const { state, client } = await signInMcpClient(url, headers, new ClientAuthState(), browser);
    expect(await whoami(client)).toBe("up-alice");
    await client.close();

    expect(state.saved!.access_token).not.toMatch(/^aisix_mcp_at_/);
    expect(state.saved!.refresh_token).toBeUndefined();
    expect(browser.consentPages).toHaveLength(1);
    expect(browser.consentPages[0]).not.toContain('id="identity"');
    expect(browser.cookies.size).toBe(0);
    expect(cp.storeWrites.filter((w) => w.includes("mcp_oauth_flow"))).toEqual([]);

    // The client's PKCE challenge went upstream, as it always has.
    expect(upstream.authorizeRequests.at(-1)!.code_challenge).toBe(
      createHash("sha256").update(state.verifier).digest("base64url"),
    );

    // A client with no credential is still turned away at the token step.
    const noKey = await tokenRequest(url, {
      grant_type: "authorization_code",
      code: "x",
      client_id: "main",
      redirect_uri: REDIRECT,
      code_verifier: VERIFIER,
    });
    expect(noKey.status).toBe(401);
    expect(await oauthError(noKey)).toBe("invalid_client");

    // The challenge is this server's own here too.
    const bare = await mcpPost(url);
    expect(bare.status).toBe(401);
    expect(bare.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${app.proxyUrl}/.well-known/oauth-protected-resource/mcp/main"`,
    );
  });
});
