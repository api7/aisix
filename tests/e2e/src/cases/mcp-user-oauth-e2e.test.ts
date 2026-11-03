import { createHash, randomUUID } from "node:crypto";

import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { afterAll, beforeAll, describe, expect, test } from "vitest";

import {
  Browser,
  ClientAuthState,
  connectMcpClient as connect,
  EtcdClient,
  followBrowser,
  OAUTH_CLIENT_REDIRECT as REDIRECT,
  OAuthMcpUpstream,
  scrapeMetrics,
  SeedClient,
  signInMcpClient as signIn,
  spawnApp,
  startMockIdp,
  startPrefixStrippingProxy,
  waitConfigPropagation,
  whoami,
  type MockIdp,
  type OAuthMcpUpstreamOptions,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: per-user OAuth to an upstream MCP server. A real MCP client, the
// official SDK, signs in through the gateway the way it would through any
// authorization server; a mock upstream that rotates its refresh tokens plays
// the real authorization server and MCP server behind it.

const hash = (key: string) => createHash("sha256").update(key).digest("hex");

const initialize = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-11-25",
    capabilities: {},
    clientInfo: { name: "mcp-user-oauth-e2e", version: "0.1" },
  },
};

const mcpPost = (url: string, headers: Record<string, string>, body: unknown = initialize) =>
  fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      ...headers,
    },
    body: JSON.stringify(body),
  });

const errorCode = async (res: Response): Promise<string | undefined> =>
  ((await res.json()) as { error?: { code?: string } }).error?.code;

const key = (name: string) => ({ "x-aisix-api-key": name });

const authorizeUrl = (serverUrl: string, params: Record<string, string> = {}) => {
  const url = new URL(`${serverUrl}/oauth/authorize`);
  url.search = new URLSearchParams({
    response_type: "code",
    client_id: "e2e",
    redirect_uri: REDIRECT,
    code_challenge: "c".repeat(43),
    code_challenge_method: "S256",
    ...params,
  }).toString();
  return url;
};

interface Stack {
  app: SpawnedApp;
  seed: SeedClient;
  base: string;
}

async function startStack(
  baseFor?: (proxyUrl: string) => Promise<string>,
): Promise<Stack> {
  const app = await spawnApp();
  const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
  const base = baseFor === undefined ? app.proxyUrl : await baseFor(app.proxyUrl);
  return { app, seed, base };
}

async function gateOnKey(stack: Stack, plaintext: string): Promise<void> {
  await waitConfigPropagation(async () => {
    const res = await fetch(`${stack.app.proxyUrl}/v1/models`, {
      headers: { authorization: `Bearer ${plaintext}` },
    });
    return res.status === 200;
  });
}

async function addServer(
  stack: Stack,
  name: string,
  upstream: OAuthMcpUpstream,
  extra: Record<string, unknown> = {},
): Promise<void> {
  await stack.seed.update("mcp_servers", randomUUID(), {
    name,
    type: "mcp",
    url: upstream.mcpUrl,
    auth_type: "oauth2_authorization_code",
    ...extra,
  });
}

const addKey = (stack: Stack, plaintext: string, extra: Record<string, unknown> = {}) =>
  stack.seed.createApiKey({
    key_hash: hash(plaintext),
    allowed_models: [],
    mcp_access: { allow: ["*"] },
    ...extra,
  });

describe("per-user OAuth to an upstream MCP server", () => {
  let stack: Stack;
  let etcdReachable = false;
  const upstreams: OAuthMcpUpstream[] = [];
  const upstream = async (options: OAuthMcpUpstreamOptions = {}) => {
    const created = await new OAuthMcpUpstream(options).start();
    upstreams.push(created);
    return created;
  };
  let main: OAuthMcpUpstream;
  let shortLived: OAuthMcpUpstream;
  let flaky: OAuthMcpUpstream;
  let revoking: OAuthMcpUpstream;
  let staticClient: OAuthMcpUpstream;
  let registering: OAuthMcpUpstream;
  let aggregatedA: OAuthMcpUpstream;
  let aggregatedB: OAuthMcpUpstream;
  let selfRevoke: OAuthMcpUpstream;
  let noRefresh: OAuthMcpUpstream;
  let jira: OAuthMcpUpstream;

  // `alice` and `bob` are members; `ALICE2` is another key of alice's.
  const ALICE = "sk-oauth-alice";
  const ALICE2 = "sk-oauth-alice-second";
  const BOB = "sk-oauth-bob";
  const CAROL = "sk-oauth-carol";
  const NO_USER = "sk-oauth-no-user";
  const NO_GRANT = "sk-oauth-no-grant";

  const url = (server: string) => `${stack.base}/mcp/${server}`;

  /** `caller` is signed in to `server` as `user`, whatever ran before. */
  const signedIn = async (target: OAuthMcpUpstream, server: string, caller: string, user: string) => {
    const reset = await fetch(`${url(server)}/oauth/credential`, { method: "DELETE", headers: key(caller) });
    expect(reset.status).toBe(204);
    target.user = user;
    const { client } = await signIn(url(server), key(caller));
    await client.close();
  };

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    stack = await startStack();
    main = await upstream();
    shortLived = await upstream({ expiresIn: 30 });
    flaky = await upstream();
    revoking = await upstream();
    staticClient = await upstream({
      discovery: false,
      registration: false,
      staticClient: { id: "configured-client", secret: "configured-secret" },
    });
    registering = await upstream();
    aggregatedA = await upstream();
    aggregatedB = await upstream();
    selfRevoke = await upstream();
    noRefresh = await upstream({ noRefreshTokens: true });
    jira = await upstream({
      discovery: false,
      registration: false,
      expiresIn: 7200,
      oauthPathPrefix: "/jira/rest/oauth2/latest",
      staticClient: { id: "jira-client", secret: "jira-secret" },
    });

    await addServer(stack, "main", main, { forward_client_headers: ["authorization"] });
    await addServer(stack, "shortlived", shortLived);
    await addServer(stack, "flaky", flaky);
    await addServer(stack, "revoking", revoking);
    await addServer(stack, "static", staticClient, {
      client_id: "configured-client",
      secret: "configured-secret",
      authorization_url: `${staticClient.url}/authorize`,
      token_url: `${staticClient.url}/token`,
      scopes: ["tools:read"],
    });
    await addServer(stack, "jira", jira, {
      client_id: "jira-client",
      secret: "jira-secret",
      authorization_url: `${jira.url}/jira/rest/oauth2/latest/authorize`,
      token_url: `${jira.url}/jira/rest/oauth2/latest/token`,
      scopes: ["READ"],
    });
    await addServer(stack, "registering", registering);
    await addServer(stack, "aggregateda", aggregatedA);
    await addServer(stack, "aggregatedb", aggregatedB);
    await addServer(stack, "selfrevoke", selfRevoke);
    await addServer(stack, "norefresh", noRefresh);
    await addServer(stack, "anonymousok", main);

    const anonymousPrincipal = await addKey(stack, "sk-oauth-anon-principal");
    await stack.seed.update("mcp_auth_settings", randomUUID(), {
      resource_url: `${stack.base}/mcp`,
      anonymous: {
        enabled: true,
        servers: ["anonymousok"],
        source_cidrs: ["127.0.0.0/8", "::1/128"],
        api_key_id: anonymousPrincipal.id,
      },
    });
    await addKey(stack, ALICE, { user_id: "alice" });
    await addKey(stack, ALICE2, { user_id: "alice" });
    await addKey(stack, BOB, { user_id: "bob" });
    await addKey(stack, CAROL, { user_id: "carol" });
    await addKey(stack, NO_GRANT, { user_id: "erin", mcp_access: { allow: [] } });
    await addKey(stack, NO_USER);
    await gateOnKey(stack, NO_USER);
  }, 90_000);

  afterAll(async () => {
    await stack?.app.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("a caller with a user and no token gets the challenge on initialize", async () => {
    if (!etcdReachable) return;
    const res = await mcpPost(url("main"), key(BOB));
    expect(res.status).toBe(401);
    expect(res.headers.get("www-authenticate")).toBe(
      `Bearer resource_metadata="${stack.base}/.well-known/oauth-protected-resource/mcp/main"`,
    );
    expect(await errorCode(res)).toBe("mcp_oauth_authorization_required");
    expect(main.mcpAuthorizations).toHaveLength(0);
  });

  test("the SDK signs in through discovery, and tools run with the user's upstream token", async () => {
    if (!etcdReachable) return;
    main.user = "alice";
    const { state, client } = await signIn(url("main"), key(ALICE));
    expect(await whoami(client)).toBe("alice");

    // The client's token is a placeholder: not the upstream's, not reusable
    // for a refresh, and it never reaches the upstream.
    const placeholder = state.saved!.access_token;
    expect(placeholder).not.toMatch(/^at-/);
    expect(state.saved!.refresh_token).toBeUndefined();
    expect(state.saved!.token_type.toLowerCase()).toBe("bearer");
    expect(state.saved!.expires_in).toBe(365 * 24 * 3600);
    const carried = main.mcpAuthorizations.filter((a) => a !== undefined);
    expect(carried.length).toBeGreaterThan(0);
    expect(carried.every((a) => a.startsWith("Bearer at-alice-"))).toBe(true);
    // Upstream, the gateway is the client, with its own callback and with
    // the client's PKCE challenge.
    const authorization = main.authorizeRequests.at(-1)!;
    expect(authorization.redirect_uri).toBe(`${stack.base}/mcp-oauth/callback`);
    expect(authorization.client_id).toMatch(/^registered-/);
    expect(authorization.resource).toBe(main.mcpUrl);
    await client.close();
  });

  test("the upstream's code stays in the gateway, and the gateway's own works once", async () => {
    if (!etcdReachable) return;
    main.user = "carol";
    const VERIFIER = "verifier-of-sufficient-length-0123456789";
    const challenge = createHash("sha256").update(VERIFIER).digest("base64url");
    const authorize = authorizeUrl(url("main"), {
      client_id: "main",
      code_challenge: challenge,
      state: "client-state",
    });
    const landed = await followBrowser(authorize);
    const code = landed.searchParams.get("code")!;
    expect(landed.searchParams.get("state")).toBe("client-state");
    expect(code).not.toMatch(/^upstream-code-/);

    const redeem = (overrides: Record<string, string> = {}) =>
      fetch(`${url("main")}/oauth/token`, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded", ...key(CAROL) },
        body: new URLSearchParams({
          grant_type: "authorization_code",
          code,
          client_id: "main",
          redirect_uri: REDIRECT,
          code_verifier: VERIFIER,
          ...overrides,
        }),
      });
    const before = main.tokenRequests.length;
    const first = await redeem();
    expect(first.status).toBe(200);
    expect(main.tokenRequests.length).toBe(before + 1);
    expect(main.tokenRequests.at(-1)!.code).toMatch(/^upstream-code-/);
    expect(main.tokenRequests.at(-1)!.code_verifier).toBe(VERIFIER);
    const tokens = await first.json();
    expect(tokens).not.toHaveProperty("refresh_token");
    expect(tokens.access_token).not.toMatch(/^at-/);

    // Single use, refused before it could reach the upstream.
    const again = await redeem();
    expect(again.status).toBe(400);
    expect(((await again.json()) as { error: string }).error).toBe("invalid_grant");

    // A code in the wrong hands, or sent back for the wrong client or
    // redirect URI, is worthless too.
    const wrong: Record<string, string>[] = [
      { code_verifier: "not-the-verifier-of-this-flow-0123456789" },
      { redirect_uri: "http://127.0.0.1:1/other" },
      { client_id: "other" },
    ];
    for (const overrides of wrong) {
      const another = await followBrowser(authorize);
      const res = await redeem({ code: another.searchParams.get("code")!, ...overrides });
      expect(res.status, JSON.stringify(overrides)).toBe(400);
    }
    expect(main.tokenRequests.length).toBe(before + 1);
  });

  test("the callback is bound to the browser that started the flow, and a code waits for consent", async () => {
    if (!etcdReachable) return;
    const browser = new Browser();
    const toUpstream = await browser.request(authorizeUrl(url("main")).toString());
    const toCallback = await browser.request(toUpstream.headers.get("location")!);
    const callback = toCallback.headers.get("location")!;

    // Another browser, with no flow of its own, gets nowhere with that link.
    const elsewhere = await fetch(callback, { redirect: "manual" });
    expect(elsewhere.status).toBe(400);
    expect(elsewhere.headers.get("location")).toBeNull();

    // This one is shown the consent page, and no code yet.
    const consent = await browser.request(callback);
    expect(consent.status).toBe(200);
    expect(consent.headers.get("location")).toBeNull();
    const page = await consent.text();
    expect(page).toContain("Allow");
    expect(page).toContain("main");
    expect(page).toContain(new URL(REDIRECT).origin);
    expect(page).not.toMatch(/upstream-code-/);

    expect((await fetch(`${stack.base}/mcp-oauth/callback?state=unknown&code=x`)).status).toBe(400);
  });

  test("another client of the same user needs no sign-in, and another user does", async () => {
    if (!etcdReachable) return;
    await signedIn(main, "main", ALICE, "alice");
    const bobReset = await fetch(`${url("main")}/oauth/credential`, { method: "DELETE", headers: key(BOB) });
    expect(bobReset.status).toBe(204);
    // Same member, another key, no OAuth configured at all.
    const { client } = await connect(url("main"), key(ALICE2));
    expect(await whoami(client)).toBe("alice");
    await client.close();

    main.user = "bob";
    const bob = await signIn(url("main"), key(BOB));
    expect(await whoami(bob.client)).toBe("bob");
    await bob.client.close();
    const stillAlice = await connect(url("main"), key(ALICE));
    expect(await whoami(stillAlice.client)).toBe("alice");
    await stillAlice.client.close();
  });

  test("the client's own Authorization never reaches the upstream, even when forwarding names it", async () => {
    if (!etcdReachable) return;
    await signedIn(main, "main", ALICE, "alice");
    const before = main.mcpAuthorizations.length;
    const { client } = await connect(url("main"), {
      ...key(ALICE),
      authorization: "Bearer client-supplied",
    });
    await whoami(client);
    await client.close();
    const seen = main.mcpAuthorizations.slice(before);
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((a) => a?.startsWith("Bearer at-alice-"))).toBe(true);
  });

  test("a token under a minute from expiry is refreshed before use, and once under concurrency", async () => {
    if (!etcdReachable) return;
    shortLived.user = "alice";
    const { client } = await signIn(url("shortlived"), key(ALICE));
    const grants = shortLived.refreshGrants().length;
    expect(await whoami(client)).toBe("alice");
    expect(shortLived.refreshGrants().length).toBeGreaterThan(grants);
    expect(shortLived.refreshGrants().at(-1)!.client_id).toMatch(/^registered-/);

    // Rotation makes a double refresh fatal: the loser would present a dead
    // refresh token, and the user would have to sign in again.
    shortLived.refreshDelayMs = 300;
    const results = await Promise.all(Array.from({ length: 5 }, () => whoami(client)));
    shortLived.refreshDelayMs = 0;
    expect(results).toEqual(Array(5).fill("alice"));
    await client.close();
  });

  test("an upstream 401 forces one refresh, and the caller does not notice", async () => {
    if (!etcdReachable) return;
    revoking.user = "alice";
    const { client } = await signIn(url("revoking"), key(ALICE));
    expect(await whoami(client)).toBe("alice");
    const grants = revoking.refreshGrants().length;
    revoking.revokeAccessTokens("alice");
    expect(await whoami(client)).toBe("alice");
    expect(revoking.refreshGrants().length).toBe(grants + 1);
    await client.close();
  });

  test("an upstream failure on refresh answers 503 and keeps the connection", async () => {
    if (!etcdReachable) return;
    flaky.user = "alice";
    const { client } = await signIn(url("flaky"), key(ALICE));
    await client.close();
    flaky.revokeAccessTokens("alice");
    flaky.refreshAnswer = { status: 503, error: "temporarily_unavailable" };

    const call = { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "whoami", arguments: {} } };
    const res = await mcpPost(url("flaky"), key(ALICE), call);
    expect(res.status).toBe(503);
    expect(res.headers.get("www-authenticate")).toBeNull();
    expect(await errorCode(res)).toBe("mcp_oauth_unavailable");

    flaky.refreshAnswer = "ok";
    const recovered = await connect(url("flaky"), key(ALICE));
    expect(await whoami(recovered.client)).toBe("alice");
    await recovered.client.close();
  });

  test("a refused refresh deletes the record and asks the client to sign in again", async () => {
    if (!etcdReachable) return;
    flaky.user = "dave";
    const state = new ClientAuthState();
    const { client } = await signIn(url("flaky"), key(BOB), state);
    expect(await whoami(client)).toBe("dave");
    flaky.revokeAccessTokens("dave");
    flaky.refreshAnswer = { status: 400, error: "invalid_grant" };
    const opens = state.browserOpens.length;

    // The SDK holds a placeholder and no refresh token, so the only thing it
    // can do with this 401 is to open the browser again.
    await expect(client.callTool({ name: "whoami", arguments: {} })).rejects.toThrow();
    expect(state.browserOpens.length).toBe(opens + 1);
    await client.close();

    // Deleted, not just refused: the refusal is not asked for again.
    flaky.refreshAnswer = "ok";
    const grants = flaky.refreshGrants().length;
    const res = await mcpPost(url("flaky"), key(BOB));
    expect(res.status).toBe(401);
    expect(flaky.refreshGrants().length).toBe(grants);

    // And signing in again works, from the state the client already holds.
    const again = await signIn(url("flaky"), key(BOB), state);
    expect(await whoami(again.client)).toBe("dave");
    await again.client.close();
  });

  test("a token the upstream keeps refusing after a good refresh ends in a sign-in", async () => {
    if (!etcdReachable) return;
    revoking.user = "bob";
    const { client } = await signIn(url("revoking"), key(BOB));
    await client.close();
    revoking.rejectEveryToken = true;
    const grants = revoking.refreshGrants().length;
    const res = await mcpPost(url("revoking"), key(BOB), { jsonrpc: "2.0", id: 2, method: "tools/list" });
    revoking.rejectEveryToken = false;
    expect(res.status).toBe(401);
    expect(res.headers.get("www-authenticate")).toContain("oauth-protected-resource/mcp/revoking");
    expect(revoking.refreshGrants().length).toBe(grants + 1);
    expect((await mcpPost(url("revoking"), key(BOB))).status).toBe(401);
  });

  test("a token with no refresh token goes straight back to sign-in on a 401", async () => {
    if (!etcdReachable) return;
    noRefresh.user = "alice";
    const { client } = await signIn(url("norefresh"), key(ALICE));
    expect(await whoami(client)).toBe("alice");
    await client.close();
    noRefresh.revokeAccessTokens("alice");
    const res = await mcpPost(url("norefresh"), key(ALICE), { jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(res.status).toBe(401);
    expect(noRefresh.refreshGrants()).toHaveLength(0);
    expect((await mcpPost(url("norefresh"), key(ALICE))).status).toBe(401);
  });

  test("a credential with no user gets an explicit error, and no browser", async () => {
    if (!etcdReachable) return;
    const res = await mcpPost(url("main"), key(NO_USER));
    expect(res.status).toBe(403);
    expect(res.headers.get("www-authenticate")).toBeNull();
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe("mcp_oauth_no_subject");
    expect(body.error.message).toContain("no user");

    const state = new ClientAuthState();
    await expect(connect(url("main"), key(NO_USER), state)).rejects.toThrow();
    expect(state.browserOpens).toHaveLength(0);

    const token = await fetch(`${url("main")}/oauth/token`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded", ...key(NO_USER) },
      body: new URLSearchParams({ grant_type: "authorization_code", code: "x" }),
    });
    expect(token.status).toBe(400);
    const error = (await token.json()) as { error: string; error_description: string };
    expect(error.error_description).toContain("no user");
  });

  test("an anonymous caller has no user either", async () => {
    if (!etcdReachable) return;
    const res = await mcpPost(url("anonymousok"), {});
    expect(res.status).toBe(403);
    expect(await errorCode(res)).toBe("mcp_oauth_no_subject");
  });

  test("a caller the ACL keeps from the server is not asked to sign in", async () => {
    if (!etcdReachable) return;
    const opened = await mcpPost(url("main"), key(NO_GRANT));
    expect(opened.status).toBe(200);
    const listed = await mcpPost(url("main"), key(NO_GRANT), { jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(listed.status).toBe(200);
    expect(((await listed.json()) as { result: { tools: unknown[] } }).result.tools).toEqual([]);
  });

  test("an authorization response from another issuer is refused", async () => {
    if (!etcdReachable) return;
    const browser = new Browser();
    const toUpstream = new URL(
      (await browser.request(authorizeUrl(url("main")).toString())).headers.get("location")!,
    );
    const back = await browser.request(
      `${stack.base}/mcp-oauth/callback?state=${toUpstream.searchParams.get("state")}&code=x&iss=https://other.example`,
    );
    const landed = new URL(back.headers.get("location")!);
    expect(landed.searchParams.get("error")).toBe("access_denied");
    expect(landed.searchParams.get("code")).toBeNull();
  });

  test("moving a server's address asks its users to sign in rather than sending their tokens there", async () => {
    if (!etcdReachable) return;
    const original = await upstream();
    const moved = await upstream();
    const id = randomUUID();
    const row = (target: OAuthMcpUpstream) => ({
      name: "moving",
      type: "mcp",
      url: target.mcpUrl,
      auth_type: "oauth2_authorization_code",
    });
    await stack.seed.update("mcp_servers", id, row(original));
    await waitConfigPropagation(
      async () => (await fetch(`${stack.base}/.well-known/oauth-protected-resource/mcp/moving`)).status === 200,
    );
    original.user = "alice";
    const { client } = await signIn(url("moving"), key(ALICE));
    expect(await whoami(client)).toBe("alice");
    await client.close();

    // The scope is only there to say when the moved row has landed.
    await stack.seed.update("mcp_servers", id, { ...row(moved), scopes: ["moved"] });
    await waitConfigPropagation(async () => {
      const document = await fetch(`${stack.base}/.well-known/oauth-protected-resource/mcp/moving`);
      return ((await document.json()) as { scopes_supported?: string[] }).scopes_supported?.[0] === "moved";
    });
    const res = await mcpPost(url("moving"), key(ALICE), { jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(res.status).toBe(401);
    expect(moved.mcpAuthorizations).toEqual([]);
    expect(moved.tokenRequests).toEqual([]);
  });

  test("a token request with no valid credential is an OAuth client error", async () => {
    if (!etcdReachable) return;
    for (const headers of [{}, key("sk-oauth-nobody")]) {
      const res = await fetch(`${url("main")}/oauth/token`, {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded", ...headers },
        body: new URLSearchParams({ grant_type: "authorization_code", code: "x" }),
      });
      expect(res.status).toBe(401);
      expect(((await res.json()) as { error: string }).error).toBe("invalid_client");
    }
  });

  test("a caller removes its own connection and nobody else's", async () => {
    if (!etcdReachable) return;
    selfRevoke.user = "alice";
    const alice = await signIn(url("selfrevoke"), key(ALICE));
    await alice.client.close();
    selfRevoke.user = "bob";
    const bob = await signIn(url("selfrevoke"), key(BOB));
    await bob.client.close();

    const remove = (headers: Record<string, string>) =>
      fetch(`${url("selfrevoke")}/oauth/credential`, { method: "DELETE", headers });
    expect((await remove({})).status).toBe(401);
    expect((await remove(key(NO_USER))).status).toBe(403);
    expect((await remove(key(ALICE))).status).toBe(204);
    expect((await remove(key(ALICE))).status).toBe(204);

    expect((await mcpPost(url("selfrevoke"), key(ALICE))).status).toBe(401);
    expect((await mcpPost(url("selfrevoke"), key(ALICE2))).status).toBe(401);
    const stillBob = await connect(url("selfrevoke"), key(BOB));
    expect(await whoami(stillBob.client)).toBe("bob");
    await stillBob.client.close();
  });

  test("the aggregated endpoint omits an unconnected server and names it on a call", async () => {
    if (!etcdReachable) return;
    aggregatedA.user = "alice";
    const connected = await signIn(url("aggregateda"), key(ALICE));
    await connected.client.close();

    const { client } = await connect(`${stack.base}/mcp`, key(ALICE));
    const tools = (await client.listTools()).tools.map((t) => t.name);
    expect(tools).toContain("aggregateda__whoami");
    expect(tools).not.toContain("aggregatedb__whoami");

    const describeCall = (c: Client, name: string) =>
      c
        .callTool({ name, arguments: {} })
        .then((r) => JSON.stringify(r))
        .catch((e: Error) => e.message);
    const call = await describeCall(client, "aggregatedb__whoami");
    expect(call).toContain("aggregatedb");
    expect(call).toContain("/mcp/aggregatedb");
    expect(aggregatedB.mcpAuthorizations).toHaveLength(0);

    const res = await client.callTool({ name: "aggregateda__whoami", arguments: {} });
    expect((res.content as { text: string }[])[0].text).toBe("alice");
    await client.close();

    // A credential with no user sees the same list, and is told why.
    const noUser = await connect(`${stack.base}/mcp`, key(NO_USER));
    expect((await noUser.client.listTools()).tools.map((t) => t.name)).not.toContain("aggregateda__whoami");
    const refused = await describeCall(noUser.client, "aggregateda__whoami");
    expect(refused).toContain("/mcp/aggregateda");
    expect(refused).toContain("no user");
    await noUser.client.close();
  });

  test("a configured client signs in with its own id and secret, without any discovery", async () => {
    if (!etcdReachable) return;
    staticClient.user = "alice";
    const { client } = await signIn(url("static"), key(ALICE));
    expect(await whoami(client)).toBe("alice");
    await client.close();
    expect(staticClient.registrations).toHaveLength(0);
    const authorization = staticClient.authorizeRequests.at(-1)!;
    expect(authorization.client_id).toBe("configured-client");
    expect(authorization.scope).toBe("tools:read");
    expect(authorization.resource).toBeUndefined();
    const exchange = staticClient.tokenRequests.at(-1)!;
    expect(exchange.client_id).toBe("configured-client");
    expect(exchange.client_secret).toBe("configured-secret");
  });

  test("a pre-registered client on an upstream publishing no metadata needs none of it", async () => {
    if (!etcdReachable) return;
    jira.user = "alice";
    const { state, client } = await signIn(url("jira"), key(ALICE));
    expect(await whoami(client)).toBe("alice");

    const authorization = jira.authorizeRequests.at(-1)!;
    expect(authorization.client_id).toBe("jira-client");
    expect(authorization.scope).toBe("READ");
    expect(authorization.code_challenge_method).toBe("S256");
    expect(authorization.code_challenge).toBe(
      createHash("sha256").update(state.verifier).digest("base64url"),
    );
    expect(authorization).not.toHaveProperty("resource");

    // PKCE is verified upstream, against the client's verifier; the secret
    // travels in the POST body and nowhere else.
    const exchange = jira.tokenRequests.at(-1)!;
    expect(exchange.grant_type).toBe("authorization_code");
    expect(exchange.code_verifier).toBe(state.verifier);
    expect(exchange.client_secret).toBe("jira-secret");
    expect(exchange).not.toHaveProperty("resource");
    expect(jira.tokenRequestQueries.at(-1)).toBe("");

    // Nothing was asked of it but the two endpoints and the MCP server.
    expect(jira.requestsSeen.filter((r) => r.includes(".well-known"))).toEqual([]);
    expect(jira.requestsSeen).not.toContain("GET /mcp");
    expect(jira.registrations).toHaveLength(0);

    // A refresh invalidates what it replaces, and carries the same client.
    jira.revokeAccessTokens("alice");
    expect(await whoami(client)).toBe("alice");
    const refresh = jira.refreshGrants().at(-1)!;
    expect(refresh.refresh_token).toMatch(/^rt-alice-/);
    expect(refresh.client_id).toBe("jira-client");
    expect(refresh.client_secret).toBe("jira-secret");
    expect(refresh.redirect_uri).toBe(`${stack.base}/mcp-oauth/callback`);
    expect(refresh).not.toHaveProperty("resource");
    expect(refresh).not.toHaveProperty("scope");

    // The rotated token is the one that works next.
    jira.revokeAccessTokens("alice");
    expect(await whoami(client)).toBe("alice");
    const spent = jira.refreshGrants().map((f) => f.refresh_token);
    expect(new Set(spent).size).toBe(2);
    expect(jira.tokenRequests.every((f) => !(f.scope ?? "").includes("offline_access"))).toBe(true);
    expect(authorization.scope).not.toContain("offline_access");
    await client.close();
  });

  test("without a client id the gateway registers once, as a public client, for everyone", async () => {
    if (!etcdReachable) return;
    registering.user = "alice";
    const alice = await signIn(url("registering"), key(ALICE));
    await alice.client.close();
    registering.user = "bob";
    const bob = await signIn(url("registering"), key(BOB));
    await bob.client.close();
    expect(registering.registrations).toHaveLength(1);
    expect(registering.registrations[0].token_endpoint_auth_method).toBe("none");
    expect(registering.registrations[0].redirect_uris).toEqual([`${stack.base}/mcp-oauth/callback`]);
    expect(new Set(registering.authorizeRequests.map((r) => r.client_id)).size).toBe(1);
    expect(registering.tokenRequests.at(-1)!.client_secret).toBeUndefined();
  });

  test("redirect URIs follow the rules and a downgrade of PKCE is refused", async () => {
    if (!etcdReachable) return;
    const authorize = (params: Record<string, string>) =>
      fetch(authorizeUrl(url("main"), params), { redirect: "manual" });
    for (const redirect_uri of [
      "http://evil.example/cb",
      "http://user:pw@127.0.0.1/cb",
      "http://127.0.0.1/cb#fragment",
      "myapp://cb",
    ]) {
      const res = await authorize({ redirect_uri });
      expect(res.status, redirect_uri).toBe(400);
      expect(res.headers.get("location"), redirect_uri).toBeNull();
    }
    expect((await authorize({ redirect_uri: "cursor://anysphere.cursor-mcp/oauth/callback" })).status).toBe(302);
    // Any https address may be asked for: the consent page names it.
    expect((await authorize({ redirect_uri: "https://chat.example.org/oauth/cb" })).status).toBe(302);

    const downgrades: Record<string, string>[] = [{ code_challenge_method: "plain" }, { code_challenge: "" }];
    for (const params of downgrades) {
      const res = await authorize(params);
      const location = new URL(res.headers.get("location")!);
      expect(location.origin, JSON.stringify(params)).toBe(new URL(REDIRECT).origin);
      expect(location.searchParams.get("error"), JSON.stringify(params)).toBe("invalid_request");
    }
  });

  test("the upstream's refusal goes back to the client with the client's state", async () => {
    if (!etcdReachable) return;
    const browser = new Browser();
    const authorize = authorizeUrl(url("main"), { state: "mine" });
    const toUpstream = new URL((await browser.request(authorize.toString())).headers.get("location")!);
    const back = await browser.request(
      `${stack.base}/mcp-oauth/callback?state=${toUpstream.searchParams.get("state")}&error=access_denied`,
    );
    const landed = new URL(back.headers.get("location")!);
    expect(landed.searchParams.get("error")).toBe("access_denied");
    expect(landed.searchParams.get("state")).toBe("mine");
    expect(landed.searchParams.get("code")).toBeNull();
  });

  test("the discovery documents are what the client relies on", async () => {
    if (!etcdReachable) return;
    const resource = await (await fetch(`${stack.base}/.well-known/oauth-protected-resource/mcp/main`)).json();
    expect(resource).toMatchObject({
      resource: `${stack.base}/mcp/main`,
      authorization_servers: [`${stack.base}/mcp/main`],
    });
    for (const path of [
      "/.well-known/oauth-authorization-server/mcp/main",
      "/mcp/main/.well-known/openid-configuration",
    ]) {
      expect(await (await fetch(`${stack.base}${path}`)).json(), path).toMatchObject({
        issuer: `${stack.base}/mcp/main`,
        authorization_endpoint: `${stack.base}/mcp/main/oauth/authorize`,
        token_endpoint: `${stack.base}/mcp/main/oauth/token`,
        registration_endpoint: `${stack.base}/mcp/main/oauth/register`,
        code_challenge_methods_supported: ["S256"],
        grant_types_supported: ["authorization_code", "refresh_token"],
        token_endpoint_auth_methods_supported: ["none"],
        jwks_uri: `${stack.base}/mcp/main/.well-known/jwks.json`,
      });
    }
    expect(await (await fetch(`${stack.base}/mcp/main/.well-known/jwks.json`)).json()).toEqual({ keys: [] });
    // Nothing is advertised for a server that does not run this flow.
    expect((await fetch(`${stack.base}/.well-known/oauth-protected-resource/mcp/absent`)).status).toBe(404);
    expect((await fetch(`${stack.base}/mcp/absent/.well-known/openid-configuration`)).status).toBe(404);

    const registered = await fetch(`${stack.base}/mcp/main/oauth/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ redirect_uris: ["http://127.0.0.1:1/x"], client_name: "n" }),
    });
    expect(registered.status).toBe(201);
    expect(await registered.json()).toMatchObject({
      client_id: "main",
      redirect_uris: ["http://127.0.0.1:1/x"],
    });
  });

  test("each route reports under its own endpoint label", async () => {
    if (!etcdReachable) return;
    const plainGets = [
      "/.well-known/oauth-protected-resource/mcp/main",
      "/.well-known/oauth-authorization-server/mcp/main",
      "/mcp/main/.well-known/openid-configuration",
      "/mcp/main/.well-known/jwks.json",
      "/mcp/main/oauth/authorize",
      "/mcp-oauth/callback",
      "/mcp-oauth/login/callback",
    ];
    await Promise.all(plainGets.map((path) => fetch(`${stack.base}${path}`, { redirect: "manual" })));
    await fetch(`${url("main")}/oauth/register`, { method: "POST", body: "{}" });
    await fetch(`${stack.base}/mcp-oauth/consent`, { method: "POST", redirect: "manual" });
    await fetch(`${url("main")}/oauth/token`, { method: "POST", headers: key(BOB) });
    await fetch(`${url("selfrevoke")}/oauth/credential`, { method: "DELETE", headers: key(NO_USER) });
    await mcpPost(url("main"), key(NO_USER));
    const samples = await scrapeMetrics(stack.app.metricsUrl);
    const endpoints = new Set(samples.map((s) => s.labels.endpoint));
    for (const endpoint of [
      "/.well-known/oauth-protected-resource/mcp/{server}",
      "/.well-known/oauth-authorization-server/mcp/{server}",
      "/mcp/{server}/.well-known/openid-configuration",
      "/mcp/{server}/.well-known/jwks.json",
      "/mcp/{server}/oauth/register",
      "/mcp/{server}/oauth/authorize",
      "/mcp/{server}/oauth/token",
      "/mcp/{server}/oauth/credential",
      "/mcp-oauth/callback",
      "/mcp-oauth/login/callback",
      "/mcp-oauth/consent",
      "/mcp/{server}",
    ]) {
      expect(endpoints, endpoint).toContain(endpoint);
    }
  });
});

describe("per-user OAuth behind a front gateway that strips a path prefix", () => {
  const PREFIX = "/team-a/gw";
  let stack: Stack;
  let front: Awaited<ReturnType<typeof startPrefixStrippingProxy>>;
  let main: OAuthMcpUpstream;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    stack = await startStack(async (proxyUrl) => {
      front = await startPrefixStrippingProxy(PREFIX, new URL(proxyUrl));
      return `${front.url}${PREFIX}`;
    });
    main = await new OAuthMcpUpstream().start();
    await addServer(stack, "main", main);
    await stack.seed.update("mcp_auth_settings", randomUUID(), {
      resource_url: `${stack.base}/mcp`,
    });
    await addKey(stack, "sk-prefix-alice", { user_id: "alice" });
    await gateOnKey(stack, "sk-prefix-alice");
  }, 60_000);

  afterAll(async () => {
    await front?.close();
    await stack?.app.exit();
    await main?.close();
  });

  test("the SDK completes the whole flow seeing only prefixed URLs", async () => {
    if (!etcdReachable) return;
    const { client } = await signIn(`${stack.base}/mcp/main`, key("sk-prefix-alice"));
    expect(await whoami(client)).toBe("alice");
    await client.close();
    // Everything that reached the gateway went through the prefix, including
    // the one metadata form a stripping proxy can still route.
    expect(front.seen).toContain(`${PREFIX}/mcp/main/.well-known/openid-configuration`);
    expect(front.seen).toContain(`${PREFIX}/.well-known/oauth-protected-resource/mcp/main`);
    expect(front.seen).toContain(`${PREFIX}/mcp-oauth/callback`);
    expect(main.authorizeRequests.at(-1)!.redirect_uri).toBe(`${stack.base}/mcp-oauth/callback`);
  });
});

describe("per-user OAuth for a caller with a JWT", () => {
  let stack: Stack;
  let idp: MockIdp;
  let main: OAuthMcpUpstream;
  let etcdReachable = false;

  const jwt = (subject: string) =>
    idp.sign({
      sub: subject,
      iss: idp.url,
      aud: "aisix-gateway",
      team: "shared",
      exp: Math.floor(Date.now() / 1000) + 600,
    });
  const url = () => `${stack.base}/mcp/main`;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    idp = await startMockIdp();
    stack = await startStack();
    main = await new OAuthMcpUpstream().start();
    await addServer(stack, "main", main);
    await stack.seed.createOidcProvider({
      name: "corp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });
    await stack.seed.update("mcp_auth_settings", randomUUID(), {
      resource_url: `${stack.base}/mcp`,
    });
    // One key for everyone, owned by a member: the JWT's subject still decides.
    const shared = await addKey(stack, "sk-jwt-shared", { user_id: "owner-of-the-shared-key" });
    await stack.seed.createClaimMapping({
      name: "everyone",
      jwt_provider: "corp",
      priority: 100,
      match: [{ claim: "team", op: "exact", values: ["shared"] }],
      resolve: { api_key_id: shared.id },
    });
    await addKey(stack, "sk-jwt-gate");
    await gateOnKey(stack, "sk-jwt-gate");
    await waitConfigPropagation(async () => {
      const res = await fetch(`${stack.app.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${jwt("gate")}` },
      });
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await stack?.app.exit();
    await main?.close();
    await idp?.close();
  });

  test("two subjects on one claim-mapped key get separate connections", async () => {
    if (!etcdReachable) return;
    const headers = (subject: string) => ({ "x-aisix-api-key": jwt(subject) });
    main.user = "upstream-alice";
    const alice = await signIn(url(), headers("alice"));
    expect(await whoami(alice.client)).toBe("upstream-alice");
    await alice.client.close();

    // Bob maps to the same key and must not be served Alice's token.
    expect((await mcpPost(url(), headers("bob"))).status).toBe(401);
    main.user = "upstream-bob";
    const bob = await signIn(url(), headers("bob"));
    expect(await whoami(bob.client)).toBe("upstream-bob");
    await bob.client.close();

    const aliceAgain = await connect(url(), headers("alice"));
    expect(await whoami(aliceAgain.client)).toBe("upstream-alice");
    await aliceAgain.client.close();

    // And the key's own owner, presenting the key itself, is a third person.
    expect((await mcpPost(url(), key("sk-jwt-shared"))).status).toBe(401);
  });
});

describe("per-user OAuth with no public address", () => {
  let stack: Stack;
  let main: OAuthMcpUpstream;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    stack = await startStack();
    main = await new OAuthMcpUpstream().start();
    await addServer(stack, "main", main);
    await addKey(stack, "sk-nobase", { user_id: "alice" });
    await gateOnKey(stack, "sk-nobase");
  }, 60_000);

  afterAll(async () => {
    await stack?.app.exit();
    await main?.close();
  });

  test("no resource_url is a 500 on the server, not a challenge", async () => {
    if (!etcdReachable) return;
    const res = await mcpPost(`${stack.app.proxyUrl}/mcp/main`, key("sk-nobase"));
    expect(res.status).toBe(500);
    expect(res.headers.get("www-authenticate")).toBeNull();
    expect(await errorCode(res)).toBe("mcp_oauth_misconfigured");
    expect((await fetch(`${stack.app.proxyUrl}/.well-known/oauth-protected-resource/mcp/main`)).status).toBe(404);
  });
});

describe("per-user OAuth under the SSRF settings", () => {
  let app: SpawnedApp;
  let main: OAuthMcpUpstream;
  let elsewhere: OAuthMcpUpstream;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    elsewhere = await new OAuthMcpUpstream().start();
    main = await new OAuthMcpUpstream({ advertise: { registrationEndpoint: `${elsewhere.url}/register` } }).start();
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.update("mcp_servers", randomUUID(), {
      name: "main",
      type: "mcp",
      url: main.mcpUrl,
      auth_type: "oauth2_authorization_code",
    });
    await seed.update("mcp_auth_settings", randomUUID(), { resource_url: `${app.proxyUrl}/mcp` });
    await seed.createApiKey({
      key_hash: hash("sk-ssrf"),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
      user_id: "alice",
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app.proxyUrl}/v1/models`, { headers: { authorization: "Bearer sk-ssrf" } });
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await main?.close();
    await elsewhere?.close();
  });

  test("an endpoint a document names on another private address is never contacted, and the client is told", async () => {
    if (!etcdReachable) return;
    const res = await fetch(authorizeUrl(`${app.proxyUrl}/mcp/main`), { redirect: "manual" });
    expect(res.status).toBe(302);
    const landed = new URL(res.headers.get("location")!);
    expect(landed.origin).toBe(new URL(REDIRECT).origin);
    expect(landed.searchParams.get("error")).toBe("server_error");
    expect(elsewhere.requestsSeen).toHaveLength(0);
    expect(main.requestsSeen).toContain("GET /.well-known/oauth-authorization-server");
    expect(main.mcpAuthorizations).toHaveLength(0);
  });
});
