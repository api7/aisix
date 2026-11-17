import { createHash, randomUUID } from "node:crypto";

import { afterAll, beforeAll, describe, expect, test } from "vitest";

import {
  Browser,
  EtcdClient,
  OAUTH_CLIENT_REDIRECT as REDIRECT,
  OAuthMcpUpstream,
  SeedClient,
  spawnApp,
  startMockOidcProvider,
  waitConfigPropagation,
  type MockOidcProvider,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the scopes the gateway shows its own MCP clients, as the authorization
// server it is for an `oauth2_authorization_code` server. The server's
// configured scopes are what the gateway asks of the upstream; to a client the
// gateway grants nothing by them, and above all not `openid`, since it issues
// no ID token.

const hash = (key: string) => createHash("sha256").update(key).digest("hex");

const VERIFIER = "client-verifier-of-sufficient-length-0123456789";
const CHALLENGE = createHash("sha256").update(VERIFIER).digest("base64url");
const UPSTREAM_SCOPES = ["openid", "profile", "offline_access", "files:read"];

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

const tokenRequest = (serverUrl: string, form: Record<string, string>) =>
  fetch(`${serverUrl}/oauth/token`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(form),
  });

const getJson = async (url: string) => {
  const res = await fetch(url);
  expect(res.status, url).toBe(200);
  return (await res.json()) as Record<string, unknown>;
};

interface Gateway {
  app: SpawnedApp;
  seed: SeedClient;
}

async function startGateway(): Promise<Gateway> {
  const app = await spawnApp();
  return { app, seed: new SeedClient(new EtcdClient(), app.etcdPrefix) };
}

/** Servers `scoped` (with `UPSTREAM_SCOPES`) and `bare` (none), then a key to gate on. */
async function seedServers(gw: Gateway, upstream: OAuthMcpUpstream, key: string): Promise<void> {
  await gw.seed.update("mcp_auth_settings", randomUUID(), { resource_url: `${gw.app.proxyUrl}/mcp` });
  for (const [name, scopes] of [["scoped", UPSTREAM_SCOPES], ["bare", undefined]] as const) {
    await gw.seed.update("mcp_servers", randomUUID(), {
      name,
      type: "mcp",
      url: upstream.mcpUrl,
      auth_type: "oauth2_authorization_code",
      ...(scopes ? { scopes } : {}),
    });
  }
  await gw.seed.createApiKey({ key_hash: hash(key), allowed_models: [], mcp_access: { allow: ["*"] } });
  await waitConfigPropagation(async () => {
    const res = await fetch(`${gw.app.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${key}` } });
    return res.status === 200;
  });
}

async function expectMetadata(base: string): Promise<void> {
  const prm = await getJson(`${base}/.well-known/oauth-protected-resource/mcp/scoped`);
  expect(prm.scopes_supported).toEqual(["profile", "files:read"]);
  const barePrm = await getJson(`${base}/.well-known/oauth-protected-resource/mcp/bare`);
  expect(barePrm).not.toHaveProperty("scopes_supported");

  for (const server of ["scoped", "bare"]) {
    for (const path of [
      `${base}/.well-known/oauth-authorization-server/mcp/${server}`,
      `${base}/mcp/${server}/.well-known/openid-configuration`,
    ]) {
      const document = await getJson(path);
      expect(document.scopes_supported, path).toEqual([]);
      expect(document.issuer, path).toBe(`${base}/mcp/${server}`);
      expect(document.grant_types_supported, path).toEqual(["authorization_code", "refresh_token"]);
    }
  }
}

describe("advertised scopes when the gateway signs users in itself", () => {
  let etcdReachable = false;
  let gw: Gateway;
  let provider: MockOidcProvider;
  let upstream: OAuthMcpUpstream;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    provider = await startMockOidcProvider();
    upstream = await new OAuthMcpUpstream().start();
    gw = await startGateway();
    const key = `sk-scopes-${randomUUID()}`;
    const access = await gw.seed.createApiKey({
      key_hash: hash(`sk-session-${randomUUID()}`),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });
    await gw.seed.createOidcProvider({
      name: "corp",
      issuer: provider.url,
      audiences: ["some-api"],
      login: { client_id: provider.clientId, scopes: ["profile"] },
    });
    await gw.seed.createClaimMapping({
      name: "everyone",
      jwt_provider: "corp",
      priority: 100,
      match: [{ claim: "sub", op: "exact", values: ["alice"] }],
      resolve: { api_key_id: access.id },
    });
    await seedServers(gw, upstream, key);
  }, 60_000);

  afterAll(async () => {
    await gw?.app.exit();
    await upstream?.close();
    await provider?.close();
  });

  test("metadata lists the upstream scopes minus openid and offline_access, and the AS documents none", async () => {
    if (!etcdReachable) return;
    await expectMetadata(gw.app.proxyUrl);
  });

  test("a client asking for openid is signed in, and no token response grants a scope or carries an ID token", async () => {
    if (!etcdReachable) return;
    provider.claims = { sub: "alice" };
    upstream.user = "up-alice";
    const server = `${gw.app.proxyUrl}/mcp/scoped`;
    const stop = await new Browser().walk(authorizeUrl(server, { scope: "openid profile" }));
    if (stop.kind !== "landed") throw new Error(`stopped: ${stop.status} ${stop.text}`);
    // What the gateway asks of the upstream is still the configured set.
    expect(upstream.authorizeRequests.at(-1)!.scope).toBe(UPSTREAM_SCOPES.join(" "));

    const issued = await tokenRequest(server, {
      grant_type: "authorization_code",
      code: stop.url.searchParams.get("code")!,
      client_id: "client",
      redirect_uri: REDIRECT,
      code_verifier: VERIFIER,
    });
    expect(issued.status).toBe(200);
    const tokens = (await issued.json()) as Record<string, unknown>;
    expect(typeof tokens.access_token).toBe("string");
    expect(tokens).not.toHaveProperty("scope");
    expect(tokens).not.toHaveProperty("id_token");

    const refreshed = await tokenRequest(server, {
      grant_type: "refresh_token",
      refresh_token: tokens.refresh_token as string,
      client_id: "client",
    });
    expect(refreshed.status).toBe(200);
    const renewed = (await refreshed.json()) as Record<string, unknown>;
    expect(typeof renewed.access_token).toBe("string");
    expect(renewed).not.toHaveProperty("scope");
    expect(renewed).not.toHaveProperty("id_token");
  });
});

describe("advertised scopes when sign-in is relayed with the caller's gateway credential", () => {
  let etcdReachable = false;
  let gw: Gateway;
  let upstream: OAuthMcpUpstream;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await new OAuthMcpUpstream().start();
    gw = await startGateway();
    await seedServers(gw, upstream, `sk-relay-${randomUUID()}`);
  }, 60_000);

  afterAll(async () => {
    await gw?.app.exit();
    await upstream?.close();
  });

  test("metadata lists the upstream scopes minus openid and offline_access, and the AS documents none", async () => {
    if (!etcdReachable) return;
    await expectMetadata(gw.app.proxyUrl);
  });

  test("a client asking for the advertised scopes still has the upstream asked for openid and offline_access", async () => {
    if (!etcdReachable) return;
    const server = `${gw.app.proxyUrl}/mcp/scoped`;
    const prm = await getJson(`${gw.app.proxyUrl}/.well-known/oauth-protected-resource/mcp/scoped`);
    const upstreamScope = async (scope: string) => {
      const res = await fetch(authorizeUrl(server, { scope }), { redirect: "manual" });
      expect(res.status).toBe(302);
      const location = new URL(res.headers.get("location")!);
      expect(`${location.origin}${location.pathname}`).toBe(`${upstream.url}/authorize`);
      return location.searchParams.get("scope")!.split(" ").sort();
    };
    expect(await upstreamScope((prm.scopes_supported as string[]).join(" "))).toEqual([...UPSTREAM_SCOPES].sort());
    // A narrower request of the client's own stays narrow otherwise.
    expect(await upstreamScope("files:read")).toEqual(["files:read", "offline_access", "openid"]);
  });
});
