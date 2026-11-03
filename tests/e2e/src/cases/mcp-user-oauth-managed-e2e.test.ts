import { createHash, randomUUID } from "node:crypto";

import { afterAll, beforeAll, describe, expect, test } from "vitest";

import {
  connectMcpClient,
  EtcdClient,
  OAuthMcpUpstream,
  SeedClient,
  signInMcpClient,
  spawnApp,
  startMockControlPlane,
  waitConfigPropagation,
  whoami,
  type MockControlPlane,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: per-user OAuth on managed gateways, whose records live behind the
// dp-manager's store API over mTLS. Two gateways of one environment share
// them, which is the point: a sign-in on one holds on the other, and a refresh
// token that rotates is only ever spent once between them.

const hash = (key: string) => createHash("sha256").update(key).digest("hex");
const ALICE = "sk-managed-oauth-alice";
const BOB = "sk-managed-oauth-bob";
const CAROL = "sk-managed-oauth-carol";

describe("per-user OAuth on managed gateways", () => {
  let cp: MockControlPlane;
  let upstream: OAuthMcpUpstream;
  let shortLived: OAuthMcpUpstream;
  let endsOldToken: OAuthMcpUpstream;
  const apps: SpawnedApp[] = [];
  let etcdReachable = false;

  const key = (name: string) => ({ "x-aisix-api-key": name });
  const headers = { "content-type": "application/json", accept: "application/json, text/event-stream" };
  const initialize = (url: string, caller: string) =>
    fetch(url, {
      method: "POST",
      headers: { ...headers, ...key(caller) },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: { protocolVersion: "2025-11-25", capabilities: {}, clientInfo: { name: "e2e", version: "0.1" } },
      }),
    });

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    cp = await startMockControlPlane();
    upstream = await new OAuthMcpUpstream().start();
    shortLived = await new OAuthMcpUpstream({ expiresIn: 30 }).start();
    endsOldToken = await new OAuthMcpUpstream({ expiresIn: 65, anonymousOnDeadToken: true }).start();
    for (let i = 0; i < 2; i++) {
      const app = await spawnApp({
        extra: {
          managed: cp.managed,
        },
        awaitProxyListener: false,
      });
      apps.push(app);
      const seed = new SeedClient(etcd, `${app.etcdPrefix}/${cp.envId}`);
      // Both gateways keep one id per server, as one environment would.
      await seed.update("mcp_servers", "11111111-1111-4111-8111-111111111111", {
        name: "main",
        type: "mcp",
        url: upstream.mcpUrl,
        auth_type: "oauth2_authorization_code",
      });
      await seed.update("mcp_servers", "22222222-2222-4222-8222-222222222222", {
        name: "short",
        type: "mcp",
        url: shortLived.mcpUrl,
        auth_type: "oauth2_authorization_code",
      });
      await seed.update("mcp_servers", "33333333-3333-4333-8333-333333333333", {
        name: "ends-old-token",
        type: "mcp",
        url: endsOldToken.mcpUrl,
        auth_type: "oauth2_authorization_code",
      });
      await seed.update("mcp_auth_settings", randomUUID(), { resource_url: `${app.proxyUrl}/mcp` });
      for (const [plaintext, user] of [
        [ALICE, "alice"],
        [BOB, "bob"],
        [CAROL, "carol"],
      ]) {
        await seed.createApiKey({
          key_hash: hash(plaintext),
          allowed_models: [],
          mcp_access: { allow: ["*"] },
          user_id: user,
        });
      }
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CAROL}` } }).catch(
          () => undefined,
        );
        return res?.status === 200;
      });
    }
  }, 120_000);

  afterAll(async () => {
    for (const app of apps) await app.exit();
    await upstream?.close();
    await shortLived?.close();
    await endsOldToken?.close();
    await cp?.close();
  });

  const url = (i: number, server = "main") => `${apps[i].proxyUrl}/mcp/${server}`;

  test("a sign-in on one gateway holds on the other, through the store API", async () => {
    if (!etcdReachable) return;
    upstream.user = "alice";
    const { client } = await signInMcpClient(url(0), key(ALICE));
    expect(await whoami(client)).toBe("alice");
    await client.close();

    const other = await connectMcpClient(url(1), key(ALICE));
    expect(await whoami(other.client)).toBe("alice");
    await other.client.close();

    const records = [...cp.store.entries()];
    const tokens = records.filter(([id]) => id.startsWith("mcp_oauth_token/"));
    expect(tokens).toHaveLength(1);
    const [id, record] = tokens[0];
    expect(id).toMatch(/^mcp_oauth_token\/[A-Za-z0-9_-]+$/);
    expect(id).not.toContain("alice");
    expect(record.ttlSeconds).toBe(365 * 24 * 3600);
    expect(record.labels).toEqual({
      mcp_server_id: "11111111-1111-4111-8111-111111111111",
      subject: "member:alice",
      member_id: "alice",
    });
    expect(JSON.parse(Buffer.from(record.value, "base64").toString("utf8")).access_token).toMatch(/^at-alice-/);

    const clients = records.filter(([id]) => id.startsWith("mcp_oauth_client/"));
    expect(clients).toHaveLength(1);
    expect(clients[0][1].ttlSeconds).toBe(365 * 24 * 3600);
    // Single-use records and leases are gone once used.
    expect(records.filter(([id]) => id.startsWith("mcp_oauth_lease/"))).toHaveLength(0);
    expect(cp.storeWrites.filter((w) => w.includes("mcp_oauth_flow"))).toEqual([]);
    expect(records.filter(([id]) => id.startsWith("mcp_oauth_code/"))).toHaveLength(0);
  });

  test("two gateways refreshing at once spend the rotating refresh token once", async () => {
    if (!etcdReachable) return;
    shortLived.user = "bob";
    const { client } = await signInMcpClient(url(0, "short"), key(BOB));
    await client.close();
    const clients = await Promise.all([0, 1, 0, 1].map((i) => connectMcpClient(url(i, "short"), key(BOB))));
    shortLived.refreshDelayMs = 500;
    const results = await Promise.all(clients.map(({ client }) => whoami(client)));
    shortLived.refreshDelayMs = 0;
    expect(results).toEqual(Array(4).fill("bob"));
    // A spent refresh token presented twice would have ended in `invalid_grant`.
    const spent = shortLived.refreshGrants().map((f) => f.refresh_token);
    expect(new Set(spent).size).toBe(spent.length);
    for (const { client } of clients) await client.close();
  });

  test("a refresh that ends the old token leaves no call on either gateway with it", async () => {
    if (!etcdReachable) return;
    endsOldToken.user = "bob";
    const { client } = await signInMcpClient(url(0, "ends-old-token"), key(BOB));
    await client.close();
    const clients = await Promise.all(
      [0, 1, 0, 1, 0, 1, 0, 1].map((i) => connectMcpClient(url(i, "ends-old-token"), key(BOB))),
    );
    // Issued for 65 s, so inside the 60 s refresh window from here on.
    await new Promise((r) => setTimeout(r, 6_000));
    const grantsBefore = endsOldToken.refreshGrants().length;
    const requestsBefore = endsOldToken.mcpAuthorizations.length;
    endsOldToken.refreshDelayMs = 800;
    const results = await Promise.all(clients.map(({ client }) => whoami(client)));
    endsOldToken.refreshDelayMs = 0;
    expect(results).toEqual(Array(8).fill("bob"));
    expect(endsOldToken.refreshGrants().length - grantsBefore).toBe(1);
    expect(new Set(endsOldToken.mcpAuthorizations.slice(requestsBefore)).size).toBe(1);
    for (const { client } of clients) await client.close();
  }, 30_000);

  test("a storage outage keeps a cached token working and never asks for a sign-in", async () => {
    if (!etcdReachable) return;
    upstream.user = "bob";
    const { client } = await signInMcpClient(url(0), key(BOB));
    expect(await whoami(client)).toBe("bob");

    cp.storeStatusOverride = 503;
    try {
      expect(await whoami(client)).toBe("bob");
      // Nothing cached for this caller: not a challenge, a retryable failure.
      const res = await initialize(url(0), CAROL);
      expect(res.status).toBe(503);
      expect(res.headers.get("www-authenticate")).toBeNull();
      expect(((await res.json()) as { error: { code: string } }).error.code).toBe("mcp_oauth_unavailable");

      const sent = await fetch(`${url(0)}/oauth/authorize?${new URLSearchParams({
        response_type: "code",
        client_id: "main",
        redirect_uri: "http://127.0.0.1:1/cb",
        code_challenge: "c".repeat(43),
        code_challenge_method: "S256",
      })}`, { redirect: "manual" });
      expect(new URL(sent.headers.get("location")!).searchParams.get("error")).toBe("temporarily_unavailable");
    } finally {
      cp.storeStatusOverride = undefined;
    }
    expect((await initialize(url(0), CAROL)).status).toBe(401);
    await client.close();
  });

  test("a store that does not know the API is an outage, not a missing token", async () => {
    if (!etcdReachable) return;
    cp.storeStatusOverride = 400;
    try {
      const res = await initialize(url(1), CAROL);
      expect(res.status).toBe(503);
    } finally {
      cp.storeStatusOverride = undefined;
    }
  });
});
