import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockIdp,
  waitConfigPropagation,
  waitForLogLine,
  type MockIdp,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a request refused for authentication writes its access-log line.
//
// An operator auditing who hit the gateway with a bad credential reads the
// access log, so every denial shape — no credential, an unknown key, a
// disabled or expired key, a rejected JWT — is one line there, on every
// proxy surface (a JWT short of a required scope is a 403, the rest 401),
// carrying the status the caller saw and never the
// credential itself. `observability.access_log: false` suppresses these
// lines like every other.

const ACCESS_LINE = "proxy request completed";
const VALID = "sk-auth-denial-valid";
const DISABLED = "sk-auth-denial-disabled";
const EXPIRED = "sk-auth-denial-expired";
const UNKNOWN = "sk-auth-denial-unknown-0123456789";

const sha = (s: string) => createHash("sha256").update(s).digest("hex");

// Every proxy surface that authenticates through the gateway key.
const SURFACES: ReadonlyArray<{ method: string; path: string }> = [
  { method: "POST", path: "/v1/chat/completions" },
  { method: "POST", path: "/v1/completions" },
  { method: "POST", path: "/v1/messages" },
  { method: "POST", path: "/v1/messages/count_tokens" },
  { method: "POST", path: "/v1/responses" },
  { method: "POST", path: "/v1/embeddings" },
  { method: "POST", path: "/v1/rerank" },
  { method: "POST", path: "/v1/images/generations" },
  { method: "POST", path: "/v1/audio/speech" },
  { method: "POST", path: "/v1/videos" },
  { method: "GET", path: "/v1/models" },
  { method: "GET", path: "/v1/files" },
  { method: "GET", path: "/v1/batches/batch_x" },
  { method: "POST", path: "/mcp" },
  { method: "POST", path: "/a2a/some-agent" },
];

interface Sent {
  method: string;
  path: string;
  id: string;
  status: number;
  secret?: string;
  label: string;
}

async function send(
  app: SpawnedApp,
  method: string,
  path: string,
  headers: Record<string, string>,
): Promise<{ status: number; id: string }> {
  const res = await fetch(`${app.proxyUrl}${path}`, {
    method,
    headers: { ...headers, "content-type": "application/json" },
    body: method === "GET" ? undefined : "{}",
  });
  await res.text();
  return { status: res.status, id: res.headers.get("x-aisix-request-id") ?? "" };
}

async function drive(app: SpawnedApp, idp: MockIdp, scopedIdp: MockIdp): Promise<Sent[]> {
  const jwtExpired = idp.sign(
    agentClaims(idp.url, { exp: Math.floor(Date.now() / 1000) - 3600 }),
  );
  const jwtUnmapped = idp.sign(agentClaims(idp.url, { sub: "nobody-bound" }));
  const jwtUnscoped = scopedIdp.sign(agentClaims(scopedIdp.url));
  const credentials: ReadonlyArray<{
    label: string;
    headers: Record<string, string>;
    secret?: string;
    status?: number;
  }> = [
    { label: "no credential", headers: {} },
    { label: "unknown key", headers: { authorization: `Bearer ${UNKNOWN}` }, secret: UNKNOWN },
    { label: "unknown x-api-key", headers: { "x-api-key": UNKNOWN }, secret: UNKNOWN },
    { label: "disabled key", headers: { authorization: `Bearer ${DISABLED}` }, secret: DISABLED },
    { label: "expired key", headers: { authorization: `Bearer ${EXPIRED}` }, secret: EXPIRED },
    { label: "expired jwt", headers: { authorization: `Bearer ${jwtExpired}` }, secret: jwtExpired },
    { label: "unmapped jwt", headers: { authorization: `Bearer ${jwtUnmapped}` }, secret: jwtUnmapped },
    {
      label: "jwt missing a required scope",
      headers: { authorization: `Bearer ${jwtUnscoped}` },
      secret: jwtUnscoped,
      status: 403,
    },
  ];
  const sent: Sent[] = [];
  for (const { method, path } of SURFACES) {
    for (const cred of credentials) {
      const { status, id } = await send(app, method, path, cred.headers);
      const label = `${cred.label} on ${method} ${path}`;
      expect(status, label).toBe(cred.status ?? 401);
      expect(id, `${label} carries x-aisix-request-id`).toBeTruthy();
      sent.push({ method, path, id, status, secret: cred.secret, label });
    }
  }
  return sent;
}

/** Stop the gateway (which drains its log queue) and return its lines. */
async function drainedLines(app: SpawnedApp): Promise<string[]> {
  await app.stop();
  await waitForLogLine(app, (l) => l.includes("aisix shut down cleanly"), "the shutdown line");
  return app.output().split("\n");
}

describe("an authentication denial writes its access-log line", () => {
  let idp: MockIdp | undefined;
  let scopedIdp: MockIdp | undefined;
  let on: SpawnedApp | undefined;
  let off: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    idp = await startMockIdp();
    scopedIdp = await startMockIdp();
    on = await spawnApp({ logLevel: "info" });
    off = await spawnApp({ logLevel: "info", accessLog: false });
    for (const app of [on, off]) {
      const seed = new SeedClient(etcd, app.etcdPrefix);
      await seed.createOidcProvider({
        name: "auth-denial-idp",
        issuer: idp.url,
        audiences: ["aisix-gateway"],
        jwks_uri: idp.jwksUrl,
      });
      await seed.createOidcProvider({
        name: "auth-denial-scoped-idp",
        issuer: scopedIdp.url,
        audiences: ["aisix-gateway"],
        jwks_uri: scopedIdp.jwksUrl,
        required_scopes: ["ai.access"],
      });
      await seed.createApiKey({ key_hash: sha(DISABLED), allowed_models: ["*"], disabled: true });
      await seed.createApiKey({
        key_hash: sha(EXPIRED),
        allowed_models: ["*"],
        expires_at: "2020-01-01T00:00:00Z",
      });
      // Seeded last: once it authenticates, every row above has loaded.
      await seed.createApiKey({ key_hash: sha(VALID), allowed_models: [] });
      const probe = new ProxyClient(app.proxyUrl, VALID);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    await on?.exit();
    await off?.exit();
    await idp?.close();
    await scopedIdp?.close();
  });

  test("each denial is exactly one line with its status and without the credential", async (ctx) => {
    if (!etcdReachable || !on || !idp || !scopedIdp) {
      ctx.skip();
      return;
    }
    const sent = await drive(on, idp, scopedIdp);
    const lines = (await drainedLines(on)).filter((l) => l.includes(ACCESS_LINE));
    for (const s of sent) {
      const mine = lines.filter((l) => l.includes(`request_id="${s.id}"`));
      expect(mine, `one access-log line for ${s.label}`).toHaveLength(1);
      expect(mine[0], s.label).toContain(`status=${s.status}`);
      expect(mine[0], s.label).toMatch(new RegExp(`method="?${s.method}"?\\s`));
      expect(mine[0], s.label).toMatch(new RegExp(`path="?${s.path}"?\\s`));
      // Nothing about the caller is established by a refused credential.
      expect(mine[0], s.label).not.toContain("api_key_id=");
      if (s.secret) {
        expect(mine[0].includes(s.secret), `${s.label}: credential not logged`).toBe(false);
      }
    }
  });

  test("access_log: false suppresses them", async (ctx) => {
    if (!etcdReachable || !off || !idp || !scopedIdp) {
      ctx.skip();
      return;
    }
    await drive(off, idp, scopedIdp);
    const lines = await drainedLines(off);
    expect(lines.filter((l) => l.includes(ACCESS_LINE))).toEqual([]);
    // Only the access log is off: the gateway's other `info` lines remain.
    expect(lines.some((l) => l.includes("tracing initialised"))).toBe(true);
  });
});
