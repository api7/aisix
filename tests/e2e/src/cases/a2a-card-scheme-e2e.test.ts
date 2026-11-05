import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, request } from "undici";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  waitConfigPropagation,
  type A2aUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// The agent card a gateway serves advertises the scheme the request actually
// arrived with: what a proxy in front says it was, or else the listener's own.

const KEY = "sk-a2a-card-scheme";
const AGENT = "scheme-agent";

describe("the served agent card's scheme", () => {
  const insecure = new Agent({ connect: { rejectUnauthorized: false } });
  let app: SpawnedApp | undefined;
  let upstream: A2aUpstream | undefined;
  let dir: string | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    dir = await mkdtemp(join(tmpdir(), "aisix-a2a-card-scheme-e2e-"));
    const certFile = join(dir, "server.crt");
    const keyFile = join(dir, "server.key");
    execFileSync("openssl", [
      "req", "-x509", "-newkey", "rsa:2048",
      "-keyout", keyFile, "-out", certFile,
      "-days", "1", "-nodes",
      "-subj", "/CN=127.0.0.1",
      "-addext", "subjectAltName=IP:127.0.0.1",
    ], { stdio: "ignore" });

    upstream = await startA2aUpstream({ cardMount: "origin" });
    app = await spawnApp({ proxyListeners: [{}, { tls: { cert_file: certFile, key_file: keyFile } }] });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.update("a2a_agents", randomUUID(), {
      name: AGENT,
      url: upstream.url,
      protocol_version: "1.0",
      auth_type: "none",
      enabled: true,
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(KEY).digest("hex"),
      allowed_models: [],
      allowed_agents: ["*"],
    });
    await waitConfigPropagation(async () => (await card(app!.proxyUrl)).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await insecure.close();
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  async function card(base: string, headers: Record<string, string> = {}) {
    const res = await request(`${base}/a2a/${AGENT}/.well-known/agent-card.json`, {
      dispatcher: insecure,
      headers: { authorization: `Bearer ${KEY}`, ...headers },
    });
    const text = await res.body.text();
    return {
      status: res.statusCode,
      url: res.statusCode === 200 ? (JSON.parse(text) as { url: string }).url : text,
    };
  }

  const authority = (base: string) => new URL(base).host;

  test("over plain HTTP with no proxy in front it is http", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const base = app.proxyUrls[0];
    expect((await card(base)).url).toBe(`http://${authority(base)}/a2a/${AGENT}`);
  });

  test("over TLS with no proxy in front it is https", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const base = app.proxyUrls[1];
    expect((await card(base)).url).toBe(`https://${authority(base)}/a2a/${AGENT}`);
  });

  test("what a TLS-terminating proxy says wins over the listener's plain HTTP", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const base = app.proxyUrls[0];
    expect((await card(base, { "x-forwarded-proto": "https" })).url).toBe(
      `https://${authority(base)}/a2a/${AGENT}`,
    );
  });
});
