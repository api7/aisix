import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  waitForLogLine,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: every request answered on the proxy listener writes one access-log
// line, including the ones no dispatching handler answers — the model list,
// the A2A agent card, the OAuth discovery document, a path no route serves,
// and a method a known route does not serve. Health probes are the
// exception: they are the platform's traffic, not a client's.
// `observability.access_log: false` suppresses all of them.

const ACCESS_LINE = "proxy request completed";
const CALLER = "sk-access-log-unrouted";

interface Probe {
  label: string;
  method: string;
  path: string;
  auth: boolean;
  status: number;
}

const LOGGED: readonly Probe[] = [
  { label: "model list", method: "GET", path: "/v1/models", auth: true, status: 200 },
  {
    label: "agent card of an unknown agent",
    method: "GET",
    path: "/a2a/no-such-agent/.well-known/agent-card.json",
    auth: true,
    status: 404,
  },
  {
    label: "OAuth discovery, surface dormant",
    method: "GET",
    path: "/.well-known/oauth-protected-resource",
    auth: false,
    status: 404,
  },
  {
    label: "OAuth discovery for /mcp, surface dormant",
    method: "GET",
    path: "/.well-known/oauth-protected-resource/mcp",
    auth: false,
    status: 404,
  },
  { label: "unrouted path", method: "GET", path: "/v1/no-such-route", auth: true, status: 404 },
  { label: "unrouted path, no credential", method: "POST", path: "/nothing/here", auth: false, status: 404 },
  { label: "wrong method on chat", method: "GET", path: "/v1/chat/completions", auth: true, status: 405 },
  { label: "wrong method on files", method: "PUT", path: "/v1/files", auth: true, status: 405 },
  { label: "wrong method on realtime", method: "POST", path: "/v1/realtime", auth: false, status: 405 },
];

const PROBES: readonly Probe[] = [
  { label: "liveness probe", method: "GET", path: "/livez", auth: false, status: 200 },
  { label: "readiness probe", method: "GET", path: "/readyz", auth: false, status: 200 },
];

interface Sent {
  probe: Probe;
  id: string;
}

async function send(app: SpawnedApp, p: Probe): Promise<Sent> {
  const res = await fetch(`${app.proxyUrl}${p.path}`, {
    method: p.method,
    headers: p.auth ? { authorization: `Bearer ${CALLER}` } : {},
  });
  await res.text();
  expect(res.status, p.label).toBe(p.status);
  if (p.status === 405) {
    expect(res.headers.get("allow"), `${p.label} still names the allowed methods`).toBeTruthy();
  }
  return { probe: p, id: res.headers.get("x-aisix-request-id") ?? "" };
}

async function drainedAccessLines(app: SpawnedApp): Promise<string[]> {
  await app.stop();
  await waitForLogLine(app, (l) => l.includes("aisix shut down cleanly"), "the shutdown line");
  return app
    .output()
    .split("\n")
    .filter((l) => l.includes(ACCESS_LINE));
}

describe("requests answered outside dispatch write their access-log line", () => {
  let on: SpawnedApp | undefined;
  let off: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    on = await spawnApp({ logLevel: "info" });
    off = await spawnApp({ logLevel: "info", accessLog: false });
    for (const app of [on, off]) {
      const seed = new SeedClient(etcd, app.etcdPrefix);
      await seed.createApiKey({
        key_hash: createHash("sha256").update(CALLER).digest("hex"),
        allowed_models: [],
      });
      const probe = new ProxyClient(app.proxyUrl, CALLER);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    await on?.exit();
    await off?.exit();
  });

  test("one line each, with the status the caller got; none for health probes", async (ctx) => {
    if (!etcdReachable || !on) {
      ctx.skip();
      return;
    }
    const logged: Sent[] = [];
    for (const p of LOGGED) logged.push(await send(on, p));
    const probes: Sent[] = [];
    for (const p of PROBES) probes.push(await send(on, p));
    const lines = await drainedAccessLines(on);
    for (const { probe, id } of logged) {
      expect(id, `${probe.label} carries x-aisix-request-id`).toBeTruthy();
      const mine = lines.filter((l) => l.includes(`request_id="${id}"`));
      expect(mine, `one access-log line for ${probe.label}`).toHaveLength(1);
      expect(mine[0], probe.label).toContain(`status=${probe.status}`);
      expect(mine[0], probe.label).toMatch(new RegExp(`method="?${probe.method}"?\\s`));
      expect(mine[0], probe.label).toMatch(new RegExp(`path="?${probe.path}"?\\s`));
    }
    for (const { probe, id } of probes) {
      expect(
        lines.filter((l) => id && l.includes(`request_id="${id}"`)),
        `${probe.label} writes no line`,
      ).toEqual([]);
      expect(
        lines.filter((l) => new RegExp(`path="?${probe.path}"?\\s`).test(l)),
        `${probe.label} writes no line`,
      ).toEqual([]);
    }
  });

  test("access_log: false suppresses them", async (ctx) => {
    if (!etcdReachable || !off) {
      ctx.skip();
      return;
    }
    for (const p of LOGGED) await send(off, p);
    expect(await drainedAccessLines(off)).toEqual([]);
  });
});
