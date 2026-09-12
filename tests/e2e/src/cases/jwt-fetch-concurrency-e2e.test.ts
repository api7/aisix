import { createHash, randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { setTimeout as sleep } from "node:timers/promises";
import { expect, test } from "vitest";
import {
  agentClaims, EtcdClient, pickFreePort, ProxyClient, SeedClient, spawnApp,
  startMockIdp, waitConfigPropagation,
} from "../harness/index.js";

async function fixture(ctx: { onTestFinished: (fn: () => Promise<void>) => void }) {
  const signer = await startMockIdp();
  const app = await spawnApp({});
  const etcd = new EtcdClient();
  const seed = new SeedClient(etcd, app.etcdPrefix);
  const port = await pickFreePort();
  const issuer = `http://127.0.0.1:${port}`;
  let keys = await (await fetch(signer.jwksUrl)).json();
  let release = () => {};
  let arrived = () => {};
  let heldPath = "";
  let hold = Promise.resolve();
  let status = 200;
  const counts = new Map<string, number>();
  const server = createServer(async (req, res) => {
    const path = (req.url ?? "").split("?")[0];
    counts.set(path, (counts.get(path) ?? 0) + 1);
    if (path === heldPath) {
      arrived();
      await hold;
    }
    res.setHeader("content-type", "application/json");
    res.statusCode = status;
    res.end(JSON.stringify(path === "/.well-known/openid-configuration"
      ? { issuer, jwks_uri: `${issuer}/jwks` } : keys));
  });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  ctx.onTestFinished(async () => {
    release();
    await app.exit();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await signer.close();
  });
  const provider = {
    name: randomUUID(), issuer, audiences: ["aisix-gateway"],
    identity_claim: "sub", jwks_uri: `${issuer}/jwks`,
  };
  const id = randomUUID();
  async function barrier() {
    const bearer = `sk-ready-${randomUUID()}`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(bearer).digest("hex"), allowed_models: [],
    });
    const proxy = new ProxyClient(app.proxyUrl, bearer);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }
  async function configure(discovery = false, suffix = "") {
    await etcd.put(`${app.etcdPrefix}/oidc_providers/${id}`, JSON.stringify({
      ...provider, jwks_uri: discovery ? undefined : `${issuer}/jwks${suffix}`,
    }));
    await barrier();
  }
  await seed.createApiKey({
    key_hash: createHash("sha256").update(`sk-bound-${id}`).digest("hex"),
    allowed_models: [], jwt_provider: provider.name, jwt_subject: "agent-1",
  });
  return {
    app, configure, counts,
    token: () => signer.sign(agentClaims(issuer)),
    request: (token: string) => fetch(`${app.proxyUrl}/v1/models`, {
      headers: { authorization: `Bearer ${token}` },
    }).then(async (response) => ({ status: response.status, body: await response.json() })),
    hold(path: string) {
      heldPath = path;
      hold = new Promise<void>((resolve) => { release = resolve; });
      const entered = new Promise<void>((resolve) => { arrived = resolve; });
      return { entered, release: () => release() };
    },
    fail() { status = 503; },
    async rotate() {
      signer.rotate();
      keys = await (await fetch(signer.jwksUrl)).json();
    },
  };
}

for (const stage of ["jwks", "discovery", "uri-change", "rotation"] as const) {
  test(`concurrent JWT requests share an in-flight ${stage} fetch`, async (ctx) => {
    if (!(await new EtcdClient().ping())) { ctx.skip(); return; }
    const f = await fixture(ctx);
    await f.configure(stage === "discovery");
    if (stage === "uri-change" || stage === "rotation") {
      expect((await f.request(f.token())).status).toBe(200);
      if (stage === "uri-change") await f.configure(false, "?new-keys=1");
      else { await f.rotate(); await sleep(1100); }
    }
    const path = stage === "discovery" ? "/.well-known/openid-configuration" : "/jwks";
    const before = f.counts.get(path) ?? 0;
    const gate = f.hold(path);
    const token = f.token();
    const first = f.request(token);
    await gate.entered;
    const followers = Array.from({ length: 8 }, () => f.request(token));
    await sleep(100);
    gate.release();
    const results = await Promise.all([first, ...followers]);
    expect(results.map((r) => ({ status: r.status, code: r.body.error?.code })))
      .toEqual(Array.from({ length: 9 }, () => ({ status: 200, code: undefined })));
    expect((f.counts.get(path) ?? 0) - before).toBe(1);
  });
}

for (const { discovery, delay } of [
  { discovery: false, delay: 100 },
  { discovery: true, delay: 100 },
  { discovery: false, delay: 1200 },
  { discovery: true, delay: 1200 },
]) {
  test(`failed ${discovery ? "discovery" : "JWKS"} fetch (${delay}ms) is shared and remains rate limited`, async (ctx) => {
    if (!(await new EtcdClient().ping())) { ctx.skip(); return; }
    const f = await fixture(ctx);
    await f.configure(discovery);
    f.fail();
    const path = discovery ? "/.well-known/openid-configuration" : "/jwks";
    const gate = f.hold(path);
    const token = f.token();
    const first = f.request(token);
    await gate.entered;
    const followers = Array.from({ length: 8 }, () => f.request(token));
    await sleep(delay);
    gate.release();
    for (const response of await Promise.all([first, ...followers])) {
      expect(response.status).toBe(503);
      expect(response.body.error.code).toBe("jwks_unavailable");
    }
    expect((await f.request(token)).status).toBe(503);
    expect(f.counts.get(path)).toBe(1);
  });
}
