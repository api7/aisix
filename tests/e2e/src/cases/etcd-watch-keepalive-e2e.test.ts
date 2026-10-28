import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startEtcdRelay,
  waitConfigPropagation,
  type EtcdRelay,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a configuration watch whose peer vanishes without a FIN or RST — a
// dropped NAT entry, a black-holed route to the control plane — must not
// leave the gateway serving a frozen configuration forever.
//
// Nothing arrives on such a connection and nothing errors on it: a watch
// stream sends nothing upstream once established, so not even TCP
// retransmission notices. Only a liveness probe on the connection itself
// (HTTP/2 PING) can tell a quiet watch from a dead one.

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");

describe("etcd watch over a silently dead connection", () => {
  let app: SpawnedApp | undefined;
  let relay: EtcdRelay | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    relay = await startEtcdRelay();
    await relay.release();
    const prefix = `/aisix-e2e-watch-keepalive-${randomUUID()}`;
    seed = new SeedClient(etcd, prefix);
    await seed.createApiKey({ key_hash: hash("sk-wk-e2e-first"), allowed_models: ["*"] });
    app = await spawnApp({ etcdPrefix: prefix, extra: { etcd: { endpoints: [relay.endpoint], prefix } } });
    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, "sk-wk-e2e-first").listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await relay?.stop();
  });

  test("a change made after the connection went dark still arrives", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    relay!.blackholeEstablished();
    await seed!.createApiKey({ key_hash: hash("sk-wk-e2e-second"), allowed_models: ["*"] });
    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, "sk-wk-e2e-second").listModels()).status === 200,
      60_000,
    );
    expect((await new ProxyClient(app!.proxyUrl, "sk-wk-e2e-second").listModels()).status).toBe(200);
  }, 90_000);
});
