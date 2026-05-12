import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, spawnApp, type SpawnedApp } from "../harness/index.js";
import { harnessRequest } from "../harness/http.js";

describe("livez e2e: public liveness route is /livez and /health is gone", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    app = await spawnApp();
  });

  afterAll(async () => {
    await app?.exit();
  });

  test("proxy and admin public /livez return plain ok, and /health is absent", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxyLivez = await harnessRequest(`${app.proxyUrl}/livez`, { method: "GET" });
    expect(proxyLivez.statusCode).toBe(200);
    expect(await proxyLivez.body.text()).toBe("ok");

    const adminLivez = await harnessRequest(`${app.adminUrl}/livez`, { method: "GET" });
    expect(adminLivez.statusCode).toBe(200);
    expect(await adminLivez.body.text()).toBe("ok");

    const proxyHealth = await harnessRequest(`${app.proxyUrl}/health`, { method: "GET" });
    expect(proxyHealth.statusCode).toBe(404);
    await proxyHealth.body.dump();

    const adminHealth = await harnessRequest(`${app.adminUrl}/health`, { method: "GET" });
    expect(adminHealth.statusCode).toBe(404);
    await adminHealth.body.dump();
  });
});
