import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, spawnApp, type SpawnedApp } from "../harness/index.js";
import { harnessRequest } from "../harness/http.js";

describe("health e2e: public health stays minimal and does not rename to /livez", () => {
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

  test("proxy and admin public health return only status, and /livez is absent", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxyHealth = await harnessRequest(`${app.proxyUrl}/health`, { method: "GET" });
    expect(proxyHealth.statusCode).toBe(200);
    expect(JSON.parse(await proxyHealth.body.text())).toEqual({ status: "ok" });

    const adminHealth = await harnessRequest(`${app.adminUrl}/health`, { method: "GET" });
    expect(adminHealth.statusCode).toBe(200);
    expect(JSON.parse(await adminHealth.body.text())).toEqual({ status: "ok" });

    const proxyLivez = await harnessRequest(`${app.proxyUrl}/livez`, { method: "GET" });
    expect(proxyLivez.statusCode).toBe(404);
    await proxyLivez.body.dump();

    const adminLivez = await harnessRequest(`${app.adminUrl}/livez`, { method: "GET" });
    expect(adminLivez.statusCode).toBe(404);
    await adminLivez.body.dump();
  });
});
