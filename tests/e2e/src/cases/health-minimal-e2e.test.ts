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

  test("proxy and admin /readyz report ready once config is applied (#591)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Readiness gates on config freshness, so poll until the supervisor's
    // first apply lands (a fresh spawn briefly reports 503 = starting up).
    const deadline = Date.now() + 5000;
    let proxyReady = false;
    while (Date.now() < deadline) {
      const r = await harnessRequest(`${app.proxyUrl}/readyz`, { method: "GET" });
      const ok = r.statusCode === 200;
      await r.body.dump();
      if (ok) {
        proxyReady = true;
        break;
      }
      await new Promise((res) => setTimeout(res, 50));
    }
    expect(proxyReady).toBe(true);

    const adminReadyz = await harnessRequest(`${app.adminUrl}/readyz`, { method: "GET" });
    expect(adminReadyz.statusCode).toBe(200);
    expect(await adminReadyz.body.text()).toBe("ok");
  });

  test("proxy /livez turns unhealthy after SIGTERM before exit", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    app.signal("SIGTERM");

    const deadline = Date.now() + 3000;
    let observedUnhealthy = false;
    while (Date.now() < deadline) {
      try {
        const res = await harnessRequest(`${app.proxyUrl}/livez`, { method: "GET" });
        if (res.statusCode !== 200) {
          observedUnhealthy = true;
          await res.body.dump();
          break;
        }
        await res.body.dump();
      } catch {
        observedUnhealthy = true;
        break;
      }
      await new Promise((r) => setTimeout(r, 50));
    }

    expect(observedUnhealthy).toBe(true);
    app = undefined;
  });
});
