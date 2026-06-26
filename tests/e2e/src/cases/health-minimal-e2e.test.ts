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

  test("admin /admin/v1/health reports an aggregate status (#618)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await harnessRequest(`${app.adminUrl}/admin/v1/health`, {
      method: "GET",
      headers: { authorization: `Bearer ${app.adminKey}` },
    });
    expect(res.statusCode).toBe(200);
    const body = (await res.body.json()) as { status: string; models: unknown[] };

    // #618: the top-level status is now a real aggregate of model health +
    // config freshness, not a fixed "ok" marker.
    expect(["ok", "degraded", "unhealthy"]).toContain(body.status);
    // A freshly spawned gateway has no upstream failures, so no model is
    // down — it must never be "unhealthy".
    expect(body.status).not.toBe("unhealthy");
    expect(Array.isArray(body.models)).toBe(true);
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
