import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, spawnApp, type SpawnedApp } from "../harness/index.js";
import { harnessRequest } from "../harness/http.js";

describe("livez e2e: public liveness route is /livez and /health is gone", () => {
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    // Held-back: this test drives the admin listener's health endpoint,
    // so it keeps admin bound (the suite default is now admin-off).
    //
    // A drain window is needed too. The harness default is
    // `min_drain_secs: 0`, which lets the process exit within
    // milliseconds of SIGTERM — leaving no interval in which to observe
    // what the health endpoints report WHILE draining, which is the
    // whole of what the last test asserts.
    app = await spawnApp({ admin: true, extra: { shutdown: { min_drain_secs: 5 } } });
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

  // A drain withdraws traffic; it does not make the process a candidate
  // for restarting. Those are the two different questions the two
  // endpoints answer, and a drain has to move exactly one of them:
  // `/readyz` reports it so the balancer stops routing here, `/livez`
  // stays `200` because a failing liveness probe asks the platform to
  // restart the instance — which would kill the very requests the drain
  // is staying alive to finish.
  //
  // Kubernetes stops probing liveness once a pod enters graceful
  // termination, so a rolling update does not act on the answer. That
  // makes this a contract about the endpoint rather than about kubelet,
  // and it is not academic: the gateway also runs as a single container
  // under docker or systemd, where a supervisor watching `/livez` does
  // act on it.
  test("a drain moves /readyz to 503 and leaves /livez at 200", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    app.signal("SIGTERM");

    // Gate on readiness having withdrawn rather than on a sleep: it
    // proves the drain has actually begun AND that the process is still
    // serving, which is the window the liveness assertion below is
    // about. `beforeAll` gives that window 5 seconds, and the poll below
    // gives up before it closes.
    const deadline = Date.now() + 3000;
    let draining = false;
    while (Date.now() < deadline) {
      const res = await harnessRequest(`${app.proxyUrl}/readyz`, { method: "GET" });
      const status = res.statusCode;
      await res.body.dump();
      if (status === 503) {
        draining = true;
        break;
      }
      await new Promise((r) => setTimeout(r, 50));
    }
    expect(draining, "/readyz never reported 503 after SIGTERM").toBe(true);

    for (const url of [`${app.proxyUrl}/livez`, `${app.adminUrl}/livez`]) {
      const res = await harnessRequest(url, { method: "GET" });
      const status = res.statusCode;
      const body = await res.body.text();
      expect(status, `${url} must stay live while draining`).toBe(200);
      expect(body).toBe("ok");
    }
  });
});
