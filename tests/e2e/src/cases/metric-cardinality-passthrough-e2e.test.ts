import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, spawnApp, type SpawnedApp } from "../harness/index.js";

// E2E: unauthenticated passthrough paths must NOT mint unbounded metric
// series (#451). The in-flight middleware runs before auth and before
// route matching; pre-fix it used the raw request path as the `endpoint`
// label, so each unique /passthrough/<provider>/<unique> path created a
// new Prometheus time series — an unauthenticated cardinality DoS.
//
// We fire many unique unauthenticated passthrough paths, then scrape
// /metrics and assert the endpoint label is collapsed to the route
// template (no unique suffix leaks into a label).

describe("metric label cardinality for passthrough (#451)", () => {
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

  test("unique unauthenticated passthrough paths collapse to one label", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // No Authorization header — these are unauthenticated requests.
    for (let i = 0; i < 25; i++) {
      await fetch(`${app.proxyUrl}/passthrough/openai/unique-path-${i}/sub-${i}`, {
        method: "POST",
        body: "{}",
      }).catch(() => {});
    }

    const scrape = await fetch(`${app.adminUrl}/metrics`).then((r) => r.text());
    const inFlightLines = scrape
      .split("\n")
      .filter((l) => l.startsWith("aisix_proxy_in_flight_requests{"));

    // No raw unique suffix may appear in any label.
    const leaked = inFlightLines.filter((l) => l.includes("unique-path-"));
    expect(leaked, `raw paths leaked into metric labels:\n${leaked.join("\n")}`).toHaveLength(0);

    // The passthrough route template is the single bounded label used.
    const passthroughLabels = new Set(
      inFlightLines
        .map((l) => l.match(/endpoint="([^"]*)"/)?.[1])
        .filter((e): e is string => !!e && e.includes("passthrough")),
    );
    expect(passthroughLabels.size).toBeLessThanOrEqual(1);
    if (passthroughLabels.size === 1) {
      expect([...passthroughLabels][0]).toBe("/passthrough/:provider/*rest");
    }
  });
});
