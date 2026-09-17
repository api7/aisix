import { describe, expect, test } from "vitest";
import { spawnApp, type SpawnedApp } from "../harness/index.js";

// `managed.cp_base_url` (env AISIX_MANAGED__CP_BASE_URL, Helm
// controlPlane.baseURL) reaches four consumers with two different
// appetites: the etcd dial strips whatever scheme is present and
// re-attaches `https://` itself, while heartbeat, telemetry and the
// budget gate concatenate a path onto the value verbatim.
//
// So a scheme-less `host:port` used to dial etcd happily — the console
// showed the gateway connected — while every REST call produced
// `host:port/dp/...`, which reqwest rejects when the request is BUILT.
// That failure is indistinguishable from an unreachable control plane,
// so the budget gate's no-cache fallback sticky-denied and every proxied
// request answered `429 budget_exceeded` (AISIX-Cloud#1643).
//
// The gateway now gives the value a scheme once, at config load. These
// specs pin the two halves of that decision at the process boundary,
// where an operator meets them: a scheme-less value is accepted, and a
// value that is no http(s) URL stops the boot instead of booting into a
// gateway that refuses all traffic.

describe("managed.cp_base_url scheme handling", () => {
  test("a scheme-less control-plane URL is accepted and the gateway boots", async () => {
    let app: SpawnedApp | undefined;
    try {
      app = await spawnApp({
        extra: { managed: { cp_base_url: "cp.example.com:7944" } },
      });
      // Booting to a serving /livez is the assertion: `spawnApp` only
      // returns once the proxy listener answers 200.
      const res = await fetch(`${app.proxyUrl}/livez`);
      expect(res.status).toBe(200);
    } finally {
      await app?.exit();
    }
  });

  test("a control-plane URL that is no http(s) URL stops the boot", async () => {
    // A misspelt scheme is the shape that must NOT be silently prefixed
    // into `https://htts://...` — the gateway reports the typo instead.
    // Resolving means the gateway booted, so the successful spawn is
    // shut down before the spec fails on it.
    const outcome = await spawnApp({
      extra: { managed: { cp_base_url: "htts://cp.example.com" } },
    }).catch((err: unknown) => err as Error);
    if (!(outcome instanceof Error)) {
      await outcome.exit();
      throw new Error("the gateway booted on a control-plane URL it cannot call");
    }
    expect(outcome.message).toMatch(/AISIX_MANAGED__CP_BASE_URL/);
  });
});
