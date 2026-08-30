import { defineConfig } from "vitest/config";

import { forkBudget } from "./src/harness/forks.js";

export default defineConfig({
  test: {
    // Fails the run when a configured etcd is not answering. Case files
    // skip themselves when etcd is unreachable, which with four clusters
    // means a dead one silently drops the quarter of the suite bound to
    // it while the leg still reports green.
    globalSetup: "./src/harness/global-setup.ts",

    // E2E tests spin up the aisix binary, so what bounds concurrency is
    // the shared infrastructure underneath. CPU is the thing to suspect
    // first if the wall-clock assertions in audio-timeout /
    // ttft-first-frame / per-attempt-telemetry start flaking — each
    // fork's gateway sizes its proxy pool to the whole machine, so four
    // forks put appreciably more threads on the same cores than two.
    // Measured over three CI runs at four forks, those three files pass
    // with stable timings (audio-timeout 1283/1409/1543ms against a
    // 2800ms bound, ttft-first-frame 24.59/24.65/24.65s); if that stops
    // holding, lower the budget rather than widening the assertions.
    // Both constraints that forced maxForks=2 are removed at the source
    // rather than by serialising:
    //
    //   - ports: harness/ports.ts carves a disjoint port range per fork
    //     off VITEST_POOL_ID, so no two forks can be issued the same one.
    //
    //   - etcd watch dispatch: every file's `aisix` opened watches
    //     against ONE shared etcd, and at 4 forks over 20+ files the
    //     dispatch latency for the last resource of a write batch blew
    //     past even a 10s `waitConfigPropagation` (#157, still flaky
    //     after the budget bump). harness/etcd.ts now gives each fork
    //     its own cluster from AISIX_E2E_ETCD_ENDPOINTS, which is what
    //     CI sets; the watchers per cluster go back to what they were
    //     at maxForks=2.
    //
    // The cap is DERIVED (see forkBudget) rather than written down next
    // to the endpoint list, so a shortened list cannot silently double
    // forks onto one cluster and bring #157 back.
    pool: "forks",
    poolOptions: {
      forks: { singleFork: false, minForks: 1, maxForks: forkBudget() },
    },
    testTimeout: 60_000,
    hookTimeout: 60_000,
    globals: false,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      reportsDirectory: "coverage",
    },
  },
});
