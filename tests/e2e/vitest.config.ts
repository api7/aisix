import { defineConfig } from "vitest/config";

import { RANGE_SLOTS } from "./src/harness/ports.js";

/**
 * One fork per etcd cluster the run was given.
 *
 * Floor 2 — the value this suite ran at before the clusters were split,
 * and what a developer with a single local etcd gets. Note the floor
 * means ONE configured endpoint still yields two forks sharing it; that
 * is the pre-existing arrangement, not a regression, but it is also why
 * "fewer endpoints => fewer forks" only holds above 2.
 *
 * Ceiling RANGE_SLOTS — ports.ts hands fork `poolId % RANGE_SLOTS` its
 * own port range, so beyond that two forks collide on a range and the
 * AddrInUse flake that module exists to prevent comes back.
 */
function forkBudget(): number {
  const override = Number(process.env.AISIX_E2E_MAX_FORKS);
  if (Number.isInteger(override) && override >= 1) return Math.min(override, RANGE_SLOTS);
  const endpoints = (process.env.AISIX_E2E_ETCD_ENDPOINTS ?? "")
    .split(",")
    .filter((s) => s.trim() !== "").length;
  return Math.min(Math.max(2, endpoints), RANGE_SLOTS);
}

export default defineConfig({
  test: {
    // Fails the run when a configured etcd is not answering. Case files
    // skip themselves when etcd is unreachable, which with four clusters
    // means a dead one silently drops the quarter of the suite bound to
    // it while the leg still reports green.
    globalSetup: "./src/harness/global-setup.ts",

    // E2E tests spin up the aisix binary, so what bounds concurrency is
    // the shared infrastructure underneath — CPU has never been the
    // observed limit here, though it is the one to suspect first if the
    // wall-clock assertions in audio-timeout / ttft-first-frame /
    // per-attempt-telemetry start flaking. Both constraints that forced
    // maxForks=2 are removed at the source rather than by serialising:
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
