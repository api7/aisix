import { defineConfig } from "vitest/config";

/** One fork per etcd cluster the run was given, never fewer than 2. */
function forkBudget(): number {
  const override = Number(process.env.AISIX_E2E_MAX_FORKS);
  if (Number.isFinite(override) && override >= 1) return override;
  const endpoints = (process.env.AISIX_E2E_ETCD_ENDPOINTS ?? "")
    .split(",")
    .filter((s) => s.trim() !== "").length;
  return Math.max(2, endpoints);
}

export default defineConfig({
  test: {
    // E2E tests spin up the aisix binary, so concurrency is bounded by
    // what the shared infrastructure underneath tolerates rather than by
    // CPU. Both constraints that once forced maxForks=2 are now removed
    // at the source rather than by serialising:
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
    // The cap is therefore DERIVED from the endpoint count rather than
    // written down next to it: one fork per cluster, floor 2 (the old
    // value, which is what a developer with a single local etcd gets).
    // Writing a number here instead would let a shortened endpoint list
    // silently double forks onto one cluster and bring #157 back.
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
