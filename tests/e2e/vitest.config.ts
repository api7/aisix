import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // E2E tests spin up the aisix binary; keep concurrency low so
    // distinct test files don't contend for ports OR for etcd watch
    // dispatch (each file's `aisix` instance opens watches against
    // a single shared etcd; under maxForks=4 with 20+ files, watch
    // dispatch latency for the LAST resource in a multi-resource
    // write batch can exceed even a 10s `waitConfigPropagation`
    // budget — observed flake on #157, persisting after the budget
    // bump). maxForks=2 halves the concurrent watcher count and
    // doubles wall time but eliminates the contention.
    pool: "forks",
    poolOptions: {
      forks: { singleFork: false, minForks: 1, maxForks: 2 },
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
