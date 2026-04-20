import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // E2E tests spin up the aisix binary; keep concurrency low so distinct
    // test files don't contend for ports. Each file picks a random port
    // internally, but lowering parallelism bounds the blast radius.
    pool: "forks",
    poolOptions: {
      forks: { singleFork: false, minForks: 1, maxForks: 4 },
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
