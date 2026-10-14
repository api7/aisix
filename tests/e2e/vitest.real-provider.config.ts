import { defineConfig } from "vitest/config";

// The credentialed smoke is intentionally outside normal CI. Keep this
// config separate so its one real-provider case cannot be discovered by the
// ordinary `pnpm test` glob.
export default defineConfig({
  test: {
    globalSetup: "./src/harness/global-setup.ts",
    include: ["src/real-provider/**/*.smoke.ts"],
    pool: "forks",
    maxWorkers: 1,
    testTimeout: 90_000,
    hookTimeout: 90_000,
    globals: false,
  },
});
