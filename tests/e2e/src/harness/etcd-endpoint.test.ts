import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { etcdEndpoint } from "./etcd.js";

const srcDir = join(dirname(fileURLToPath(import.meta.url)), "..");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return entry.name.endsWith(".ts") ? [path] : [];
  });
}

describe("etcdEndpoint", () => {
  // The whole point of the helper is that a fork's gateway and the test
  // driving it agree on which cluster they mean. A file that resolves
  // AISIX_E2E_ETCD itself pins fork 1's endpoint while the gateway it
  // spawned talks to fork N's, and the symptom is a config-propagation
  // timeout that looks like a product bug.
  it("is the only place the etcd endpoint is resolved from the environment", () => {
    const offenders = sourceFiles(srcDir).filter(
      (file) =>
        !file.endsWith(join("harness", "etcd.ts")) &&
        !file.endsWith(join("harness", "etcd-endpoint.test.ts")) &&
        /process\.env\.AISIX_E2E_ETCD\b/.test(readFileSync(file, "utf8")),
    );
    expect(offenders.map((f) => f.slice(srcDir.length + 1))).toEqual([]);
  });

  it("spreads forks across the configured endpoints", () => {
    const prev = { ...process.env };
    try {
      process.env.AISIX_E2E_ETCD_ENDPOINTS = "http://a:2379, http://b:2379,http://c:2379";
      const seen = new Set<string>();
      for (const poolId of ["1", "2", "3"]) {
        process.env.VITEST_POOL_ID = poolId;
        seen.add(etcdEndpoint());
      }
      expect(seen.size).toBe(3);
    } finally {
      process.env = prev;
    }
  });

  it("falls back to the single-endpoint form when no list is set", () => {
    const prev = { ...process.env };
    try {
      delete process.env.AISIX_E2E_ETCD_ENDPOINTS;
      process.env.AISIX_E2E_ETCD = "http://only:2379";
      process.env.VITEST_POOL_ID = "7";
      expect(etcdEndpoint()).toBe("http://only:2379");
    } finally {
      process.env = prev;
    }
  });
});
