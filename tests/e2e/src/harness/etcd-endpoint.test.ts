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
  // the endpoint itself pins fork 1's cluster while the gateway it
  // spawned talks to fork N's, and the symptom is a config-propagation
  // timeout that looks like a product bug.
  //
  // Three spellings count: reading AISIX_E2E_ETCD, reading the
  // _ENDPOINTS list (`\b` does NOT separate them — `_` is a word
  // character — so a case doing `AISIX_E2E_ETCD_ENDPOINTS.split(",")[0]`
  // would pin fork 1's cluster and pass a narrower guard), and writing a
  // literal in the 2379-2382 range CI publishes its clusters on, in any
  // of its three host spellings. (A deliberately dead endpoint like
  // 127.0.0.1:1, which several cases use to prove a failure path, is
  // outside that range and stays allowed.)
  it("is the only place the etcd endpoint is resolved", () => {
    const offenders = sourceFiles(srcDir).filter((file) => {
      if (
        file.endsWith(join("harness", "etcd.ts")) ||
        file.endsWith(join("harness", "global-setup.ts")) ||
        file.endsWith(join("harness", "etcd-endpoint.test.ts"))
      ) {
        return false;
      }
      const src = readFileSync(file, "utf8");
      return (
        /process\.env\.AISIX_E2E_ETCD/.test(src) ||
        /(?:127\.0\.0\.1|localhost|\[::1\]):(?:2379|238[0-2])\b/.test(src)
      );
    });
    expect(offenders.map((f) => f.slice(srcDir.length + 1))).toEqual([]);
  });

  // Restore the individual keys rather than reassigning process.env:
  // replacing the object detaches it from the live environment, which
  // spawnApp reads to build a child's env.
  const withEnv = (vars: Record<string, string | undefined>, fn: () => void) => {
    const keys = Object.keys(vars);
    const prev = Object.fromEntries(keys.map((k) => [k, process.env[k]]));
    for (const [k, v] of Object.entries(vars)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    try {
      fn();
    } finally {
      for (const k of keys) {
        if (prev[k] === undefined) delete process.env[k];
        else process.env[k] = prev[k];
      }
    }
  };

  it("spreads forks across the configured endpoints", () => {
    withEnv(
      { AISIX_E2E_ETCD_ENDPOINTS: "http://a:2379, http://b:2379,http://c:2379", VITEST_POOL_ID: "1" },
      () => {
        const seen = new Set<string>();
        for (const poolId of ["1", "2", "3"]) {
          process.env.VITEST_POOL_ID = poolId;
          seen.add(etcdEndpoint());
        }
        expect(seen.size).toBe(3);
      },
    );
  });

  it("falls back to the single-endpoint form when no list is set", () => {
    withEnv(
      {
        AISIX_E2E_ETCD_ENDPOINTS: undefined,
        AISIX_E2E_ETCD: "http://only:2379",
        VITEST_POOL_ID: "7",
      },
      () => {
        expect(etcdEndpoint()).toBe("http://only:2379");
      },
    );
  });
});
