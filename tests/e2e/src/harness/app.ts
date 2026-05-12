import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { stringify as yamlStringify } from "yaml";

import { pickFreePorts } from "./ports.js";
import { EtcdClient } from "./etcd.js";
import { harnessRequest } from "./http.js";

export interface AppOverrides {
  /** Inserted into `admin.admin_keys`. Defaults to a fresh random key. */
  adminKey?: string;
  /** Whether to enable Prometheus on `/metrics`. Defaults to true. */
  prometheus?: boolean;
  /** Extra raw config keys merged into the YAML at the top level. */
  extra?: Record<string, unknown>;
}

export interface SpawnedApp {
  proxyUrl: string;
  adminUrl: string;
  adminKey: string;
  etcdPrefix: string;
  signal(signal: NodeJS.Signals): void;
  exit(): Promise<void>;
}

const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");
const READY_TIMEOUT_MS = 10_000;
const SHUTDOWN_GRACE_MS = 3_000;

/**
 * Per-test handle to a spawned `aisix` binary. Each call writes a fresh
 * config YAML into a tmp dir, picks two free ports, picks a unique etcd
 * prefix, and waits up to 10s for `/livez` on the proxy and `/admin/v1/health`
 * on the admin listener to respond 200. `exit()` issues SIGTERM and waits up to
 * 3s, escalating to SIGKILL.
 */
export async function spawnApp(overrides: AppOverrides = {}): Promise<SpawnedApp> {
  const etcd = new EtcdClient();
  if (!(await etcd.ping())) {
    throw new Error(
      `etcd not reachable at ${process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379"} ` +
        "(set AISIX_E2E_ETCD or run `docker run --rm -p 2379:2379 quay.io/coreos/etcd:v3.5.15`)",
    );
  }

  const [proxyPort, adminPort] = await pickFreePorts(2);
  const adminKey = overrides.adminKey ?? `admin-${randomUUID()}`;
  const etcdPrefix = `/aisix-e2e-${randomUUID()}`;

  const cfg = {
    etcd: {
      endpoints: [process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379"],
      prefix: etcdPrefix,
      dial_timeout_ms: 5000,
      request_timeout_ms: 5000,
    },
    proxy: { addr: `127.0.0.1:${proxyPort}`, request_body_limit_bytes: 10485760 },
    admin: { addr: `127.0.0.1:${adminPort}`, admin_keys: [adminKey] },
    observability: {
      service_name: "aisix-e2e",
      log_level: "warn",
      access_log: false,
      metrics: {
        prometheus: { enabled: overrides.prometheus ?? true, path: "/metrics" },
        otlp: { enabled: false, endpoint: "http://127.0.0.1:4317" },
      },
      tracing: { otlp: { enabled: false, endpoint: "http://127.0.0.1:4317", sample_ratio: 1 } },
    },
    cache: { backend: "memory" },
    ...(overrides.extra ?? {}),
  };

  const dir = await mkdtemp(join(tmpdir(), "aisix-e2e-"));
  const cfgPath = join(dir, "config.yaml");
  await writeFile(cfgPath, yamlStringify(cfg), "utf8");

  // Strip AISIX_* env vars so they don't leak into the binary's
  // config loader (which treats AISIX_<KEY> as config overrides).
  const childEnv: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined && !k.startsWith("AISIX_")) childEnv[k] = v;
  }
  childEnv.RUST_LOG = process.env.RUST_LOG ?? "warn";
  childEnv.HTTP_PROXY = "";
  childEnv.HTTPS_PROXY = "";
  childEnv.ALL_PROXY = "";
  childEnv.http_proxy = "";
  childEnv.https_proxy = "";
  childEnv.all_proxy = "";
  childEnv.NO_PROXY = "127.0.0.1,localhost";
  childEnv.no_proxy = "127.0.0.1,localhost";

  const child = spawn(BIN_PATH, ["--config", cfgPath], {
    stdio: ["ignore", "pipe", "pipe"],
    env: childEnv,
  });

  let stderrBuf = "";
  child.stderr?.on("data", (c: Buffer) => {
    stderrBuf += c.toString("utf8");
  });
  child.stdout?.on("data", (c: Buffer) => {
    stderrBuf += c.toString("utf8");
  });
  let exitErr: string | undefined;
  child.once("exit", (code, signal) => {
    if (code !== 0 && code !== null) {
      exitErr = `aisix exited early with code=${code} signal=${signal}`;
    }
  });

  const proxyUrl = `http://127.0.0.1:${proxyPort}`;
  const adminUrl = `http://127.0.0.1:${adminPort}`;

  try {
    await Promise.all([
      waitForReady(`${proxyUrl}/livez`, READY_TIMEOUT_MS),
      waitForReady(`${adminUrl}/admin/v1/health`, READY_TIMEOUT_MS, adminKey),
    ]);
  } catch (err) {
    const detail = exitErr ?? "still running";
    const stderr = stderrBuf.slice(-2000);
    await terminate(child);
    await cleanup(etcd, etcdPrefix, dir);
    throw new Error(
      `${(err as Error).message}\n  binary state: ${detail}\n  stderr tail:\n${stderr}`,
    );
  }

  return {
    proxyUrl,
    adminUrl,
    adminKey,
    etcdPrefix,
    signal(signal: NodeJS.Signals) {
      if (child.exitCode === null) child.kill(signal);
    },
    async exit() {
      await terminate(child);
      await cleanup(etcd, etcdPrefix, dir);
    },
  };
}

async function waitForReady(url: string, timeoutMs: number, bearer?: string): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  let lastStatus: number | undefined;
  let attempts = 0;
  while (Date.now() < deadline) {
    attempts++;
    try {
      const headers: Record<string, string> = {};
      if (bearer) headers.authorization = `Bearer ${bearer}`;
      const res = await harnessRequest(url, { method: "GET", headers });
      lastStatus = res.statusCode;
      if (res.statusCode === 200) {
        await res.body.dump();
        return;
      }
      await res.body.dump();
    } catch (err) {
      lastErr = err;
    }
    await sleep(100);
  }
  throw new Error(
    `timed out waiting for ${url} after ${attempts} attempts (lastStatus=${lastStatus ?? "n/a"}): ${lastErr ?? "no response"}`,
  );
}

async function terminate(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null) return;
  child.kill("SIGTERM");
  const exited = await Promise.race([
    new Promise<boolean>((r) => child.once("exit", () => r(true))),
    sleep(SHUTDOWN_GRACE_MS).then(() => false),
  ]);
  if (!exited && child.exitCode === null) {
    child.kill("SIGKILL");
    await new Promise<void>((r) => child.once("exit", () => r()));
  }
}

async function cleanup(etcd: EtcdClient, prefix: string, dir: string): Promise<void> {
  // Best-effort — never throw from cleanup.
  await Promise.allSettled([
    etcd.deletePrefix(prefix),
    rm(dir, { recursive: true, force: true }),
  ]);
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
