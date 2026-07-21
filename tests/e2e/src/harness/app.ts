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
  /**
   * Whether to bind the admin listener. Defaults to `true`. When `false`,
   * the gateway runs with `admin.enabled = false` — no admin listener is
   * bound even in etcd/file mode, mirroring the post-removal world. Seed
   * resources through `SeedClient`/`EtcdClient` (never the Admin API), and
   * readiness is gated on the proxy `/livez` plus the metrics listener
   * instead of the admin health endpoint. `adminUrl`/`adminKey` are still
   * returned but point at an unbound port.
   */
  admin?: boolean;
  /** Inserted into `admin.admin_keys`. Defaults to a fresh random key. */
  adminKey?: string;
  /** Whether to enable the Prometheus scrape endpoint. Defaults to true. */
  prometheus?: boolean;
  /** Prometheus scrape path. Defaults to `/metrics`. */
  prometheusPath?: string;
  /** Extra raw config keys merged into the YAML at the top level. */
  extra?: Record<string, unknown>;
  /**
   * `proxy.real_ip` block (#492). Merged into the base proxy config so
   * the listener addr is preserved. Configures nginx-style trusted-proxy
   * real-client-IP resolution from `x-forwarded-for`.
   */
  realIp?: {
    trusted_proxies?: string[];
    recursive?: boolean;
    header?: string;
  };
  /**
   * Extra environment variables for the spawned binary, applied AFTER the
   * `AISIX_*` strip. Use for non-config secrets the DP reads from its own
   * environment rather than from the kine config — e.g.
   * `SLS_CRED_<REF>_AK_ID` / `_AK_SECRET` for an `aliyun_sls` exporter, whose
   * AccessKey deliberately never travels on the config path.
   */
  extraEnv?: Record<string, string>;
  /**
   * `observability.metrics.client_type_rules` (AISIX-Cloud#1045): operator
   * UA→client_type regex rules, tried before the built-in allowlist.
   * A dedicated override because `extra` replaces whole top-level blocks
   * and the observability block carries the harness-picked metrics port.
   */
  clientTypeRules?: Array<{ pattern: string; client: string }>;
  /**
   * FILE MODE: contents of a standalone `resources.yaml`. When set, the
   * generated config carries `resources_file` (pointing at this content
   * written into the tmp dir) and NO `etcd` section — the gateway loads
   * every resource from the file and etcd is never contacted (no ping,
   * no prefix cleanup). Rewrite the file at `SpawnedApp.resourcesPath`
   * and send SIGHUP to exercise reloads.
   */
  resourcesFile?: string;
}

export interface SpawnedApp {
  proxyUrl: string;
  adminUrl: string;
  adminKey: string;
  etcdPrefix: string;
  /**
   * Dedicated metrics listener URL — the only Prometheus scrape surface.
   * The port is reserved even when `prometheus: false` (nothing listens
   * there in that case).
   */
  metricsUrl: string;
  /**
   * FILE MODE only: absolute path of the resources.yaml the gateway
   * loads. Rewrite it and `signal("SIGHUP")` to trigger a reload.
   */
  resourcesPath?: string;
  /**
   * Combined stdout+stderr captured so far. Lets tests wait
   * deterministically on log lines (e.g. the reload-failed WARN)
   * instead of sleeping.
   */
  output(): string;
  signal(signal: NodeJS.Signals): void;
  exit(): Promise<void>;
}

const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");
const READY_TIMEOUT_MS = 10_000;
const SHUTDOWN_GRACE_MS = 3_000;

/**
 * Per-test handle to a spawned `aisix` binary. Each call writes a fresh
 * config YAML into a tmp dir, picks three free ports (proxy, admin,
 * metrics), picks a unique etcd prefix, and waits up to 10s for `/livez`
 * on the proxy, `/admin/v1/health` on the admin listener, and the scrape
 * path on the metrics listener to respond 200. `exit()` issues SIGTERM
 * and waits up to 3s, escalating to SIGKILL.
 *
 * A startup that dies to a port collision (an external process bound one
 * of the picked ports in the pick→bind window; see `ports.ts`) is retried
 * with fresh ports rather than failing the test.
 */
export async function spawnApp(overrides: AppOverrides = {}): Promise<SpawnedApp> {
  const MAX_ATTEMPTS = 3;
  for (let attempt = 1; ; attempt++) {
    try {
      return await spawnAppOnce(overrides);
    } catch (err) {
      if (attempt < MAX_ATTEMPTS && isAddrInUseStartupFailure(err)) {
        console.warn(
          `spawnApp: aisix died to a port collision (attempt ${attempt}/${MAX_ATTEMPTS}), retrying with fresh ports`,
        );
        continue;
      }
      throw err;
    }
  }
}

/**
 * True when the spawn failure is `aisix` exiting at startup because one
 * of its listeners hit AddrInUse — the only failure class `spawnApp`
 * retries (anything else is a real bug the test must surface).
 */
export function isAddrInUseStartupFailure(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  // Matches both the OS error text ("Address already in use (os error
  // 98)") and Rust's ErrorKind rendering ("AddrInUse").
  return msg.includes("exited early") && /addr(?:ess)?\s*(?:already\s*)?in\s*use/i.test(msg);
}

async function spawnAppOnce(overrides: AppOverrides = {}): Promise<SpawnedApp> {
  const fileMode = overrides.resourcesFile !== undefined;
  const etcd = new EtcdClient();
  // FILE MODE never contacts etcd — skip the availability gate so the
  // file source stays exercisable even without the shared etcd.
  if (!fileMode && !(await etcd.ping())) {
    throw new Error(
      `etcd not reachable at ${process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379"} ` +
        "(set AISIX_E2E_ETCD or run `docker run --rm -p 2379:2379 quay.io/coreos/etcd:v3.5.15`)",
    );
  }

  const prometheusEnabled = overrides.prometheus ?? true;
  const adminEnabled = overrides.admin ?? true;
  const [proxyPort, adminPort, metricsPort] = await pickFreePorts(3);
  const adminKey = overrides.adminKey ?? `admin-${randomUUID()}`;
  const etcdPrefix = `/aisix-e2e-${randomUUID()}`;

  const dir = await mkdtemp(join(tmpdir(), "aisix-e2e-"));
  let resourcesPath: string | undefined;
  if (fileMode) {
    resourcesPath = join(dir, "resources.yaml");
    await writeFile(resourcesPath, overrides.resourcesFile!, "utf8");
  }

  const cfg = {
    // Exactly one resource source: the standalone file, or etcd.
    ...(fileMode
      ? { resources_file: resourcesPath }
      : {
          etcd: {
            endpoints: [process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379"],
            prefix: etcdPrefix,
            dial_timeout_ms: 5000,
            request_timeout_ms: 5000,
          },
        }),
    proxy: {
      addr: `127.0.0.1:${proxyPort}`,
      request_body_limit_bytes: 10485760,
      ...(overrides.realIp ? { real_ip: overrides.realIp } : {}),
    },
    admin: adminEnabled
      ? { addr: `127.0.0.1:${adminPort}`, admin_keys: [adminKey] }
      : { addr: `127.0.0.1:${adminPort}`, enabled: false },
    observability: {
      service_name: "aisix-e2e",
      log_level: "warn",
      access_log: false,
      metrics: {
        prometheus: {
          enabled: prometheusEnabled,
          path: overrides.prometheusPath ?? "/metrics",
          addr: `127.0.0.1:${metricsPort}`,
        },
        otlp: { enabled: false, endpoint: "http://127.0.0.1:4317" },
        ...(overrides.clientTypeRules
          ? { client_type_rules: overrides.clientTypeRules }
          : {}),
      },
      tracing: { otlp: { enabled: false, endpoint: "http://127.0.0.1:4317", sample_ratio: 1 } },
    },
    cache: { backend: "memory" },
    ...(overrides.extra ?? {}),
  };

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

  // Non-config secrets the DP reads straight from its environment (e.g.
  // SLS AccessKeys). Applied last so they survive the AISIX_* strip above.
  for (const [k, v] of Object.entries(overrides.extraEnv ?? {})) {
    childEnv[k] = v;
  }

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
  // Reject the readiness wait the moment the binary exits non-zero, so
  // an intentional boot failure (e.g. a malformed resources file)
  // surfaces immediately instead of after the full readiness timeout.
  const exitedEarly = new Promise<never>((_, reject) => {
    child.once("exit", (code, signal) => {
      if (code !== 0 && code !== null) {
        exitErr = `aisix exited early with code=${code} signal=${signal}`;
        reject(new Error(exitErr));
      }
    });
  });

  const proxyUrl = `http://127.0.0.1:${proxyPort}`;
  const adminUrl = `http://127.0.0.1:${adminPort}`;
  const metricsUrl = `http://127.0.0.1:${metricsPort}`;

  try {
    await Promise.race([
      Promise.all([
        waitForReady(`${proxyUrl}/livez`, READY_TIMEOUT_MS),
        // The admin health endpoint only exists when the admin listener is
        // bound; with `admin: false` there is no admin surface, so gate on
        // the proxy `/livez` and the metrics listener alone.
        ...(adminEnabled
          ? [waitForReady(`${adminUrl}/admin/v1/health`, READY_TIMEOUT_MS, adminKey)]
          : []),
        // Gate on the dedicated metrics listener too, so scrapes in the test
        // never race the listener coming up. Skipped when prometheus is
        // disabled — nothing binds the metrics port then.
        ...(prometheusEnabled
          ? [
              waitForReady(
                `${metricsUrl}${overrides.prometheusPath ?? "/metrics"}`,
                READY_TIMEOUT_MS,
              ),
            ]
          : []),
      ]),
      exitedEarly,
    ]);
  } catch (err) {
    const detail = exitErr ?? "still running";
    // Keep the head too — a startup error (anyhow's `Error: …` line)
    // prints before its backtrace, and a tail-only excerpt used to cut
    // exactly the line that says what went wrong.
    const stderr =
      stderrBuf.length <= 3000
        ? stderrBuf
        : `${stderrBuf.slice(0, 1500)}\n  […]\n${stderrBuf.slice(-1500)}`;
    await terminate(child);
    await cleanup(fileMode ? undefined : etcd, etcdPrefix, dir);
    throw new Error(
      `${(err as Error).message}\n  binary state: ${detail}\n  stderr:\n${stderr}`,
    );
  }

  return {
    proxyUrl,
    adminUrl,
    adminKey,
    etcdPrefix,
    metricsUrl,
    resourcesPath,
    output() {
      return stderrBuf;
    },
    signal(signal: NodeJS.Signals) {
      if (child.exitCode === null) child.kill(signal);
    },
    async exit() {
      await terminate(child);
      await cleanup(fileMode ? undefined : etcd, etcdPrefix, dir);
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

async function cleanup(
  etcd: EtcdClient | undefined,
  prefix: string,
  dir: string,
): Promise<void> {
  // Best-effort — never throw from cleanup. `etcd` is undefined in file
  // mode, where no prefix was ever written.
  await Promise.allSettled([
    ...(etcd ? [etcd.deletePrefix(prefix)] : []),
    rm(dir, { recursive: true, force: true }),
  ]);
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
