import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { stringify as yamlStringify } from "yaml";
import { Agent, request } from "undici";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, pickFreePorts } from "../harness/index.js";

// E2E: `proxy.tls` / `admin.tls` actually serve HTTPS.
//
// Pins issue #473: both TLS config blocks were parsed into the config
// structs but never wired into the Axum listeners, so the ports kept
// serving plain HTTP while the docs claimed TLS. This spec configures a
// self-signed cert on both listeners and asserts:
//   1. HTTPS works on the proxy listener (`/livez`).
//   2. HTTPS works on the admin listener (`/metrics`, unauthenticated).
//   3. A plain-HTTP request to the TLS proxy port does NOT succeed.

const BIN_PATH =
  process.env.AISIX_BIN ??
  join(process.cwd(), "..", "..", "target", "debug", "aisix");

const insecureAgent = new Agent({ connect: { rejectUnauthorized: false } });

async function waitForHttpsReady(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  while (Date.now() < deadline) {
    try {
      const res = await request(url, { dispatcher: insecureAgent });
      res.body.dump();
      if (res.statusCode === 200) return;
    } catch (err) {
      lastErr = err;
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`timed out waiting for ${url}: ${String(lastErr)}`);
}

describe("listener TLS (#473)", () => {
  let child: ChildProcess | undefined;
  let dir: string | undefined;
  let proxyPort = 0;
  let adminPort = 0;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    dir = await mkdtemp(join(tmpdir(), "aisix-tls-e2e-"));
    const certFile = join(dir, "server.crt");
    const keyFile = join(dir, "server.key");
    // Self-signed cert valid for 127.0.0.1.
    execFileSync("openssl", [
      "req", "-x509", "-newkey", "rsa:2048",
      "-keyout", keyFile, "-out", certFile,
      "-days", "1", "-nodes",
      "-subj", "/CN=127.0.0.1",
      "-addext", "subjectAltName=IP:127.0.0.1",
    ]);

    [proxyPort, adminPort] = await pickFreePorts(2);
    const adminKey = `admin-${randomUUID()}`;
    const cfg = {
      etcd: {
        endpoints: [process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379"],
        prefix: `/aisix-e2e-tls-${randomUUID()}`,
        dial_timeout_ms: 5000,
        request_timeout_ms: 5000,
      },
      proxy: {
        addr: `127.0.0.1:${proxyPort}`,
        request_body_limit_bytes: 10485760,
        tls: { cert_file: certFile, key_file: keyFile },
      },
      admin: {
        addr: `127.0.0.1:${adminPort}`,
        admin_keys: [adminKey],
        tls: { cert_file: certFile, key_file: keyFile },
      },
      observability: {
        service_name: "aisix-e2e-tls",
        log_level: "warn",
        access_log: false,
        metrics: { prometheus: { enabled: true, path: "/metrics" } },
      },
      cache: { backend: "memory" },
    };
    const cfgPath = join(dir, "config.yaml");
    await writeFile(cfgPath, yamlStringify(cfg), "utf8");

    const childEnv: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined && !k.startsWith("AISIX_")) childEnv[k] = v;
    }
    childEnv.RUST_LOG = process.env.RUST_LOG ?? "warn";
    childEnv.NO_PROXY = "127.0.0.1,localhost";
    childEnv.no_proxy = "127.0.0.1,localhost";

    child = spawn(BIN_PATH, ["--config", cfgPath], {
      stdio: ["ignore", "pipe", "pipe"],
      env: childEnv,
    });
    let buf = "";
    child.stderr?.on("data", (c: Buffer) => (buf += c.toString("utf8")));
    child.stdout?.on("data", (c: Buffer) => (buf += c.toString("utf8")));

    try {
      await waitForHttpsReady(`https://127.0.0.1:${proxyPort}/livez`, 15_000);
    } catch (err) {
      throw new Error(`${String(err)}\n--- aisix output ---\n${buf}`);
    }
  }, 30_000);

  afterAll(async () => {
    if (child) {
      child.kill("SIGTERM");
      await new Promise((r) => setTimeout(r, 500));
      child.kill("SIGKILL");
    }
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  test("proxy listener serves HTTPS", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await request(`https://127.0.0.1:${proxyPort}/livez`, {
      dispatcher: insecureAgent,
    });
    res.body.dump();
    expect(res.statusCode).toBe(200);
  });

  test("admin listener serves HTTPS", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await request(`https://127.0.0.1:${adminPort}/metrics`, {
      dispatcher: insecureAgent,
    });
    res.body.dump();
    expect(res.statusCode).toBe(200);
  });

  test("plain HTTP to the TLS proxy port does not succeed", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    // A plain-HTTP request hitting a TLS listener must not yield a 200.
    let ok = false;
    try {
      const res = await request(`http://127.0.0.1:${proxyPort}/livez`, {
        dispatcher: insecureAgent,
        headersTimeout: 2000,
        bodyTimeout: 2000,
      });
      res.body.dump();
      ok = res.statusCode === 200;
    } catch {
      ok = false;
    }
    expect(ok).toBe(false);
  });
});
