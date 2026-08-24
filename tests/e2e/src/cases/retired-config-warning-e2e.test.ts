import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { describe, expect, test } from "vitest";
import { stringify as yamlStringify } from "yaml";

const execFileP = promisify(execFile);

const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

// E2E for the retired static OTLP blocks (AISIX-Cloud#1380).
//
// `observability.tracing.otlp` and `observability.metrics.otlp` were
// placeholders no code ever read, and the gateway used to log "OTLP tracer
// configured" for them — a success signal for a pipeline that did not
// exist. They are gone from the shipped example configs, but a config
// copied from an older one still carries them, so they still parse. What
// must not come back is the silence: boot names each key it is ignoring.
//
// Black-box on purpose: the contract is what an operator sees in the log
// of a real process, and the env form is how a container deployment sets
// these. Nothing here needs etcd — the endpoint below is never reachable
// and the gateway is stopped while it is still retrying.

/** Boot the gateway, stop it, and return everything it logged. */
async function bootAndCollectLogs(
  observability: Record<string, unknown>,
  extraEnv: Record<string, string> = {},
): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "aisix-retired-cfg-"));
  try {
    const cfgPath = join(dir, "config.yaml");
    await writeFile(
      cfgPath,
      yamlStringify({
        etcd: { endpoints: ["http://127.0.0.1:1"], prefix: "/aisix" },
        proxy: { addr: "127.0.0.1:0" },
        admin: { enabled: false },
        observability,
        // No load balancer fronts this process; drain immediately so the
        // SIGTERM below is a clean exit rather than a 30s wait.
        shutdown: { min_drain_secs: 0 },
      }),
      "utf8",
    );

    // Strip AISIX_* so the ambient harness environment cannot override the
    // config under test, then add back only this case's own override.
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined && !k.startsWith("AISIX_")) env[k] = v;
    }
    Object.assign(env, extraEnv);

    // The gateway does not exit on its own — it keeps retrying the dead
    // etcd endpoint — so `timeout` is how this run ends. Either outcome
    // carries the logs: a clean SIGTERM exit resolves, anything else
    // rejects with the same streams attached.
    try {
      const ok = await execFileP(BIN_PATH, ["--config", cfgPath], { env, timeout: 5_000 });
      return `${ok.stdout}${ok.stderr}`;
    } catch (e) {
      const err = e as Error & { stdout?: string; stderr?: string };
      return `${err.stdout ?? ""}${err.stderr ?? ""}`;
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

const PROMETHEUS_OFF = { prometheus: { enabled: false, path: "/metrics", addr: "127.0.0.1:0" } };

describe("retired startup settings are named at boot, not silently ignored", () => {
  test("a config copied from an older example still boots, and says why the blocks do nothing", async () => {
    const logs = await bootAndCollectLogs({
      log_level: "info",
      metrics: {
        ...PROMETHEUS_OFF,
        otlp: { enabled: false, endpoint: "http://127.0.0.1:4317" },
      },
      // Disabled is the copied-in shape almost every affected config has,
      // and it must warn too — the operator is being told to delete it,
      // not that it failed to turn on.
      tracing: { otlp: { enabled: false, endpoint: "http://127.0.0.1:4317", sample_ratio: 1 } },
    });

    expect(logs).toContain("observability.tracing.otlp");
    expect(logs).toContain("otlp_http");
    expect(logs).toContain("observability.metrics.otlp");
    expect(logs).toContain("observability.metrics.prometheus.addr");
    // The claim the issue was filed about must be gone for good.
    expect(logs).not.toContain("OTLP tracer configured");
  });

  test("the env-var form a container deployment uses is caught too", async () => {
    const logs = await bootAndCollectLogs(
      { log_level: "info", metrics: PROMETHEUS_OFF },
      {
        AISIX_OBSERVABILITY__TRACING__OTLP__ENABLED: "true",
        AISIX_OBSERVABILITY__TRACING__OTLP__ENDPOINT: "http://127.0.0.1:4317",
      },
    );
    expect(logs).toContain("observability.tracing.otlp");
  });

  test("a config that never carried the blocks stays quiet", async () => {
    const logs = await bootAndCollectLogs({ log_level: "info", metrics: PROMETHEUS_OFF });
    expect(logs).not.toContain("observability.tracing.otlp");
    expect(logs).not.toContain("observability.metrics.otlp");
  });
});
