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

// A startup config the gateway cannot deserialize stops the boot, and the
// error has to say where the bad key is: its full dotted path, and for the
// env form the variable that set it. Black-box: the contract is what an
// operator reads from a process that refused to start.

/** Boot with `extra` merged into a minimal config; return exit code and output. */
async function boot(
  extra: Record<string, unknown>,
  extraEnv: Record<string, string> = {},
): Promise<{ code: number | null; logs: string }> {
  const dir = await mkdtemp(join(tmpdir(), "aisix-cfg-keypath-"));
  try {
    const cfgPath = join(dir, "config.yaml");
    await writeFile(
      cfgPath,
      yamlStringify({
        etcd: { endpoints: ["http://127.0.0.1:1"], prefix: "/aisix" },
        proxy: { addr: "127.0.0.1:0" },
        admin: { enabled: false },
        shutdown: { min_drain_secs: 0 },
        ...extra,
      }),
      "utf8",
    );
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined && !k.startsWith("AISIX_")) env[k] = v;
    }
    Object.assign(env, extraEnv);
    try {
      const ok = await execFileP(BIN_PATH, ["--config", cfgPath], { env, timeout: 5_000 });
      return { code: 0, logs: `${ok.stdout}${ok.stderr}` };
    } catch (e) {
      const err = e as Error & { code?: number | string; stdout?: string; stderr?: string };
      if (err.code === "ENOENT") {
        throw new Error(`aisix binary not found at ${BIN_PATH}; set AISIX_BIN or build it`);
      }
      return {
        code: typeof err.code === "number" ? err.code : null,
        logs: `${err.stdout ?? ""}${err.stderr ?? ""}`,
      };
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

describe("a startup config error names the key it is about", () => {
  test("a bad nested key in the file is named by its full path", async () => {
    const { code, logs } = await boot({
      upstream: { media_fetch: { allowed_hosts: ["media.qa"] } },
    });
    expect(code).toBe(1);
    expect(logs).toContain("upstream.media_fetch.allowed_hosts");
  });

  test("a bad nested key from the environment names the variable too", async () => {
    const { code, logs } = await boot(
      {},
      { AISIX_SHUTDOWN__BOGUS: "1" },
    );
    expect(code).toBe(1);
    expect(logs).toContain("shutdown.bogus");
    expect(logs).toContain("AISIX_SHUTDOWN__BOGUS");
  });

  test("the env form of a removed nested key names the variable and the path", async () => {
    const { code, logs } = await boot(
      {},
      { AISIX_UPSTREAM__MEDIA_FETCH__ALLOWED_HOSTS: "media.qa" },
    );
    expect(code).toBe(1);
    expect(logs).toContain("upstream.media_fetch.allowed_hosts");
    expect(logs).toContain("AISIX_UPSTREAM__MEDIA_FETCH__ALLOWED_HOSTS");
  });
});
