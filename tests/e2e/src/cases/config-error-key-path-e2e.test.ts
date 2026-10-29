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
// error has to say where the bad value is: its full dotted path, and for the
// env form the variable that set it. A key the gateway does not know does
// not stop it: it is named in a warning, and in the error of a load that
// fails for another reason. Black-box: the contract is what an operator
// reads from the process.

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
  test("a bad nested value in the file is named by its full path", async () => {
    const { code, logs } = await boot({
      upstream: { media_fetch: { max_bytes: "lots" } },
    });
    expect(code).toBe(1);
    expect(logs).toContain("upstream.media_fetch.max_bytes");
  });

  test("a bad nested value from the environment names the variable too", async () => {
    const { code, logs } = await boot({}, { AISIX_SHUTDOWN__MIN_DRAIN_SECS: "soon" });
    expect(code).toBe(1);
    expect(logs).toContain("shutdown.min_drain_secs (set by AISIX_SHUTDOWN__MIN_DRAIN_SECS)");
  });

  test("a load error also names the keys it ignored", async () => {
    // The misspelling is usually the cause: `adress` is why the required
    // `proxy.addr` is missing.
    const { code, logs } = await boot({ proxy: { adress: "127.0.0.1:0" } });
    expect(code).toBe(1);
    expect(logs).toContain("also ignored, as no setting this gateway knows: proxy.adress");
  });
});

describe("a key the gateway does not know is ignored and named", () => {
  // Before 1.8.0 each of these stopped the boot with `unknown field`. The
  // gateway now keeps running (here: retrying the dead etcd endpoint until
  // the harness timeout stops it) and names the key.
  test("a removed nested key in the file is named by its full path", async () => {
    const { code, logs } = await boot({
      upstream: { media_fetch: { allowed_hosts: ["media.qa"] } },
    });
    expect(code).not.toBe(1);
    expect(logs).toContain(
      "upstream.media_fetch.allowed_hosts is not a setting this gateway knows and was ignored.",
    );
  });

  test("an unknown nested key from the environment names the variable too", async () => {
    const { code, logs } = await boot({}, { AISIX_SHUTDOWN__BOGUS: "1" });
    expect(code).not.toBe(1);
    expect(logs).toContain(
      "shutdown.bogus (set by AISIX_SHUTDOWN__BOGUS) is not a setting this gateway knows and was ignored.",
    );
  });
});
