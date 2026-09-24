import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, test, vi } from "vitest";

const dirs: string[] = [];

afterEach(async () => {
  vi.unstubAllEnvs();
  vi.resetModules();
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function failingProcess(source: string) {
  const dir = await mkdtemp(join(tmpdir(), "aisix-harness-output-"));
  dirs.push(dir);
  const executable = join(dir, "process.cjs");
  await writeFile(executable, `#!/usr/bin/env node\n${source}`);
  await chmod(executable, 0o755);
  vi.stubEnv("AISIX_BIN", executable);
  vi.resetModules();
  return (await import("./app.js")).spawnApp;
}

test("startup assertions retain an error between long logs and a backtrace", async () => {
  const spawnApp = await failingProcess(`
    const { writeFileSync } = require("node:fs");
    writeFileSync(2, "startup warning\\n".repeat(300));
    writeFileSync(2, 'Error: unsupported variable "request_id"\\n');
    writeFileSync(2, "backtrace frame\\n".repeat(300));
    process.exit(1);
  `);
  await expect(spawnApp({ resourcesFile: '_format_version: "1"\n' })).rejects.toThrow(/unsupported variable.*request_id/);
});

test("startup failure includes output drained after the child exits", async () => {
  const spawnApp = await failingProcess(`
    const { spawn } = require("node:child_process");
    spawn(process.execPath, ["-e", 'setTimeout(() => require("node:fs").writeFileSync(2, "final startup diagnostic\\\\n"), 100)'], {
      stdio: ["ignore", "inherit", "inherit"],
    }).unref();
    process.exit(1);
  `);
  await expect(spawnApp({ resourcesFile: '_format_version: "1"\n' })).rejects.toThrow("final startup diagnostic");
});

test("a real gateway startup error survives surrounding output", async () => {
  const binary = process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");
  const spawnApp = await failingProcess(`
    const { writeFileSync } = require("node:fs");
    const { spawnSync } = require("node:child_process");
    writeFileSync(2, "startup warning\\n".repeat(300));
    const result = spawnSync(${JSON.stringify(binary)}, process.argv.slice(2), {
      encoding: "utf8", timeout: 5000,
    });
    writeFileSync(1, result.stdout ?? "");
    writeFileSync(2, result.stderr ?? "");
    writeFileSync(2, "shutdown diagnostic\\n".repeat(300));
    process.exit(result.status ?? 2);
  `);
  await expect(spawnApp({ resourcesFile: '_format_version: "1"\n', extraEnv: {
    AISIX_OBSERVABILITY__METRICS__LABELS: JSON.stringify({ aisix_request_ttft_seconds: ["request_id"] }),
  } })).rejects.toThrow(/unsupported variable.*request_id/);
});
