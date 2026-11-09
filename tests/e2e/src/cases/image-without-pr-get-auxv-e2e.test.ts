import { execFile as execFileCb } from "node:child_process";
import { randomUUID } from "node:crypto";
import { copyFile, link, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { scrapeMetrics, suiteThreadPerCore, sumMetric, type MetricSample } from "../harness/index.js";

// The published image, started the way it ships, on a kernel older than 6.4.
//
// The image gives the binary a file capability and runs it as a non-root user.
// A process started with a file capability is not dumpable, so its own
// /proc/<pid>/auxv belongs to root and it cannot read it. Linux 6.4 added
// prctl(PR_GET_AUXV), which works regardless. Without that prctl the process
// has only what libc kept from its start, and anything that reaches for the
// proc file gets nothing.
//
// CI kernels have the prctl, so a seccomp profile takes it away: it answers
// EINVAL, exactly what an older kernel does for an option it does not know.
// The image is built from the repository's own Dockerfile, so the capability
// and the user are the ones that ship.

const execFile = promisify(execFileCb);

const REPO_ROOT = join(process.cwd(), "..", "..");
const BINARY = process.env.AISIX_BIN ?? join(REPO_ROOT, "target", "debug", "aisix");
const IMAGE_BUILD_TIMEOUT_MS = 300_000;

const PR_GET_AUXV = 0x41555856;
const EINVAL = 22;
const PRCTL_SYSCALL: Record<string, number> = { x64: 157, arm64: 167 };
const CAP_NET_BIND_SERVICE_BIT = 10n;

const docker = async (...args: string[]) =>
  (await execFile("docker", args, { maxBuffer: 64 * 1024 * 1024 })).stdout.trim();

const dockerAvailable = async () => {
  try {
    await docker("version", "--format", "{{.Server.Version}}");
    return true;
  } catch {
    return false;
  }
};

const sample = (samples: MetricSample[], name: string) => samples.find((s) => s.name === name);

const hasDocker = await dockerAvailable();
if (!hasDocker && process.env.CI) throw new Error("this spec needs a docker daemon, and CI has one");

// The test binary goes into a Linux image as it is.
describe.skipIf(!hasDocker || process.platform !== "linux")("image default start on a kernel without PR_GET_AUXV", () => {
  const id = randomUUID().slice(0, 8);
  const image = `aisix-e2e-no-pr-get-auxv:${id}`;
  const container = `aisix-e2e-no-pr-get-auxv-${id}`;
  let dir: string | undefined;
  let metricsUrl = "";
  let gatewayPid = "";

  const inContainer = (script: string) => docker("exec", container, "sh", "-c", script);

  beforeAll(async () => {
    dir = await mkdtemp(join(tmpdir(), "aisix-e2e-no-pr-get-auxv-"));
    const context = join(dir, "context");
    await mkdir(join(context, "bin"), { recursive: true });
    await mkdir(join(context, "docker"));
    await copyFile(join(REPO_ROOT, "Dockerfile"), join(context, "Dockerfile"));
    await copyFile(join(REPO_ROOT, "config.managed.yaml"), join(context, "config.managed.yaml"));
    await copyFile(join(REPO_ROOT, "docker", "entrypoint.sh"), join(context, "docker", "entrypoint.sh"));
    const arch = process.arch === "arm64" ? "arm64" : "amd64";
    const staged = join(context, "bin", `aisix-${arch}`);
    // The debug binary is hundreds of megabytes; /tmp is often another device.
    await link(BINARY, staged).catch(() => copyFile(BINARY, staged));
    await docker("build", "--quiet", "-t", image, context);

    const seccomp = join(dir, "seccomp.json");
    await writeFile(
      seccomp,
      JSON.stringify({
        defaultAction: "SCMP_ACT_ALLOW",
        syscalls: [
          {
            names: ["prctl"],
            action: "SCMP_ACT_ERRNO",
            errnoRet: EINVAL,
            args: [{ index: 0, value: PR_GET_AUXV, op: "SCMP_CMP_EQ" }],
          },
        ],
      }),
    );
    const config = join(dir, "config.yaml");
    await writeFile(
      config,
      [
        "resources_file: /etc/aisix/resources.yaml",
        "proxy:",
        '  addr: "0.0.0.0:3000"',
        ...(suiteThreadPerCore !== undefined ? [`  thread_per_core: ${suiteThreadPerCore}`] : []),
        "admin:",
        '  addr: "0.0.0.0:3001"',
        "  admin_keys: [e2e-admin-key]",
        "observability:",
        "  heap_profiling:",
        "    auto_dump:",
        "      thresholds: [0.000001]",
        "",
      ].join("\n"),
    );
    const resources = join(dir, "resources.yaml");
    await writeFile(resources, '_format_version: "1"\n');

    await docker(
      "run",
      "--detach",
      "--name",
      container,
      "--security-opt",
      `seccomp=${seccomp}`,
      "--publish",
      "127.0.0.1:0:9090",
      "--volume",
      `${config}:/etc/aisix/config.yaml:ro`,
      "--volume",
      `${resources}:/etc/aisix/resources.yaml:ro`,
      image,
    );
    metricsUrl = `http://${(await docker("port", container, "9090/tcp")).split("\n")[0]}`;

    const until = Date.now() + 30_000;
    for (;;) {
      // Connection refused is the normal state until the listener binds.
      const ready = await fetch(`${metricsUrl}/status/ready`)
        .then((r) => r.ok)
        .catch(() => false);
      if (ready) break;
      const running = (await docker("inspect", "--format", "{{.State.Running}}", container)) === "true";
      if (!running || Date.now() > until) {
        throw new Error(`gateway never became ready (running=${running}):\n${await containerLogs()}`);
      }
      await new Promise((r) => setTimeout(r, 200));
    }
    gatewayPid = await inContainer(
      'for p in /proc/[0-9]*; do [ "$(cat "$p/comm" 2>/dev/null)" = aisix ] && echo "${p#/proc/}"; done; true',
    );
  }, IMAGE_BUILD_TIMEOUT_MS);

  afterAll(async () => {
    if (dir === undefined) return;
    await docker("rm", "--force", container).catch(() => {});
    await docker("image", "rm", "--force", image).catch(() => {});
    await rm(dir, { recursive: true, force: true });
  }, 60_000);

  const containerLogs = async () => {
    const { stdout, stderr } = await execFile("docker", ["logs", container], {
      maxBuffer: 64 * 1024 * 1024,
    });
    return stdout + stderr;
  };

  test("the container reproduces the condition: no prctl, and the proc file is root's", async () => {
    const nr = PRCTL_SYSCALL[process.arch];
    expect(nr, process.arch).toBeDefined();
    const errno = await docker(
      "exec",
      container,
      "perl",
      "-e",
      `$! = 0; my $buf = "\\0" x 512; syscall(${nr}, ${PR_GET_AUXV}, $buf, 512, 0, 0); print $! + 0;`,
    );
    expect(Number(errno), "PR_GET_AUXV must answer EINVAL inside the container").toBe(EINVAL);

    expect(gatewayPid).toMatch(/^\d+$/);
    const status = await inContainer(`cat /proc/${gatewayPid}/status`);
    expect(status).toMatch(/^Uid:\s+10001\s/m);
    expect(status).toMatch(/^NoNewPrivs:\s+0$/m);
    const capEff = BigInt(`0x${/^CapEff:\s+([0-9a-f]+)$/m.exec(status)![1]}`);
    expect((capEff >> CAP_NET_BIND_SERVICE_BIT) & 1n, "file capability took effect").toBe(1n);

    const auxv = await inContainer(`stat -c '%u %a' /proc/${gatewayPid}/auxv`);
    expect(auxv, "a non-dumpable process's auxv belongs to root").toBe("0 400");
  });

  test("process and allocator families are all there, with believable values", async () => {
    for (let scrape = 0; scrape < 2; scrape += 1) {
      const s = await scrapeMetrics(metricsUrl);
      const rss = sample(s, "process_resident_memory_bytes")?.value;
      expect(rss, "process_resident_memory_bytes").toBeGreaterThan(1 << 20);
      expect(sample(s, "process_virtual_memory_bytes")?.value).toBeGreaterThanOrEqual(rss!);
      expect(sample(s, "process_threads")?.value).toBeGreaterThanOrEqual(2);
      const fds = sample(s, "process_open_fds")?.value;
      expect(fds, "process_open_fds").toBeGreaterThan(0);
      expect(sample(s, "process_max_fds")?.value).toBeGreaterThanOrEqual(fds!);
      expect(sample(s, "process_cpu_seconds_total")?.value).toBeGreaterThan(0);
      const started = sample(s, "process_start_time_seconds")?.value;
      expect(started, "process_start_time_seconds").toBeGreaterThan(Date.now() / 1000 - 600);
      expect(started).toBeLessThanOrEqual(Date.now() / 1000);
      expect(sumMetric(s, "aisix_allocator_bytes", { stat: "allocated" })).toBeGreaterThan(0);
    }
  });

  test("the automatic heap dump runs: its thread lives and writes a profile", async () => {
    const until = Date.now() + 30_000;
    let dumps = 0;
    while (Date.now() < until && dumps === 0) {
      dumps = sumMetric(await scrapeMetrics(metricsUrl), "aisix_heap_profile_dumps_total", {
        trigger: "auto",
        result: "ok",
      });
      if (dumps === 0) await new Promise((r) => setTimeout(r, 200));
    }
    expect(dumps, "automatic heap dumps written").toBeGreaterThan(0);
    const threads = await inContainer(`cat /proc/${gatewayPid}/task/*/comm`);
    expect(threads.split("\n")).toContain("heap-autodump");
  });

  test("nothing panicked", async () => {
    expect(await containerLogs()).not.toContain("panicked");
  });
});
