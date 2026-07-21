import { spawn, type ChildProcess } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { stringify as yamlStringify } from "yaml";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePorts,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

// E2E for the load-observability contract: `GET /status/config`,
// `GET /status/ready`, and the `aisix_config_*` Prometheus series on the
// dedicated metrics/status listener. Exercises both load paths (etcd watch
// and standalone file) against the real binary — the endpoints report the
// gateway's own internal load state, so these are source-blind black-box
// checks of the observable contract.

const CALLER_PLAINTEXT = "sk-status-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

interface StatusConfig {
  state: string;
  source: {
    type: string;
    connected?: boolean;
    observed_revision?: number;
    source_hash?: string;
    observed_at?: string;
  };
  applied?: {
    applied_revision?: number;
    config_hash: string;
    apply_seq: number;
    applied_at: string;
    resource_counts: Record<string, number>;
  };
  last_reload?: { successful: boolean; at: string };
  last_failure: { at: string; last_error_kind: string; last_error: string } | null;
  rejected: Array<{
    resource_kind: string;
    resource_id: string;
    last_error_kind: string;
    last_error: string;
    first_seen_at: string;
    last_seen_at: string;
  }>;
}

async function getStatusConfig(app: SpawnedApp): Promise<StatusConfig> {
  const res = await fetch(`${app.metricsUrl}/status/config`);
  expect(res.status).toBe(200);
  return (await res.json()) as StatusConfig;
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

/** Value of a single-sample (no-label) gauge line from a scrape. */
function gaugeValue(scrapeText: string, metric: string): number | undefined {
  for (const line of scrapeText.split("\n")) {
    if (line.startsWith(`${metric} `)) {
      const v = Number(line.slice(metric.length + 1).trim());
      if (!Number.isNaN(v)) return v;
    }
  }
  return undefined;
}

describe("status/config: etcd watch source", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "status-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "status-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["status-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("clean config reports synced with revisions and resource counts", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    let cfg: StatusConfig | undefined;
    // "synced" alone races the seed: the state flips synced on an EARLY
    // revision before all three seeded resources are applied, and the
    // count asserts below then read undefined. Wait until the applied
    // snapshot actually contains the seeded resources.
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return (
        cfg.state === "synced" &&
        cfg.applied?.resource_counts.models === 1 &&
        cfg.applied?.resource_counts.provider_keys === 1 &&
        cfg.applied?.resource_counts.api_keys === 1
      );
    });

    expect(cfg!.source.type).toBe("etcd");
    expect(cfg!.source.connected).toBe(true);
    // observed >= applied revision, both present in etcd mode.
    expect(typeof cfg!.source.observed_revision).toBe("number");
    expect(typeof cfg!.applied?.applied_revision).toBe("number");
    expect(cfg!.source.observed_revision!).toBeGreaterThanOrEqual(
      cfg!.applied!.applied_revision!,
    );
    expect(cfg!.source.source_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(cfg!.applied!.config_hash).toMatch(/^[0-9a-f]{64}$/);
    // Clean load: applied hash equals the observed source hash.
    expect(cfg!.applied!.config_hash).toBe(cfg!.source.source_hash);
    expect(cfg!.applied!.resource_counts.models).toBe(1);
    expect(cfg!.applied!.resource_counts.provider_keys).toBe(1);
    expect(cfg!.applied!.resource_counts.api_keys).toBe(1);
    expect(cfg!.last_reload?.successful).toBe(true);
    expect(cfg!.last_failure).toBeNull();
    expect(cfg!.rejected).toHaveLength(0);
  });

  test("metrics listener exposes aisix_config_* series", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => (await getStatusConfig(app!)).state === "synced");

    const text = await scrape(app);
    expect(gaugeValue(text, "aisix_config_last_reload_successful")).toBe(1);
    expect(gaugeValue(text, "aisix_config_source_connected")).toBe(1);
    expect(text).toContain("aisix_config_last_reload_success_timestamp_seconds");
    expect(text).toContain("aisix_config_reloads_total");
    expect(text).toMatch(/aisix_config_observed_revision \d+/);
    expect(text).toMatch(/aisix_config_applied_revision \d+/);
    expect(text).toMatch(/aisix_config_hash_info\{hash="[0-9a-f]{64}"\} 1/);
  });

  test("a bad doc alongside good ones degrades without dropping the good config", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const etcd = new EtcdClient();
    // A schema-invalid model written straight to etcd (empty display_name
    // violates minLength) — the CP would reject it, but the DP loader must
    // skip it and keep serving the rest.
    const badId = randomUUID();
    await etcd.put(
      `${app.etcdPrefix}/models/${badId}`,
      JSON.stringify({
        display_name: "",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: "pk-nonexistent",
      }),
    );

    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.state === "degraded" && cfg.rejected.length > 0;
    });

    // The good model is still applied (serving), not dropped.
    expect(cfg!.applied!.resource_counts.models).toBe(1);
    const rej = cfg!.rejected.find((r) => r.resource_id === badId);
    expect(rej, JSON.stringify(cfg!.rejected)).toBeDefined();
    expect(rej!.resource_kind).toBe("models");
    expect(rej!.last_error_kind).toBe("schema_failed");
    expect(rej!.first_seen_at).toMatch(/Z$/);

    // The bad doc surfaces on the metrics listener too.
    const text = await scrape(app);
    expect(text).toMatch(/aisix_config_rejected_resources\{kind="models"\} 1/);
    expect(gaugeValue(text, "aisix_config_last_reload_successful")).toBe(0);

    // End-to-end: a real chat on the still-good model returns 200.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const chat = await proxy.chat({
      model: "status-model",
      messages: [{ role: "user", content: "still serving despite the bad doc" }],
    });
    expect(chat.status, JSON.stringify(chat.body)).toBe(200);

    // Cleanup the bad doc so it doesn't bleed into sibling tests.
    await etcd.delete(`${app.etcdPrefix}/models/${badId}`);
  });

  test("a rejected provider_key never leaks its secret into status or metrics", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const SECRET = "sk-LEAKED-SECRET-abcdef0123456789";
    const leakId = randomUUID();
    const etcd = new EtcdClient();
    // The credential itself is the schema-failing instance: `api_key` must be
    // a string, so an array fails validation and the UNMASKED error message
    // would echo the secret. This asserts the masking actually redacts it —
    // not that the secret merely happens to sit in a valid sibling field.
    await etcd.put(
      `${app.etcdPrefix}/provider_keys/${leakId}`,
      JSON.stringify({ display_name: "leak-pk", api_key: [SECRET] }),
    );

    await waitConfigPropagation(async () => {
      const cfg = await getStatusConfig(app!);
      return cfg.rejected.some((r) => r.resource_id === leakId);
    });

    // The secret must be absent from the whole /status/config body …
    const raw = await fetch(`${app.metricsUrl}/status/config`).then((r) => r.text());
    expect(raw).not.toContain(SECRET);
    // … and from the metrics scrape.
    const text = await scrape(app);
    expect(text).not.toContain(SECRET);

    await etcd.delete(`${app.etcdPrefix}/provider_keys/${leakId}`);
  });
});

describe("status/config: standalone file source", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      resourcesFile: `
_format_version: "1"
provider_keys:
  - display_name: file-status-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstream.baseUrl}/v1
models:
  - display_name: file-status-model
    provider: openai
    model_name: gpt-4o-mini
    provider_key: file-status-pk
api_keys:
  - display_name: file-status-caller
    key_hash: ${CALLER_KEY_HASH}
    allowed_models: ["file-status-model"]
`,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("file mode boots synced with no revision fields", async () => {
    if (!app) throw new Error("setup failed");
    // File mode is synchronous at boot — no propagation wait needed.
    const cfg = await getStatusConfig(app);
    expect(cfg.state).toBe("synced");
    expect(cfg.source.type).toBe("file");
    // Revision + connected fields are etcd-only and must be absent in file mode.
    expect(cfg.source.connected).toBeUndefined();
    expect(cfg.source.observed_revision).toBeUndefined();
    expect(cfg.applied?.applied_revision).toBeUndefined();
    expect(cfg.source.source_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(cfg.applied!.resource_counts.models).toBe(1);
    expect(cfg.applied!.resource_counts.provider_keys).toBe(1);
    expect(cfg.rejected).toHaveLength(0);

    // The scrape omits the etcd-only revision/connected series in file mode.
    const text = await scrape(app);
    expect(gaugeValue(text, "aisix_config_last_reload_successful")).toBe(1);
    expect(text).not.toContain("aisix_config_observed_revision");
    expect(text).not.toContain("aisix_config_source_connected");
  });

  test("status/ready is 200 once the file config is applied", async () => {
    if (!app) throw new Error("setup failed");
    const res = await fetch(`${app.metricsUrl}/status/ready`);
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("ok");
  });
});

// A DP pointed at an unreachable etcd never applies a config, so it stays in
// `never_loaded` — but that also makes `/admin/v1/health` (an etcd-backed
// store read) fail, which the standard harness gate waits on. So this case
// spawns the binary directly and gates only on the metrics/status listener,
// which binds independently of the config source.
interface MinimalApp {
  metricsUrl: string;
  stop(): Promise<void>;
}

async function spawnPointedAtDeadEtcd(): Promise<MinimalApp> {
  const [proxyPort, adminPort, metricsPort, deadPort] = await pickFreePorts(4);
  const dir = await mkdtemp(join(tmpdir(), "aisix-status-nl-"));
  const cfg = {
    // etcd-client's connect is lazy (no eager dial without auth), so the
    // gateway boots and binds its listeners even though nothing answers here;
    // the watch supervisor's first load never succeeds → never_loaded.
    etcd: {
      endpoints: [`http://127.0.0.1:${deadPort}`],
      prefix: "/aisix-never-loaded",
      dial_timeout_ms: 2000,
      request_timeout_ms: 2000,
    },
    proxy: { addr: `127.0.0.1:${proxyPort}`, request_body_limit_bytes: 10485760 },
    admin: { addr: `127.0.0.1:${adminPort}`, admin_keys: [`admin-${randomUUID()}`] },
    observability: {
      service_name: "aisix-status-nl",
      log_level: "warn",
      access_log: false,
      metrics: {
        prometheus: { enabled: true, path: "/metrics", addr: `127.0.0.1:${metricsPort}` },
        otlp: { enabled: false, endpoint: "http://127.0.0.1:4317" },
      },
      tracing: { otlp: { enabled: false, endpoint: "http://127.0.0.1:4317", sample_ratio: 1 } },
    },
    cache: { backend: "memory" },
  };
  const cfgPath = join(dir, "config.yaml");
  await writeFile(cfgPath, yamlStringify(cfg), "utf8");

  // Strip AISIX_* so the harness's own env can't override the config.
  const childEnv: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined && !k.startsWith("AISIX_")) childEnv[k] = v;
  }
  childEnv.RUST_LOG = "warn";

  const child: ChildProcess = spawn(BIN_PATH, ["--config", cfgPath], {
    stdio: ["ignore", "ignore", "ignore"],
    env: childEnv,
  });
  const metricsUrl = `http://127.0.0.1:${metricsPort}`;

  // Gate only on the metrics/status listener answering (any status — a 503
  // from /status/ready still means the listener is up).
  const deadline = Date.now() + 10_000;
  for (;;) {
    if (child.exitCode !== null) throw new Error(`aisix exited early code=${child.exitCode}`);
    try {
      const res = await fetch(`${metricsUrl}/status/ready`);
      await res.text();
      break;
    } catch {
      if (Date.now() > deadline) throw new Error("metrics listener never came up");
      await new Promise((r) => setTimeout(r, 100));
    }
  }

  return {
    metricsUrl,
    async stop() {
      if (child.exitCode === null) child.kill("SIGTERM");
      await Promise.race([
        new Promise<void>((r) => child.once("exit", () => r())),
        new Promise<void>((r) => setTimeout(r, 3000)),
      ]);
      if (child.exitCode === null) child.kill("SIGKILL");
      await rm(dir, { recursive: true, force: true });
    },
  };
}

describe("status/ready: never_loaded before any config", () => {
  let app: MinimalApp | undefined;

  beforeAll(async () => {
    app = await spawnPointedAtDeadEtcd();
  });

  afterAll(async () => {
    await app?.stop();
  });

  test("status/ready is 503 and status/config reports never_loaded", async () => {
    if (!app) throw new Error("setup failed");

    const ready = await fetch(`${app.metricsUrl}/status/ready`);
    expect(ready.status).toBe(503);
    expect(await ready.text()).toBe("no configuration available");

    const res = await fetch(`${app.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    const cfg = (await res.json()) as StatusConfig;
    expect(cfg.state).toBe("never_loaded");
    expect(cfg.applied).toBeUndefined();
    // The etcd source shows disconnected while it keeps retrying.
    expect(cfg.source.type).toBe("etcd");
    expect(cfg.source.connected).toBe(false);
  });
});
