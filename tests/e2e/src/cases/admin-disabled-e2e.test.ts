import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the gateway runs with the admin listener switched off
// (`admin.enabled = false`), the shape it takes once the Admin API is
// removed. Resources reach the gateway declaratively — never the Admin
// API — and the full request path must still work, with the
// metrics/status listener as the only feedback surface. Both resource
// sources are covered, because they bind the admin surface through
// distinct store variants (etcd vs the file-managed snapshot):
//   - etcd source, seeded straight to etcd via SeedClient;
//   - standalone `resources_file`, loaded from a declarative file.

const ETCD_CALLER_PLAINTEXT = "sk-admin-off-etcd-caller";
const ETCD_CALLER_KEY_HASH = createHash("sha256")
  .update(ETCD_CALLER_PLAINTEXT)
  .digest("hex");

const FILE_CALLER_PLAINTEXT = "sk-admin-off-file-caller";
// `key_env` sugar: the plaintext travels to the gateway via this env var
// (never `AISIX_`-prefixed — the config loader claims that namespace).
const FILE_CALLER_KEY_ENV = "ADMIN_OFF_FILE_CALLER_KEY";

/**
 * With `admin.enabled = false` the admin port is never bound, so a
 * connection to it is refused (the port was reserved by the harness, but
 * nothing listens). Assert the specific connection-refused cause rather
 * than any throw, so a wrongly-bound listener — or a DNS/abort error —
 * can't satisfy the check.
 */
async function expectAdminPortRefused(adminUrl: string): Promise<void> {
  let caught: unknown;
  try {
    await fetch(`${adminUrl}/admin/v1/health`);
  } catch (e) {
    caught = e;
  }
  expect(caught).toBeDefined();
  const code = (caught as { cause?: { code?: string } } | undefined)?.cause
    ?.code;
  expect(code).toBe("ECONNREFUSED");
}

describe("admin disabled, etcd source: gateway serves seeded-via-etcd config with no admin listener", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // No admin listener is bound — spawnApp gates readiness on the proxy
    // and the metrics listener instead of the admin health endpoint.
    app = await spawnApp({ admin: false });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "admin-off-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "admin-off-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: ETCD_CALLER_KEY_HASH,
      allowed_models: ["admin-off-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a request seeded only through etcd succeeds with the admin listener off", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: ETCD_CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // A 200 means the ProviderKey + Model + ApiKey — all written to etcd,
    // none through the Admin API — propagated into the snapshot and the
    // proxy dispatched to the upstream.
    let responded = false;
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "admin-off-model",
          messages: [{ role: "user", content: "admin-off-probe" }],
        });
        responded = r.choices[0]?.message.role === "assistant";
        return responded;
      } catch {
        return false;
      }
    });
    expect(responded).toBe(true);
    expect(upstream!.receivedRequests.length).toBeGreaterThan(0);
  });

  test("the metrics/status listener reports exactly the applied configuration", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // The load-observability contract is the operational feedback that
    // replaces admin reads: served on the metrics listener, so it stays
    // available with the admin listener off. The etcd prefix is unique to
    // this spawn, so the seeded one-of-each is the whole population.
    const res = await fetch(`${app.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      state?: string;
      applied?: { resource_counts?: Record<string, number> };
    };
    expect(typeof body.state).toBe("string");
    const counts = body.applied?.resource_counts ?? {};
    expect(counts.models).toBe(1);
    expect(counts.provider_keys).toBe(1);
    expect(counts.api_keys).toBe(1);
  });

  test("the proxy health endpoints serve with the admin listener off", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // /livez and /readyz live on the proxy listener — the health surface
    // that survives the Admin API's removal. /readyz carries the full
    // readiness semantics (config freshness from the watch + shutdown);
    // it reports ready once the initial load applies (an empty prefix
    // counts), so it settles at 200 regardless of seed timing.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/readyz`);
        await r.text();
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const livez = await fetch(`${app!.proxyUrl}/livez`);
    expect(livez.status).toBe(200);
    expect(await livez.text()).toBe("ok");

    const verbose = await fetch(`${app!.proxyUrl}/readyz?verbose`);
    expect(verbose.status).toBe(200);
    const body = await verbose.text();
    expect(body).toContain("[+]shutdown ok");
    expect(body).toContain("[+]config ok");
  });

  test("no admin listener is bound", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await expectAdminPortRefused(app.adminUrl);
  });
});

describe("admin disabled, file source: gateway serves a declarative resources.yaml with no admin listener", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    // File mode never contacts etcd; combined with admin off, the gateway
    // has neither an admin surface nor a configuration store — the shape a
    // single-container standalone gateway takes post-removal.
    app = await spawnApp({
      admin: false,
      resourcesFile: `
_format_version: "1"
provider_keys:
  - display_name: admin-off-file-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstream.baseUrl}/v1
models:
  - display_name: admin-off-file-model
    provider: openai
    model_name: gpt-4o-mini
    provider_key: admin-off-file-pk
api_keys:
  - display_name: admin-off-file-caller
    key_env: ${FILE_CALLER_KEY_ENV}
    allowed_models: ["admin-off-file-model"]
`,
      extraEnv: { [FILE_CALLER_KEY_ENV]: FILE_CALLER_PLAINTEXT },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a request from a declarative file serves with the admin listener off", async () => {
    if (!app || !upstream) throw new Error("setup failed");

    const client = new OpenAI({
      apiKey: FILE_CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    let responded = false;
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "admin-off-file-model",
          messages: [{ role: "user", content: "admin-off-file-probe" }],
        });
        responded = r.choices[0]?.message.role === "assistant";
        return responded;
      } catch {
        return false;
      }
    });
    expect(responded).toBe(true);
    expect(upstream.receivedRequests.length).toBeGreaterThan(0);
  });

  test("the metrics/status listener reports the file-loaded configuration", async () => {
    if (!app) throw new Error("setup failed");

    const res = await fetch(`${app.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      state?: string;
      source?: { type?: string };
      applied?: { resource_counts?: Record<string, number> };
    };
    expect(body.source?.type).toBe("file");
    expect(body.applied?.resource_counts?.models).toBe(1);
  });

  test("the runtime health view serves from the file snapshot on the status listener", async () => {
    if (!app) throw new Error("setup failed");

    // /status/models reads through the file-mode FileManagedStore — the
    // exact read surface that must stay live with the admin listener off
    // (the admin endpoint that used to back this view is unbound here).
    const res = await fetch(`${app.metricsUrl}/status/models`);
    expect(res.status).toBe(200);
    const rows = (await res.json()) as Array<{
      display_name?: string;
      kind?: string;
      status?: string;
    }>;
    expect(rows).toHaveLength(1);
    expect(rows[0]?.display_name).toBe("admin-off-file-model");
    expect(rows[0]?.kind).toBe("direct");
    expect(rows[0]?.status).toBe("healthy");
  });

  test("the proxy health endpoints serve in file mode with the admin listener off", async () => {
    if (!app) throw new Error("setup failed");

    // File mode loads at boot and has no watch to go stale, so /readyz on
    // the proxy listener reports ready as soon as the file has applied.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/readyz`);
        await r.text();
        return r.status === 200;
      } catch {
        return false;
      }
    });

    const livez = await fetch(`${app!.proxyUrl}/livez`);
    expect(livez.status).toBe(200);
    expect(await livez.text()).toBe("ok");

    const verbose = await fetch(`${app!.proxyUrl}/readyz?verbose`);
    expect(verbose.status).toBe(200);
    const body = await verbose.text();
    expect(body).toContain("[+]shutdown ok");
    expect(body).toContain("[+]config ok");
  });

  test("no admin listener is bound in file mode", async () => {
    if (!app) throw new Error("setup failed");
    await expectAdminPortRefused(app.adminUrl);
  });
});
