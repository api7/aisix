import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for `GET /status/models` on the dedicated metrics/status listener:
// the per-model runtime health view (cooldown / background-check state) as
// an operational read that does not depend on the admin listener. While
// both endpoints exist, the status-listener response must be exactly the
// admin listener's `GET /admin/v1/models/status` response — same JSON,
// same ordering — which these cases pin against the live gateway in both
// resource-source modes (etcd watch and standalone file).

const CALLER_PLAINTEXT = "sk-status-models-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

async function fetchStatusModels(
  app: SpawnedApp,
): Promise<{ raw: string; rows: Array<Record<string, unknown>>; contentType: string | null }> {
  const res = await fetch(`${app.metricsUrl}/status/models`);
  expect(res.status).toBe(200);
  const raw = await res.text();
  return {
    raw,
    rows: JSON.parse(raw) as Array<Record<string, unknown>>,
    contentType: res.headers.get("content-type"),
  };
}

async function fetchAdminModelStatuses(
  app: SpawnedApp,
): Promise<{ raw: string; rows: Array<Record<string, unknown>>; contentType: string | null }> {
  const res = await fetch(`${app.adminUrl}/admin/v1/models/status`, {
    headers: { authorization: `Bearer ${app.adminKey}` },
  });
  expect(res.status).toBe(200);
  const raw = await res.text();
  return {
    raw,
    rows: JSON.parse(raw) as Array<Record<string, unknown>>,
    contentType: res.headers.get("content-type"),
  };
}

describe("status/models: etcd mode — equivalence with the admin endpoint", () => {
  let app: SpawnedApp | undefined;
  let authFailUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  let authFailModelID = "";
  let stableModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    authFailUpstream = await startOpenAiUpstream({
      status: 401,
      errorBody: {
        error: { message: "Incorrect API key provided", type: "invalid_request_error" },
      },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-status-models-stable",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "status-models stable" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // Held-back: compares /status/models to /admin/v1/models/status, so it
    // keeps admin bound (the suite default is now admin-off).
    app = await spawnApp({ admin: true });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const failPk = await seed.createProviderKey({
      display_name: "status-models-401-pk",
      secret: "sk-mock",
      api_base: `${authFailUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "status-models-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });
    authFailModelID = (
      await seed.createModel({
        display_name: "status-models-401",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: failPk.id,
      })
    ).id;
    stableModelID = (
      await seed.createModel({
        display_name: "status-models-stable",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: stablePk.id,
      })
    ).id;
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["status-models-401", "status-models-stable"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await authFailUpstream?.close();
    await stableUpstream?.close();
  });

  test("serves the same JSON as the admin endpoint, with one model in cooldown", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Readiness: both models visible on the status listener AND the
    // stable model actually dispatchable end-to-end.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const { rows } = await fetchStatusModels(app!);
      if (rows.length < 2) return false;
      const probe = await proxy.chat({
        model: "status-models-stable",
        messages: [{ role: "user", content: "ready-status-models" }],
      });
      return probe.status === 200;
    });

    // Trip the auth-failing model once: the 401 surfaces to the caller
    // and puts the direct model into cooldown.
    const tripped = await proxy.chat({
      model: "status-models-401",
      messages: [{ role: "user", content: "trip cooldown" }],
    });
    expect(tripped.status).toBe(401);
    await waitConfigPropagation(async () => {
      const { rows } = await fetchStatusModels(app!);
      return rows.some((row) => row.id === authFailModelID && row.status === "cooldown");
    });

    // The status-listener body is the admin body: same JSON, same order,
    // same content type. Raw-text equality is the strongest form; the
    // parsed deep-equal repeats it with a readable diff on failure.
    const admin = await fetchAdminModelStatuses(app);
    const status = await fetchStatusModels(app);
    expect(status.rows).toEqual(admin.rows);
    expect(status.raw).toBe(admin.raw);
    expect(status.contentType).toBe(admin.contentType);

    // And the shared body carries the expected runtime state.
    const cooled = status.rows.find((row) => row.id === authFailModelID)!;
    expect(cooled.kind).toBe("direct");
    expect(cooled.status).toBe("cooldown");
    expect(cooled.status_reason).toBe("upstream_auth_failure");
    expect(cooled.cooldown_until).toBeTruthy();
    const stable = status.rows.find((row) => row.id === stableModelID)!;
    expect(stable.status).toBe("healthy");
  });

  test("status listener needs no key while the admin route still requires one", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Unauthenticated operational read — same trust domain as
    // /status/config on the same listener.
    const status = await fetch(`${app.metricsUrl}/status/models`);
    expect(status.status).toBe(200);
    await status.text();

    // The admin listener's endpoint keeps its admin-key auth.
    const admin = await fetch(`${app.adminUrl}/admin/v1/models/status`);
    expect(admin.status).toBe(401);
    await admin.text();
  });
});

describe("status/models: standalone file source", () => {
  let app: SpawnedApp | undefined;

  beforeAll(async () => {
    app = await spawnApp({
      admin: true,
      resourcesFile: `
_format_version: "1"
provider_keys:
  - display_name: file-status-models-pk
    provider: openai
    api_key: sk-mock
    api_base: http://127.0.0.1:9/v1
models:
  - display_name: file-status-models-direct
    provider: openai
    model_name: gpt-4o-mini
    provider_key: file-status-models-pk
`,
    });
  });

  afterAll(async () => {
    await app?.exit();
  });

  test("file mode serves the runtime health view on the status listener", async () => {
    if (!app) throw new Error("setup failed");

    // File mode is synchronous at boot — no propagation wait needed.
    const { rows } = await fetchStatusModels(app);
    expect(rows).toHaveLength(1);
    const row = rows[0];
    expect(row.display_name).toBe("file-status-models-direct");
    expect(typeof row.id).toBe("string");
    expect(row.id).toBeTruthy();
    expect(row.kind).toBe("direct");
    expect(row.status).toBe("healthy");
  });

  test("file mode status listener matches the admin endpoint too", async () => {
    if (!app) throw new Error("setup failed");

    const admin = await fetchAdminModelStatuses(app);
    const status = await fetchStatusModels(app);
    expect(status.rows).toEqual(admin.rows);
    expect(status.raw).toBe(admin.raw);
  });
});
