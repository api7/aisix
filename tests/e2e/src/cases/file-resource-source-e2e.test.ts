import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  fileId,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the standalone file-based resource source (`resources_file`).
// A single-container gateway loads its resources from one resources.yaml —
// no etcd, no Admin API writes. Each entry is an etcd document written
// down: its `id` plus the document etcd would store, references by id. It
// must be observably identical to an etcd-backed gateway serving the same
// documents, including what happens to a document that does not load.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for the
// proxy surface the cases drive.

const execFileP = promisify(execFile);

const CALLER_PLAINTEXT = "sk-file-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const PK = fileId("file-pk");
const ALLOWED = fileId("file-allowed");
const FORBIDDEN = fileId("file-forbidden");
const UNSCOPED = fileId("file-unscoped");
const NO_SECRETS = fileId("file-no-secrets");
const MODEL_SCOPED = fileId("file-model-scoped");

function fileModeResources(upstreamBase: string): string {
  return `
_format_version: "2"
provider_keys:
  - id: ${PK}
    display_name: file-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v1
models:
  - id: ${ALLOWED}
    display_name: file-allowed
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${PK}
  - id: ${FORBIDDEN}
    display_name: file-forbidden
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${PK}
  # Reachable by the caller, and deliberately NOT the model the scoped
  # guardrail names — the negative half of the scope assertion.
  - id: ${UNSCOPED}
    display_name: file-unscoped
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${PK}
api_keys:
  - id: ${fileId("file-caller")}
    key_hash: ${CALLER_KEY_HASH}
    allowed_model_ids: ["${ALLOWED}", "${UNSCOPED}"]
guardrails:
  - id: ${NO_SECRETS}
    name: file-no-secrets
    kind: keyword
    patterns:
      - kind: literal
        value: file-mode-forbidden-phrase
  - id: ${MODEL_SCOPED}
    name: file-model-scoped
    kind: keyword
    patterns:
      - kind: literal
        value: file-mode-model-scoped-phrase
  - id: ${fileId("file-unattached")}
    name: file-unattached
    kind: keyword
    patterns:
      - kind: literal
        value: file-mode-unattached-phrase
guardrail_attachments:
  - id: ${fileId("attach-env")}
    guardrail_id: ${NO_SECRETS}
    scope_type: env
    priority: 100
  - id: ${fileId("attach-model")}
    guardrail_id: ${MODEL_SCOPED}
    scope_type: model
    scope_id: ${ALLOWED}
    priority: 100
`;
}

describe("file resource source: smoke + read-only admin surface", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    // Exercises the file-mode admin read surface (and pins that the
    // removed write endpoints stay gone), so the admin listener stays
    // bound (the suite default is admin-off).
    app = await spawnApp({
      admin: true,
      resourcesFile: fileModeResources(upstream.baseUrl),
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("full chain serves a chat completion from the file-defined resources", async () => {
    if (!app || !upstream) throw new Error("setup failed");

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // File mode is synchronous at boot: no propagation wait needed —
    // the first request must already serve.
    const completion = await client.chat.completions.create({
      model: "file-allowed",
      messages: [{ role: "user", content: "hello from file mode" }],
    });
    expect(completion.choices[0]?.message.role).toBe("assistant");
    expect(completion.choices[0]?.message.content).toBe("mock reply");
    expect(upstream.receivedRequests.length).toBeGreaterThan(0);

    // The caller key's allowed_models still gates access: the sibling
    // model exists in the file but is not granted → 403, upstream
    // untouched by the blocked call.
    const hitsBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "file-forbidden",
        messages: [{ role: "user", content: "must be blocked" }],
      });
    } catch (e) {
      caught = e;
    }
    if (!(caught instanceof APIError)) {
      throw new Error(`expected APIError, got: ${String(caught)}`);
    }
    expect(caught.status).toBe(403);
    expect(upstream.receivedRequests.length).toBe(hitsBefore);
  });

  test("playground forwards through the proxy stack in file mode", async () => {
    if (!app) throw new Error("setup failed");
    // The playground endpoint lives on the admin listener but
    // authenticates with a *proxy* caller key — a POST that must keep
    // working on the otherwise read-only admin listener.
    const res = await fetch(`${app.adminUrl}/playground/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "file-allowed",
        messages: [{ role: "user", content: "playground in file mode" }],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      choices: Array<{ message: { role: string } }>;
    };
    expect(body.choices[0]?.message.role).toBe("assistant");
  });

  test("admin surface is read-only: reads serve the file contents, writes answer 405, rotate 404", async () => {
    if (!app) throw new Error("setup failed");
    const auth = { authorization: `Bearer ${app.adminKey}` };

    // GET list works and reflects the file, each resource under the id
    // its entry carries.
    const listRes = await fetch(`${app.adminUrl}/admin/v1/models`, { headers: auth });
    expect(listRes.status).toBe(200);
    const listed = (await listRes.json()) as Array<{ id: string; value: { display_name: string } }>;
    expect(
      listed.map((e) => [e.id, e.value.display_name]).sort((a, z) => a[1].localeCompare(z[1])),
    ).toEqual([
      [ALLOWED, "file-allowed"],
      [FORBIDDEN, "file-forbidden"],
      [UNSCOPED, "file-unscoped"],
    ]);

    // The write endpoints were removed: the routes serve GET only, so
    // POST/DELETE answer 405 with an Allow header advertising GET.
    const postRes = await fetch(`${app.adminUrl}/admin/v1/models`, {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify({
        display_name: "sneaky",
        provider: "openai",
        model_name: "gpt-4o",
        provider_key_id: "11111111-1111-1111-1111-111111111111",
      }),
    });
    expect(postRes.status).toBe(405);
    expect(postRes.headers.get("allow")).toContain("GET");
    await postRes.text();

    // The refused write did not change the resource set.
    const relist = await fetch(`${app.adminUrl}/admin/v1/models`, { headers: auth });
    expect(((await relist.json()) as unknown[]).length).toBe(3);

    const delRes = await fetch(`${app.adminUrl}/admin/v1/models/any-id`, {
      method: "DELETE",
      headers: auth,
    });
    expect(delRes.status).toBe(405);
    expect(delRes.headers.get("allow")).toContain("GET");
    await delRes.text();

    // The rotate route was removed outright (it was POST-only), so the
    // path no longer exists at all.
    const rotateRes = await fetch(`${app.adminUrl}/admin/v1/api_keys/any-id/rotate`, {
      method: "POST",
      headers: auth,
    });
    expect(rotateRes.status).toBe(404);
    await rotateRes.text();

    // Method routing answers before auth: an unauthenticated write gets
    // the same 405 — there is no write endpoint left to protect, and
    // the 405 body carries no resources-file detail to leak.
    const unauthed = await fetch(`${app.adminUrl}/admin/v1/models`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ display_name: "nope" }),
    });
    expect(unauthed.status).toBe(405);
    expect((await unauthed.text())).not.toContain(app.resourcesPath!);
  });

  test("file-defined guardrail fires on matching input", async () => {
    if (!app || !upstream) throw new Error("setup failed");
    // A guardrail fires because an attachment says so — the file declares
    // an env-scoped one alongside it, naming the guardrail by its id. There
    // is no implicit scope (AISIX-Cloud#1450), so this pins both halves:
    // the reference resolves, and the attachment reaches the index.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const hitsBefore = upstream.receivedRequests.length;
    const blocked = await proxy.chat({
      model: "file-allowed",
      messages: [{ role: "user", content: "contains file-mode-forbidden-phrase here" }],
    });
    expect(blocked.status).toBe(422);
    // Blocked before dispatch: the upstream never sees the request.
    expect(upstream.receivedRequests.length).toBe(hitsBefore);

    // Clean input still passes end-to-end.
    const clean = await proxy.chat({
      model: "file-allowed",
      messages: [{ role: "user", content: "clean input" }],
    });
    expect(clean.status).toBe(200);
  });

  test("a model-scoped attachment resolves its target by id", async () => {
    if (!app || !upstream) throw new Error("setup failed");
    // `scope_id` is the model entry's id, and the request must resolve to
    // that same id for the scope to match.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const hitsBefore = upstream.receivedRequests.length;
    const blocked = await proxy.chat({
      model: "file-allowed",
      messages: [{ role: "user", content: "contains file-mode-model-scoped-phrase here" }],
    });
    expect(blocked.status).toBe(422);
    expect(upstream.receivedRequests.length).toBe(hitsBefore);

    // …and the other half: the same phrase on a model the attachment does
    // NOT name passes. Blocking on the scoped model alone would also be
    // satisfied by a guardrail that runs everywhere, which is exactly the
    // behaviour this release removed.
    const allowed = await proxy.chat({
      model: "file-unscoped",
      messages: [{ role: "user", content: "contains file-mode-model-scoped-phrase here" }],
    });
    expect(allowed.status).toBe(200);
    expect(upstream.receivedRequests.length).toBe(hitsBefore + 1);
  });

  test("a guardrail with no attachment governs nothing", async () => {
    if (!app || !upstream) throw new Error("setup failed");
    // AISIX-Cloud#1450: a guardrail's scope is its attachments and nothing
    // else. `file-unattached` is declared and loads cleanly — declaring it
    // is not an error — but nothing attaches it, so its pattern must pass
    // straight through to the upstream.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const hitsBefore = upstream.receivedRequests.length;
    const allowed = await proxy.chat({
      model: "file-allowed",
      messages: [{ role: "user", content: "contains file-mode-unattached-phrase here" }],
    });
    expect(allowed.status).toBe(200);
    expect(upstream.receivedRequests.length).toBe(hitsBefore + 1);
  });
});

describe("file resource source: differential vs etcd mode", () => {
  let fileApp: SpawnedApp | undefined;
  let etcdApp: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    // Local runs may skip quietly. On CI they cannot: ping() throws
    // there, for every case file, so the equivalence contract this file
    // pins can no longer rot behind a silent skip.
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();

    // The SAME logical resource set on both gateways: one provider key,
    // an allowed model, a forbidden model, one caller key.
    fileApp = await spawnApp({
      admin: true,
      resourcesFile: `
_format_version: "2"
provider_keys:
  - id: ${fileId("diff-pk")}
    display_name: diff-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstream.baseUrl}/v1
models:
  - id: ${fileId("diff-allowed")}
    display_name: diff-allowed
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${fileId("diff-pk")}
  - id: ${fileId("diff-forbidden")}
    display_name: diff-forbidden
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${fileId("diff-pk")}
api_keys:
  - id: ${fileId("diff-caller")}
    key_hash: ${CALLER_KEY_HASH}
    allowed_models: ["diff-allowed"]
`,
    });

    etcdApp = await spawnApp({ admin: true });
    const seed = new SeedClient(etcd, etcdApp.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "diff-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "diff-allowed",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "diff-forbidden",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["diff-allowed"],
    });
  });

  afterAll(async () => {
    await fileApp?.exit();
    await etcdApp?.exit();
    await upstream?.close();
  });

  test("identical observable behavior: /v1/models, chat 200, allowed_models 403", async (ctx) => {
    if (!etcdReachable || !fileApp || !etcdApp || !upstream) {
      ctx.skip();
      return;
    }
    const fileProxy = new ProxyClient(fileApp.proxyUrl, CALLER_PLAINTEXT);
    const etcdProxy = new ProxyClient(etcdApp.proxyUrl, CALLER_PLAINTEXT);

    // etcd mode propagates asynchronously — gate on the allowed model
    // serving. File mode is ready at boot by contract.
    await waitConfigPropagation(async () => {
      const r = await etcdProxy.chat({
        model: "diff-allowed",
        messages: [{ role: "user", content: "ready-probe" }],
      });
      return r.status === 200;
    });

    // 1) /v1/models — same listing (modulo the `created` timestamp).
    const fileModels = await fileProxy.listModels();
    const etcdModels = await etcdProxy.listModels();
    expect(fileModels.status).toBe(200);
    expect(etcdModels.status).toBe(200);
    const normalize = (body: unknown) => {
      const b = body as { object: string; data: Array<Record<string, unknown>> };
      return {
        object: b.object,
        data: b.data
          .map(({ id, object, owned_by }) => ({ id, object, owned_by }))
          .sort((a, z) => String(a.id).localeCompare(String(z.id))),
      };
    };
    expect(normalize(fileModels.body)).toEqual(normalize(etcdModels.body));

    // 2) Chat on the allowed model — 200 with the same completion body
    // from the shared mock upstream.
    const fileChat = await fileProxy.chat({
      model: "diff-allowed",
      messages: [{ role: "user", content: "differential" }],
    });
    const etcdChat = await etcdProxy.chat({
      model: "diff-allowed",
      messages: [{ role: "user", content: "differential" }],
    });
    expect(fileChat.status).toBe(200);
    expect(etcdChat.status).toBe(200);
    const content = (r: unknown) =>
      (r as { choices: Array<{ message: { role: string; content: string } }> }).choices[0]
        ?.message;
    expect(content(fileChat.body)).toEqual(content(etcdChat.body));

    // 3) allowed_models enforcement — same 403 error envelope.
    const fileBlocked = await fileProxy.chat({
      model: "diff-forbidden",
      messages: [{ role: "user", content: "blocked" }],
    });
    const etcdBlocked = await etcdProxy.chat({
      model: "diff-forbidden",
      messages: [{ role: "user", content: "blocked" }],
    });
    expect(fileBlocked.status).toBe(403);
    expect(etcdBlocked.status).toBe(403);
    const envelope = (r: unknown) => {
      const e = (r as { error: { type?: string; code?: unknown } }).error;
      return { type: e.type, code: e.code };
    };
    expect(envelope(fileBlocked.body)).toEqual(envelope(etcdBlocked.body));
  });
});

describe("file resource source: SIGHUP reload", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  function reloadFile(upstreamBase: string, models: string[]): string {
    const blocks = models
      .map(
        (name) => `  - id: ${fileId(name)}
    display_name: ${name}
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${fileId("reload-pk")}`,
      )
      .join("\n");
    return `
_format_version: "2"
provider_keys:
  - id: ${fileId("reload-pk")}
    display_name: reload-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v1
models:
${blocks}
api_keys:
  - id: ${fileId("reload-caller")}
    key_hash: ${CALLER_KEY_HASH}
    allow_all_models: true
`;
  }

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      admin: true,
      resourcesFile: reloadFile(upstream.baseUrl, ["reload-a"]),
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const statusConfig = async () => {
    const res = await fetch(`${app!.metricsUrl}/status/config`);
    expect(res.status).toBe(200);
    return (await res.json()) as {
      state: string;
      rejected: Array<{ resource_kind: string; resource_id: string; last_error_kind: string }>;
    };
  };

  test("a SIGHUP applies an edit row by row, the way etcd applies writes", async () => {
    if (!app || !upstream || !app.resourcesPath) throw new Error("setup failed");
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const modelIds = async () => {
      const r = await proxy.listModels();
      expect(r.status).toBe(200);
      return (r.body as { data: Array<{ id: string }> }).data.map((m) => m.id).sort();
    };

    expect(await modelIds()).toEqual(["reload-a"]);

    // Valid edit: add reload-b, SIGHUP, and the new model becomes
    // callable end-to-end.
    await writeFile(
      app.resourcesPath,
      reloadFile(upstream.baseUrl, ["reload-a", "reload-b"]),
      "utf8",
    );
    app.signal("SIGHUP");
    await waitConfigPropagation(async () => {
      const r = await proxy.chat({
        model: "reload-b",
        messages: [{ role: "user", content: "post-reload" }],
      });
      return r.status === 200;
    });
    expect(await modelIds()).toEqual(["reload-a", "reload-b"]);

    // An edit carrying one document that does not load: like a bad etcd
    // write, that row is rejected and reported while every other row of the
    // edit applies — reload-c is picked up beside it.
    const brokenId = fileId("broken-entry");
    const withBrokenRow = `${reloadFile(upstream.baseUrl, ["reload-a", "reload-b", "reload-c"])}
guardrails:
  - id: ${brokenId}
    name: broken-entry
    kind: no-such-guardrail-kind
`;
    await writeFile(app.resourcesPath, withBrokenRow, "utf8");
    app.signal("SIGHUP");
    await waitConfigPropagation(async () => (await statusConfig()).rejected.length > 0);
    expect(await modelIds()).toEqual(["reload-a", "reload-b", "reload-c"]);
    const status = await statusConfig();
    expect(status.state).toBe("degraded");
    expect(status.rejected).toEqual([
      expect.objectContaining({
        resource_kind: "guardrails",
        resource_id: brokenId,
        last_error_kind: "schema_failed",
      }),
    ]);
  });

  test("a file that cannot be read keeps the configuration that is serving", async () => {
    if (!app || !upstream || !app.resourcesPath) throw new Error("setup failed");
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const failuresBefore = (app.output().match(/resources file could not be read/g) ?? []).length;
    await writeFile(app.resourcesPath, "_format_version: \"2\"\nmodels: [\n", "utf8");
    app.signal("SIGHUP");
    await waitConfigPropagation(
      async () =>
        (app!.output().match(/resources file could not be read/g) ?? []).length > failuresBefore,
    );
    const stillServing = await proxy.chat({
      model: "reload-a",
      messages: [{ role: "user", content: "still-alive" }],
    });
    expect(stillServing.status).toBe(200);
  });
});

describe("file resource source: boot", () => {
  test("a file that cannot be read as rows exits non-zero with every problem named", async () => {
    let caught: unknown;
    try {
      await spawnApp({
        admin: true,
        resourcesFile: `
_format_version: "2"
models:
  - display_name: no-id
api_keys:
  - id: not-a-uuid
    key_hash: aa
  - id: ${fileId("dup")}
    key_hash: bb
  - id: ${fileId("dup")}
    key_hash: cc
`,
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    const msg = (caught as Error).message;
    // Non-zero exit, not a readiness timeout.
    expect(msg).toContain("exited early with code=1");
    expect(msg).toMatch(/3 error\(s\)/);
    expect(msg).toContain("models[0]: `id` is required");
    expect(msg).toContain('api_keys[0]: `id` must be a UUID string, found "not-a-uuid"');
    expect(msg).toContain(`api_keys[2]: duplicate id "${fileId("dup")}"`);
  }, 30_000);

  test("a row the loader rejects does not stop the gateway", async () => {
    const upstream = await startOpenAiUpstream();
    const app = await spawnApp({
      resourcesFile: `
_format_version: "2"
provider_keys:
  - id: ${fileId("boot-pk")}
    display_name: boot-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstream.baseUrl}/v1
models:
  - id: ${fileId("boot-model")}
    display_name: boot-model
    provider: openai
    model_name: gpt-4o-mini
    provider_key_id: ${fileId("boot-pk")}
  - id: ${fileId("boot-broken")}
    display_name: 42
api_keys:
  - id: ${fileId("boot-caller")}
    key_hash: ${CALLER_KEY_HASH}
    allow_all_models: true
`,
    });
    try {
      const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      const r = await proxy.chat({
        model: "boot-model",
        messages: [{ role: "user", content: "hello" }],
      });
      expect(r.status).toBe(200);
      const res = await fetch(`${app.metricsUrl}/status/config`);
      const status = (await res.json()) as {
        rejected: Array<{ resource_kind: string; resource_id: string }>;
      };
      expect(status.rejected).toEqual([
        expect.objectContaining({ resource_kind: "models", resource_id: fileId("boot-broken") }),
      ]);
    } finally {
      await app.exit();
      await upstream.close();
    }
  }, 30_000);
});

describe("file resource source: validate subcommand", () => {
  const BIN_PATH =
    process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

  test("valid file exits 0; a rejected row or a malformed file exits 1", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aisix-validate-e2e-"));
    try {
      const lines = [
        '_format_version: "2"',
        "provider_keys:",
        `  - id: ${fileId("pk")}`,
        "    display_name: pk",
        "    api_key: sk-x",
        "models:",
        `  - id: ${fileId("m1")}`,
        "    display_name: m1",
        "    provider: openai",
        "    model_name: gpt-4o",
        `    provider_key_id: ${fileId("pk")}`,
        "",
      ];
      const good = join(dir, "good.yaml");
      await writeFile(good, lines.join("\n"), "utf8");
      const ok = await execFileP(BIN_PATH, ["validate", "--resources", good]);
      expect(ok.stdout).toContain("OK:");

      const failure = async (contents: string) => {
        const path = join(dir, "bad.yaml");
        await writeFile(path, contents, "utf8");
        try {
          await execFileP(BIN_PATH, ["validate", "--resources", path]);
        } catch (e) {
          return e as Error & { code?: number; stderr?: string };
        }
        throw new Error("expected `aisix validate` to exit non-zero");
      };

      const rejected = await failure(
        [...lines, `  - id: ${fileId("m2")}`, "    display_name: 42", ""].join("\n"),
      );
      expect(rejected.code).toBe(1);
      expect(String(rejected.stderr)).toContain(
        `/aisix/models/${fileId("m2")} (schema_failed)`,
      );

      const malformed = await failure("models: []\n");
      expect(malformed.code).toBe(1);
      expect(String(malformed.stderr)).toContain("missing mandatory _format_version");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
