import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import OpenAI, { APIError } from "openai";
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

// E2E: the standalone file-based resource source (`resources_file`).
// A single-container gateway loads provider keys / models / caller keys
// from one resources.yaml — no etcd, no Admin API writes — and must be
// observably identical to an etcd-backed gateway serving the same
// logical resources.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for the
// proxy surface the cases drive.
//
// NOTE on env names: the caller-key plaintext travels to the gateway via
// an environment variable (`key_env` sugar). The variable name must NOT
// start with `AISIX_` — the binary's config loader treats that prefix as
// config overrides (same reason the harness strips them).

const execFileP = promisify(execFile);

const CALLER_PLAINTEXT = "sk-file-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CALLER_KEY_ENV = "FILE_E2E_CALLER_KEY";

function fileModeResources(upstreamBase: string): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: file-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v1
models:
  - display_name: file-allowed
    provider: openai
    model_name: gpt-4o-mini
    provider_key: file-pk
  - display_name: file-forbidden
    provider: openai
    model_name: gpt-4o-mini
    provider_key: file-pk
api_keys:
  - display_name: file-caller
    key_env: ${CALLER_KEY_ENV}
    allowed_models: ["file-allowed"]
guardrails:
  - name: file-no-secrets
    kind: keyword
    patterns:
      - kind: literal
        value: file-mode-forbidden-phrase
`;
}

describe("file resource source: smoke + admin write-guard", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      resourcesFile: fileModeResources(upstream.baseUrl),
      extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT },
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
    // authenticates with a *proxy* caller key and must not be caught
    // by the file-managed write guard.
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

  test("admin write endpoints answer 409 file-managed; reads serve the file contents", async () => {
    if (!app) throw new Error("setup failed");
    const auth = { authorization: `Bearer ${app.adminKey}` };

    // GET list works and reflects the file.
    const listRes = await fetch(`${app.adminUrl}/admin/v1/models`, { headers: auth });
    expect(listRes.status).toBe(200);
    const listed = (await listRes.json()) as Array<{ value: { display_name: string } }>;
    expect(listed.map((e) => e.value.display_name).sort()).toEqual([
      "file-allowed",
      "file-forbidden",
    ]);

    // POST is refused with a clear 409 naming the file.
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
    expect(postRes.status).toBe(409);
    const postBody = (await postRes.json()) as { error_msg: string };
    expect(postBody.error_msg).toContain("file-managed");
    expect(postBody.error_msg).toContain(app.resourcesPath!);

    // The refused write did not change the resource set.
    const relist = await fetch(`${app.adminUrl}/admin/v1/models`, { headers: auth });
    expect(((await relist.json()) as unknown[]).length).toBe(2);

    // DELETE and rotate are covered by the same guard.
    const delRes = await fetch(`${app.adminUrl}/admin/v1/models/any-id`, {
      method: "DELETE",
      headers: auth,
    });
    expect(delRes.status).toBe(409);
    const rotateRes = await fetch(`${app.adminUrl}/admin/v1/api_keys/any-id/rotate`, {
      method: "POST",
      headers: auth,
    });
    expect(rotateRes.status).toBe(409);

    // Auth ordering: an UNAUTHENTICATED write still gets 401, and the
    // 401 body must not leak the resources-file path (that detail is
    // only for authenticated admins).
    const unauthed = await fetch(`${app.adminUrl}/admin/v1/models`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ display_name: "nope" }),
    });
    expect(unauthed.status).toBe(401);
    const unauthedBody = (await unauthed.json()) as { error_msg: string };
    expect(unauthedBody.error_msg).not.toContain(app.resourcesPath!);
  });

  test("file-defined guardrail fires on matching input", async () => {
    if (!app || !upstream) throw new Error("setup failed");
    // The file format has no attachment collection, so file-defined
    // guardrails apply env-globally. Pin that they actually fire — the
    // runtime executes them through the zero-attachment fallback in the
    // guardrail index, and this test is the regression trap for that
    // dependency.
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
});

describe("file resource source: differential vs etcd mode", () => {
  let fileApp: SpawnedApp | undefined;
  let etcdApp: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) {
      // Local runs may skip quietly; on CI the differential pin is the
      // point of this file — a silent skip would let the equivalence
      // contract rot invisibly.
      if (process.env.CI) {
        throw new Error("differential case requires etcd on CI (AISIX_E2E_ETCD)");
      }
      return;
    }

    upstream = await startOpenAiUpstream();

    // The SAME logical resource set on both gateways: one provider key,
    // an allowed model, a forbidden model, one caller key.
    fileApp = await spawnApp({
      resourcesFile: `
_format_version: "1"
provider_keys:
  - display_name: diff-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstream.baseUrl}/v1
models:
  - display_name: diff-allowed
    provider: openai
    model_name: gpt-4o-mini
    provider_key: diff-pk
  - display_name: diff-forbidden
    provider: openai
    model_name: gpt-4o-mini
    provider_key: diff-pk
api_keys:
  - display_name: diff-caller
    key_hash: ${CALLER_KEY_HASH}
    allowed_models: ["diff-allowed"]
`,
    });

    etcdApp = await spawnApp();
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
        (name) => `  - display_name: ${name}
    provider: openai
    model_name: gpt-4o-mini
    provider_key: reload-pk`,
      )
      .join("\n");
    return `
_format_version: "1"
provider_keys:
  - display_name: reload-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v1
models:
${blocks}
api_keys:
  - display_name: reload-caller
    key_hash: ${CALLER_KEY_HASH}
    allowed_models: ["*"]
`;
  }

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      resourcesFile: reloadFile(upstream.baseUrl, ["reload-a"]),
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("valid edit applies on SIGHUP; invalid edit keeps last-good", async () => {
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

    // Invalid edit: reload-c appears alongside a broken entry. The
    // whole reload must be rejected — nothing from the bad file applies
    // (no partial pickup of reload-c), and the last-good snapshot keeps
    // serving.
    const broken = `${reloadFile(upstream.baseUrl, ["reload-a", "reload-b", "reload-c"])}
guardrails:
  - name: broken-entry
    kind: no-such-guardrail-kind
`;
    const failuresBefore = (app.output().match(/reload failed/g) ?? []).length;
    await writeFile(app.resourcesPath, broken, "utf8");
    app.signal("SIGHUP");
    // Deterministic wait: the gateway logs the aggregated reload
    // failure; poll the captured output instead of sleeping.
    await waitConfigPropagation(async () => {
      return (app!.output().match(/reload failed/g) ?? []).length > failuresBefore;
    });

    // Old models still serve; the new one from the rejected file is
    // absent.
    expect(await modelIds()).toEqual(["reload-a", "reload-b"]);
    const stillServing = await proxy.chat({
      model: "reload-a",
      messages: [{ role: "user", content: "still-alive" }],
    });
    expect(stillServing.status).toBe(200);
    const notPickedUp = await proxy.chat({
      model: "reload-c",
      messages: [{ role: "user", content: "must not exist" }],
    });
    expect(notPickedUp.status).toBe(404);
  });
});

describe("file resource source: fail-fast boot", () => {
  test("malformed file exits non-zero with the aggregated named errors on stderr", async () => {
    let caught: unknown;
    try {
      await spawnApp({
        resourcesFile: `
_format_version: "1"
models:
  - display_name: broken
    provider: openai
    provider_key: ghost-pk
api_keys:
  - display_name: dup
    key_hash: aa
    allowed_models: ["nope"]
  - display_name: dup
    key_hash: bb
    allowed_models: []
not_a_collection: []
`,
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    const msg = (caught as Error).message;
    // Non-zero exit, not a readiness timeout.
    expect(msg).toContain("exited early with code=1");
    // The aggregated report names every problem with its context.
    expect(msg).toMatch(/4 error\(s\)/);
    expect(msg).toContain("unknown top-level key `not_a_collection`");
    expect(msg).toContain('duplicate api_keys entry');
    expect(msg).toContain('unknown provider key "ghost-pk"');
    expect(msg).toContain('unknown model "nope"');
  }, 30_000);
});

describe("file resource source: validate subcommand", () => {
  const BIN_PATH =
    process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

  test("valid file exits 0; invalid file exits 1 with the aggregated report on stderr", async () => {
    const dir = await mkdtemp(join(tmpdir(), "aisix-validate-e2e-"));
    try {
      const good = join(dir, "good.yaml");
      await writeFile(
        good,
        [
          '_format_version: "1"',
          "provider_keys:",
          "  - display_name: pk",
          "    api_key: sk-x",
          "models:",
          "  - display_name: m1",
          "    provider: openai",
          "    model_name: gpt-4o",
          "    provider_key: pk",
          "",
        ].join("\n"),
        "utf8",
      );
      const ok = await execFileP(BIN_PATH, ["validate", "--resources", good]);
      expect(ok.stdout).toContain("OK:");

      const bad = join(dir, "bad.yaml");
      await writeFile(bad, "models: []\n", "utf8");
      let failure: (Error & { code?: number; stderr?: string }) | undefined;
      try {
        await execFileP(BIN_PATH, ["validate", "--resources", bad]);
      } catch (e) {
        failure = e as Error & { code?: number; stderr?: string };
      }
      if (!failure) throw new Error("expected `aisix validate` to exit non-zero");
      expect(failure.code).toBe(1);
      expect(String(failure.stderr)).toContain("missing mandatory _format_version");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
