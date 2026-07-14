import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { join } from "node:path";
import { promisify } from "node:util";
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

// E2E: the `aisix export` round-trip — the migration path that closes the
// loop etcd → export → file → same behavior.
//
// A gateway is populated via the seed front door (canonical documents
// written straight to etcd, as the control plane does in managed mode).
// `aisix export` reads that etcd store and emits a resources.yaml; a
// second, file-mode gateway loads the exported file and must be
// observably identical to the etcd-backed one — same /v1/models listing,
// same chat completion, same allowed_models 403. The exported file also
// must carry no live credential at default settings; the provider
// secret is replaced with a ${VAR} placeholder the file-mode gateway
// interpolates from its environment.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for the
// proxy surface the case drives.

const execFileP = promisify(execFile);

const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");
const ETCD_ENDPOINT = process.env.AISIX_E2E_ETCD ?? "http://127.0.0.1:2379";

const CALLER_PLAINTEXT = "sk-export-roundtrip-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
// A distinctive marker so the no-leak assertion is unambiguous.
const PROVIDER_SECRET = "sk-provider-secret-must-not-leak-1a2b3c";

/** Extract every `${VAR}` interpolation reference the export emitted. */
function placeholderVars(yaml: string): string[] {
  return [...yaml.matchAll(/\$\{([^}]+)\}/g)].map((m) => m[1]!);
}

describe("aisix export: etcd → export → file round-trip", () => {
  let etcdApp: SpawnedApp | undefined;
  let fileApp: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) {
      // Follow the harness convention: defer to each test's ctx.skip()
      // rather than throwing when etcd is unavailable.
      return;
    }

    upstream = await startOpenAiUpstream();

    // Populate the source-of-truth gateway's etcd exactly as the control
    // plane would: one provider key (with a real secret), an allowed
    // model, a forbidden model, and a caller key scoped to the allowed
    // model only.
    etcdApp = await spawnApp();
    const seed = new SeedClient(etcd, etcdApp.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "rt-pk",
      api_key: PROVIDER_SECRET,
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "rt-allowed",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "rt-forbidden",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["rt-allowed"],
    });

    // etcd propagates asynchronously — gate on the allowed model serving
    // before the test reads etcd via export.
    const probe = new ProxyClient(etcdApp.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const r = await probe.chat({
        model: "rt-allowed",
        messages: [{ role: "user", content: "ready-probe" }],
      });
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await fileApp?.exit();
    await etcdApp?.exit();
    await upstream?.close();
  });

  test("exported file has no live secret and reproduces the gateway's behavior", async (ctx) => {
    if (!etcdReachable || !etcdApp || !upstream) {
      ctx.skip();
      return;
    }

    // 1) Run `aisix export` against the seeded etcd store.
    const { stdout: yaml, stderr } = await execFileP(BIN_PATH, [
      "export",
      "--etcd",
      ETCD_ENDPOINT,
      "--prefix",
      etcdApp.etcdPrefix,
    ]);

    // 2) No live credential at default settings — the provider secret is
    // replaced with a placeholder, and the companion list on stderr tells
    // the operator which variable to set.
    expect(yaml).not.toContain(PROVIDER_SECRET);
    // Diagnostics on stderr must not leak the secret into CI logs either.
    expect(stderr).not.toContain(PROVIDER_SECRET);
    expect(yaml).toContain("_format_version");
    const vars = placeholderVars(yaml);
    expect(vars.length).toBeGreaterThan(0);
    expect(stderr).toContain("rt-pk");
    for (const v of vars) {
      expect(stderr).toContain(v);
    }

    // 3) Load the exported file into a file-mode gateway, supplying the
    // real secret through the placeholder variables the export emitted.
    const extraEnv = Object.fromEntries(vars.map((v) => [v, PROVIDER_SECRET]));
    fileApp = await spawnApp({ resourcesFile: yaml, extraEnv });

    const fileProxy = new ProxyClient(fileApp.proxyUrl, CALLER_PLAINTEXT);
    const etcdProxy = new ProxyClient(etcdApp.proxyUrl, CALLER_PLAINTEXT);

    // 4a) /v1/models — same listing (modulo the per-response timestamp).
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

    // 4b) Chat on the allowed model — 200 with the same completion body
    // from the shared mock upstream. Proves the ${VAR}-interpolated
    // provider credential reached the upstream on the file side.
    const fileChat = await fileProxy.chat({
      model: "rt-allowed",
      messages: [{ role: "user", content: "round-trip" }],
    });
    const etcdChat = await etcdProxy.chat({
      model: "rt-allowed",
      messages: [{ role: "user", content: "round-trip" }],
    });
    expect(fileChat.status).toBe(200);
    expect(etcdChat.status).toBe(200);
    const message = (r: unknown) =>
      (r as { choices: Array<{ message: { role: string; content: string } }> }).choices[0]
        ?.message;
    expect(message(fileChat.body)).toEqual(message(etcdChat.body));

    // 4c) allowed_models enforcement — same 403 envelope on the model the
    // caller key does not grant.
    const fileBlocked = await fileProxy.chat({
      model: "rt-forbidden",
      messages: [{ role: "user", content: "blocked" }],
    });
    const etcdBlocked = await etcdProxy.chat({
      model: "rt-forbidden",
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

  test("default export redacts a live secret on every secret-bearing kind", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    // A dedicated prefix seeded with one distinctively-marked secret per
    // redacted field/kind. Reads etcd directly — no gateway needed. Each
    // entry is asserted present (by name) AND its marker absent, so a
    // rejected-and-dropped seed can't make a leak assertion pass vacuously.
    const etcd = new EtcdClient();
    const prefix = `/aisix-export-noleak-${randomUUID()}`;
    const seed = new SeedClient(etcd, prefix);
    const marker = {
      providerApiKey: "leak-provider-apikey-AAA111",
      providerHeader: "leak-provider-header-BBB222",
      mcpSecret: "leak-mcp-secret-CCC333",
      guardrailApiKey: "leak-guardrail-apikey-DDD444",
      otlpHeader: "leak-otlp-header-EEE555",
    };
    try {
      await seed.createProviderKey({
        display_name: "noleak-pk",
        api_key: marker.providerApiKey,
        request: { default_headers: { "x-tenant-token": marker.providerHeader } },
      });
      await seed.update("mcp_servers", randomUUID(), {
        name: "noleak-mcp",
        url: "https://mcp.example.com/mcp",
        auth_type: "bearer",
        secret: marker.mcpSecret,
      });
      const guardrail = await seed.createGuardrail({
        name: "noleak-guardrail",
        kind: "openai_moderation",
        api_key: marker.guardrailApiKey,
      });
      // Attach it env-wide so it is already gateway-wide in etcd and the
      // exporter emits it (exercising guardrail-credential redaction);
      // an attachment-scoped guardrail would be omitted by design.
      await seed.update("guardrail_attachments", randomUUID(), {
        guardrail_id: guardrail.id,
        scope_type: "env",
        priority: 1,
      });
      await seed.createObservabilityExporter({
        name: "noleak-otlp",
        kind: "otlp_http",
        endpoint: "https://otlp.example.com/v1/traces",
        headers: { "x-honeycomb-team": marker.otlpHeader },
      });

      const { stdout: yaml, stderr } = await execFileP(BIN_PATH, [
        "export",
        "--etcd",
        ETCD_ENDPOINT,
        "--prefix",
        prefix,
      ]);

      // Each entry made it into the export (not silently dropped)…
      for (const name of ["noleak-pk", "noleak-mcp", "noleak-guardrail", "noleak-otlp"]) {
        expect(yaml, `${name} missing from export`).toContain(name);
      }
      // …and not one live secret survived at default settings, in the
      // file or in the stderr diagnostics.
      for (const [field, value] of Object.entries(marker)) {
        expect(yaml, `secret leaked to file: ${field}`).not.toContain(value);
        expect(stderr, `secret leaked to stderr: ${field}`).not.toContain(value);
      }
      expect(yaml).toContain("${AISIXSECRET");
      expect(stderr).toContain("secret value(s) were replaced");
    } finally {
      await etcd.deletePrefix(prefix);
    }
  });
});
