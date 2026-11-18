import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { parse as parseYaml } from "yaml";
import {
  awaitWindowHeadroom,
  droppedRows,
  EtcdClient,
  etcdEndpoint,
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
// observably identical to the etcd-backed one — every resource under the
// id the control plane gave it, the same /v1/models listing, the same chat
// completion, the same allowed_models 403, and a cache policy scoped to
// the caller key by that key's id still applying. The exported file also
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
const ETCD_ENDPOINT = etcdEndpoint();

const CALLER_PLAINTEXT = "sk-export-roundtrip-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const OTHER_PLAINTEXT = "sk-export-roundtrip-uncached";
const OTHER_KEY_HASH = createHash("sha256").update(OTHER_PLAINTEXT).digest("hex");
// A distinctive marker so the no-leak assertion is unambiguous.
const PROVIDER_SECRET = "sk-provider-secret-must-not-leak-1a2b3c";

// For the kinds that shape a request without being a route: a token cap far
// below what one request is estimated at, and a prompt and answer long
// enough to exceed it.
const TPM = 5;
const ESTIMATE_PROMPT = "Please summarise the quarterly report for the regional sales team in detail.";
const ESTIMATE_ANSWER =
  "The regional sales team grew revenue in every month of the quarter, led by new accounts.";

function chatBody(content: string, usage = true) {
  return {
    id: `chatcmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    ...(usage ? { usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } } : {}),
  };
}

async function resourceCounts(app: SpawnedApp): Promise<Record<string, number>> {
  const res = await fetch(`${app.metricsUrl}/status/config`);
  const body = (await res.json()) as { applied?: { resource_counts?: Record<string, number> } };
  return body.applied?.resource_counts ?? {};
}

/** Extract every `${VAR}` interpolation reference the export emitted. */
function placeholderVars(yaml: string): string[] {
  return [...yaml.matchAll(/\$\{([^}]+)\}/g)].map((m) => m[1]!);
}

describe("aisix export: etcd → export → file round-trip", () => {
  let etcdApp: SpawnedApp | undefined;
  let fileApp: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  const ids: Record<string, string> = {};

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
    const allowed = await seed.createModel({
      display_name: "rt-allowed",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const forbidden = await seed.createModel({
      display_name: "rt-forbidden",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Scoped to the caller key by its id: the reference a file could not
    // carry while its ids were derived from entry names. The key itself is
    // seeded last, so it authenticating means everything else has landed.
    const key = { id: randomUUID() };
    const policy = await seed.createCachePolicy({
      name: "rt-per-key-cache",
      enabled: true,
      applies_to: `api_key:${key.id}`,
    });
    // A second caller the policy does not scope.
    await seed.createApiKey({
      key_hash: OTHER_KEY_HASH,
      allowed_model_ids: [allowed.id],
    });
    await seed.update("api_keys", key.id, {
      key_hash: CALLER_KEY_HASH,
      allowed_model_ids: [allowed.id],
    });
    Object.assign(ids, {
      pk: pk.id,
      allowed: allowed.id,
      forbidden: forbidden.id,
      key: key.id,
      policy: policy.id,
    });

    // etcd propagates asynchronously — gate on the last-seeded key
    // authenticating before the test reads etcd via export.
    const probe = new ProxyClient(etcdApp.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const { status } = await probe.listModels();
      if (status === 401) return false;
      if (status !== 200) throw new Error(`/v1/models answered ${status} while waiting`);
      return true;
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

    // Every entry is written under its etcd id, references as stored.
    const doc = parseYaml(yaml) as Record<string, Array<Record<string, unknown>>>;
    const idsOf = (kind: string) => doc[kind]!.map((e) => e.id).sort();
    expect(idsOf("provider_keys")).toEqual([ids.pk]);
    expect(idsOf("models")).toEqual([ids.allowed, ids.forbidden].sort());
    expect(idsOf("api_keys")).toContain(ids.key);
    expect(doc.api_keys!.find((k) => k.id === ids.key)!.allowed_model_ids).toEqual([ids.allowed]);
    expect(doc.cache_policies![0]).toMatchObject({
      id: ids.policy,
      applies_to: `api_key:${ids.key}`,
    });

    // 3) Load the exported file into a file-mode gateway, supplying the
    // real secret through the placeholder variables the export emitted.
    const extraEnv = Object.fromEntries(vars.map((v) => [v, PROVIDER_SECRET]));
    fileApp = await spawnApp({
      resourcesFile: yaml,
      extraEnv,
      admin: true,
      exposeDebugHeaders: true,
    });
    // Every exported document loads: a row the loader rejected would be
    // dropped while the gateway boots anyway.
    expect(await droppedRows(fileApp)).toEqual({ rejected: [], unknown_kinds: [] });

    // The file-mode gateway serves each resource under the control-plane id.
    const listIds = async (kind: string) => {
      const res = await fetch(`${fileApp!.adminUrl}/admin/v1/${kind}`, {
        headers: { authorization: `Bearer ${fileApp!.adminKey}` },
      });
      expect(res.status).toBe(200);
      return ((await res.json()) as Array<{ id: string }>).map((e) => e.id).sort();
    };
    expect(await listIds("provider_keys")).toEqual([ids.pk]);
    expect(await listIds("models")).toEqual([ids.allowed, ids.forbidden].sort());
    expect(await listIds("api_keys")).toContain(ids.key);
    expect(await listIds("cache_policies")).toEqual([ids.policy]);

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

    // 4d) The cache policy scoped by the caller key's id applies on the
    // file side: the repeat is served from cache and never reaches the
    // upstream.
    const prompt = `cache-${randomUUID()}`;
    const cachedChat = (caller = CALLER_PLAINTEXT) =>
      fetch(`${fileApp!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${caller}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "rt-allowed", messages: [{ role: "user", content: prompt }] }),
      });
    const first = await cachedChat();
    expect(first.status).toBe(200);
    expect(first.headers.get("x-aisix-cache")).toBe("miss");
    await first.text();
    const hitsBefore = upstream.receivedRequests.length;
    const second = await cachedChat();
    expect(second.status).toBe(200);
    expect(second.headers.get("x-aisix-cache")).toBe("hit");
    await second.text();
    expect(upstream.receivedRequests.length).toBe(hitsBefore);

    // A caller the policy does not scope is not served from that cache.
    for (let i = 0; i < 2; i++) {
      const before = upstream.receivedRequests.length;
      const res = await cachedChat(OTHER_PLAINTEXT);
      expect(res.status).toBe(200);
      expect(res.headers.get("x-aisix-cache")).not.toBe("hit");
      await res.text();
      expect(upstream.receivedRequests.length).toBe(before + 1);
    }
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
      }, { attach: false });
      // Attach it env-wide explicitly (hence `attach: false` above — this
      // test writes the attachment it means). The exporter emits the
      // guardrail and this attachment, exercising guardrail-credential
      // redaction on the round-trip.
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
      expect(yaml).not.toContain("$${AISIXSECRET");
      expect(stderr).toContain("secret value(s) were replaced");

      // The redacted file still loads whole once the placeholders are
      // set: every secret-bearing kind's document survives the export.
      const dir = await mkdtemp(join(tmpdir(), "aisix-export-noleak-"));
      try {
        const path = join(dir, "resources.yaml");
        await writeFile(path, yaml, "utf8");
        const env = {
          ...process.env,
          ...Object.fromEntries(placeholderVars(yaml).map((v) => [v, "placeholder-value"])),
        };
        const { stdout } = await execFileP(BIN_PATH, ["validate", "--resources", path], { env });
        expect(stdout).toContain("OK:");
      } finally {
        await rm(dir, { recursive: true, force: true });
      }
    } finally {
      await etcd.deletePrefix(prefix);
    }
  });
});

describe("aisix export: the kinds that are not routes round-trip too", () => {
  const apps: SpawnedApp[] = [];
  const upstreams: OpenAiUpstream[] = [];

  afterAll(async () => {
    for (const app of apps) await app.exit();
    for (const u of upstreams) await u.close();
  });

  test("usage accounting and the global price catalog behave the same from the file; budgets stay out", async (ctx) => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) {
      ctx.skip();
      return;
    }
    const cheap = await startOpenAiUpstream({ nonStreamBody: chatBody("rt-cheap-served") });
    const pricey = await startOpenAiUpstream({ nonStreamBody: chatBody("rt-pricey-served") });
    const estimated = await startOpenAiUpstream({ nonStreamBody: chatBody(ESTIMATE_ANSWER, false) });
    upstreams.push(cheap, pricey, estimated);

    const source = await spawnApp();
    apps.push(source);
    const seed = new SeedClient(etcd, source.etcdPrefix);
    // The shared catalog the gateway reads beside its own prefix.
    const catalog = new SeedClient(etcd, `${source.etcdPrefix}/global`);

    await seed.update("usage_accounting", randomUUID(), { exclude_estimated_usage: true });
    const ids = {
      team: randomUUID(),
      user: randomUUID(),
      policy: randomUUID(),
      budget: randomUUID(),
    };
    await seed.update("teams", ids.team, { name: "rt-team" });
    await seed.update("users", ids.user, { name: "rt-user" });
    await seed.update("budget_policy", ids.policy, { fail_mode: "sticky", closed_grace_seconds: 600 });
    // Exhausted, as the control plane's aggregator would leave it.
    await seed.update("budgets", ids.budget, {
      scope: "api_key",
      scope_ref: randomUUID(),
      period: "month",
      exhausted: { reason: {} },
    });
    const cheapPrice = await catalog.createPricing({
      key: "vendor/rt-cheap",
      input_per_1k: 0.1,
      output_per_1k: 0.1,
    });
    const priceyPrice = await catalog.createPricing({
      key: "vendor/rt-pricey",
      input_per_1k: 10,
      output_per_1k: 10,
    });

    const model = async (name: string, upstream: OpenAiUpstream, extra: Record<string, unknown> = {}) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        api_key: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
        ...extra,
      });
    };
    // The inline `cost` says the opposite of the catalog, so a gateway
    // without the catalog routes to the other target.
    await model("rt-cheap", cheap, {
      pricing_key: "vendor/rt-cheap",
      cost: { input_per_1k: 50, output_per_1k: 50 },
    });
    await model("rt-pricey", pricey, {
      pricing_key: "vendor/rt-pricey",
      cost: { input_per_1k: 0.01, output_per_1k: 0.01 },
    });
    await seed.createModel({
      display_name: "rt-least-cost",
      routing: { strategy: "least_cost", targets: [{ model: "rt-pricey" }, { model: "rt-cheap" }] },
    });
    await model("rt-estimated", estimated);
    const priceCaller = `sk-export-rt-price-${randomUUID()}`;
    const capCaller = `sk-export-rt-cap-${randomUUID()}`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(capCaller).digest("hex"),
      allowed_models: ["*"],
      rate_limit: { tpm: TPM },
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(priceCaller).digest("hex"),
      allowed_models: ["*"],
    });
    for (const caller of [capCaller, priceCaller]) {
      const probe = new ProxyClient(source.proxyUrl, caller);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
    // The catalog is on its own watch, so the keys above do not imply it.
    await waitConfigPropagation(async () => ((await resourceCounts(source)).global_pricing ?? 0) >= 2);

    const { stdout: yaml, stderr } = await execFileP(BIN_PATH, [
      "export",
      "--etcd",
      ETCD_ENDPOINT,
      "--prefix",
      source.etcdPrefix,
    ]);
    const doc = parseYaml(yaml) as Record<string, unknown> & {
      global?: Record<string, Array<Record<string, unknown>>>;
    };
    const idsOf = (entries: unknown) => (entries as Array<{ id: string }>).map((e) => e.id).sort();
    expect(idsOf(doc.global?.pricing)).toEqual([cheapPrice.id, priceyPrice.id].sort());
    expect(idsOf(doc.teams)).toEqual([ids.team]);
    expect(idsOf(doc.users)).toEqual([ids.user]);
    expect(idsOf(doc.budget_policy)).toEqual([ids.policy]);
    expect(doc.usage_accounting).toEqual([
      expect.objectContaining({ exclude_estimated_usage: true }),
    ]);
    // A budget's exhausted state is the control plane's to clear, so it
    // is never frozen into a file.
    expect(doc.budgets).toBeUndefined();
    expect(stderr).toContain("1 budgets document(s) not exported");

    const vars = placeholderVars(yaml);
    const file = await spawnApp({
      resourcesFile: yaml,
      extraEnv: Object.fromEntries(vars.map((v) => [v, "sk-mock"])),
    });
    apps.push(file);
    expect(await droppedRows(file)).toEqual({ rejected: [], unknown_kinds: [] });
    expect(await resourceCounts(file)).toMatchObject({
      global_pricing: 2,
      usage_accounting: 1,
      teams: 1,
      users: 1,
      budget_policy: 1,
    });
    expect((await resourceCounts(file)).budgets).toBeUndefined();

    const post = (app: SpawnedApp, caller: string, model: string, content: string) =>
      fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
      });

    for (const app of [source, file]) {
      // The catalog price, not the inline cost, ranks the targets.
      const res = await post(app, priceCaller, "rt-least-cost", "cheapest please");
      expect(res.status).toBe(200);
      const body = (await res.json()) as { choices: Array<{ message: { content: string } }> };
      expect(body.choices[0]?.message.content).toBe("rt-cheap-served");

      // The upstream reports no usage, so every request is estimated at
      // more than the key's TPM — and none of it counts.
      await awaitWindowHeadroom(20);
      const statuses: number[] = [];
      for (let i = 0; i < 4; i++) {
        const r = await post(app, capCaller, "rt-estimated", ESTIMATE_PROMPT);
        await r.text();
        statuses.push(r.status);
        await new Promise((done) => setTimeout(done, 300));
      }
      expect(statuses).toEqual([200, 200, 200, 200]);
    }
  });
});
