import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Characterization: a resource document seeded straight to etcd
// (`SeedClient`) must be indistinguishable from the same resource
// created through the Admin API — both after the store's serde
// round-trip (read back via admin GET) and behaviorally on the proxy
// (chat succeeds through seeded resources; allowed_models authz reads
// a seeded caller key exactly like an admin-created one).
//
// This pins the contract the e2e seed migration relies on: the Admin
// API's create handlers add nothing to the stored document beyond the
// id in the key, so tests may write canonical documents directly —
// the same front door the control plane uses in managed mode.
// Transitional: this pin de-risks migrating case seeding to
// direct document writes; once that migration completes it can be
// folded into the general suite or retired.

const ADMIN_CALLER = "sk-char-admin-caller";
const SEED_CALLER = "sk-char-seed-caller";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** Strip the fields a pair intentionally differs by (identity and
 * cross-references), leaving defaults and everything the handler might
 * have added — which is exactly what the comparison is about. */
function normalize(
  value: Record<string, unknown>,
  varied: string[],
): Record<string, unknown> {
  const copy: Record<string, unknown> = { ...value };
  for (const f of varied) delete copy[f];
  return copy;
}

type Entry = { id: string; value: Record<string, unknown> };

describe("seed-vs-admin characterization: direct etcd writes ≡ Admin API writes", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdClient: EtcdClient | undefined;
  let etcdReachable = false;

  let adminPk: Entry, seedPk: Entry;
  let adminModel: Entry, seedModel: Entry;
  let adminKey: Entry, seedKey: Entry;
  let adminExporter: Entry, seedExporter: Entry;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdClient = etcd;
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    seed = new SeedClient(etcd, app.etcdPrefix);

    // One resource of each kind through each front door. Bodies are
    // identical except identity fields (display_name/name) and
    // cross-references (provider_key_id, allowed_models, key_hash).
    adminPk = await admin.createProviderKey({
      display_name: "char-admin-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    seedPk = await seed.createProviderKey({
      display_name: "char-seed-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    adminModel = await admin.createModel({
      display_name: "char-admin-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: adminPk.id,
    });
    seedModel = await seed.createModel({
      display_name: "char-seed-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: seedPk.id,
    });

    // Carry optional fields too so the projection comparison below has
    // non-empty residue after the varied fields are stripped.
    adminKey = await admin.createApiKey({
      key_hash: sha256(ADMIN_CALLER),
      allowed_models: ["char-admin-model"],
      rate_limit: { rpm: 1000 },
      expires_at: "2030-01-01T00:00:00Z",
    });
    seedKey = await seed.createApiKey({
      key_hash: sha256(SEED_CALLER),
      allowed_models: ["char-seed-model"],
      rate_limit: { rpm: 1000 },
      expires_at: "2030-01-01T00:00:00Z",
    });

    // Exporters only need to load — point at a closed port, disabled.
    adminExporter = await admin.createObservabilityExporter({
      name: "char-admin-exporter",
      kind: "otlp_http",
      endpoint: "http://127.0.0.1:9/otlp",
      enabled: false,
    });
    seedExporter = await seed.createObservabilityExporter({
      name: "char-seed-exporter",
      kind: "otlp_http",
      endpoint: "http://127.0.0.1:9/otlp",
      enabled: false,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("seeded resources serve traffic exactly like admin-created ones", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const seedClient = new OpenAI({
      apiKey: SEED_CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    const adminClient = new OpenAI({
      apiKey: ADMIN_CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Positive probes double as propagation gates: a 200 through each
    // caller proves its ProviderKey + Model + ApiKey all propagated,
    // whichever front door wrote them.
    await waitConfigPropagation(async () => {
      try {
        const [a, s] = await Promise.all([
          adminClient.chat.completions.create({
            model: "char-admin-model",
            messages: [{ role: "user", content: "ready-probe" }],
          }),
          seedClient.chat.completions.create({
            model: "char-seed-model",
            messages: [{ role: "user", content: "ready-probe" }],
          }),
        ]);
        return (
          a.choices[0]?.message.role === "assistant" &&
          s.choices[0]?.message.role === "assistant"
        );
      } catch {
        return false;
      }
    });

    // authz reads the seeded caller key exactly like an admin-created
    // one: the seeded key is NOT allowed the admin model → 403.
    await expect(
      seedClient.chat.completions.create({
        model: "char-admin-model",
        messages: [{ role: "user", content: "must-403" }],
      }),
    ).rejects.toSatisfy((err: unknown) => err instanceof APIError && err.status === 403);
  });

  test("seeded documents read back identical to admin-created ones after the store round-trip", async (ctx) => {
    if (!etcdReachable || !admin || !app || !etcdClient) {
      ctx.skip();
      return;
    }

    // Admin GET deserializes every etcd value through the same serde
    // models the loader uses, then re-serializes — a serde-malformed
    // seeded document would be skipped by the store and fail the entry
    // lookup below. (The proxy loader additionally applies JSON-Schema
    // validation, which this lens does not; loader acceptance of the
    // traffic-bearing kinds is covered by the 200-probes in the
    // behavioral test above. api_keys read back through a public
    // projection that drops attribution fields, hence the raw-bytes
    // check below.)
    const find = (entries: Entry[], id: string, label: string): Entry => {
      const e = entries.find((x) => x.id === id);
      if (!e) throw new Error(`${label} (${id}) missing from admin GET — seeded document rejected by the store's serde round-trip?`);
      return e;
    };

    const pks = await admin.json<Entry[]>("GET", "/admin/v1/provider_keys");
    expect(
      normalize(find(pks, seedPk.id, "seed provider_key").value, ["display_name"]),
    ).toEqual(
      normalize(find(pks, adminPk.id, "admin provider_key").value, ["display_name"]),
    );

    const models = await admin.json<Entry[]>("GET", "/admin/v1/models");
    // The identity fields themselves must round-trip byte-exact — a
    // handler-side canonicalization (trim/case) would otherwise hide
    // behind normalize().
    expect(find(models, seedModel.id, "seed model").value.display_name).toBe(
      "char-seed-model",
    );
    expect(find(models, adminModel.id, "admin model").value.display_name).toBe(
      "char-admin-model",
    );
    expect(
      normalize(find(models, seedModel.id, "seed model").value, [
        "display_name",
        "provider_key_id",
      ]),
    ).toEqual(
      normalize(find(models, adminModel.id, "admin model").value, [
        "display_name",
        "provider_key_id",
      ]),
    );

    const keys = await admin.json<Entry[]>("GET", "/admin/v1/apikeys");
    expect(
      normalize(find(keys, seedKey.id, "seed api_key").value, [
        "key_hash",
        "allowed_models",
      ]),
    ).toEqual(
      normalize(find(keys, adminKey.id, "admin api_key").value, [
        "key_hash",
        "allowed_models",
      ]),
    );

    // The apikeys GET is a public projection that omits attribution
    // fields — pin the raw stored bytes too, so handler-side enrichment
    // of the stored document cannot hide behind the projection.
    const rawSeedKey = JSON.parse(
      (await etcdClient.get(`${app.etcdPrefix}/api_keys/${seedKey.id}`))!,
    ) as Record<string, unknown>;
    const rawAdminKey = JSON.parse(
      (await etcdClient.get(`${app.etcdPrefix}/api_keys/${adminKey.id}`))!,
    ) as Record<string, unknown>;
    expect(normalize(rawSeedKey, ["key_hash", "allowed_models"])).toEqual(
      normalize(rawAdminKey, ["key_hash", "allowed_models"]),
    );

    const exporters = await admin.json<Entry[]>("GET", "/admin/v1/observability_exporters");
    expect(
      normalize(find(exporters, seedExporter.id, "seed exporter").value, ["name"]),
    ).toEqual(
      normalize(find(exporters, adminExporter.id, "admin exporter").value, ["name"]),
    );
  });

  // Declared last on purpose: it mutates then removes seed-created
  // resources the earlier tests rely on.
  test("seed update rewrites the stored document; seed delete revokes it", async (ctx) => {
    if (!etcdReachable || !admin || !app || !upstream || !seed) {
      ctx.skip();
      return;
    }

    // -- update: overwrite the seeded model's document with a new
    // upstream model_name. Admin GET reads etcd directly, so the store
    // must reflect the write immediately; the proxy snapshot follows
    // after propagation.
    const updated = { ...seedModel.value, model_name: "gpt-4o-mini-updated" };
    await seed.update("models", seedModel.id, updated);

    const models = await admin.json<Entry[]>("GET", "/admin/v1/models");
    expect(models.find((m) => m.id === seedModel.id)?.value.model_name).toBe(
      "gpt-4o-mini-updated",
    );

    const seedCaller = new OpenAI({
      apiKey: SEED_CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    // Propagated when a chat through the alias forwards the NEW
    // model_name to the upstream — the strongest observable effect of
    // the update (a 200 alone would pass on the stale snapshot too).
    await waitConfigPropagation(async () => {
      try {
        const r = await seedCaller.chat.completions.create({
          model: "char-seed-model",
          messages: [{ role: "user", content: "update-probe" }],
        });
        if (r.choices[0]?.message.role !== "assistant") return false;
        const last = upstream!.receivedRequests.at(-1);
        return (
          last !== undefined &&
          (JSON.parse(last.body) as { model?: string }).model ===
            "gpt-4o-mini-updated"
        );
      } catch {
        return false;
      }
    });

    // -- delete: remove the seeded caller key. The store no longer
    // returns it, and after propagation its bearer stops
    // authenticating (fail-closed), while the admin-created caller is
    // untouched.
    await seed.delete("api_keys", seedKey.id);
    expect(
      await etcdClient!.get(`${app.etcdPrefix}/api_keys/${seedKey.id}`),
    ).toBeUndefined();

    await waitConfigPropagation(async () => {
      try {
        await seedCaller.chat.completions.create({
          model: "char-seed-model",
          messages: [{ role: "user", content: "revoked-probe" }],
        });
        return false; // still authenticating — not propagated yet
      } catch (err) {
        return (
          err instanceof APIError && (err.status === 401 || err.status === 403)
        );
      }
    });

    // Control probe: the delete revoked only that key — the
    // admin-created caller still serves (rules out "gateway fell over"
    // as the reason the seed caller went dark).
    const adminCaller = new OpenAI({
      apiKey: ADMIN_CALLER,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    const control = await adminCaller.chat.completions.create({
      model: "char-admin-model",
      messages: [{ role: "user", content: "control-probe" }],
    });
    expect(control.choices[0]?.message.role).toBe("assistant");
  });
});
