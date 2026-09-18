import { createHash, randomUUID } from "node:crypto";
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

// E2E for config forward compatibility (issue #871). Under the supported
// rolling-upgrade order the control plane upgrades first and may write
// resource documents carrying fields this data-plane version does not
// know. The observable contract:
//
// - such a document LOADS and behaves (an api_key authenticates real
//   traffic) with the unknown fields ignored — not whole-row rejected;
// - the tolerance is never silent: `GET /status/config` reports the
//   ignored fields as `partially_compatible[]` next to `rejected[]`,
//   and the metrics listener exposes a per-kind gauge;
// - a converged same-version deployment (documents written by this
//   version's own canonical shapes) reports ZERO partially-compatible
//   rows — the strictness that catches typos lives in the declarative
//   writers (`aisix validate`, the file source, the control plane).

const CALLER_PLAINTEXT = "sk-forward-compat-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

interface StatusConfig {
  state: string;
  applied?: { resource_counts: Record<string, number> };
  last_reload?: { successful: boolean; at: string };
  last_failure: { last_error_kind: string } | null;
  rejected: Array<{ resource_kind: string; resource_id: string }>;
  unknown_kinds: Array<{
    resource_kind: string;
    resource_id: string;
    last_error: string;
  }>;
  partially_compatible: Array<{
    resource_kind: string;
    field: string;
    count: number;
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

/** One gauge sample's value, or `undefined` when the series is absent. */
function gauge(text: string, name: string, labels = ""): number | undefined {
  const line = text
    .split("\n")
    .find((l) => l.startsWith(labels ? `${name}{${labels}}` : `${name} `));
  return line === undefined ? undefined : Number(line.split(" ").pop());
}

describe("config forward-compat: unknown fields from a newer control plane", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let yellowKeyId: string;
  let pkId: string;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "fc-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    pkId = pk.id;
    await seed.createModel({
      display_name: "fc-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("an api_key document with an unknown field authenticates and is reported partially compatible", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    // A document as a newer CP would write it: canonical api_key fields
    // plus one this DP version has never heard of.
    yellowKeyId = randomUUID();
    await etcd.put(
      `${app.etcdPrefix}/api_keys/${yellowKeyId}`,
      JSON.stringify({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["fc-model"],
        quota_profile: "gold",
      }),
    );

    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return (cfg.applied?.resource_counts.api_keys ?? 0) >= 1;
    });

    // The credential WORKS — the user journey the strict reader broke:
    // pre-#871 this row was whole-row rejected and the key 401'd
    // identically to "no such key".
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const chat = await proxy.chat({
      model: "fc-model",
      messages: [{ role: "user", content: "does the forward-compat key work?" }],
    });
    expect(chat.status, JSON.stringify(chat.body)).toBe(200);

    // A second traffic-bearing kind: a model document with an unknown
    // field must also load and serve chat.
    const yellowModelId = randomUUID();
    await etcd.put(
      `${app.etcdPrefix}/models/${yellowModelId}`,
      JSON.stringify({
        display_name: "fc-model-yellow",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pkId,
        future_model_knob: true,
      }),
    );
    await etcd.put(
      `${app.etcdPrefix}/api_keys/${randomUUID()}`,
      JSON.stringify({
        key_hash: createHash("sha256").update(`${CALLER_PLAINTEXT}-2`).digest("hex"),
        allowed_models: ["fc-model-yellow"],
      }),
    );
    // Both puts have to land: the model was written first, so waiting on
    // the model count alone can be satisfied while the credential that
    // addresses it is still in flight.
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return (
        (cfg.applied?.resource_counts.models ?? 0) >= 2 &&
        (cfg.applied?.resource_counts.api_keys ?? 0) >= 2
      );
    });
    const proxy2 = new ProxyClient(app.proxyUrl, `${CALLER_PLAINTEXT}-2`);
    const chat2 = await proxy2.chat({
      model: "fc-model-yellow",
      messages: [{ role: "user", content: "does the forward-compat model serve?" }],
    });
    expect(chat2.status, JSON.stringify(chat2.body)).toBe(200);
    await etcd.delete(`${app.etcdPrefix}/models/${yellowModelId}`);

    // The tolerance is reported, not silent: the exact ignored field with
    // a row count, next to an empty rejected[]. The row is served, so the
    // state stays synced rather than degraded.
    cfg = await getStatusConfig(app);
    expect(cfg.state).toBe("synced");
    expect(cfg.rejected).toHaveLength(0);
    expect(cfg.partially_compatible).toContainEqual({
      resource_kind: "api_keys",
      field: "quota_profile",
      count: 1,
    });

    // And on the metrics listener as a per-kind gauge.
    const text = await scrape(app);
    expect(text).toMatch(
      /aisix_config_partially_compatible_resources\{kind="api_keys"\} 1/,
    );
  });

  test("a value the gateway cannot interpret stays rejected (unknown enum value)", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    // An unknown VALUE has no lenient fallback — there is no old behavior
    // to run for a routing strategy this version cannot interpret. The
    // row must reject (RED), not load partially.
    const badId = randomUUID();
    await etcd.put(
      `${app.etcdPrefix}/models/${badId}`,
      JSON.stringify({
        display_name: "fc-router",
        routing: {
          strategy: "strategy-from-the-future",
          targets: [{ model: "fc-model" }],
        },
      }),
    );

    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.rejected.some((r) => r.resource_id === badId);
    });
    expect(cfg!.state).toBe("degraded");
    expect(cfg!.rejected.find((r) => r.resource_id === badId)!.resource_kind).toBe(
      "models",
    );

    await etcd.delete(`${app.etcdPrefix}/models/${badId}`);
  });

  test("deleting the forward-compat row clears the report; converged config has zero partially-compatible rows", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    await etcd.delete(`${app.etcdPrefix}/api_keys/${yellowKeyId}`);

    // Zero-YELLOW invariant at equal versions: every remaining document
    // was written through this version's own canonical shapes
    // (SeedClient), so nothing may report as partially compatible. A
    // failure here means the seed shapes and the DP models drifted —
    // exactly the typo class the old strictness caught.
    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.partially_compatible.length === 0 && cfg.state === "synced";
    });
    expect(cfg!.applied?.resource_counts.models).toBe(1);
    expect(cfg!.applied?.resource_counts.provider_keys).toBe(1);
    expect(cfg!.rejected).toHaveLength(0);

    // The gauge zeroes rather than lingering at its stale value.
    const text = await scrape(app);
    expect(text).toMatch(
      /aisix_config_partially_compatible_resources\{kind="api_keys"\} 0/,
    );
  });
});

// E2E for a resource KIND from a newer control plane (issue #1207). The
// supported upgrade order is control plane first, and a new resource kind is
// a free change under the compatibility policy, so it ships without waiting
// for the support floor to move: every gateway in the field then reads a
// document whose `kind` segment it has never heard of. That is forward
// compatibility, not a load failure, and the observable contract says so:
//
// - `aisix_config_last_reload_successful` stays `1` while unknown kinds are
//   the only thing the gateway did not load — that gauge is what operators
//   alert on, and the row is nothing they can fix or delete;
// - the rows are counted in their own series, not in
//   `aisix_config_rejected_resources`, so a real rejection stays visible;
// - `/status/config` agrees: `unknown_kinds[]`, not `rejected[]`;
// - a genuine rejection still flips everything, in the same snapshot;
// - the boot full load and an incremental watch event classify identically.
describe("config forward-compat: a resource kind from a newer control plane", () => {
  const FUTURE_KIND = "quota_pools";
  // Written once in beforeAll and never reassigned, so no case can read it
  // undefined and the deletion case cannot target a key that never existed.
  const futureRowId = randomUUID();
  let app: SpawnedApp | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "fk-pk",
      secret: "sk-mock",
      api_base: "http://127.0.0.1:1/v1",
    });
    await seed.createModel({
      display_name: "fk-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // A live watch event on a running gateway, as a newer control plane
    // projecting a kind this build predates would deliver it.
    await etcd.put(
      `${app.etcdPrefix}/${FUTURE_KIND}/${futureRowId}`,
      JSON.stringify({ display_name: "next release's resource", limit: 10 }),
    );
  });

  afterAll(async () => {
    await app?.exit();
  });

  test("an unknown kind is reported apart and leaves the reload successful", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    let cfg: StatusConfig | undefined;
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.unknown_kinds.some((r) => r.resource_id === futureRowId);
    });

    expect(cfg!.unknown_kinds).toContainEqual(
      expect.objectContaining({
        resource_kind: FUTURE_KIND,
        resource_id: futureRowId,
      }),
    );
    // Not a rejection, in either view.
    expect(cfg!.rejected).toHaveLength(0);
    expect(cfg!.state).toBe("synced");
    expect(cfg!.last_reload!.successful).toBe(true);
    expect(cfg!.last_failure).toBeNull();

    const text = await scrape(app);
    expect(gauge(text, "aisix_config_last_reload_successful")).toBe(1);
    expect(
      gauge(text, "aisix_config_unknown_kind_resources", `kind="${FUTURE_KIND}"`),
    ).toBe(1);
    expect(
      gauge(text, "aisix_config_rejected_resources", `kind="${FUTURE_KIND}"`) ?? 0,
    ).toBe(0);
  });

  test("a genuine rejection in the same snapshot still fails the reload", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    // An unknown routing strategy has no lenient fallback: the row rejects,
    // and THAT is what a failed reload means.
    const badId = randomUUID();
    let cfg: StatusConfig | undefined;
    try {
      await etcd.put(
        `${app.etcdPrefix}/models/${badId}`,
        JSON.stringify({
          display_name: "fk-router",
          routing: {
            strategy: "strategy-from-the-future",
            targets: [{ model: "fk-model" }],
          },
        }),
      );

      await waitConfigPropagation(async () => {
        cfg = await getStatusConfig(app!);
        return cfg.rejected.some((r) => r.resource_id === badId);
      });
      expect(cfg!.state).toBe("degraded");
      expect(cfg!.last_reload!.successful).toBe(false);
      // The classes never mix: the unknown kind is still reported, apart.
      expect(cfg!.unknown_kinds.some((r) => r.resource_id === futureRowId)).toBe(
        true,
      );
      expect(cfg!.rejected.some((r) => r.resource_id === futureRowId)).toBe(false);

      const text = await scrape(app);
      expect(gauge(text, "aisix_config_last_reload_successful")).toBe(0);
      expect(gauge(text, "aisix_config_rejected_resources", 'kind="models"')).toBe(1);
      expect(
        gauge(text, "aisix_config_unknown_kind_resources", `kind="${FUTURE_KIND}"`),
      ).toBe(1);
    } finally {
      await etcd.delete(`${app.etcdPrefix}/models/${badId}`);
    }

    // Fixing the real problem restores the gauge even though the unknown
    // kind is still there — the discriminating step: before the fix the
    // unknown-kind row alone held this at 0 until the gateway was upgraded.
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.rejected.length === 0;
    });
    expect(cfg!.last_reload!.successful).toBe(true);
    expect(cfg!.state).toBe("synced");
    const text = await scrape(app);
    expect(gauge(text, "aisix_config_last_reload_successful")).toBe(1);
  });

  test("the boot full load classifies the unknown kind the same way", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    // The watch path and the initial full load are separate call sites in
    // the loader; a successor on the same prefix reads the unknown-kind row
    // through the boot load instead of a watch event.
    const successor = await spawnApp({ etcdPrefix: app.etcdPrefix });
    try {
      let cfg: StatusConfig | undefined;
      await waitConfigPropagation(async () => {
        cfg = await getStatusConfig(successor);
        return (cfg.applied?.resource_counts.models ?? 0) >= 1;
      });
      expect(cfg!.unknown_kinds).toContainEqual(
        expect.objectContaining({
          resource_kind: FUTURE_KIND,
          resource_id: futureRowId,
        }),
      );
      expect(cfg!.rejected).toHaveLength(0);
      expect(cfg!.state).toBe("synced");
      expect(cfg!.last_reload!.successful).toBe(true);

      const text = await scrape(successor);
      expect(gauge(text, "aisix_config_last_reload_successful")).toBe(1);
      expect(
        gauge(text, "aisix_config_unknown_kind_resources", `kind="${FUTURE_KIND}"`),
      ).toBe(1);
    } finally {
      // The original app owns the prefix cleanup.
      await successor.stop();
    }
  });

  test("deleting the unknown-kind row clears the report and zeroes the gauge", async (ctx) => {
    if (!etcdReachable || !app || !etcd) {
      ctx.skip();
      return;
    }

    // Precondition, not decoration: without it an empty `unknown_kinds[]`
    // would satisfy the wait below even if the row had never been written.
    let cfg = await getStatusConfig(app);
    expect(cfg.unknown_kinds.some((r) => r.resource_id === futureRowId)).toBe(true);

    await etcd.delete(`${app.etcdPrefix}/${FUTURE_KIND}/${futureRowId}`);
    await waitConfigPropagation(async () => {
      cfg = await getStatusConfig(app!);
      return cfg.unknown_kinds.length === 0;
    });
    expect(cfg.state).toBe("synced");
    expect(cfg.last_reload!.successful).toBe(true);

    // Zero, not a lingering stale count and not NaN: the series counts rows
    // in a state, so 0 is the true reading and `sum()` keeps working.
    const text = await scrape(app);
    expect(
      gauge(text, "aisix_config_unknown_kind_resources", `kind="${FUTURE_KIND}"`),
    ).toBe(0);
  });
});
