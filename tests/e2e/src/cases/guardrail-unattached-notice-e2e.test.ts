import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for the notice a gateway logs about an enabled guardrail that
// nothing attaches. The interesting property is WHEN it must stay quiet.
//
// A guardrail and its attachment are separate documents arriving as
// separate writes, so every ordinary creation has an index build where the
// gateway holds the guardrail and not yet its attachment. Warning there
// named correctly-attached guardrails as inspecting nothing, permanently —
// the notice is deduplicated per id for the life of the process, so the
// false alarm never ages out. Nothing in the unit tests could see it: they
// build the index once, and the race needs two builds with a write between.
//
// Both halves are asserted here because either alone is satisfiable the
// wrong way: silence proves nothing if the notice never fires at all, and
// firing proves nothing if it fires on everything.

const MODEL = "un-model";
const CALLER_PLAINTEXT = "sk-unattached-notice-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const NOTICE = /guardrail is enabled but has no attachment/;

function noticesFor(output: string, name: string): number {
  return output.split("\n").filter((l) => NOTICE.test(l) && l.includes(name))
    .length;
}

describe("guardrail unattached notice", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let proxy: ProxyClient | undefined;
  let etcdReachable = false;
  let churn = 0;

  async function chat(content: string): Promise<number> {
    return (
      await proxy!.chat({ model: MODEL, messages: [{ role: "user", content }] })
    ).status;
  }

  async function appliedRevision(): Promise<number> {
    return sumMetric(
      await scrapeMetrics(app!.metricsUrl),
      "aisix_config_applied_revision",
    );
  }

  // rebuildIndex forces exactly one guardrail-index build.
  //
  // It takes BOTH a write and a request, and neither alone will do: the
  // index is keyed on the snapshot version, so requests without a write
  // reuse the cached index, and a write without a request builds nothing —
  // the rebuild is lazy, deferred to the first request after the version
  // moves. A loop missing either half observes no builds at all and passes
  // while asserting nothing.
  //
  // The write is gated on the gateway having APPLIED it rather than on a
  // fixed delay, because that is the event a build hangs off: a request
  // sent while the snapshot is still at the old revision reuses the cached
  // index, so the round silently builds nothing and the loop does fewer
  // builds than it reads as having done.
  async function rebuildIndex(): Promise<void> {
    const before = await appliedRevision();
    await seed!.createProviderKey({
      display_name: `un-churn-pk-${churn++}`,
      secret: "sk-mock",
      api_base: `${upstream!.baseUrl}/v1`,
    });
    await waitConfigPropagation(async () => (await appliedRevision()) > before);
    await chat("harmless");
  }

  // awaitInForce gates on the guardrail actually screening traffic — its
  // own pattern coming back blocked. The notice assertions are all
  // negative, so without a positive gate they pass just as well on a
  // guardrail that never reached the snapshot. Blocking is a different
  // surface from the log line under test, and `chat` returns a status
  // rather than throwing, so a real transport failure still surfaces as
  // itself rather than as this timeout.
  async function awaitInForce(probe: string): Promise<void> {
    await waitConfigPropagation(async () => (await chat(probe)) === 422);
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "un-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Caller key last, then gate on it authenticating: that one condition
    // implies every resource above it is in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL],
    });
    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy!.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("an attached guardrail is never reported, however the two writes interleave", async (ctx) => {
    if (!etcdReachable || !seed || !app || !proxy) {
      ctx.skip();
      return;
    }

    // The ordinary creation shape first: guardrail and env attachment
    // written back to back, then traffic.
    const settled = "un-attached-guard";
    await seed.createGuardrail({
      name: settled,
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: "un-attached-probe" }],
    });
    await awaitInForce("un-attached-probe");
    for (let i = 0; i < 3; i++) await rebuildIndex();
    expect(noticesFor(app.output(), settled)).toBe(0);

    // …then the window the notice actually fired in, opened deliberately.
    // The two documents are separate writes, so a request landing between
    // them builds an index holding the guardrail alone — routine under live
    // traffic, and never reached by a test that writes both back to back.
    const raced = "un-raced-guard";
    const g = await seed.createGuardrail(
      {
        name: raced,
        enabled: true,
        hook_point: "input",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "un-raced-probe" }],
      },
      { attach: false },
    );
    await rebuildIndex();
    await seed.attachGuardrailToEnv(g.id);
    await awaitInForce("un-raced-probe");
    for (let i = 0; i < 3; i++) await rebuildIndex();

    // Nothing, and nothing later either: the notice is deduplicated per id
    // for the life of the process, so one report here could never be
    // withdrawn once the attachment landed.
    expect(noticesFor(app.output(), raced)).toBe(0);
  });

  test("a guardrail nothing attaches is reported at startup, once", async (ctx) => {
    if (!etcdReachable || !seed || !app || !proxy) {
      ctx.skip();
      return;
    }
    const name = "un-orphan-guard";
    await seed.createGuardrail(
      {
        name,
        enabled: true,
        hook_point: "input",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "un-orphan-probe" }],
      },
      { attach: false },
    );

    // Asserted through a fresh process on the same configuration rather
    // than by waiting out the grace on the running one. Two reasons: it is
    // seconds instead of half a minute, and it covers the half the grace
    // cannot — a gateway that RESTARTS onto standing configuration. The
    // notice fires from an index build, builds happen only when the
    // snapshot version moves, and configuration nobody is editing moves
    // nothing, so without the startup report a restarted gateway would say
    // nothing about a rule that inspects no traffic until some unrelated
    // write happened to come along.
    //
    // The grace's own arithmetic is covered by the unit tests, which can
    // inject `now` instead of sleeping.
    const restarted = await spawnApp({ etcdPrefix: app.etcdPrefix });
    try {
      await waitConfigPropagation(
        async () => noticesFor(restarted.output(), name) > 0,
      );
      // Once, not once per rebuild: it is a standing property of the row.
      const seenOnce = noticesFor(restarted.output(), name);
      expect(seenOnce).toBe(1);

      // And the guardrails that ARE attached are not swept up in it — the
      // startup report has no grace, so if it were reading the snapshot
      // wrongly this is where it would show.
      expect(noticesFor(restarted.output(), "un-attached-guard")).toBe(0);
      expect(noticesFor(restarted.output(), "un-raced-guard")).toBe(0);
    } finally {
      await restarted.exit();
    }
  });
});
