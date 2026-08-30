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

  // rebuildIndex forces exactly one guardrail-index build.
  //
  // It takes BOTH a write and a request, and neither alone will do: the
  // index is keyed on the snapshot version, so requests without a write
  // reuse the cached index, and a write without a request builds nothing —
  // the rebuild is lazy, deferred to the first request after the version
  // moves. A loop missing either half observes no builds at all and passes
  // while asserting nothing.
  async function rebuildIndex(): Promise<void> {
    await seed!.createProviderKey({
      display_name: `un-churn-pk-${churn++}`,
      secret: "sk-mock",
      api_base: `${upstream!.baseUrl}/v1`,
    });
    await waitConfigPropagation();
    await proxy!.chat({
      model: "un-model",
      messages: [{ role: "user", content: "harmless" }],
    });
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
      display_name: "un-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["un-model"],
    });
    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("an attached guardrail is never reported, however the two writes interleave", async () => {
    if (!etcdReachable || !seed || !app) return;

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
    for (let i = 0; i < 3; i++) await rebuildIndex();

    // Nothing, and nothing later either: the notice is deduplicated per id
    // for the life of the process, so one report here could never be
    // withdrawn once the attachment landed.
    expect(noticesFor(app.output(), raced)).toBe(0);
  });

  test("a guardrail nothing attaches is reported, once", async () => {
    if (!etcdReachable || !seed || !app) return;
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

    // Two builds have to pass before the notice is due — the first sighting
    // is indistinguishable from an attachment still in flight. A couple of
    // spare rounds absorb an unrelated write landing in between; the
    // assertion is on the count, so a late notice still fails.
    for (let i = 0; i < 4 && noticesFor(app.output(), name) === 0; i++) {
      await rebuildIndex();
    }
    expect(noticesFor(app.output(), name)).toBe(1);

    // …and it is a standing property said once, not an event repeated on
    // every later build.
    for (let i = 0; i < 3; i++) await rebuildIndex();
    expect(noticesFor(app.output(), name)).toBe(1);
  });
});
