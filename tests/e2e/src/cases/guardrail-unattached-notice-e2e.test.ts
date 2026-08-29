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
//
// The index is rebuilt lazily — on the first request after a snapshot
// version change, not on the write itself — so every step that needs a
// rebuild sends traffic. A test that only wrote config would observe no
// builds at all and pass while asserting nothing.

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
  let etcdReachable = false;

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
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("an attached guardrail is never reported, however the two writes interleave", async () => {
    if (!etcdReachable || !seed || !app) return;
    // `createGuardrail` writes the guardrail and its env attachment as two
    // separate etcd keys — the ordinary creation shape, and the window the
    // notice used to fire in.
    const name = "un-attached-guard";
    await seed.createGuardrail({
      name,
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: "un-attached-probe" }],
    });

    // Force several index rebuilds with the attachment in place. If the
    // notice were going to fire on this guardrail it would have by now, and
    // being deduplicated per id it could never be withdrawn afterwards.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    for (let i = 0; i < 3; i++) {
      await seed.createProviderKey({
        display_name: `un-churn-pk-${i}`,
        secret: "sk-mock",
        api_base: `${upstream!.baseUrl}/v1`,
      });
      await waitConfigPropagation();
      await proxy.chat({
        model: "un-model",
        messages: [{ role: "user", content: "harmless" }],
      });
    }

    expect(noticesFor(app.output(), name)).toBe(0);
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

    // Two index builds have to pass before the notice is due — the first
    // sighting is indistinguishable from an attachment still in flight — and
    // a build only happens on a request.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      await proxy.chat({
        model: "un-model",
        messages: [{ role: "user", content: "harmless" }],
      });
      return noticesFor(app!.output(), name) > 0;
    });

    // …and it is a standing property said once, not an event repeated on
    // every later rebuild. Each iteration needs a WRITE as well as a
    // request: the index is keyed on the snapshot version, so requests
    // alone reuse the cached index and would prove nothing.
    const after = noticesFor(app.output(), name);
    for (let i = 0; i < 3; i++) {
      await seed.createProviderKey({
        display_name: `un-orphan-churn-pk-${i}`,
        secret: "sk-mock",
        api_base: `${upstream!.baseUrl}/v1`,
      });
      await waitConfigPropagation();
      await proxy.chat({
        model: "un-model",
        messages: [{ role: "user", content: "harmless" }],
      });
    }
    expect(noticesFor(app.output(), name)).toBe(after);
  });
});
