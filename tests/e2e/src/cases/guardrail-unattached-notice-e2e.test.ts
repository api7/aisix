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
// nothing attaches.
//
// The gateway sweeps for these on a timer rather than reporting from an
// index build, and this file exists because both halves of that are easy
// to get wrong in ways nothing else catches.
//
// It must stay QUIET on a guardrail that is correctly attached. The
// guardrail and its attachment are separate documents arriving as separate
// writes, so there is always a window where the gateway holds one and not
// the other; naming a row there is a log line saying a security control
// inspects no traffic, about one that does — and, deduplicated per id for
// the life of the process, it never ages out.
//
// It must SPEAK on a guardrail that really is attached to nothing, and the
// case that matters is not the one that is easy to test: a scope target
// deleted out from under a rule that was screening traffic yesterday.
// Nobody writes anything afterwards, so a notice that rides a config change
// is never delivered.
//
// The timings below are the gateway's own (`UNATTACHED_GRACE`,
// `UNATTACHED_SWEEP_INTERVAL`). They are wall time, which is why this file
// is slow; the arithmetic of the grace itself is unit-tested with an
// injected clock.

const MODEL = "un-model";
const GRACE = 30_000;
const SWEEP = 10_000;
const CALLER_PLAINTEXT = "sk-unattached-notice-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const NOTICE = /guardrail is enabled but has no attachment/;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Matches the `guardrail_name` field exactly, not as a substring: with
// names like `un-inert` and `un-orphaned` in the same file, a substring
// test silently counts one row's notice against another and the assertion
// that a guardrail was named ONCE quietly reads two.
function noticesFor(output: string, name: string): number {
  const named = new RegExp(`guardrail_name=${name}(\\s|$)`);
  return output.split("\n").filter((l) => NOTICE.test(l) && named.test(l))
    .length;
}

describe("guardrail unattached notice", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let proxy: ProxyClient | undefined;
  let etcdReachable = false;

  async function chat(content: string): Promise<number> {
    return (
      await proxy!.chat({ model: MODEL, messages: [{ role: "user", content }] })
    ).status;
  }

  // awaitInForce gates on a guardrail actually screening traffic — its own
  // pattern coming back blocked. Every assertion about the notice is
  // negative, so without a positive gate they pass just as well on a
  // guardrail that never reached the snapshot. Blocking is a different
  // surface from the log line under test, and `chat` returns a status
  // rather than throwing, so a transport failure still surfaces as itself
  // rather than as this timeout.
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

  test(
    "named only when attached to nothing, and then exactly once",
    async (ctx) => {
      if (!etcdReachable || !seed || !app || !proxy) {
        ctx.skip();
        return;
      }
      const guard = (name: string, probe: string) => ({
        name,
        enabled: true,
        hook_point: "input",
        kind: "keyword",
        patterns: [{ kind: "literal", value: probe }],
      });

      // ORDINARY: guardrail and env attachment written back to back.
      await seed.createGuardrail(guard("un-settled", "un-settled-probe"));
      // INERT: attached to nothing, ever. The one row that must be named.
      await seed.createGuardrail(guard("un-inert", "un-inert-probe"), {
        attach: false,
      });
      // RACED: written alone now, attached after a sweep has already seen
      // it unattached — the window every ordinary creation passes through
      // under live traffic, held open deliberately here.
      const raced = await seed.createGuardrail(guard("un-raced", "un-raced-probe"), {
        attach: false,
      });
      // ORPHANED-LATER: attached now, its attachment removed after the
      // grace. A scope target deleted out from under a rule that has been
      // screening traffic — the case the notice exists for.
      // `attach: false` plus one explicit attachment, so there is exactly
      // one row to remove later — the default would add a second and the
      // guardrail would still be attached after the delete.
      const orphaned = await seed.createGuardrail(
        guard("un-orphaned", "un-orphaned-probe"),
        { attach: false },
      );
      const orphanedAttachment = await seed.attachGuardrailToEnv(orphaned.id);

      await awaitInForce("un-settled-probe");

      // Just over one sweep, so at least one lands with `un-raced` present
      // and unattached — the window every ordinary creation passes through.
      //
      // Deliberately NOT sized from the grace. The row's clock starts at
      // the first sweep after it is WRITTEN, which is before this sleep
      // begins — the `awaitInForce` above sits in between, and the harness
      // budgets up to 30s for guardrail propagation on a loaded runner. A
      // window of `GRACE - SWEEP` would leave a correctly-attached
      // guardrail unattached for that plus the propagation wait, and past
      // the grace it gets named: a false red on the assertion below. The
      // grace's lower bound is asserted directly in the unit tests, which
      // is where a value can be checked without racing anything.
      await sleep(SWEEP + 2_000);
      await seed.attachGuardrailToEnv(raced.id);
      await awaitInForce("un-raced-probe");

      // Past the grace, plus a sweep to act on it.
      await sleep(GRACE + SWEEP);
      const out = () => app!.output();
      expect(noticesFor(out(), "un-inert")).toBe(1);
      expect(noticesFor(out(), "un-settled")).toBe(0);
      expect(noticesFor(out(), "un-raced")).toBe(0);
      expect(noticesFor(out(), "un-orphaned")).toBe(0);

      // Now take the attachment away. `un-orphaned` has been present since
      // well before the grace, so it is due on the next sweep — it does NOT
      // serve out a fresh one as if it were a new row.
      await seed.delete("guardrail_attachments", orphanedAttachment.id);
      // Gate on the rule actually going inert before waiting on the log, so
      // a delete that never landed fails as itself rather than as a silent
      // sweep.
      await waitConfigPropagation(async () => (await chat("un-orphaned-probe")) === 200);
      // Two sweeps, not more: the row is already older than the grace, so
      // the very next one is due. A window wide enough to also cover a
      // fresh grace would pass on an implementation that restarts the
      // clock when the attachment goes — which is the whole bug.
      await waitConfigPropagation(
        async () => noticesFor(out(), "un-orphaned") > 0,
        SWEEP * 2,
      );
      expect(noticesFor(out(), "un-orphaned")).toBe(1);

      // Said once, not once per sweep, and the others still silent.
      await sleep(SWEEP * 2);
      expect(noticesFor(out(), "un-orphaned")).toBe(1);
      expect(noticesFor(out(), "un-inert")).toBe(1);
      expect(noticesFor(out(), "un-settled")).toBe(0);
      expect(noticesFor(out(), "un-raced")).toBe(0);
    },
    3 * (GRACE + SWEEP * 3),
  );
});
