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

// E2E for the gauge-retirement sweep.
//
// Three gauge families are written only from the request (or health-check)
// path, and the recorder registers no idle timeout — so a series whose key
// is deleted is never written again and keeps reporting its last value as
// though it described the present. The documented budget alert has no
// defence against that: the stranded sample carries `details_present=1`
// too, so a key that was over its threshold when it was deleted latches
// the alert until the process restarts.
//
// `aisix_ratelimit_remaining_*` is the family that can be driven with no
// control plane — the value comes from the data plane's own limiter — so
// it is what proves the whole chain here against a real binary: the series
// is registered on a request, the key is deleted from etcd, and the sweep
// running on the periodic upkeep task retires it.
//
// The retired value is asserted as the literal `NaN`, not "absent" and not
// zero. Zero is a MEANINGFUL value on this metric — `remaining_requests 0`
// says the caller is out of quota — so a retirement that zeroed would
// replace a stale claim with a false one. Checking the rendered text
// rather than the parsed samples is deliberate: the harness's sample
// regex does not match `NaN`, so a parsed-only assertion would pass
// equally for "retired", "never emitted" and "series dropped".

const PLAINTEXT = "sk-gauge-retirement";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const MODEL = "gr-model";

describe("gauge retirement e2e: a deleted key stops reporting", () => {
  let upstream: OpenAiUpstream | undefined;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let keyId = "";
  let etcdReachable = false;

  async function scrapeText(): Promise<string> {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.ok).toBe(true);
    return res.text();
  }

  /** The `aisix_ratelimit_remaining_requests` line for the seeded key. */
  function remainingLine(scrape: string): string | undefined {
    return scrape
      .split("\n")
      .find(
        (l) =>
          l.startsWith("aisix_ratelimit_remaining_requests{") &&
          l.includes(`api_key_id="${keyId}"`),
      );
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp({});
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gr-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const key = await seed.createApiKey({
      key_hash: hash(PLAINTEXT),
      allowed_models: [MODEL],
      // Any limit will do: the gauge is emitted from the post-dispatch
      // peek, which only runs for a key that has one.
      rate_limit: { rpm: 100 },
    });
    keyId = key.id;

    const proxy = new ProxyClient(app.proxyUrl, PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await proxy.chat({ model: MODEL, messages: [{ role: "user", content: "hi" }] });
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.stop();
    await upstream?.close();
  });

  test("the series reports a real value while the key exists", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const line = remainingLine(await scrapeText());
    expect(line, "the request must have registered the gauge").toBeTruthy();
    expect(line).toContain(`model="${MODEL}"`);
    // A number, not NaN: nothing has been retired yet.
    const value = line!.slice(line!.lastIndexOf(" ") + 1);
    expect(Number.isFinite(Number(value))).toBe(true);
  });

  test("deleting the key retires the series to NaN, not to zero", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    await seed.delete("api_keys", keyId);
    // The key is gone from the snapshot once it stops authenticating.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "x" }] }),
      });
      await res.text();
      return res.status === 401;
    });

    // The sweep rides the periodic upkeep task, so this is the one place
    // the test has to wait on a tick rather than on a request.
    const deadline = Date.now() + 30_000;
    let line: string | undefined;
    for (;;) {
      line = remainingLine(await scrapeText());
      if (line?.endsWith(" NaN")) break;
      if (Date.now() > deadline) break;
      await new Promise((r) => setTimeout(r, 500));
    }

    expect(line, "the retired series must still be exported").toBeTruthy();
    expect(
      line,
      "a retired remaining-quota gauge must read NaN — 0 would say the caller is throttled",
    ).toMatch(/ NaN$/);
  }, 60_000);
});
