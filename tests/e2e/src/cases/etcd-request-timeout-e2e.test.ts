import { createHash, randomUUID } from "node:crypto";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  etcdEndpoint,
  ProxyClient,
  SeedClient,
  spawnApp,
  startEtcdRelay,
  startOpenAiUpstream,
  type EtcdRelay,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// `etcd.request_timeout_ms` bounds one unary etcd call. Until it was
// wired, both etcd timeout keys were accepted by the config parser and
// read by nothing: a configuration range read that never answered stayed
// in flight forever, and the instance served nothing while producing no
// error that explained why.
//
// The three specs pin the three halves of the contract — the bound aborts
// such a read and hands the supervisor a retryable failure; unset leaves
// the read unbounded, with no implicit default, which is what keeps a
// large configuration set bootable; and the watch stream is exempt, so a
// set bound does not tear down a quiet watch.
//
// The first two drive the gateway through a TCP relay standing in for
// etcd, which is what makes a read that never completes reproducible:
// `hold()` accepts the connection and forwards nothing.

const MODEL = "etcd-timeout-model";
const SECOND_MODEL = "etcd-timeout-model-after-idle";
const CALLER_PLAINTEXT = "sk-etcd-timeout-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

/** The supervisor's retry log line, emitted at WARN on every failed cycle. */
const BACKOFF_LINE = "etcd watch failed; backing off before reconnect";
/** The abort's own wording, so a failure of a different kind can't pass. */
const TIMEOUT_LINE = "exceeded etcd.request_timeout_ms";

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Poll the captured output until `needle` appears at least `count` times. */
async function waitForOutput(
  app: SpawnedApp,
  needle: string,
  timeoutMs: number,
  count = 1,
): Promise<number> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const seen = app.output().split(needle).length - 1;
    if (seen >= count || Date.now() >= deadline) return seen;
    await sleep(100);
  }
}

/**
 * Poll `model` until the gateway answers 200 for it, or the budget runs
 * out. A refused connection counts as "not yet": the proxy listener binds
 * only once a configuration has been applied, which in these specs is
 * after the read the relay was holding finally lands.
 *
 * Two ways to run out of budget, and they need different reports. If the
 * gateway answered at all, the last status is returned and the caller's
 * `toBe(200)` names it. If every attempt was refused, there is no status
 * to report — returning `0` would fail as "expected 0 to be 200" and lose
 * the transport cause — so the last error is raised instead.
 */
async function waitForChat(app: SpawnedApp, model: string, timeoutMs: number): Promise<number> {
  const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
  const deadline = Date.now() + timeoutMs;
  let status = 0;
  let lastError: unknown;
  for (;;) {
    try {
      status = (await proxy.chat({ model, messages: [{ role: "user", content: "hi" }] })).status;
      lastError = undefined;
    } catch (err) {
      status = 0;
      lastError = err;
    }
    if (status === 200) return status;
    if (Date.now() >= deadline) {
      if (lastError !== undefined) {
        throw new Error(
          `no response for model ${model} within ${timeoutMs} ms — the proxy listener never bound`,
          { cause: lastError },
        );
      }
      return status;
    }
    await sleep(500);
  }
}

describe("etcd.request_timeout_ms", () => {
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  const apps: SpawnedApp[] = [];
  const relays: EtcdRelay[] = [];

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
  });

  afterEach(async () => {
    await Promise.all(apps.splice(0).map((a) => a.exit()));
    await Promise.all(relays.splice(0).map((r) => r.stop()));
  });

  afterAll(async () => {
    await upstream?.close();
  });

  /** A model reachable through a mock upstream, written to the real etcd. */
  async function seedModel(prefix: string, displayName: string): Promise<void> {
    const seed = new SeedClient(etcd!, prefix);
    const pk = await seed.createProviderKey({
      display_name: `etcd-timeout-pk-${displayName}`,
      secret: "sk-mock",
      api_base: `${upstream!.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
  }

  /** Everything a caller needs for a 200. One API key, both models allowed. */
  async function seedFixtures(prefix: string): Promise<void> {
    await seedModel(prefix, MODEL);
    await new SeedClient(etcd!, prefix).createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL, SECOND_MODEL],
    });
  }

  async function spawnAgainst(
    endpoint: string,
    prefix: string,
    etcdExtra: Record<string, unknown>,
    opts: { awaitProxyListener?: boolean } = {},
  ): Promise<SpawnedApp> {
    const app = await spawnApp({
      etcdPrefix: prefix,
      awaitProxyListener: opts.awaitProxyListener ?? true,
      // The supervisor's backoff line is WARN, which this level keeps.
      logLevel: "warn",
      extra: { etcd: { endpoints: [endpoint], prefix, ...etcdExtra } },
    });
    apps.push(app);
    return app;
  }

  test("aborts a black-holed read and retries it on the supervisor's backoff", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-etcd-rt-bounded-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    // Connection accepted, nothing forwarded: the range read is issued and
    // never answered. Unbounded, it stays in flight forever.
    await relay.hold();
    await seedFixtures(prefix);

    const app = await spawnAgainst(
      relay.endpoint,
      prefix,
      { request_timeout_ms: 1000 },
      { awaitProxyListener: false },
    );

    // Two occurrences, not one: a single abort would also be produced by a
    // read that failed and then gave up. The retry loop is the subject.
    expect(await waitForOutput(app, BACKOFF_LINE, 30_000, 2)).toBeGreaterThanOrEqual(2);
    // …and the abort was the timeout, not some other transport failure.
    expect(app.output()).toContain(TIMEOUT_LINE);

    // The loop being live is what lets the gateway recover on its own once
    // the read can complete: no restart, no operator action.
    await relay.release();
    expect(await waitForChat(app, MODEL, 60_000)).toBe(200);
  }, 150_000);

  test("leaves the read unbounded when unset", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-etcd-rt-unset-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    await relay.hold();
    await seedFixtures(prefix);

    // No `request_timeout_ms` — the shipped default, and what the example
    // config files now leave commented out.
    const app = await spawnAgainst(relay.endpoint, prefix, {}, { awaitProxyListener: false });

    // Well past the 5000 ms this key was previously documented as
    // defaulting to. Any implicit default reintroduced here would abort
    // the read inside this window and log the backoff line; unbounded
    // logs neither. A deployment whose range read simply takes a while —
    // the reason the default is unset — depends on exactly this.
    await sleep(12_000);
    expect(app.output()).not.toContain(BACKOFF_LINE);
    expect(app.output()).not.toContain(TIMEOUT_LINE);

    // Still a live read rather than a wedged one: releasing it serves that
    // very snapshot.
    await relay.release();
    expect(await waitForChat(app, MODEL, 60_000)).toBe(200);
  }, 150_000);

  test("does not bound the watch: a quiet interval past the timeout keeps it live", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    // Straight at the real etcd — the subject is the watch surviving, not
    // a controllable read. The regression this guards is a bound reaching
    // the watch stream itself: the supervisor then treats every quiet
    // interval as a failed cycle, logs the backoff line and reconnects,
    // and a configuration that changes rarely never stops churning.
    const prefix = `/aisix-e2e-etcd-rt-watch-${randomUUID()}`;
    await seedFixtures(prefix);
    const app = await spawnAgainst(etcdEndpoint(), prefix, { request_timeout_ms: 1000 });

    expect(await waitForChat(app, MODEL, 30_000)).toBe(200);

    // Assert on the log written during the idle window, not on the whole
    // log. The boot range read is bounded by this same 1000 ms, and on a
    // loaded runner a cold TCP + HTTP/2 handshake plus the read can
    // exceed it once — the supervisor then retries and succeeds, and the
    // spec still reaches this point. Reading the cumulative output would
    // report that as the watch being torn down, which it is not.
    const beforeIdle = app.output().length;

    // Nothing written under the prefix for many multiples of the bound.
    await sleep(6_000);
    expect(app.output().slice(beforeIdle)).not.toContain(BACKOFF_LINE);

    // The watch opened before that idle window must still deliver this.
    await seedModel(prefix, SECOND_MODEL);
    expect(await waitForChat(app, SECOND_MODEL, 30_000)).toBe(200);
  }, 150_000);
});
