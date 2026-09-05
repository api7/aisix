import { randomUUID } from "node:crypto";
import { connect } from "node:net";
import { afterEach, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  etcdEndpoint,
  spawnApp,
  startEtcdRelay,
  type EtcdRelay,
  type SpawnedApp,
} from "../harness/index.js";

// `Client::connect` performs I/O only when etcd credentials are
// configured — it issues the `Authenticate` RPC — so the same gateway
// used to start in two opposite ways depending on a key that has nothing
// to do with reachability. Without credentials it started, held the proxy
// listener closed and waited for etcd however long that took; with them,
// an etcd that was down failed the connect and the process exited.
//
// These specs pin the split that replaced it. An etcd that never
// answered is waited out on the supervisor's retry loop, exactly as the
// unauthenticated deployment is. Credentials etcd answered and refused
// still end the boot, because no amount of waiting fixes them.
//
// The password is real config here: the gateway reads it from the env
// var `etcd.password_env` names.
const ETCD_USER = "e2e-auth-probe";
const PASSWORD_ENV = "ETCD_E2E_PASSWORD";

/** The supervisor's retry log line, emitted at WARN on every failed cycle. */
const BACKOFF_LINE = "etcd watch failed; backing off before reconnect";
/** The boot's own wording for "not reachable, carrying on anyway". */
const DEFERRED_LINE = "etcd is not reachable yet";
/** What a refusal has to be reported as, whatever etcd's own message says. */
const REFUSED_LINE = "etcd rejected the connection";
/** The supervisor's line for a refusal it meets after the boot is past. */
const REFUSED_AFTER_BOOT_LINE = "etcd refused this gateway's credentials";

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

/** True when a TCP connection to `port` on loopback is accepted. */
function tcpAccepts(port: number, timeoutMs = 1_000): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = connect(port, "127.0.0.1");
    const settle = (accepted: boolean) => {
      sock.destroy();
      resolve(accepted);
    };
    sock.once("connect", () => settle(true));
    sock.once("error", () => settle(false));
    sock.setTimeout(timeoutMs, () => settle(false));
  });
}

describe("etcd credentials: unreachable is waited out, refused is not", () => {
  let etcdReachable = false;
  const apps: SpawnedApp[] = [];
  const relays: EtcdRelay[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
  });

  afterEach(async () => {
    await Promise.all(apps.splice(0).map((a) => a.exit()));
    await Promise.all(relays.splice(0).map((r) => r.stop()));
  });

  /**
   * Spawn a gateway that authenticates to etcd. `awaitProxyListener` is
   * off throughout: every case here keeps the first configuration from
   * ever landing, which is exactly when the listener stays closed.
   */
  async function spawnAuthenticated(
    endpoint: string,
    prefix: string,
    etcdExtra: Record<string, unknown> = {},
  ): Promise<SpawnedApp> {
    const app = await spawnApp({
      etcdPrefix: prefix,
      awaitProxyListener: false,
      // The supervisor's backoff line and the boot's deferral line are
      // both WARN, which this level keeps.
      logLevel: "warn",
      extraEnv: { [PASSWORD_ENV]: "not-the-password-etcd-has" },
      extra: {
        etcd: {
          endpoints: [endpoint],
          prefix,
          user: ETCD_USER,
          password_env: PASSWORD_ENV,
          ...etcdExtra,
        },
      },
    });
    apps.push(app);
    return app;
  }

  test("an etcd that never answers no longer ends the boot", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-etcd-auth-refused-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    // Not listening: every dial is refused, which is what a gateway
    // scheduled before its etcd — or during an etcd restart — meets.
    await relay.refuse();

    // Reaching this line is most of the assertion: the boot used to
    // spend 5s × 5 attempts here and then exit.
    const app = await spawnAuthenticated(relay.endpoint, prefix);
    expect(app.output()).toContain(DEFERRED_LINE);

    // Two occurrences, not one: a single line would also be produced by
    // a connection that failed once and gave up. The loop is the subject.
    expect(await waitForOutput(app, BACKOFF_LINE, 30_000, 2)).toBeGreaterThanOrEqual(2);

    // Still up, and still refusing to serve: the metrics listener is
    // bound (it answered during spawn) while the proxy listener stays
    // closed because no configuration has been applied.
    const metrics = await fetch(`${app.metricsUrl}/metrics`);
    expect(metrics.status).toBe(200);
    expect(await tcpAccepts(Number(new URL(app.proxyUrl).port))).toBe(false);
  }, 120_000);

  test("a silent endpoint is bounded by dial_timeout_ms and retried", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-etcd-auth-silent-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    // Connection accepted, nothing forwarded: TCP completes and the
    // `Authenticate` round trip is never answered. `dial_timeout_ms`
    // reached only the TCP connect, so this hung the boot outright —
    // before any listener was up and with no line saying why.
    await relay.hold();

    const app = await spawnAuthenticated(relay.endpoint, prefix, { dial_timeout_ms: 1000 });
    expect(app.output()).toContain(DEFERRED_LINE);
    // …and the abort was the dial bound, not some other failure.
    expect(app.output()).toContain("etcd.dial_timeout_ms");
    expect(await waitForOutput(app, BACKOFF_LINE, 30_000, 2)).toBeGreaterThanOrEqual(2);

    const metrics = await fetch(`${app.metricsUrl}/metrics`);
    expect(metrics.status).toBe(200);
  }, 120_000);

  test("a refusal that only arrives after boot is reported, not buried", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    // The boot can only exit on a refusal it is given. Scheduled before
    // its etcd — the ordering this whole change is about — it takes the
    // unreachable branch instead, and a refusal that lands later reaches
    // the supervisor rather than the boot path. The process is up and
    // serving from whatever it has by then, so it keeps retrying; what it
    // must not do is file "your credentials are wrong" under the same
    // routine warn line as "etcd is restarting", which is how a
    // configuration mistake stays invisible for a day.
    const prefix = `/aisix-e2e-etcd-auth-late-${randomUUID()}`;
    const relay = await startEtcdRelay();
    relays.push(relay);
    await relay.refuse();

    const app = await spawnAuthenticated(relay.endpoint, prefix);
    expect(app.output()).toContain(DEFERRED_LINE);

    // The endpoint comes back — as the suite's own etcd, which has
    // authentication disabled and therefore answers the credentials with
    // a refusal rather than going quiet.
    await relay.release();
    expect(await waitForOutput(app, REFUSED_AFTER_BOOT_LINE, 60_000)).toBeGreaterThanOrEqual(1);

    // Still up: a refusal after boot is loud, not fatal.
    const metrics = await fetch(`${app.metricsUrl}/metrics`);
    expect(metrics.status).toBe(200);
  }, 120_000);

  test("credentials the cluster refuses end the boot, promptly and by name", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    // Straight at the real etcd, which has authentication disabled, so
    // it answers the `Authenticate` call with a refusal rather than
    // going quiet. That is the half that must NOT be waited out: an
    // instance that hung here would be a configuration mistake turned
    // into a gateway that is up and permanently empty.
    const prefix = `/aisix-e2e-etcd-auth-refusal-${randomUUID()}`;
    const started = Date.now();
    await expect(spawnAuthenticated(etcdEndpoint(), prefix)).rejects.toThrow(
      new RegExp(REFUSED_LINE),
    );
    // The old path reached the same exit only after 5s × 5 attempts.
    expect(Date.now() - started).toBeLessThan(15_000);
  }, 120_000);
});
