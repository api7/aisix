import { randomUUID } from "node:crypto";
import { connect } from "node:net";
import { beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, etcdEndpoint, spawnApp, type SpawnedApp } from "../harness/index.js";

// E2E: a shared Redis that ANSWERS and refuses the credential must end
// the boot, not be waited out.
//
// The gateway treats an unreachable shared backend as a degradation: the
// listeners bind, the limiter counts per replica, the cache misses, and a
// background task keeps dialling. That is right for an outage and wrong
// for a password, because no amount of retrying turns a wrong password
// into a right one — the gateway comes up healthy and quietly never
// enforces a shared limit again.
//
// It could not tell the two apart. All three drivers report a refused
// credential as an ordinary connectivity failure, so release QA against
// 1.4.0-rc.2 saw a gateway point at a `requirepass` Redis with the wrong
// password, bind every listener, and log a five-second TIMEOUT against a
// server that had answered in 0.26s. The credential was never mentioned.
//
// The Redis here needs no password, so a configured one is refused
// outright ("Client sent AUTH, but no password is set"). That is the same
// refusal an operator's typo produces and it needs no second Redis — and
// it doubles as the check that `password` reaches the handshake at all,
// which in `single` mode it did not: the field was parsed and dropped,
// the boot logged `connected`, and every counter operation failed from
// then on.
//
// Both backends, because they are one connect path with two callers and
// this is exactly the kind of thing that gets fixed in one of them.

const ETCD_ENDPOINT = etcdEndpoint();
const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

/** RESP-level PING, so the suite skips honestly when no redis is up. */
async function redisPing(url: string): Promise<boolean> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return false;
  const host = m[1];
  const port = m[2] ? Number(m[2]) : 6379;
  return new Promise((resolve) => {
    const sock = connect({ host, port }, () => sock.write("PING\r\n"));
    const done = (ok: boolean) => {
      sock.destroy();
      resolve(ok);
    };
    sock.once("data", (buf) => done(buf.toString().startsWith("+PONG")));
    sock.once("error", () => done(false));
    sock.setTimeout(1000, () => done(false));
  });
}

describe("a shared Redis that refuses the credential ends the boot", () => {
  let ready = false;

  beforeAll(async () => {
    ready = (await new EtcdClient().ping()) && (await redisPing(REDIS_URL));
  });

  /**
   * Boot with one of the two Redis-backed subsystems pointed at the live
   * Redis, authenticating with a password it will not accept, and wait
   * for the process to be gone.
   *
   * `awaitListeners: false` because the assertion is that no listener
   * ever comes up: gating on readiness would turn the expected outcome
   * into a harness timeout instead of an exit.
   */
  async function bootRefused(block: "ratelimit" | "cache"): Promise<SpawnedApp> {
    const prefix = `/aisix-e2e-redis-refused-${block}-${randomUUID()}`;
    const app = await spawnApp({
      awaitListeners: false,
      logLevel: "warn",
      extra: {
        etcd: { endpoints: [ETCD_ENDPOINT], prefix },
        [block]: {
          backend: "redis",
          redis: {
            url: REDIS_URL,
            // Supplied as the FIELD, not inside the URL: that is the
            // documented way to keep the secret out of the config file,
            // and in `single` mode it used to reach nothing at all.
            password: "a-password-this-redis-does-not-want",
            timeout_secs: 5,
          },
        },
      },
    });
    await app.waitForExit(30_000);
    return app;
  }

  for (const block of ["ratelimit", "cache"] as const) {
    test(`${block}.redis: a refused password exits instead of falling open`, async (ctx) => {
      if (!ready) {
        ctx.skip();
        return;
      }
      const app = await bootRefused(block);
      const out = app.output();

      // It never served. A gateway that bound here is the bug: it would
      // run for the life of the process with the shared backend it was
      // configured with silently absent.
      expect(out).not.toContain("aisix listening");
      // It says which block, and it says the server refused. The
      // discriminator is the absence of "timed out": that is what this
      // reported before, and it sends an operator to look at the network
      // instead of at the password.
      expect(out).toContain(`${block}.redis`);
      expect(out.toLowerCase()).toContain("authentication");
      expect(out).not.toContain("timed out");
      // And never the URL, which carries the credential in
      // `redis://user:pass@host` form.
      expect(out).not.toContain("redis://");
    }, 60_000);
  }
});
