import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: what happens between SIGTERM and the process exiting.
//
// A load balancer learns that a replica is going away by polling its
// health check, so it keeps routing new connections for one check
// interval after `/readyz` starts answering 503. Closing the listener at
// signal time refuses every connection routed inside that window — which
// is how a rolling update surfaces at the caller as 502/503 even though
// nothing is actually broken.
//
// So the gateway keeps serving after the signal: `/readyz` flips
// immediately, new connections are still accepted for at least
// `shutdown.min_drain_secs`, and the listener only closes once nothing is
// left in flight. Responses carry `Connection: close` throughout, so a
// pooling client retires its connections as it uses them rather than
// holding idle ones open until the listener disappears.

const CALLER_PLAINTEXT = "sk-graceful-drain-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const DRAIN_WINDOW_SECS = 5;
/** Longer than the drain window, so it is still running when it elapses. */
const SLOW_UPSTREAM_MS = 9_000;

function chatBody(): string {
  return JSON.stringify({
    model: "graceful-drain",
    messages: [{ role: "user", content: "hi" }],
  });
}

async function chat(proxyUrl: string): Promise<Response> {
  return fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: chatBody(),
  });
}

async function readyzStatus(proxyUrl: string): Promise<number | "refused"> {
  try {
    const res = await fetch(`${proxyUrl}/readyz`);
    await res.text();
    return res.status;
  } catch {
    return "refused";
  }
}

/** Poll until `check` holds, or fail with `what`. */
async function waitUntil(
  check: () => boolean,
  what: string,
  timeoutMs = 10_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (check()) return;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(`timed out waiting for ${what}`);
}

/** Poll until `/readyz` reports the expected status, or time out. */
async function waitReadyz(
  proxyUrl: string,
  want: number | "refused",
  timeoutMs = 5_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let last: number | "refused" = "refused";
  while (Date.now() < deadline) {
    last = await readyzStatus(proxyUrl);
    if (last === want) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`/readyz never reported ${want} (last: ${last})`);
}

describe("graceful drain e2e: SIGTERM stops readiness, not service", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Request 1 outlives the drain window; request 2 answers at once.
    // `/v1/models` (the propagation gate) never reaches the upstream, so
    // the two chat calls below are requests 1 and 2 in arrival order.
    upstream = await startOpenAiUpstream({
      scriptedResponses: [{ responseDelayMs: SLOW_UPSTREAM_MS }, {}],
    });
    app = await spawnApp({
      // The drain phases are reported at INFO; the suite default is WARN.
      logLevel: "info",
      extra: { shutdown: { min_drain_secs: DRAIN_WINDOW_SECS } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "graceful-drain-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "graceful-drain",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["graceful-drain"],
    });

    // Gate on the caller key authenticating — seeded last, so it implies
    // the whole seed set is in the snapshot.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "keeps serving through the drain window, then exits once nothing is in flight",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      const proxyUrl = app.proxyUrl;

      expect(await readyzStatus(proxyUrl)).toBe(200);

      // A request that outlives the drain window, started before the
      // signal. It must complete rather than be cut off, and it must
      // hold the listener open past the window.
      const inFlight = chat(proxyUrl);
      // Gate on the upstream having actually received it, so the signal
      // below lands with the request genuinely in flight rather than after
      // a sleep that only usually wins the race.
      await waitUntil(
        () => upstream!.receivedRequests.length > 0,
        "the slow request never reached the upstream",
      );

      const signalledAt = Date.now();
      app.signal("SIGTERM");

      // Readiness withdraws immediately — this is what the balancer polls.
      await waitReadyz(proxyUrl, 503, 3_000);

      // …but the listener is still accepting. Probe well past the point
      // where it used to close — about a second after the signal — and
      // still comfortably inside the window, which is the interval a
      // balancer that has not yet re-checked would route into. Serve a
      // fast response so this assertion is about acceptance, not latency.
      await new Promise((r) => setTimeout(r, 2_500));
      expect(Date.now() - signalledAt).toBeLessThan(DRAIN_WINDOW_SECS * 1000);
      const duringWindow = await chat(proxyUrl);
      expect(duringWindow.status).toBe(200);
      await duringWindow.text();
      // A pooling client must retire the connection instead of holding it
      // idle until the listener goes away.
      expect(duringWindow.headers.get("connection")).toBe("close");

      // The window is a minimum, not a deadline: the slow request is
      // still running when it elapses, so the listener stays open for it.
      const slow = await inFlight;
      expect(slow.status).toBe(200);
      await slow.text();
      expect(Date.now() - signalledAt).toBeGreaterThan(DRAIN_WINDOW_SECS * 1000);

      // Nothing in flight now — the process closes the listener and exits
      // on its own. No SIGKILL, so the clean-shutdown path runs.
      await app.waitForExit(15_000);
      expect(await readyzStatus(proxyUrl)).toBe("refused");
      expect(app.output()).toContain("drain complete");
    },
    60_000,
  );
});
