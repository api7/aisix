import { createHash } from "node:crypto";
import http2 from "node:http2";
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

// E2E: the retirement signal an HTTP/2 downstream gets during a drain.
//
// `graceful-drain-e2e` covers the HTTP/1.1 half, where retirement is a
// response header: a client that pools connections sees `Connection:
// close` on whatever it is served during the window and retires the
// connection as it uses it. HTTP/2 forbids that header (RFC 9113 §8.2.2)
// and has no header in its place — its retirement signal is GOAWAY, a
// connection-level frame.
//
// So the gateway sends one when the drain STARTS, and that is the whole
// of what this spec pins:
//
//   - the frame arrives at SIGTERM, not when the listener finally closes
//     — otherwise an h2 downstream keeps dispatching onto the connection
//     for the entire grace period and the drain never converges;
//   - GOAWAY does not cut off the streams already running on it;
//   - and it does not take the listener with it, because a balancer is
//     still routing new connections here until its next health check.
//
// Node's `http2.connect()` over `http://` speaks h2c with prior
// knowledge, which is how a fronting proxy configured for HTTP/2
// upstreams reaches the gateway — no TLS, no ALPN.

const CALLER_PLAINTEXT = "sk-drain-h2-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const DRAIN_WINDOW_SECS = 5;
/** Longer than the drain window, so it is still running when GOAWAY lands. */
const SLOW_UPSTREAM_MS = 9_000;

interface H2Response {
  status: number;
  body: string;
}

function chatOverH2(session: http2.ClientHttp2Session): Promise<H2Response> {
  return new Promise((resolve, reject) => {
    const stream = session.request({
      ":method": "POST",
      ":path": "/v1/chat/completions",
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    });
    let status = 0;
    const chunks: Buffer[] = [];
    stream.on("response", (headers) => {
      status = Number(headers[":status"] ?? 0);
    });
    stream.on("data", (chunk: Buffer) => chunks.push(chunk));
    stream.on("end", () => resolve({ status, body: Buffer.concat(chunks).toString() }));
    stream.on("error", reject);
    stream.end(
      JSON.stringify({ model: "drain-h2", messages: [{ role: "user", content: "hi" }] }),
    );
  });
}

/** Resolves with the GOAWAY's last-stream-id, or rejects on timeout. */
function goaway(session: http2.ClientHttp2Session, timeoutMs: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`no GOAWAY within ${timeoutMs}ms`)),
      timeoutMs,
    );
    session.once("goaway", (_code, lastStreamId) => {
      clearTimeout(timer);
      resolve(lastStreamId);
    });
  });
}

async function waitUntil(check: () => boolean, what: string, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (check()) return;
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(`timed out waiting for ${what}`);
}

describe("graceful drain e2e: an HTTP/2 downstream is retired with GOAWAY", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Request 1 answers at once; request 2 outlives the drain window.
    // `/v1/models` (the propagation gate) never reaches the upstream.
    upstream = await startOpenAiUpstream({
      scriptedResponses: [{}, { responseDelayMs: SLOW_UPSTREAM_MS }],
    });
    app = await spawnApp({
      logLevel: "info",
      extra: { shutdown: { min_drain_secs: DRAIN_WINDOW_SECS } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "drain-h2-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "drain-h2",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["drain-h2"],
    });

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
    "GOAWAY lands at SIGTERM, in-flight streams finish, and the listener stays open",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      const proxyUrl = app.proxyUrl;

      const session = http2.connect(proxyUrl);
      await new Promise<void>((resolve, reject) => {
        session.once("connect", () => resolve());
        session.once("error", reject);
      });

      // The gateway really is serving this over h2c — otherwise the rest
      // of the spec would pass by testing nothing.
      const first = await chatOverH2(session);
      expect(first.status).toBe(200);

      // A stream that outlives the drain window, started before the
      // signal, so GOAWAY has something to not cut off.
      const inFlight = chatOverH2(session);
      await waitUntil(
        () => upstream!.receivedRequests.length > 1,
        "the slow request never reached the upstream",
      );

      const goawayArrives = goaway(session, 5_000);
      const signalledAt = Date.now();
      app.signal("SIGTERM");

      // The frame has to arrive at the START of the drain. The slow
      // stream above is still running, so the listener cannot close for
      // several more seconds — a GOAWAY sent with the listener would
      // miss this deadline by design.
      await goawayArrives;
      expect(Date.now() - signalledAt).toBeLessThan(SLOW_UPSTREAM_MS);

      // The listener is not going anywhere: a balancer that has not yet
      // re-checked `/readyz` is still routing new connections here, and
      // an h2c one still negotiates.
      const during = http2.connect(proxyUrl);
      await new Promise<void>((resolve, reject) => {
        during.once("connect", () => resolve());
        during.once("error", reject);
      });
      during.close();
      // Answered rather than refused is the point; 503 is what `/readyz`
      // has been reporting since the signal.
      const probe = await fetch(`${proxyUrl}/readyz`);
      expect(probe.status).toBe(503);
      await probe.text();

      // GOAWAY asks the peer to stop opening streams; it does not cut off
      // the ones already running.
      const slow = await inFlight;
      expect(slow.status).toBe(200);
      expect(Date.now() - signalledAt).toBeGreaterThan(DRAIN_WINDOW_SECS * 1000);

      session.close();
      await app.waitForExit(15_000);
      expect(app.output()).toContain("drain complete");
    },
    60_000,
  );
});
