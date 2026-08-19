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

// E2E: a streaming response counts as in-flight for the whole time bytes
// may still flow, not just until the handler hands the response back.
//
// Axum polls a response body AFTER the middleware that produced it has
// returned, so a drain gate released at that point reads zero while an SSE
// stream is still running. The listener would then close as soon as the
// minimum window elapsed — under exactly the traffic shape an AI gateway
// carries most of, and for exactly the load balancer the window exists to
// outlast: one slower to withdraw the instance than the window is long.

const CALLER_PLAINTEXT = "sk-graceful-drain-sse-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const DRAIN_WINDOW_SECS = 3;
/** 10 events, one per second: the stream outlives the window several times over. */
const SSE_EVENT_COUNT = 10;
const SSE_EVENT_DELAY_MS = 1_000;

async function tcpAccepts(proxyUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${proxyUrl}/readyz`);
    await res.text();
    return true;
  } catch {
    return false;
  }
}

describe("graceful drain e2e: a live SSE stream holds the listener open", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      streamEvents: [
        ...Array.from({ length: SSE_EVENT_COUNT }, (_, i) =>
          JSON.stringify({
            id: `chunk-${i}`,
            object: "chat.completion.chunk",
            created: 1,
            model: "gpt-4o-mini",
            choices: [{ index: 0, delta: { content: `t${i}` }, finish_reason: null }],
          }),
        ),
        "[DONE]",
      ],
      eventDelayMs: SSE_EVENT_DELAY_MS,
    });
    app = await spawnApp({
      logLevel: "info",
      extra: { shutdown: { min_drain_secs: DRAIN_WINDOW_SECS } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "graceful-drain-sse-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "graceful-drain-sse",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["graceful-drain-sse"],
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
    "keeps accepting past the window while a stream is still producing",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      const proxyUrl = app.proxyUrl;

      const res = await fetch(`${proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "graceful-drain-sse",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
      });
      expect(res.status).toBe(200);

      // Read the first chunk so the response head is committed and the
      // handler has returned — the exact point a body-blind drain gate
      // would drop back to zero.
      const reader = res.body!.getReader();
      const first = await reader.read();
      expect(first.done).toBe(false);

      const signalledAt = Date.now();
      app.signal("SIGTERM");

      // Well past the window, with the stream still producing.
      await new Promise((r) => setTimeout(r, (DRAIN_WINDOW_SECS + 2) * 1000));
      expect(Date.now() - signalledAt).toBeGreaterThan(DRAIN_WINDOW_SECS * 1000);
      expect(await tcpAccepts(proxyUrl)).toBe(true);

      // Drain the rest, then the process may finish and exit on its own.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done } = await reader.read();
        if (done) break;
      }
      await app.waitForExit(20_000);
      expect(app.output()).toContain("drain complete");
    },
    90_000,
  );
});
