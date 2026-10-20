import { createServer, type IncomingMessage, type Server } from "node:http";
import { connect, type AddressInfo, type Socket } from "node:net";
import { afterEach, describe, expect, test } from "vitest";
import { spawnApp, waitForLogLine, type SpawnedApp } from "../harness/index.js";

// E2E: a request still being served when thread-per-core workers exit at
// the end of a drain.
//
// The drain ends the first moment nothing is in flight. A downstream
// connection that was accepted before that moment but has not yet sent its
// request (a pre-connected pool entry, a request still in transit) is
// served after it: its worker keeps running until that request is done,
// while every other worker exits as soon as its own connections have
// finished. Such a request must still be answered once, and reach the
// upstream once.
//
// The Provider Key below carries a connection override, so its upstream
// connections come from a pool that is not per-worker.

const CALLER_PLAINTEXT = "sk-tpc-shared-pool-drain";
const CALLER_KEY_ENV = "TPC_SHARED_POOL_CALLER_KEY";
const MODEL = "tpc-shared-pool";
const VENDOR_HOST = "tpc-shared-pool.invalid";

const WORKERS = 4;
const WARM_REQUESTS = 32;
const WARM_DELAY_MS = 300;
/** Fresh connections that send nothing, so every worker outlives the drain. */
const HOLDERS = 16;
/**
 * Late requests, each slower than the one before, so the workers serving
 * them exit one after another while the slower ones are still running.
 */
const LATE_DELAYS_MS = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000];
/** Fresh gateways to try before concluding the sequence was not hit. */
const ITERATIONS = 3;

interface Mock {
  server: Server;
  port: number;
  /** Marker → how many times the upstream received it. */
  received: Map<string, number>;
}

async function startMock(): Promise<Mock> {
  const received = new Map<string, number>();
  const server = createServer(async (req: IncomingMessage, res) => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    const marker: string = body.messages?.[0]?.content ?? "";
    received.set(marker, (received.get(marker) ?? 0) + 1);
    const delay = Number(/delay=(\d+)/.exec(marker)?.[1] ?? 0);
    await new Promise((r) => setTimeout(r, delay));
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        id: `cmpl-${marker}`,
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: marker }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );
  });
  server.keepAliveTimeout = 120_000;
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return { server, port: (server.address() as AddressInfo).port, received };
}

function resources(port: number): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: tpc-shared-pool-pk
    provider: openai
    api_key: sk-mock
    api_base: http://${VENDOR_HOST}:${port}/v1
    resolve_addresses: ["127.0.0.1"]
models:
  - display_name: ${MODEL}
    provider: openai
    model_name: gpt-4o-mini
    provider_key: tpc-shared-pool-pk
api_keys:
  - display_name: tpc-shared-pool-caller
    key_env: ${CALLER_KEY_ENV}
    allowed_models: ["${MODEL}"]
`;
}

function rawChat(marker: string): string {
  const body = JSON.stringify({ model: MODEL, messages: [{ role: "user", content: marker }] });
  return (
    "POST /v1/chat/completions HTTP/1.1\r\n" +
    "Host: 127.0.0.1\r\n" +
    `Authorization: Bearer ${CALLER_PLAINTEXT}\r\n` +
    "Content-Type: application/json\r\n" +
    `Content-Length: ${Buffer.byteLength(body)}\r\n` +
    "Connection: close\r\n\r\n" +
    body
  );
}

async function openSocket(port: number): Promise<Socket> {
  return new Promise((resolve, reject) => {
    const s = connect(port, "127.0.0.1", () => resolve(s));
    s.once("error", reject);
  });
}

/** Send `marker` on `socket` and resolve with the status line's code, or "no-response". */
function exchange(socket: Socket, marker: string): Promise<number | "no-response"> {
  return new Promise((resolve) => {
    let raw = "";
    socket.on("data", (d) => (raw += d.toString("utf8")));
    socket.on("error", () => {});
    socket.on("close", () => {
      const m = /^HTTP\/1\.1 (\d{3})/.exec(raw);
      resolve(m ? Number(m[1]) : "no-response");
    });
    socket.write(rawChat(marker));
  });
}

async function waitUntil(check: () => boolean, what: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 20));
  }
}

interface LateOutcome {
  marker: string;
  status: number | "no-response";
  upstreamCount: number;
}

async function runOnce(iteration: number, cleanup: Array<() => Promise<void>>): Promise<LateOutcome[]> {
  const mock = await startMock();
  cleanup.push(() => new Promise<void>((r) => mock.server.close(() => r())));
  const app: SpawnedApp = await spawnApp({
    // "drain complete" is reported at INFO; the suite default is WARN.
    logLevel: "info",
    threadPerCore: true,
    resourcesFile: resources(mock.port),
    extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT, AISIX_PROXY__WORKERS: String(WORKERS) },
    extra: { shutdown: { min_drain_secs: 1 } },
  });
  cleanup.push(() => app.exit());
  const proxyPort = Number(new URL(app.proxyUrl).port);

  // Fill the shared pool with upstream connections opened by every worker.
  const warm = await Promise.all(
    Array.from({ length: WARM_REQUESTS }, async (_, i) =>
      exchange(await openSocket(proxyPort), `it${iteration}-warm-${i}-delay=${WARM_DELAY_MS}`),
    ),
  );
  expect(warm.every((s) => s === 200), `warm-up statuses: ${warm.join(",")}`).toBe(true);

  const holders = await Promise.all(Array.from({ length: HOLDERS }, () => openSocket(proxyPort)));
  const lates = await Promise.all(LATE_DELAYS_MS.map(() => openSocket(proxyPort)));
  // Let every worker accept its share before the drain starts.
  await new Promise((r) => setTimeout(r, 300));

  app.signal("SIGTERM");
  await waitForLogLine(app, (l) => l.includes("drain complete"), "the drain to end", 10_000);

  const markers = LATE_DELAYS_MS.map((d, i) => `it${iteration}-late-${i}-delay=${d}`);
  const settled = new Set<number>();
  const pending = lates.map((s, i) =>
    exchange(s, markers[i]).then((st) => {
      settled.add(i);
      return st;
    }),
  );
  await waitUntil(
    () => markers.every((m, i) => mock.received.has(m) || settled.has(i)),
    "every late request to reach the upstream or be answered",
    5_000,
  );
  // The holders were all that kept the idle workers alive.
  for (const h of holders) h.destroy();

  const statuses = await Promise.all(pending);
  await app.waitForExit(20_000);
  return markers.map((marker, i) => ({
    marker,
    status: statuses[i],
    upstreamCount: mock.received.get(marker) ?? 0,
  }));
}

describe("thread-per-core drain: shared Provider Key pool", () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.reverse()) await fn().catch(() => {});
    cleanup.length = 0;
  });

  // The spec pins thread-per-core itself, so the work-stealing leg would
  // only run the same scenario a second time.
  test.skipIf(process.env.E2E_THREAD_PER_CORE === "false")(
    "a request served after the drain ends is answered once and sent upstream once",
    async () => {
      const all: LateOutcome[] = [];
      for (let it = 0; it < ITERATIONS; it++) {
        const outcomes = await runOnce(it, cleanup);
        all.push(...outcomes);
        for (const fn of cleanup.reverse()) await fn().catch(() => {});
        cleanup.length = 0;
        if (outcomes.some((o) => o.upstreamCount !== 1 || o.status !== 200)) break;
      }
      for (const o of all) {
        expect({ marker: o.marker, status: o.status, upstreamCount: o.upstreamCount }).toEqual({
          marker: o.marker,
          status: 200,
          upstreamCount: 1,
        });
      }
    },
    180_000,
  );
});
