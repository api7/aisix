import { execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { connect, type AddressInfo, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect as tlsConnect } from "node:tls";
import { afterEach, describe, expect, test } from "vitest";
import { spawnApp, waitForLogLine, type SpawnedApp } from "../harness/index.js";

// E2E: a connection accepted during the drain whose request only arrives
// after the drain has ended.
//
// The drain ends the first moment nothing is in flight, and a connection
// that has been accepted but has not sent its request yet carries nothing
// in flight: a client's pre-connected pool entry, or a request still in
// transit when the balancer routed it here. The gateway does not cut a
// connection during a drain, so that request must still be served and
// answered — with `Connection: close`, like every response during the
// drain — rather than the connection closing with no response at all.
// That holds on a TLS listener too, where the handshake has completed
// but not a byte of the request has been read when the drain ends.

const CALLER_PLAINTEXT = "sk-drain-late-request";
const CALLER_KEY_ENV = "DRAIN_LATE_REQUEST_CALLER_KEY";
const MODEL = "drain-late-request";

/** Enough connections that a per-connection coin flip cannot pass by luck. */
const LATE_CONNECTIONS = 8;

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
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return { server, port: (server.address() as AddressInfo).port, received };
}

function resources(port: number): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: drain-late-request-pk
    provider: openai
    api_key: sk-mock
    api_base: http://127.0.0.1:${port}/v1
models:
  - display_name: ${MODEL}
    provider: openai
    model_name: gpt-4o-mini
    provider_key: drain-late-request-pk
api_keys:
  - display_name: drain-late-request-caller
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
    `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n` +
    body
  );
}

async function openSocket(port: number): Promise<Socket> {
  return new Promise((resolve, reject) => {
    const s = connect(port, "127.0.0.1", () => resolve(s));
    s.once("error", reject);
  });
}

/** A TLS connection whose handshake has completed, with nothing sent on it. */
async function openTlsSocket(port: number): Promise<Socket> {
  return new Promise((resolve, reject) => {
    const s = tlsConnect(
      { host: "127.0.0.1", port, rejectUnauthorized: false, ALPNProtocols: ["http/1.1"] },
      () => resolve(s),
    );
    s.once("error", reject);
  });
}

/** Send `marker` on `socket` and resolve with whatever came back before the server closed it. */
function exchange(socket: Socket, marker: string): Promise<string> {
  return new Promise((resolve) => {
    let raw = "";
    socket.on("data", (d) => (raw += d.toString("utf8")));
    socket.on("error", () => {});
    socket.on("close", () => resolve(raw));
    socket.write(rawChat(marker));
  });
}

describe("drain: a request that arrives after the drain ended", () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.reverse()) await fn().catch(() => {});
    cleanup.length = 0;
  });

  async function run(tls: boolean): Promise<void> {
    const mock = await startMock();
    cleanup.push(() => new Promise<void>((r) => mock.server.close(() => r())));
    let proxyListeners: Array<{ tls?: { cert_file: string; key_file: string } }> | undefined;
    if (tls) {
      const dir = await mkdtemp(join(tmpdir(), "aisix-drain-late-request-"));
      cleanup.push(() => rm(dir, { recursive: true, force: true }));
      const certFile = join(dir, "server.crt");
      const keyFile = join(dir, "server.key");
      execFileSync("openssl", [
        "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", keyFile, "-out", certFile,
        "-days", "1", "-nodes",
        "-subj", "/CN=127.0.0.1",
        "-addext", "subjectAltName=IP:127.0.0.1",
      ]);
      // The harness needs a plaintext listener for its own readiness check.
      proxyListeners = [{}, { tls: { cert_file: certFile, key_file: keyFile } }];
    }
    const app: SpawnedApp = await spawnApp({
      // "drain complete" is reported at INFO; the suite default is WARN.
      logLevel: "info",
      resourcesFile: resources(mock.port),
      extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT },
      extra: { shutdown: { min_drain_secs: 1 } },
      proxyListeners,
    });
    cleanup.push(() => app.exit());
    const target = new URL(tls ? app.proxyUrls[1] : app.proxyUrl);
    expect(target.protocol).toBe(tls ? "https:" : "http:");
    const port = Number(target.port);

    // Connected well before the drain ends — the window alone is a
    // second — and silent until it has.
    const sockets = await Promise.all(
      Array.from({ length: LATE_CONNECTIONS }, () => (tls ? openTlsSocket(port) : openSocket(port))),
    );
    app.signal("SIGTERM");
    await waitForLogLine(app, (l) => l.includes("drain complete"), "the drain to end", 10_000);

    const markers = sockets.map((_, i) => `late-${tls ? "tls" : "plain"}-${i}`);
    const answers = await Promise.all(sockets.map((s, i) => exchange(s, markers[i])));

    for (const [i, raw] of answers.entries()) {
      const head = raw.split("\r\n\r\n")[0] ?? "";
      expect(
        {
          marker: markers[i],
          status: /^HTTP\/1\.1 (\d{3})/.exec(head)?.[1] ?? "no response",
          connectionClose: /^connection: close$/im.test(head),
          upstreamCount: mock.received.get(markers[i]) ?? 0,
        },
        raw,
      ).toEqual({ marker: markers[i], status: "200", connectionClose: true, upstreamCount: 1 });
    }

    // Every connection is answered and closed, so nothing holds the
    // gateway any longer.
    await app.waitForExit(20_000);
  }

  test("is served and answered with Connection: close, not dropped unanswered", () => run(false), 60_000);

  test(
    "is served on a TLS connection that completed its handshake but sent nothing before the drain ended",
    () => run(true),
    60_000,
  );
});
