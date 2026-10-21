import { randomBytes } from "node:crypto";
import { connect, type AddressInfo, type Socket } from "node:net";
import { WebSocketServer } from "ws";
import { afterEach, describe, expect, test } from "vitest";
import { spawnApp, waitForLogLine, type SpawnedApp } from "../harness/index.js";

// E2E: a WebSocket upgrade (`/v1/realtime`) during a drain.
//
// Responses served while draining carry `Connection: close`, so a pooling
// client stops reusing the connection. A 101 is the exception: its
// `Connection: upgrade` is what completes the handshake (RFC 6455 §4.1),
// and once upgraded the connection is no longer HTTP, so there is nothing
// to retire. A client that checks the header — as RFC 6455 requires —
// fails the handshake if the gateway rewrites it.

const CALLER_PLAINTEXT = "sk-drain-ws-upgrade";
const CALLER_KEY_ENV = "DRAIN_WS_UPGRADE_CALLER_KEY";
const MODEL = "drain-ws-upgrade";

function resources(port: number): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: drain-ws-upgrade-pk
    provider: openai
    adapter: openai
    api_key: sk-mock
    api_base: http://127.0.0.1:${port}/v1
models:
  - display_name: ${MODEL}
    provider: openai
    model_name: gpt-realtime-mock
    provider_key: drain-ws-upgrade-pk
api_keys:
  - display_name: drain-ws-upgrade-caller
    key_env: ${CALLER_KEY_ENV}
    allowed_models: ["${MODEL}"]
`;
}

async function openSocket(port: number): Promise<Socket> {
  return new Promise((resolve, reject) => {
    const s = connect(port, "127.0.0.1", () => resolve(s));
    s.once("error", reject);
  });
}

/** Send a WebSocket handshake on `socket` and resolve with the response head. */
function handshake(socket: Socket): Promise<string> {
  return new Promise((resolve) => {
    let raw = "";
    const done = () => resolve(raw.split("\r\n\r\n")[0] ?? "");
    socket.on("data", (d) => {
      raw += d.toString("latin1");
      if (raw.includes("\r\n\r\n")) done();
    });
    socket.on("error", () => {});
    socket.on("close", done);
    socket.write(
      `GET /v1/realtime?model=${MODEL} HTTP/1.1\r\n` +
        "Host: 127.0.0.1\r\n" +
        `Authorization: Bearer ${CALLER_PLAINTEXT}\r\n` +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Key: ${randomBytes(16).toString("base64")}\r\n` +
        "Sec-WebSocket-Version: 13\r\n\r\n",
    );
  });
}

function summarise(head: string) {
  return {
    status: /^HTTP\/1\.1 (\d{3})/.exec(head)?.[1] ?? "no response",
    connection: /^connection: *(.*)$/im.exec(head)?.[1]?.trim().toLowerCase() ?? "absent",
  };
}

describe("drain: a WebSocket upgrade", () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const fn of cleanup.reverse()) await fn().catch(() => {});
    cleanup.length = 0;
  });

  async function start(minDrainSecs: number): Promise<{ app: SpawnedApp; port: number }> {
    const upstream = new WebSocketServer({ host: "127.0.0.1", port: 0 });
    await new Promise<void>((r) => upstream.on("listening", () => r()));
    cleanup.push(
      () =>
        new Promise<void>((r) => {
          for (const c of upstream.clients) c.terminate();
          upstream.close(() => r());
        }),
    );
    const app = await spawnApp({
      // The drain phases are reported at INFO; the suite default is WARN.
      logLevel: "info",
      resourcesFile: resources((upstream.address() as AddressInfo).port),
      extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT },
      extra: { shutdown: { min_drain_secs: minDrainSecs } },
    });
    cleanup.push(() => app.exit());
    return { app, port: Number(new URL(app.proxyUrl).port) };
  }

  test(
    "inside the drain window keeps Connection: upgrade on its 101",
    async () => {
      const { app, port } = await start(30);
      app.signal("SIGTERM");
      await waitForLogLine(app, (l) => l.includes("draining"), "the drain to start", 10_000);

      const socket = await openSocket(port);
      const head = await handshake(socket);
      socket.destroy();
      expect(summarise(head), head).toEqual({ status: "101", connection: "upgrade" });
    },
    60_000,
  );

  test(
    "on a connection accepted before the drain ended keeps Connection: upgrade on its 101",
    async () => {
      const { app, port } = await start(1);
      const socket = await openSocket(port);
      app.signal("SIGTERM");
      await waitForLogLine(app, (l) => l.includes("drain complete"), "the drain to end", 10_000);

      const head = await handshake(socket);
      socket.destroy();
      expect(summarise(head), head).toEqual({ status: "101", connection: "upgrade" });
      // The upgraded session was the last thing holding the gateway.
      await app.waitForExit(20_000);
    },
    60_000,
  );
});
