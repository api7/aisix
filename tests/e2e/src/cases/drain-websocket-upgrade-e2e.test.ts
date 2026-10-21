import { randomBytes } from "node:crypto";
import { createServer } from "node:http";
import { connect, type AddressInfo, type Socket } from "node:net";
import { WebSocket as WsClient, WebSocketServer } from "ws";
import { afterEach, describe, expect, test } from "vitest";
import {
  metricDelta,
  scrapeMetrics,
  spawnApp,
  waitForLogLine,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a WebSocket upgrade (`/v1/realtime`) during a drain.
//
// Responses served while draining carry `Connection: close`, so a pooling
// client stops reusing the connection. A 101 is the exception: its
// `Connection: upgrade` is what completes the handshake (RFC 6455 §4.1),
// and once upgraded the connection is no longer HTTP, so there is nothing
// to retire. A client that checks the header — as RFC 6455 requires —
// fails the handshake if the gateway rewrites it.
//
// An open session is in flight for the drain like a streaming response:
// the drain does not end under it, and its usage is recorded when it
// closes.

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
    // One mock for both protocols: the Realtime WebSocket, which answers
    // a client event with a usage-bearing `response.done`, and a chat
    // completion that takes as long as its `delay=` marker asks.
    const http = createServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(c as Buffer);
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      const marker: string = body.messages?.[0]?.content ?? "";
      await new Promise((r) => setTimeout(r, Number(/delay=(\d+)/.exec(marker)?.[1] ?? 0)));
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "cmpl",
          object: "chat.completion",
          created: 1,
          model: "gpt-realtime-mock",
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      );
    });
    const upstream = new WebSocketServer({ server: http });
    upstream.on("connection", (socket) => {
      socket.on("message", () =>
        socket.send(
          JSON.stringify({
            type: "response.done",
            response: {
              usage: { input_tokens: 9, output_tokens: 4, input_token_details: { cached_tokens: 0 } },
            },
          }),
        ),
      );
    });
    await new Promise<void>((r) => http.listen(0, "127.0.0.1", () => r()));
    cleanup.push(
      () =>
        new Promise<void>((r) => {
          for (const c of upstream.clients) c.terminate();
          upstream.close();
          http.closeAllConnections();
          http.close(() => r());
        }),
    );
    const app = await spawnApp({
      // The drain phases are reported at INFO; the suite default is WARN.
      logLevel: "info",
      resourcesFile: resources((http.address() as AddressInfo).port),
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
      expect(summarise(head), head).toEqual({ status: "101", connection: "upgrade" });
      // The session that handshake opened is in flight, so the gateway
      // stays up for it although the drain has already ended.
      const exitedEarly = await app.waitForExit(3_000).then(
        () => true,
        () => false,
      );
      expect(exitedEarly, "the gateway exited under an open session").toBe(false);
      socket.destroy();
      await app.waitForExit(20_000);
    },
    60_000,
  );
  test(
    "an open session holds the drain, and its usage is recorded when it closes",
    async () => {
      const { app, port } = await start(1);
      const session = new WsClient(`ws://127.0.0.1:${port}/v1/realtime?model=${MODEL}`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await new Promise<void>((resolve, reject) => {
        session.once("open", () => resolve());
        session.once("error", reject);
      });
      const before = await scrapeMetrics(app.metricsUrl);

      app.signal("SIGTERM");
      // Well past the one-second window: a drain that ignored the session
      // would have ended by now.
      await new Promise((r) => setTimeout(r, 3_000));
      expect(app.output()).not.toContain("drain complete");

      // A chat request still in flight keeps the drain — and with it the
      // metrics listener — open after the session has closed, so the
      // session's usage can be read off the scrape.
      const held = fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}`, "content-type": "application/json" },
        body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "delay=3000" }] }),
      });
      await new Promise((r) => setTimeout(r, 300));

      const done = new Promise<void>((r) => session.once("message", () => r()));
      session.send(JSON.stringify({ type: "response.create" }));
      await done;
      session.close();
      await new Promise<void>((r) => session.once("close", () => r()));

      let recorded = 0;
      const deadline = Date.now() + 2_000;
      while (Date.now() < deadline) {
        const after = await scrapeMetrics(app.metricsUrl);
        recorded = metricDelta(
          before,
          after,
          "aisix_llm_input_tokens_total",
          (labels) => !Object.values(labels).includes("/v1/chat/completions"),
        );
        if (recorded > 0) break;
        await new Promise((r) => setTimeout(r, 50));
      }
      expect(recorded, "the session's input tokens on the scrape").toBe(9);

      expect((await held).status).toBe(200);
      await waitForLogLine(app, (l) => l.includes("drain complete"), "the drain to end", 10_000);
      await app.waitForExit(20_000);
    },
    60_000,
  );
});
