import { createHash } from "node:crypto";
import { WebSocket as WsClient, WebSocketServer, type WebSocket as WsSocket } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the /v1/realtime close handshake (RFC 6455 §5.5.1) in both
// directions. A client that closes gets a Close back instead of an
// abnormal 1006; an upstream that closes reaches the client with its
// code and reason, preceded by an `error` event unless the close was
// normal, and with 1011 where the upstream's code may not be sent on.

const CALLER = "sk-realtime-close-e2e";

interface CloseUpstream {
  port: number;
  /** Close codes the upstream received from the gateway. */
  closesSeen: number[];
  close(): Promise<void>;
}

/** Mock Realtime upstream: greets with `session.created`, then obeys
 * `test.close` / `test.close-empty` / `test.close-raw` / `test.drop`
 * control events. */
async function startUpstream(): Promise<CloseUpstream> {
  const closesSeen: number[] = [];
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.send(JSON.stringify({ type: "session.created", session: {} }));
    socket.on("close", (code) => closesSeen.push(code));
    socket.on("message", (data) => {
      const ev = JSON.parse(data.toString()) as { type: string; code?: number; reason?: string };
      if (ev.type === "test.close") socket.close(ev.code, ev.reason);
      // A Close frame with no status code: reads as 1005, which is reserved.
      if (ev.type === "test.close-empty") socket.close();
      if (ev.type === "test.drop") socket.terminate();
      // A Close frame written byte by byte, for what `ws` refuses to send:
      // a code that may not be sent, or a control frame over 125 bytes.
      if (ev.type === "test.close-raw") {
        const payload = Buffer.concat([
          Buffer.from([ev.code! >> 8, ev.code! & 0xff]),
          Buffer.from(ev.reason ?? ""),
        ]);
        const header =
          payload.length < 126
            ? Buffer.from([0x88, payload.length])
            : Buffer.from([0x88, 126, payload.length >> 8, payload.length & 0xff]);
        (socket as unknown as { _socket: NodeJS.WritableStream })._socket.write(
          Buffer.concat([header, payload]),
        );
      }
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    closesSeen,
    close: () =>
      new Promise<void>((resolve) => {
        for (const c of wss.clients) c.terminate();
        wss.close(() => resolve());
      }),
  };
}

interface Session {
  events: Array<{ type: string; error?: { type?: string; message?: string } }>;
  closed: Promise<{ code: number; reason: string }>;
  ws: WsClient;
}

/** Open a session and resolve once the upstream's greeting has arrived. */
async function openSession(app: SpawnedApp): Promise<Session> {
  const ws = new WsClient(`${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=rt-close-model`, {
    headers: { authorization: `Bearer ${CALLER}` },
  });
  const events: Session["events"] = [];
  const closed = new Promise<{ code: number; reason: string }>((resolve) =>
    ws.on("close", (code, reason) => resolve({ code, reason: reason.toString() })),
  );
  await new Promise<void>((resolve, reject) => {
    ws.on("message", (data) => {
      const ev = JSON.parse(data.toString());
      events.push(ev);
      if (ev.type === "session.created") resolve();
    });
    ws.on("error", reject);
  });
  return { events, closed, ws };
}

describe("/v1/realtime close handshake", () => {
  let app: SpawnedApp | undefined;
  let upstream: CloseUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;
    upstream = await startUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["*"],
    });
    const pk = await seed.createProviderKey({
      display_name: "rt-close-pk",
      secret: "sk-upstream-realtime",
      api_base: `http://127.0.0.1:${upstream.port}/v1`,
    });
    await seed.createModel({
      display_name: "rt-close-model",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: pk.id,
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "rt-close-model");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const ready = (ctx: { skip: () => void }) => {
    if (!app || !upstream) {
      ctx.skip();
      return false;
    }
    return true;
  };

  test("a client's close is answered, and passed on upstream", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstream!.closesSeen.length;
    const s = await openSession(app!);
    s.ws.close(1000, "client done");
    expect(await s.closed).toEqual({ code: 1000, reason: "client done" });
    await expect.poll(() => upstream!.closesSeen.length).toBe(before + 1);
    expect(upstream!.closesSeen[before]).toBe(1000);
  });

  test("a normal upstream close reaches the client as is, with no error event", async (ctx) => {
    if (!ready(ctx)) return;
    const before = upstream!.closesSeen.length;
    const s = await openSession(app!);
    s.ws.send(JSON.stringify({ type: "test.close", code: 1000, reason: "session over" }));
    expect(await s.closed).toEqual({ code: 1000, reason: "session over" });
    expect(s.events.filter((e) => e.type === "error")).toEqual([]);
    // The upstream's close is answered too, not left to a reset (1006).
    await expect.poll(() => upstream!.closesSeen.length).toBe(before + 1);
    expect(upstream!.closesSeen[before]).toBe(1000);
  });

  test("an application close code reaches the client after an error event naming it", async (ctx) => {
    if (!ready(ctx)) return;
    const s = await openSession(app!);
    s.ws.send(JSON.stringify({ type: "test.close", code: 4001, reason: "quota exhausted" }));
    expect(await s.closed).toEqual({ code: 4001, reason: "quota exhausted" });
    const errors = s.events.filter((e) => e.type === "error");
    expect(errors).toHaveLength(1);
    expect(errors[0].error?.type).toBe("server_error");
    expect(errors[0].error?.message).toContain("4001");
    expect(errors[0].error?.message).toContain("quota exhausted");
  });

  /** Trigger an upstream end and return the close and the one error event's message. */
  async function upstreamEnds(control: object) {
    const s = await openSession(app!);
    s.ws.send(JSON.stringify(control));
    const closed = await s.closed;
    const errors = s.events.filter((e) => e.type === "error");
    expect(errors).toHaveLength(1);
    expect(errors[0].error?.type).toBe("server_error");
    return { closed, message: errors[0].error?.message ?? "" };
  }

  test("a close without a status code becomes 1011, without claiming a code", async (ctx) => {
    if (!ready(ctx)) return;
    const { closed, message } = await upstreamEnds({ type: "test.close-empty" });
    expect(closed).toEqual({ code: 1011, reason: "" });
    expect(message).toBe("upstream websocket closed without a status code");
  });

  test("an upstream that drops the connection becomes 1011, without claiming a code", async (ctx) => {
    if (!ready(ctx)) return;
    const { closed, message } = await upstreamEnds({ type: "test.drop" });
    expect(closed.code).toBe(1011);
    expect(message).toMatch(
      /^upstream websocket (connection failed: |connection ended without a close frame$)/,
    );
    expect(message).not.toMatch(/code \d/);
  });

  // The gateway's WebSocket library replaces such a Close with its own
  // 1002 "Protocol violation" and drops the upstream's code.
  test.for([1004, 2999])(
    "an upstream close with code %i, which may not be sent, becomes 1011",
    async (code, ctx) => {
      if (!ready(ctx)) return;
      const { closed, message } = await upstreamEnds({
        type: "test.close-raw",
        code,
        reason: `r${code}`,
      });
      expect(closed).toEqual({ code: 1011, reason: "" });
      expect(message).toBe(
        'upstream websocket sent a close frame the gateway cannot relay: an invalid close code, or 1002 "Protocol violation"',
      );
    },
  );

  test("an upstream 1002 with its own reason is relayed as received", async (ctx) => {
    if (!ready(ctx)) return;
    const { closed, message } = await upstreamEnds({
      type: "test.close",
      code: 1002,
      reason: "bad frame",
    });
    expect(closed).toEqual({ code: 1002, reason: "bad frame" });
    expect(message).toBe("upstream websocket closed with code 1002: bad frame");
  });

  test("an oversized Close frame is a protocol error: 1011, naming the failure", async (ctx) => {
    if (!ready(ctx)) return;
    const { closed, message } = await upstreamEnds({
      type: "test.close-raw",
      code: 4003,
      reason: "x".repeat(140),
    });
    expect(closed).toEqual({ code: 1011, reason: "upstream error" });
    expect(message).toMatch(/^upstream websocket connection failed: /);
    expect(message).not.toMatch(/code \d/);
  });
});
