import { createHash } from "node:crypto";
import { WebSocket } from "undici";
import { WebSocketServer, type WebSocket as WsSocket } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: /v1/realtime WebSocket relay (#721, AISIX-Cloud#873 §⑤) against a
// real `aisix` binary. Verifies with a live WS handshake what unit tests
// can't fully pin:
//
//   1. Browser-flow auth: the caller key rides the
//      `openai-insecure-api-key.<key>` subprotocol item (Node's native
//      WebSocket client can't set headers — exactly like a browser), and
//      the gateway echoes the `realtime` subprotocol.
//   2. Bidirectional frame relay: a client event reaches the mock
//      upstream verbatim; the upstream's `response.done` reaches the
//      client verbatim.
//   3. The upstream handshake carries the PROVIDER credential and the
//      UPSTREAM model id (`?model=gpt-realtime-mock`), not the caller's
//      key or the gateway alias.
//   4. Auth failure rejects the HTTP upgrade (native client fires
//      an error/close, never `open`).

const CALLER_PLAINTEXT = "sk-realtime-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface RealtimeUpstream {
  port: number;
  handshakes: { url: string; authorization: string }[];
  frames: string[];
  close(): Promise<void>;
}

/** Mock OpenAI Realtime upstream: records the handshake, then answers the
 * first client event with a usage-bearing `response.done` frame. */
async function startRealtimeUpstream(): Promise<RealtimeUpstream> {
  const handshakes: RealtimeUpstream["handshakes"] = [];
  const frames: string[] = [];
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket, req) => {
    handshakes.push({
      url: req.url ?? "",
      authorization: (req.headers.authorization as string) ?? "",
    });
    socket.on("message", (data) => {
      frames.push(data.toString());
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: {
            usage: {
              input_tokens: 9,
              output_tokens: 4,
              input_token_details: { cached_tokens: 0 },
            },
          },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    handshakes,
    frames,
    close: () =>
      new Promise<void>((resolve, reject) =>
        wss.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

describe("realtime e2e: /v1/realtime WebSocket relay (#721)", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: RealtimeUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    upstream = await startRealtimeUpstream();

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    const pk = await seed.createProviderKey({
      display_name: "realtime-e2e-pk",
      secret: "sk-upstream-realtime",
      api_base: `http://127.0.0.1:${upstream.port}/v1`,
    });
    await seed.createModel({
      display_name: "realtime-e2e-model",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: pk.id,
    });

    // Gate on the DP snapshot via /v1/models — the WS upgrade below
    // authenticates against the same snapshot, and a handshake fired
    // before the caller key propagates is rejected outright.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "realtime-e2e-model");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("browser-flow subprotocol auth + bidirectional relay + upstream credential swap", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const wsUrl = `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=realtime-e2e-model`;
    // undici's browser-style WebSocket — cannot set headers, exactly the
    // browser constraint the subprotocol flow exists for (Node 20 CI has
    // no global WebSocket, so import it from undici explicitly).
    const ws = new WebSocket(wsUrl, [
      "realtime",
      `openai-insecure-api-key.${CALLER_PLAINTEXT}`,
      "openai-beta.realtime-v1",
    ]);

    const opened = new Promise<void>((resolve, reject) => {
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener("error", () => reject(new Error("handshake failed")), {
        once: true,
      });
    });
    await opened;
    expect(ws.protocol).toBe("realtime");

    const done = new Promise<string>((resolve) => {
      ws.addEventListener("message", (ev) => resolve(String(ev.data)), {
        once: true,
      });
    });
    ws.send(
      JSON.stringify({ type: "session.update", session: { instructions: "hi" } }),
    );
    const frame = JSON.parse(await done) as {
      type: string;
      response: { usage: { input_tokens: number } };
    };
    expect(frame.type).toBe("response.done");
    expect(frame.response.usage.input_tokens).toBe(9);

    // Upstream saw the relayed event, the provider credential, and the
    // upstream model id.
    expect(upstream.frames.some((f) => f.includes("session.update"))).toBe(true);
    expect(upstream.handshakes.length).toBe(1);
    expect(upstream.handshakes[0].authorization).toBe(
      "Bearer sk-upstream-realtime",
    );
    expect(upstream.handshakes[0].url).toContain("model=gpt-realtime-mock");
    expect(upstream.handshakes[0].url).not.toContain(CALLER_PLAINTEXT);

    ws.close();
  });

  test("bad credentials reject the upgrade handshake", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const wsUrl = `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=realtime-e2e-model`;
    const ws = new WebSocket(wsUrl, [
      "realtime",
      "openai-insecure-api-key.sk-wrong",
    ]);
    const failed = await new Promise<boolean>((resolve) => {
      ws.addEventListener("open", () => resolve(false), { once: true });
      ws.addEventListener("error", () => resolve(true), { once: true });
    });
    expect(failed).toBe(true);
  });
});
