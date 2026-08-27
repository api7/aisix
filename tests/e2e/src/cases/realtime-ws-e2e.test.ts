import { createHash } from "node:crypto";
import { WebSocket } from "undici";
import {
  WebSocket as WsClient,
  WebSocketServer,
  type WebSocket as WsSocket,
} from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockIdp,
  waitConfigPropagation,
  type MockIdp,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

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
//   5. The `openai-beta: realtime=v1` opt-in is CLIENT-driven: forwarded
//      upstream only when the caller asked for it. Sending it
//      unconditionally made OpenAI's GA endpoint kill the session with
//      `beta_api_shape_disabled`.

const CALLER_PLAINTEXT = "sk-realtime-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface RealtimeUpstream {
  port: number;
  handshakes: { url: string; authorization: string; openaiBeta?: string }[];
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
      openaiBeta: req.headers["openai-beta"] as string | undefined,
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
  let idp: MockIdp | undefined;
  let otlp: MockOtlp | undefined;
  let restrictedKey: { id: string } | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // The access log emits at `info` (asserted by the #932 case below).
    app = await spawnApp({ logLevel: "info" });
    seed = new SeedClient(etcd, app.etcdPrefix);
    upstream = await startRealtimeUpstream();
    idp = await startMockIdp();
    otlp = await startMockOtlp();

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

    // JWT identity resolving (via a claim mapping) to a key that may NOT
    // use the realtime model — drives the post-auth refusal attribution
    // test (#932). The OTLP receiver is how that test reads the error
    // usage event.
    await seed.createObservabilityExporter({
      name: "rt-otlp",
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await seed.createOidcProvider({
      name: "rt-idp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });
    restrictedKey = await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-rt-restricted").digest("hex"),
      allowed_models: ["some-other-model"],
    });
    await seed.createClaimMapping({
      name: "rt-dept",
      jwt_provider: "rt-idp",
      match: [{ claim: "department", op: "exact", values: ["realtime"] }],
      resolve: { api_key_id: restrictedKey.id },
    });
    // Readiness probe seeded LAST: watch events apply in revision order,
    // so this key authenticating implies every earlier seed is live (same
    // pattern as claim-mapping-e2e).
    await seed.createApiKey({
      key_hash: createHash("sha256").update("sk-rt-ready-probe").digest("hex"),
      allowed_models: [],
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
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: "Bearer sk-rt-ready-probe" },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await idp?.close();
    await otlp?.close();
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
    // This client offered `openai-beta.realtime-v1`, so the opt-in is
    // forwarded upstream in the header form.
    expect(upstream.handshakes[0].openaiBeta).toBe("realtime=v1");

    ws.close();
  });

  test("a caller that did not opt in gets NO openai-beta header upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    // The regression: the gateway used to send `openai-beta: realtime=v1`
    // on every upstream dial. OpenAI's GA /v1/realtime answers that with
    // `beta_api_shape_disabled` and closes the session before the client
    // can send anything, so a plain GA caller must dial clean.
    const before = upstream.handshakes.length;
    const wsUrl = `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=realtime-e2e-model`;
    // `ws` sets headers: a server-side caller with no beta opt-in.
    const c = new WsClient(wsUrl, {
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
    });
    const relayed = await new Promise<string>((resolve, reject) => {
      c.on("open", () => c.send(JSON.stringify({ type: "session.update" })));
      c.on("message", (d) => resolve(d.toString()));
      c.on("unexpected-response", (_q, res) =>
        reject(new Error(`upgrade refused: ${res.statusCode}`)),
      );
      c.on("error", (e) => reject(e));
    });
    expect(JSON.parse(relayed).type).toBe("response.done");
    c.terminate();

    const hs = upstream.handshakes[before];
    expect(hs).toBeDefined();
    expect(hs.openaiBeta).toBeUndefined();
  });

  test("a plain http GET answers the envelope, not a bare rejection", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // No upgrade headers: pre-#885 this got axum's bare 400 with no
    // telemetry. It must now wear the endpoint's error envelope.
    const res = await fetch(`${app.proxyUrl}/v1/realtime`);
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error?: { type?: string } };
    expect(body.error?.type).toBe("websocket_upgrade_required");
  });

  test("a post-auth refusal attributes the caller and JWT identity (#932)", async (ctx) => {
    if (!etcdReachable || !app || !idp || !otlp || !restrictedKey) {
      ctx.skip();
      return;
    }
    // JWT auth resolves (through the `rt-dept` claim mapping) to a key
    // that may not use the realtime model, so auth succeeds and the
    // model ACL then refuses — the error usage event must carry the
    // resolved key and the JWT identity, not an anonymous shape.
    const token = idp.sign(
      agentClaims(idp.url, { sub: "rt-alice", department: "realtime" }),
    );
    const wsUrl = `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=realtime-e2e-model`;
    // The `ws` client can set headers (undici's browser-style client
    // cannot), which also makes this the header-auth coverage for the
    // endpoint — every other case rides the subprotocol flow.
    const refusal = await new Promise<{ status: number; requestId: string }>(
      (resolve, reject) => {
        const c = new WsClient(wsUrl, {
          headers: { authorization: `Bearer ${token}` },
        });
        c.on("unexpected-response", (_req, res) => {
          res.resume();
          resolve({
            status: res.statusCode ?? 0,
            requestId: String(res.headers["x-aisix-request-id"] ?? ""),
          });
          c.terminate();
        });
        c.on("open", () => {
          c.terminate();
          reject(new Error("handshake must fail on model ACL"));
        });
        c.on("error", (e) => reject(e));
      },
    );
    expect(refusal.status).toBe(403);
    expect(refusal.requestId).not.toBe("");

    // The error event fans out to OTLP exporters like every other usage
    // event; poll the mock for the span keyed on the handshake's id.
    let span;
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
      // Attempt carrier only — the AISIX-Cloud#1279 hierarchy also ships
      // structural spans sharing the request id without usage attributes.
      span = otlp.spans.find(
        (s) =>
          s.attributes["aisix.request_id"] === refusal.requestId &&
          s.attributes["aisix.attempt_index"] !== undefined,
      );
      if (span) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    if (!span) {
      throw new Error(
        `no usage span for request_id=${refusal.requestId} ` +
          `(spans=${otlp.spans.length}, parse failures=${otlp.parseFailures.length})`,
      );
    }
    expect(span.attributes["aisix.api_key_id"]).toBe(restrictedKey.id);
    expect(span.attributes["aisix.jwt_subject"]).toBe("rt-alice");
    expect(span.attributes["aisix.jwt_provider"]).toBe("rt-idp");
    expect(span.attributes["aisix.jwt_claim_mapping"]).toBe("rt-dept");

    // Third surface of the same fix: the refusal's access-log line must
    // name the caller (the realtime access log carried no api_key_id on
    // any path before #932).
    const logLine = app
      .output()
      .split("\n")
      .find(
        (l) =>
          l.includes("proxy request completed") &&
          l.includes(refusal.requestId),
      );
    expect(logLine, "access log line for the refusal").toBeTruthy();
    expect(logLine).toContain(restrictedKey.id);
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
