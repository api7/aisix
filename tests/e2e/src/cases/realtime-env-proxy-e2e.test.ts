import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { connect as netConnect, createServer as createNetServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WebSocket as WsClient, WebSocketServer } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the Realtime upstream WebSocket honours the proxy environment the
// same way every other upstream call does. A `wss://` upstream is reached
// through an HTTP CONNECT tunnel to `HTTPS_PROXY`, with the proxy URL's
// credentials as `Proxy-Authorization` and the upstream TLS (under
// `upstream.tls`) run inside the tunnel; a host `NO_PROXY` covers is
// dialled directly.
//
// The proxied upstream's host name resolves nowhere: only the proxy knows
// where it lives, so a gateway that dials it directly fails the session.

const CALLER = "sk-realtime-env-proxy";
const PROXIED_HOST = "realtime-upstream.invalid";
const PROXY_USER = "rt-user";
const PROXY_PASS = "rt-pass";

interface Upstream {
  port: number;
  handshakes: string[];
  close(): Promise<void>;
}

/** Answers the first client event with a `response.done` frame and
 * records each handshake's request URL. */
function serveRealtime(wss: WebSocketServer, handshakes: string[]): void {
  wss.on("connection", (socket, req) => {
    handshakes.push(req.url ?? "");
    socket.on("message", () =>
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: { usage: { input_tokens: 3, output_tokens: 2 } },
        }),
      ),
    );
  });
}

function listening(server: Server): Promise<number> {
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      if (addr === null || typeof addr === "string") throw new Error("no port");
      resolve(addr.port);
    }),
  );
}

function closing(server: Server, wss: WebSocketServer): Promise<void> {
  for (const c of wss.clients) c.terminate();
  return new Promise((resolve) => {
    wss.close();
    server.closeAllConnections();
    server.close(() => resolve());
  });
}

interface KeyPair {
  key: Buffer;
  cert: Buffer;
}

/** A self-signed certificate for `PROXIED_HOST` and `localhost`, or
 * `undefined` without openssl. */
function selfSigned(): KeyPair | undefined {
  try {
    execFileSync("openssl", ["version"], { stdio: "ignore" });
  } catch {
    return undefined;
  }
  const dir = mkdtempSync(join(tmpdir(), "aisix-rt-proxy-tls-"));
  execFileSync(
    "openssl",
    [
      "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
      "-keyout", join(dir, "key.pem"), "-out", join(dir, "cert.pem"),
      "-subj", `/CN=${PROXIED_HOST}`,
      "-addext", `subjectAltName=DNS:${PROXIED_HOST},DNS:localhost`,
    ],
    { stdio: "ignore" },
  );
  return { key: readFileSync(join(dir, "key.pem")), cert: readFileSync(join(dir, "cert.pem")) };
}

/** A `wss://` Realtime upstream. */
async function startTlsUpstream(tls: KeyPair): Promise<Upstream> {
  const server = createHttpsServer(tls);
  const wss = new WebSocketServer({ server });
  const handshakes: string[] = [];
  serveRealtime(wss, handshakes);
  const port = await listening(server);
  return { port, handshakes, close: () => closing(server, wss) };
}

async function startPlainUpstream(): Promise<Upstream> {
  const server = createServer();
  const wss = new WebSocketServer({ server });
  const handshakes: string[] = [];
  serveRealtime(wss, handshakes);
  const port = await listening(server);
  return { port, handshakes, close: () => closing(server, wss) };
}

/** A CONNECT-only proxy that demands credentials and sends every tunnel
 * to loopback, whatever host it names; served over TLS when given a
 * certificate. */
async function startConnectProxy(tls?: KeyPair): Promise<{
  port: number;
  connects: { target: string; auth: string }[];
  close(): Promise<void>;
}> {
  const connects: { target: string; auth: string }[] = [];
  const refuse = (_req: unknown, res: { statusCode: number; end(): void }) => {
    res.statusCode = 405;
    res.end();
  };
  const server: Server = tls ? createHttpsServer(tls, refuse) : createServer(refuse);
  server.on("connect", (req, client, head) => {
    const auth = String(req.headers["proxy-authorization"] ?? "");
    connects.push({ target: req.url ?? "", auth });
    const expected = `Basic ${Buffer.from(`${PROXY_USER}:${PROXY_PASS}`).toString("base64")}`;
    if (auth !== expected) {
      client.end("HTTP/1.1 407 Proxy Authentication Required\r\n\r\n");
      return;
    }
    const port = Number((req.url ?? "").split(":").pop());
    const up = netConnect(port, "127.0.0.1", () => {
      client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
      up.write(head);
      up.pipe(client);
      client.pipe(up);
    });
    up.on("error", () => client.destroy());
    client.on("error", () => up.destroy());
  });
  const port = await listening(server);
  return {
    port,
    connects,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/** A listener that records how many connections reach it and never
 * answers: whatever speaks to it first, nothing gets through. */
async function startSink(): Promise<{ port: number; hits: () => number; close(): Promise<void> }> {
  let hits = 0;
  const server = createNetServer((sock) => {
    hits += 1;
    sock.destroy();
  });
  const port = await new Promise<number>((resolve) =>
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      if (addr === null || typeof addr === "string") throw new Error("no port");
      resolve(addr.port);
    }),
  );
  return { port, hits: () => hits, close: () => new Promise((r) => server.close(() => r())) };
}

/** Open a session on `model`, send one event, and return the first frame
 * the gateway relays back. */
function firstFrame(app: SpawnedApp, model: string): Promise<{ type: string }> {
  const url = `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${model}`;
  const c = new WsClient(url, { headers: { authorization: `Bearer ${CALLER}` } });
  return new Promise((resolve, reject) => {
    c.on("open", () => c.send(JSON.stringify({ type: "session.update" })));
    c.on("message", (d) => {
      resolve(JSON.parse(d.toString()));
      c.terminate();
    });
    c.on("unexpected-response", (_q, res) => reject(new Error(`upgrade refused: ${res.statusCode}`)));
    c.on("error", reject);
  });
}

describe("realtime upstream WebSocket follows the proxy environment", () => {
  let app: SpawnedApp | undefined;
  let tlsUpstream: Upstream | undefined;
  let plainUpstream: Upstream | undefined;
  let proxy: Awaited<ReturnType<typeof startConnectProxy>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    const tls = selfSigned();
    tlsUpstream = tls && (await startTlsUpstream(tls));
    plainUpstream = await startPlainUpstream();
    proxy = await startConnectProxy();

    const proxyUrl = `http://${PROXY_USER}:${PROXY_PASS}@127.0.0.1:${proxy.port}`;
    app = await spawnApp({
      extraEnv: {
        HTTPS_PROXY: proxyUrl,
        https_proxy: proxyUrl,
        HTTP_PROXY: proxyUrl,
        http_proxy: proxyUrl,
        NO_PROXY: "127.0.0.1,localhost",
        no_proxy: "127.0.0.1,localhost",
      },
      // The proxied upstream's certificate is self-signed.
      extra: { upstream: { tls: { verify: false } } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["*"],
    });
    if (tlsUpstream) {
      const pk = await seed.createProviderKey({
        display_name: "rt-proxy-tls-pk",
        secret: "sk-upstream-realtime",
        api_base: `https://${PROXIED_HOST}:${tlsUpstream.port}/v1`,
      });
      await seed.createModel({
        display_name: "rt-proxied",
        provider: "openai",
        model_name: "gpt-realtime-mock",
        provider_key_id: pk.id,
      });
    }
    const directPk = await seed.createProviderKey({
      display_name: "rt-proxy-direct-pk",
      secret: "sk-upstream-realtime",
      api_base: `http://127.0.0.1:${plainUpstream.port}/v1`,
    });
    await seed.createModel({
      display_name: "rt-direct",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: directPk.id,
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "rt-direct");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await tlsUpstream?.close();
    await plainUpstream?.close();
    await proxy?.close();
  });

  test("a wss:// upstream is tunnelled through HTTPS_PROXY with the proxy credentials", async (ctx) => {
    if (!etcdReachable || !app || !proxy) return ctx.skip();
    if (!tlsUpstream) return ctx.skip();
    const frame = await firstFrame(app, "rt-proxied");
    expect(frame.type).toBe("response.done");
    expect(proxy.connects).toContainEqual({
      target: `${PROXIED_HOST}:${tlsUpstream.port}`,
      auth: `Basic ${Buffer.from(`${PROXY_USER}:${PROXY_PASS}`).toString("base64")}`,
    });
    expect(tlsUpstream.handshakes).toHaveLength(1);
    expect(tlsUpstream.handshakes[0]).toContain("model=gpt-realtime-mock");
  });

  test("a host NO_PROXY covers is dialled directly", async (ctx) => {
    if (!etcdReachable || !app || !proxy || !plainUpstream) return ctx.skip();
    const frame = await firstFrame(app, "rt-direct");
    expect(frame.type).toBe("response.done");
    expect(plainUpstream.handshakes).toHaveLength(1);
    expect(proxy.connects.filter((c) => c.target.endsWith(`:${plainUpstream!.port}`))).toEqual([]);
  });
});

describe("realtime upstream WebSocket through an https:// proxy, and a SOCKS one", () => {
  let app: SpawnedApp | undefined;
  let tlsUpstream: Upstream | undefined;
  let plainUpstream: Upstream | undefined;
  let proxy: Awaited<ReturnType<typeof startConnectProxy>> | undefined;
  let socks: Awaited<ReturnType<typeof startSink>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    const tls = selfSigned();
    if (!tls) return;
    tlsUpstream = await startTlsUpstream(tls);
    plainUpstream = await startPlainUpstream();
    proxy = await startConnectProxy(tls);
    socks = await startSink();

    // `wss` takes HTTPS_PROXY (served over TLS); `ws` takes HTTP_PROXY,
    // here a SOCKS proxy, which the gateway cannot speak.
    const httpsProxy = `https://${PROXY_USER}:${PROXY_PASS}@localhost:${proxy.port}`;
    const socksProxy = `socks5://127.0.0.1:${socks.port}`;
    app = await spawnApp({
      extraEnv: {
        HTTPS_PROXY: httpsProxy,
        https_proxy: httpsProxy,
        HTTP_PROXY: socksProxy,
        http_proxy: socksProxy,
        NO_PROXY: "127.0.0.1",
        no_proxy: "127.0.0.1",
      },
      // Both the proxy's and the upstream's certificates are self-signed.
      extra: { upstream: { tls: { verify: false } } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["*"],
    });
    const tlsPk = await seed.createProviderKey({
      display_name: "rt-https-proxy-pk",
      secret: "sk-upstream-realtime",
      api_base: `https://${PROXIED_HOST}:${tlsUpstream.port}/v1`,
    });
    await seed.createModel({
      display_name: "rt-via-https-proxy",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: tlsPk.id,
    });
    // `localhost` is outside NO_PROXY, so this one is the SOCKS proxy's.
    const socksPk = await seed.createProviderKey({
      display_name: "rt-socks-pk",
      secret: "sk-upstream-realtime",
      api_base: `http://localhost:${plainUpstream.port}/v1`,
    });
    await seed.createModel({
      display_name: "rt-via-socks",
      provider: "openai",
      model_name: "gpt-realtime-mock",
      provider_key_id: socksPk.id,
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "rt-via-socks");
    });
  });

  afterAll(async () => {
    await app?.exit();
    await tlsUpstream?.close();
    await plainUpstream?.close();
    await proxy?.close();
    await socks?.close();
  });

  test("a wss:// upstream is tunnelled through a TLS-speaking proxy", async (ctx) => {
    if (!etcdReachable || !app || !proxy || !tlsUpstream) return ctx.skip();
    const frame = await firstFrame(app, "rt-via-https-proxy");
    expect(frame.type).toBe("response.done");
    expect(proxy.connects.map((c) => c.target)).toContain(`${PROXIED_HOST}:${tlsUpstream.port}`);
    expect(tlsUpstream.handshakes).toHaveLength(1);
  });

  test("a SOCKS proxy fails the session instead of falling back to a direct dial", async (ctx) => {
    if (!etcdReachable || !app || !socks || !plainUpstream) return ctx.skip();
    const frame = await firstFrame(app, "rt-via-socks");
    expect(frame.type).toBe("error");
    expect(plainUpstream.handshakes).toEqual([]);
    expect(socks.hits()).toBe(0);
  });
});
