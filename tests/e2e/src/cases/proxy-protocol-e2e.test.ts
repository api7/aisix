import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect as tlsConnect } from "node:tls";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (AISIX-Cloud#1915): PROXY protocol v1 / v2 on the proxy listeners.
//
// A layer-4 balancer that rewrites the source address prepends a PROXY
// header naming the real client. The gateway detects it on its own, and
// honours it only from a socket peer inside `proxy.real_ip.trusted_proxies`.
// The e2e client is loopback, so trusting 127.0.0.1 makes it the balancer.
//
// Pinned contract:
//   - trusted peer + v2 / v1 header: the header's source is the client —
//     UsageEvent `client_source_ip`, the request span's `peer`, and what a
//     model's `allowed_cidrs` is checked against;
//   - the same on a TLS listener, header in front of the ClientHello;
//   - a trusted peer that sends no header is served as before;
//   - v2 LOCAL keeps the socket address;
//   - the header's source is the peer `x-forwarded-for` trust is judged
//     on: untrusted → XFF ignored, trusted → XFF walked;
//   - a header from a peer outside the list closes the connection;
//   - on a dual-stack `[::]` listener an IPv4 client is the IPv4 address
//     (not `::ffff:a.b.c.d`), so an IPv4 `trusted_proxies` entry matches
//     it and it is reported as plain IPv4.

const CALLER_PLAINTEXT = "sk-proxy-protocol-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const OPEN_MODEL = "pp-open-model";
const RESTRICTED_MODEL = "pp-restricted-model";
const LOGSTORE = "pp-usage";
const CREDENTIAL_REF = "ppsls";

const V2_SIGNATURE = Buffer.from("\r\n\r\n\0\r\nQUIT\n", "binary");

function ipv4Bytes(ip: string): number[] {
  return ip.split(".").map(Number);
}

/** A v2 PROXY (or LOCAL) header over TCP/IPv4, with one ignorable TLV. */
function v2Header(src: string, srcPort: number, command: "proxy" | "local" = "proxy"): Buffer {
  const block = Buffer.from([
    ...ipv4Bytes(src),
    ...ipv4Bytes("127.0.0.1"),
    srcPort >> 8,
    srcPort & 0xff,
    0x01,
    0xbb,
    // PP2_TYPE_NOOP TLV, 2 bytes of padding.
    0x04,
    0x00,
    0x02,
    0xaa,
    0xbb,
  ]);
  const head = Buffer.from([
    command === "proxy" ? 0x21 : 0x20,
    0x11,
    block.length >> 8,
    block.length & 0xff,
  ]);
  return Buffer.concat([V2_SIGNATURE, head, block]);
}

function v1Header(src: string, srcPort: number): Buffer {
  return Buffer.from(`PROXY TCP4 ${src} 127.0.0.1 ${srcPort} 443\r\n`, "ascii");
}

interface RawResponse {
  /** `null` when the gateway closed the connection without answering. */
  status: number | null;
  headers: Record<string, string>;
  body: string;
}

/**
 * One HTTP/1.1 request on a fresh connection that opens with `preamble`.
 * Written by hand because no HTTP client lets a test put bytes ahead of
 * the request (or ahead of the TLS ClientHello).
 */
async function rawChat(
  port: number,
  preamble: Buffer,
  opts: { model: string; marker: string; tls?: boolean; headers?: Record<string, string> },
): Promise<RawResponse> {
  const body = JSON.stringify({
    model: opts.model,
    messages: [{ role: "user", content: opts.marker }],
  });
  const extra = Object.entries(opts.headers ?? {})
    .map(([k, v]) => `${k}: ${v}\r\n`)
    .join("");
  const request =
    `POST /v1/chat/completions HTTP/1.1\r\n` +
    `host: 127.0.0.1:${port}\r\n` +
    `authorization: Bearer ${CALLER_PLAINTEXT}\r\n` +
    `content-type: application/json\r\n` +
    `content-length: ${Buffer.byteLength(body)}\r\n` +
    `connection: close\r\n` +
    extra +
    `\r\n` +
    body;

  const raw = await new Promise<Buffer>((resolve, reject) => {
    const tcp = connect(port, "127.0.0.1");
    const chunks: Buffer[] = [];
    let stream: Socket = tcp;
    const finish = () => resolve(Buffer.concat(chunks));
    const connectFailed = (e: Error) => reject(e);
    tcp.once("error", connectFailed);
    tcp.once("connect", () => {
      tcp.off("error", connectFailed);
      tcp.on("error", finish);
      if (preamble.length > 0) tcp.write(preamble);
      if (opts.tls) {
        stream = tlsConnect({ socket: tcp, rejectUnauthorized: false });
        stream.once("secureConnect", () => stream.write(request));
      } else {
        stream.write(request);
      }
      stream.on("data", (c: Buffer) => chunks.push(c));
      stream.once("end", finish);
      stream.once("close", finish);
      // A refused connection is a reset or a TLS failure, not a test error.
      stream.once("error", finish);
    });
    setTimeout(() => reject(new Error("raw request timed out")), 15_000).unref();
  });

  const text = raw.toString("utf8");
  const split = text.indexOf("\r\n\r\n");
  if (!text.startsWith("HTTP/1.1 ") || split < 0) {
    return { status: null, headers: {}, body: text };
  }
  const [statusLine, ...headerLines] = text.slice(0, split).split("\r\n");
  const headers: Record<string, string> = {};
  for (const line of headerLines) {
    const i = line.indexOf(":");
    headers[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
  }
  return { status: Number(statusLine.split(" ")[1]), headers, body: text.slice(split + 4) };
}

function portOf(url: string): number {
  return Number(new URL(url).port);
}

describe("PROXY protocol e2e (AISIX-Cloud#1915): the header's source is the client", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let dir: string | undefined;
  // Trusts the loopback balancer and 10.0.0.0/8 (for the XFF walk).
  let trusted: SpawnedApp | undefined;
  // Default `trusted_proxies`: nobody may send a header.
  let untrusted: SpawnedApp | undefined;
  // Dual-stack `[::]` listener trusting the IPv4 loopback.
  let dualStack: SpawnedApp | undefined;
  let plainPort = 0;
  let tlsPort = 0;

  const usageFor = (res: RawResponse) => {
    const requestId = res.headers["x-aisix-request-id"];
    expect(requestId, "x-aisix-request-id on the response").toBeTruthy();
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("request_id") === requestId,
      `the usage event for ${requestId}`,
    );
  };

  const accessLogFor = (app: SpawnedApp, requestId: string) =>
    waitForLogLine(
      app,
      (line) => line.includes("proxy request completed") && line.includes(`request_id="${requestId}"`),
      `the access-log line for ${requestId}`,
    );

  async function seed(app: SpawnedApp) {
    const s = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await s.createObservabilityExporter({
      name: "pp-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls!.url,
      project: "pp-project",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const pk = await s.createProviderKey({
      display_name: "pp-pk",
      secret: "sk-mock",
      api_base: `${upstream!.baseUrl}/v1`,
    });
    await s.createModel({
      display_name: OPEN_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await s.createModel({
      display_name: RESTRICTED_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      allowed_cidrs: ["203.0.113.0/24"],
    });
    await s.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [OPEN_MODEL, RESTRICTED_MODEL],
    });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });
  }

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    sls = await startMockSls();

    dir = await mkdtemp(join(tmpdir(), "aisix-pp-e2e-"));
    const certFile = join(dir, "server.crt");
    const keyFile = join(dir, "server.key");
    execFileSync("openssl", [
      "req", "-x509", "-newkey", "rsa:2048",
      "-keyout", keyFile, "-out", certFile,
      "-days", "1", "-nodes",
      "-subj", "/CN=127.0.0.1",
      "-addext", "subjectAltName=IP:127.0.0.1",
    ], { stdio: "ignore" });

    const slsEnv = {
      [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
      [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
    };
    trusted = await spawnApp({
      realIp: { trusted_proxies: ["127.0.0.1/32", "10.0.0.0/8"], recursive: true },
      proxyListeners: [{}, { tls: { cert_file: certFile, key_file: keyFile } }],
      logLevel: "info",
      extraEnv: slsEnv,
    });
    plainPort = portOf(trusted.proxyUrls[0]);
    tlsPort = portOf(trusted.proxyUrls[1]);
    untrusted = await spawnApp({ extraEnv: slsEnv });
    dualStack = await spawnApp({
      realIp: { trusted_proxies: ["127.0.0.1/32"], recursive: true },
      proxyListeners: [{ host: "::" }],
      extraEnv: slsEnv,
    });
    await seed(trusted);
    await seed(untrusted);
    await seed(dualStack);
  }, 90_000);

  afterAll(async () => {
    await trusted?.exit();
    await untrusted?.exit();
    await dualStack?.exit();
    await upstream?.close();
    await sls?.close();
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  test("a v2 header from a trusted peer names the client everywhere the peer is read", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const marker = "pp-v2-open";
    const res = await rawChat(plainPort, v2Header("203.0.113.7", 51234), {
      model: OPEN_MODEL,
      marker,
    });
    expect(res.status).toBe(200);

    const usage = await usageFor(res);
    expect(usage.get("client_source_ip")).toBe("203.0.113.7");

    const requestId = res.headers["x-aisix-request-id"];
    expect(requestId).toBeTruthy();
    const line = await accessLogFor(trusted!, requestId);
    expect(line).toMatch(/\bpeer=203\.0\.113\.7:51234\b/);
  });

  test("allowed_cidrs is checked against the header's source", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const inRange = await rawChat(plainPort, v2Header("203.0.113.7", 40000), {
      model: RESTRICTED_MODEL,
      marker: "pp-v2-restricted-in",
    });
    expect(inRange.status).toBe(200);

    const outOfRange = await rawChat(plainPort, v2Header("198.51.100.7", 40000), {
      model: RESTRICTED_MODEL,
      marker: "pp-v2-restricted-out",
    });
    expect(outOfRange.status).toBe(403);
    expect(outOfRange.body).toContain("ip_restricted");
  });

  test("a v1 header from a trusted peer names the client", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const marker = "pp-v1-restricted";
    const res = await rawChat(plainPort, v1Header("203.0.113.9", 40001), {
      model: RESTRICTED_MODEL,
      marker,
    });
    expect(res.status).toBe(200);
    const usage = await usageFor(res);
    expect(usage.get("client_source_ip")).toBe("203.0.113.9");
  });

  test("a v2 header ahead of the TLS ClientHello is honoured on a TLS listener", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const marker = "pp-v2-tls";
    const res = await rawChat(tlsPort, v2Header("203.0.113.11", 40002), {
      model: RESTRICTED_MODEL,
      marker,
      tls: true,
    });
    expect(res.status).toBe(200);
    const usage = await usageFor(res);
    expect(usage.get("client_source_ip")).toBe("203.0.113.11");
  });

  test("a trusted peer that sends no header is served with the socket address", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const marker = "pp-no-header";
    const res = await rawChat(plainPort, Buffer.alloc(0), { model: OPEN_MODEL, marker });
    expect(res.status).toBe(200);
    const usage = await usageFor(res);
    expect(usage.get("client_source_ip")).toBe("127.0.0.1");
  });

  test("a v2 LOCAL header keeps the socket address", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const marker = "pp-v2-local";
    const res = await rawChat(plainPort, v2Header("203.0.113.7", 40003, "local"), {
      model: OPEN_MODEL,
      marker,
    });
    expect(res.status).toBe(200);
    const usage = await usageFor(res);
    expect(usage.get("client_source_ip")).toBe("127.0.0.1");
  });

  test("x-forwarded-for is trusted only when the header's source is", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const ignored = "pp-xff-ignored";
    const a = await rawChat(plainPort, v2Header("203.0.113.7", 40004), {
      model: OPEN_MODEL,
      marker: ignored,
      headers: { "x-forwarded-for": "198.51.100.1" },
    });
    expect(a.status).toBe(200);
    expect((await usageFor(a)).get("client_source_ip")).toBe("203.0.113.7");

    const walked = "pp-xff-walked";
    const b = await rawChat(plainPort, v2Header("10.0.0.5", 40005), {
      model: OPEN_MODEL,
      marker: walked,
      headers: { "x-forwarded-for": "198.51.100.1" },
    });
    expect(b.status).toBe(200);
    expect((await usageFor(b)).get("client_source_ip")).toBe("198.51.100.1");
  });

  test("a header from a peer outside trusted_proxies closes the connection", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const port = portOf(untrusted!.proxyUrl);
    const refused = await rawChat(port, v2Header("203.0.113.7", 40006), {
      model: OPEN_MODEL,
      marker: "pp-untrusted",
    });
    expect(refused.status).toBeNull();
    await waitForLogLine(
      untrusted!,
      (line) =>
        line.includes("WARN") &&
        line.includes("PROXY protocol header from a peer outside proxy.real_ip.trusted_proxies") &&
        /\bpeer=127\.0\.0\.1:\d+\b/.test(line),
      "the untrusted-header warning",
    );

    // The same peer without a header is still served.
    const served = await rawChat(port, Buffer.alloc(0), {
      model: OPEN_MODEL,
      marker: "pp-untrusted-plain",
    });
    expect(served.status).toBe(200);
  });

  test("an IPv4 client on a dual-stack listener is matched and reported as IPv4", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const port = portOf(dualStack!.proxyUrl);
    const forwarded = await rawChat(port, Buffer.alloc(0), {
      model: OPEN_MODEL,
      marker: "pp-dual-xff",
      headers: { "x-forwarded-for": "203.0.113.21" },
    });
    expect(forwarded.status).toBe(200);
    expect((await usageFor(forwarded)).get("client_source_ip")).toBe("203.0.113.21");

    const direct = await rawChat(port, Buffer.alloc(0), {
      model: OPEN_MODEL,
      marker: "pp-dual-direct",
    });
    expect(direct.status).toBe(200);
    expect((await usageFor(direct)).get("client_source_ip")).toBe("127.0.0.1");
  });
});
