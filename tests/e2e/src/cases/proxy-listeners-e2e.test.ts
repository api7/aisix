import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, request } from "undici";
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

// E2E: `proxy.listeners` — one gateway answering HTTPS and plaintext
// HTTP at the same time (AISIX-Cloud#1662).
//
// Before it, `proxy.addr` + `proxy.tls` described the only listener
// there was, so configuring a certificate made the single port
// HTTPS-only and every plaintext client lost its way in.
//
// Pinned contract:
//   - a proxied chat completes on a TLS listener and on a plaintext one,
//     within the same gateway, against the same upstream;
//   - both listeners share one router and one configuration — the same
//     caller key and the same model work on either;
//   - with `proxy.listeners` set, `proxy.addr` is not bound, and the
//     gateway says so once at startup.

const PLAINTEXT_KEY = "sk-proxy-listeners-e2e";
const KEY_HASH = createHash("sha256").update(PLAINTEXT_KEY).digest("hex");
const MODEL = "proxy-listeners";

// The listener's certificate is generated per run and trusted by nobody.
const insecureAgent = new Agent({ connect: { rejectUnauthorized: false } });

describe("proxy.listeners e2e: HTTPS and plaintext HTTP side by side", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let dir: string | undefined;
  let etcdReachable = false;

  const chat = async (baseUrl: string, marker: string) => {
    const res = await request(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      dispatcher: insecureAgent,
      headers: {
        authorization: `Bearer ${PLAINTEXT_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "user", content: marker }],
      }),
    });
    const text = await res.body.text();
    return { status: res.statusCode, body: text ? JSON.parse(text) : null };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    dir = await mkdtemp(join(tmpdir(), "aisix-proxy-listeners-e2e-"));
    const certFile = join(dir, "server.crt");
    const keyFile = join(dir, "server.key");
    execFileSync("openssl", [
      "req", "-x509", "-newkey", "rsa:2048",
      "-keyout", keyFile, "-out", certFile,
      "-days", "1", "-nodes",
      "-subj", "/CN=127.0.0.1",
      "-addext", "subjectAltName=IP:127.0.0.1",
    ]);

    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      // TLS first, so the plaintext listener is not simply the first
      // thing bound — the order the gateway binds them in must not
      // decide which one works.
      proxyListeners: [{ tls: { cert_file: certFile, key_file: keyFile } }, {}],
      // The `proxy.addr` ignored notice is an INFO line.
      logLevel: "info",
    });

    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "proxy-listeners-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Seeded last, so its key authenticating implies the whole set has
    // reached the gateway.
    await seed.createApiKey({ key_hash: KEY_HASH, allowed_models: [MODEL] });
    await waitConfigPropagation(async () => {
      const res = await request(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${PLAINTEXT_KEY}` },
      });
      await res.body.dump();
      return res.statusCode === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    if (dir) await rm(dir, { recursive: true, force: true });
  });

  test("the TLS listener and the plaintext listener both proxy a chat", async (ctx) => {
    if (!etcdReachable || !app || !upstream) return ctx.skip();

    const [httpsUrl, httpUrl] = app.proxyUrls;
    expect(httpsUrl.startsWith("https://")).toBe(true);
    expect(httpUrl.startsWith("http://")).toBe(true);

    const before = upstream.receivedRequests.length;

    const overTls = await chat(httpsUrl, "over-tls");
    expect(overTls.status).toBe(200);
    expect((overTls.body as { model?: string }).model).toBeDefined();

    const overPlaintext = await chat(httpUrl, "over-plaintext");
    expect(overPlaintext.status).toBe(200);

    // Both requests reached the one upstream the one configuration
    // names: the listeners share a router and a snapshot, they do not
    // each carry their own.
    const seen = upstream.receivedRequests.slice(before);
    expect(seen.length).toBe(2);
  });

  test("a plaintext request to the TLS listener does not succeed", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const httpsUrl = app.proxyUrls[0];
    let ok = false;
    try {
      const res = await request(`${httpsUrl.replace("https://", "http://")}/livez`, {
        dispatcher: insecureAgent,
        headersTimeout: 2000,
        bodyTimeout: 2000,
      });
      await res.body.dump();
      ok = res.statusCode === 200;
    } catch {
      ok = false;
    }
    expect(ok).toBe(false);
  });

  test("proxy.addr is reported ignored, and nothing listens on it", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();

    const notice = app
      .output()
      .split("\n")
      .find((line) => line.includes("proxy.addr is ignored because proxy.listeners is set"));
    expect(notice, `no ignored-addr notice in:\n${app.output()}`).toBeDefined();

    // The notice names the address it is ignoring; nothing may answer there.
    const ignored = /addr=?"?(127\.0\.0\.1:\d+)/.exec(notice!);
    expect(ignored, `notice does not name the ignored address: ${notice}`).not.toBeNull();
    expect(app.proxyUrls.some((url) => url.endsWith(ignored![1]))).toBe(false);

    let reachable = false;
    try {
      const res = await request(`http://${ignored![1]}/livez`, {
        headersTimeout: 2000,
        bodyTimeout: 2000,
      });
      await res.body.dump();
      reachable = true;
    } catch {
      reachable = false;
    }
    expect(reachable).toBe(false);
  });
});
