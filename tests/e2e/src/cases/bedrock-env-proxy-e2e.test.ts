import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, request as httpRequest, type IncomingMessage, type Server } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { connect as netConnect } from "node:net";
import { networkInterfaces, tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

// E2E: the Bedrock clients honour the proxy environment the way every other
// upstream client does.
//
// The gateway's reqwest clients route through `HTTP_PROXY` / `HTTPS_PROXY` /
// `ALL_PROXY` unless `NO_PROXY` covers the host. The AWS SDK stack Bedrock
// runs on — the runtime calls, the guardrail's ApplyGuardrail, and the STS
// calls that turn a `role_arn` into credentials — connected directly instead,
// past an egress proxy every other provider went through.
//
// The AWS client always reaches loopback directly (a local credential agent
// can be on loopback), so the mocks the proxy must carry a call to listen on
// this machine's own non-loopback address instead; the proxy's request log
// says which way each call went.

/** This machine's first non-internal IPv4 address. */
function ownAddress(): string | undefined {
  for (const nets of Object.values(networkInterfaces())) {
    for (const net of nets ?? []) {
      if (net.family === "IPv4" && !net.internal) return net.address;
    }
  }
  return undefined;
}
const OWN = ownAddress() ?? "127.0.0.1";

const CALLER = "sk-bedrock-env-proxy-e2e";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const ACCOUNT = "111122223333";
const MODEL = "meta.llama3-3-70b-instruct-v1:0";

type Reply = { status?: number; headers?: Record<string, string>; body: string };

/** A throwaway CA and a leaf for this machine's address, or `undefined` without
 * openssl. The AWS SDK stack cannot skip verification, so the gateway is
 * handed the CA through `upstream.tls.ca_file`. */
function makePki(): { caPath: string; key: Buffer; cert: Buffer } | undefined {
  try {
    execFileSync("openssl", ["version"], { stdio: "ignore" });
  } catch {
    return undefined;
  }
  const dir = mkdtempSync(join(tmpdir(), "aisix-bedrock-proxy-tls-"));
  const p = (name: string) => join(dir, name);
  const openssl = (args: string[]) => execFileSync("openssl", args, { stdio: "ignore" });
  openssl(["genrsa", "-out", p("ca.key"), "2048"]);
  openssl(["req", "-x509", "-new", "-key", p("ca.key"), "-out", p("ca.crt"), "-days", "1", "-sha256", "-subj", "/CN=aisix-e2e-proxy-ca"]);
  openssl(["genrsa", "-out", p("server.key"), "2048"]);
  openssl(["req", "-new", "-key", p("server.key"), "-out", p("server.csr"), "-subj", `/CN=${OWN}`]);
  writeFileSync(p("ext.cnf"), `subjectAltName=IP:${OWN}\n`, "utf8");
  openssl([
    "x509", "-req", "-in", p("server.csr"), "-CA", p("ca.crt"), "-CAkey", p("ca.key"),
    "-CAcreateserial", "-out", p("server.crt"), "-days", "1", "-sha256", "-extfile", p("ext.cnf"),
  ]);
  return { caPath: p("ca.crt"), key: readFileSync(p("server.key")), cert: readFileSync(p("server.crt")) };
}

async function listen(
  handler: (path: string, raw: string) => Reply,
  tls?: { key: Buffer; cert: Buffer },
  host = OWN,
) {
  const paths: string[] = [];
  const onRequest = (req: IncomingMessage, res: import("node:http").ServerResponse) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const path = decodeURIComponent(req.url ?? "");
      paths.push(path);
      const reply = handler(path, raw);
      res.writeHead(reply.status ?? 200, { "content-type": "application/json", ...(reply.headers ?? {}) });
      res.end(reply.body);
    });
  };
  const server: Server = tls ? createHttpsServer(tls, onRequest) : createServer(onRequest);
  await new Promise<void>((resolve) => server.listen(0, host, resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { server, port: addr.port, paths };
}

/** Relays absolute-form `http://` requests, tunnels `CONNECT`, and records
 * each request line. */
async function startForwardProxy() {
  const seen: string[] = [];
  const server: Server = createServer((req, res) => {
    seen.push(`${req.method} ${req.url}`);
    const target = new URL(req.url ?? "");
    const out = httpRequest(
      {
        host: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
        method: req.method,
        headers: req.headers,
      },
      (up) => {
        res.writeHead(up.statusCode ?? 502, up.headers);
        up.pipe(res);
      },
    );
    out.on("error", () => res.destroy());
    req.pipe(out);
  });
  server.on("connect", (req, client, head) => {
    seen.push(`CONNECT ${req.url}`);
    const [host, port] = (req.url ?? "").split(":");
    const up = netConnect(Number(port), host, () => {
      client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
      up.write(head);
      up.pipe(client);
      client.pipe(up);
    });
    up.on("error", () => client.destroy());
    client.on("error", () => up.destroy());
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { server, url: `http://127.0.0.1:${addr.port}`, seen };
}

const converse = (): Reply => ({
  body: JSON.stringify({
    output: { message: { role: "assistant", content: [{ text: "hello" }] } },
    stopReason: "end_turn",
    usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
    metrics: { latencyMs: 1 },
  }),
});

function sts(): Reply {
  const ns = 'xmlns="https://sts.amazonaws.com/doc/2011-06-15/"';
  return {
    headers: { "content-type": "text/xml" },
    body:
      `<AssumeRoleResponse ${ns}><AssumeRoleResult><Credentials>` +
      `<AccessKeyId>ASIA-proxied</AccessKeyId><SecretAccessKey>secret-proxied</SecretAccessKey>` +
      `<SessionToken>token-proxied</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration></Credentials>` +
      `<AssumedRoleUser><Arn>arn:aws:sts::${ACCOUNT}:assumed-role/Proxied/s</Arn><AssumedRoleId>AROA:s</AssumedRoleId></AssumedRoleUser>` +
      `</AssumeRoleResult><ResponseMetadata><RequestId>req-sts</RequestId></ResponseMetadata></AssumeRoleResponse>`,
  };
}

describe("Bedrock clients honour the proxy environment", () => {
  let app: SpawnedApp | undefined;
  let bedrock: Awaited<ReturnType<typeof listen>> | undefined;
  let loopbackBedrock: Awaited<ReturnType<typeof listen>> | undefined;
  let stsMock: Awaited<ReturnType<typeof listen>> | undefined;
  let guardrail: Awaited<ReturnType<typeof listen>> | undefined;
  let tlsBedrock: Awaited<ReturnType<typeof listen>> | undefined;
  let proxy: Awaited<ReturnType<typeof startForwardProxy>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    bedrock = await listen(converse);
    loopbackBedrock = await listen(converse, undefined, "127.0.0.1");
    stsMock = await listen(sts);
    guardrail = await listen(() => ({ body: JSON.stringify({ action: "NONE", outputs: [] }) }));
    proxy = await startForwardProxy();
    const pki = makePki();
    if (pki) tlsBedrock = await listen(converse, pki);

    app = await spawnApp({
      extraEnv: {
        HTTP_PROXY: proxy.url,
        HTTPS_PROXY: proxy.url,
        http_proxy: proxy.url,
        https_proxy: proxy.url,
        NO_PROXY: "127.0.0.1",
        no_proxy: "127.0.0.1",
        AWS_ACCESS_KEY_ID: "AKIDAMBIENTPROXY0001",
        AWS_SECRET_ACCESS_KEY: "ambient-secret",
        AWS_ENDPOINT_URL_STS: `http://${OWN}:${stsMock.port}`,
        AWS_EC2_METADATA_DISABLED: "true",
        AWS_CONFIG_FILE: "/nonexistent/aisix-e2e-aws-config",
        AWS_SHARED_CREDENTIALS_FILE: "/nonexistent/aisix-e2e-aws-credentials",
      },
      extra: {
        bedrock_endpoint_url: `http://${OWN}:${guardrail.port}`,
        ...(pki ? { upstream: { tls: { ca_file: pki.caPath } } } : {}),
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const keys: Array<[string, string, Record<string, unknown>]> = [
      [
        "proxied",
        `http://${OWN}:${bedrock.port}`,
        { access_key_id: "AKIDPROXIED000000001", secret_access_key: "s", region: "us-east-1" },
      ],
      [
        "direct",
        `http://127.0.0.1:${loopbackBedrock.port}`,
        { access_key_id: "AKIDDIRECT0000000001", secret_access_key: "s", region: "us-east-1" },
      ],
      [
        "role",
        `http://127.0.0.1:${loopbackBedrock.port}`,
        { auth_type: "default_chain", role_arn: `arn:aws:iam::${ACCOUNT}:role/Proxied`, region: "us-east-1" },
      ],
    ];
    if (tlsBedrock) {
      keys.push([
        "tunnel",
        `https://${OWN}:${tlsBedrock.port}`,
        { access_key_id: "AKIDTUNNEL0000000001", secret_access_key: "s", region: "us-east-1" },
      ]);
    }
    for (const [name, apiBase, secret] of keys) {
      const pk = await seed.createProviderKey({
        display_name: `envproxy-${name}`,
        provider: "bedrock",
        adapter: "bedrock",
        secret: JSON.stringify(secret),
        api_base: apiBase,
      });
      await seed.createModel({
        display_name: `envproxy-${name}`,
        provider: "bedrock",
        model_name: MODEL,
        provider_key_id: pk.id,
      });
    }
    // Fail-closed, so a guardrail call the proxy could not carry fails the
    // request. Seeded before the caller key, which the readiness gate waits on.
    await seed.createGuardrail({
      name: "envproxy-guardrail",
      enabled: true,
      hook_point: "input",
      fail_open: false,
      kind: "bedrock",
      guardrail_id: "envproxygr01",
      guardrail_version: "DRAFT",
      region: "us-east-1",
      aws_credentials: { kind: "static", access_key_id: "AKIDGUARDPROXY000001", secret_access_key: "s" },
      latency_mode: { kind: "serial" },
    });
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: keys.map(([name]) => `envproxy-${name}`),
    });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      const body = await r.text();
      return r.status === 200 && keys.every(([name]) => body.includes(`envproxy-${name}`));
    });
  });

  afterAll(async () => {
    await app?.exit();
    proxy?.server.closeAllConnections();
    for (const s of [bedrock?.server, loopbackBedrock?.server, stsMock?.server, guardrail?.server, tlsBedrock?.server, proxy?.server]) {
      if (s) await new Promise<void>((resolve) => s.close(() => resolve()));
    }
  });

  function skip(ctx: { skip: () => void }) {
    // Without a non-loopback address no mock can sit behind the proxy.
    if (!etcdReachable || !app || OWN === "127.0.0.1") {
      ctx.skip();
      return true;
    }
    return false;
  }

  async function chat(model: string) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
    });
    return { status: res.status, text: await res.text() };
  }

  test("a Bedrock runtime call goes through the proxy", async (ctx) => {
    if (skip(ctx)) return;
    const res = await chat("envproxy-proxied");
    expect(res.status, res.text).toBe(200);
    expect(JSON.parse(res.text).choices[0].message.content).toBe("hello");
    expect(proxy!.seen.some((l) => l.startsWith(`POST http://${OWN}:${bedrock!.port}/model/`))).toBe(true);
  });

  test("an https Bedrock endpoint is tunnelled through the proxy", async (ctx) => {
    if (skip(ctx)) return;
    if (!tlsBedrock) return ctx.skip();
    const res = await chat("envproxy-tunnel");
    expect(res.status, res.text).toBe(200);
    expect(proxy!.seen).toContain(`CONNECT ${OWN}:${tlsBedrock.port}`);
    // The SDK's HTTP/1 client sends the absolute-form target inside the
    // tunnel too; servers must accept it (RFC 9112 §3.2.2), and Bedrock does.
    expect(tlsBedrock.paths.some((p) => p.includes(`${OWN}:${tlsBedrock!.port}/model/`) || p.startsWith("/model/"))).toBe(true);
  });

  test("a loopback Bedrock endpoint is reached directly", async (ctx) => {
    if (skip(ctx)) return;
    const before = proxy!.seen.length;
    const res = await chat("envproxy-direct");
    expect(res.status, res.text).toBe(200);
    expect(proxy!.seen.slice(before).filter((l) => l.includes(`:${loopbackBedrock!.port}/`))).toEqual([]);
    expect(loopbackBedrock!.paths.length).toBeGreaterThan(0);
  });

  test("the STS call that assumes a role goes through the proxy", async (ctx) => {
    if (skip(ctx)) return;
    const res = await chat("envproxy-role");
    expect(res.status, res.text).toBe(200);
    expect(stsMock!.paths.length).toBeGreaterThan(0);
    expect(proxy!.seen).toContain(`POST http://${OWN}:${stsMock!.port}/`);
  });

  test("the Bedrock guardrail call goes through the proxy", async (ctx) => {
    if (skip(ctx)) return;
    const before = guardrail!.paths.length;
    const res = await chat("envproxy-direct");
    expect(res.status, res.text).toBe(200);
    expect(guardrail!.paths.length).toBeGreaterThan(before);
    expect(proxy!.seen.some((l) => l.startsWith(`POST http://${OWN}:${guardrail!.port}/guardrail/`))).toBe(true);
  });
});
