import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { connect as netConnect } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WebSocket as WsClient, WebSocketServer } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an `azure-openai` Provider Key whose `api_base` is the resource
// endpoint with the `/openai/v1` path the Azure portal shows pasted on —
// `https://<resource>.openai.azure.com/openai/v1` — reaches the same
// resource root on realtime and on the files / batches surface as it does on
// chat completions, with each route's own path appended once:
//
// - unset or a dated `api_version`: `/openai/realtime?api-version=…&deployment=<d>`
//   and `/openai/files?api-version=…`;
// - `v1`: `/openai/v1/realtime?model=<d>` and `/openai/v1/files`.
//
// The resource host is real Azure, which a spec cannot stand up, so the
// gateway runs behind a forward proxy (`HTTPS_PROXY`) that tunnels every
// CONNECT to a local TLS mock; nothing leaves the machine.
//
// Azure v1 API: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle

const CALLER = "sk-azure-api-base-paths";
const RESOURCE_HOST = "aisixe2e.openai.azure.com";
const API_BASE = `https://${RESOURCE_HOST}/openai/v1`;
const SECRET = "az-resource-key";
const DATED = "2025-03-01-preview";
const DEPLOYMENT = "dep-api-base";

const VERSIONS = { unset: undefined, dated: DATED, v1: "v1" } as const;
type Version = keyof typeof VERSIONS;
const versions = Object.keys(VERSIONS) as Version[];

interface Mock {
  port: number;
  /** Request targets (path + query) of every HTTP request and WebSocket handshake. */
  http: string[];
  handshakes: string[];
  close(): Promise<void>;
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

function selfSigned(): { key: Buffer; cert: Buffer } {
  const dir = mkdtempSync(join(tmpdir(), "aisix-azure-api-base-"));
  try {
    execFileSync(
      "openssl",
      [
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-keyout", join(dir, "key.pem"), "-out", join(dir, "cert.pem"),
        "-subj", `/CN=${RESOURCE_HOST}`,
        "-addext", `subjectAltName=DNS:${RESOURCE_HOST}`,
      ],
      { stdio: "ignore" },
    );
    return { key: readFileSync(join(dir, "key.pem")), cert: readFileSync(join(dir, "cert.pem")) };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** The Azure resource: lists over HTTPS, and a Realtime WebSocket. */
async function startResource(): Promise<Mock> {
  const http: string[] = [];
  const handshakes: string[] = [];
  const server = createHttpsServer(selfSigned(), (req, res) => {
    http.push(req.url ?? "");
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ object: "list", data: [], has_more: false }));
  });
  const wss = new WebSocketServer({ server });
  wss.on("connection", (socket, req) => {
    handshakes.push(req.url ?? "");
    socket.on("message", () => socket.send(JSON.stringify({ type: "session.updated" })));
  });
  const port = await listening(server);
  return {
    port,
    http,
    handshakes,
    close: () =>
      new Promise((resolve) => {
        for (const c of wss.clients) c.terminate();
        wss.close();
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/** A CONNECT proxy that sends every tunnel to `port` on loopback. */
async function startConnectProxy(port: number): Promise<{ url: string; targets: string[]; close(): Promise<void> }> {
  const targets: string[] = [];
  const server: Server = createServer((_req, res) => res.writeHead(405).end());
  server.on("connect", (req, client, head) => {
    targets.push(req.url ?? "");
    const up = netConnect(port, "127.0.0.1", () => {
      client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
      up.write(head);
      up.pipe(client);
      client.pipe(up);
    });
    up.on("error", () => client.destroy());
    client.on("error", () => up.destroy());
  });
  const proxyPort = await listening(server);
  return {
    url: `http://127.0.0.1:${proxyPort}`,
    targets,
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

describe("azure-openai api_base with a pasted /openai/v1 path on realtime and files/batches", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let resource: Mock | undefined;
  let proxy: Awaited<ReturnType<typeof startConnectProxy>> | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    resource = await startResource();
    proxy = await startConnectProxy(resource.port);
    app = await spawnApp({
      extraEnv: {
        HTTPS_PROXY: proxy.url,
        https_proxy: proxy.url,
        NO_PROXY: "127.0.0.1,localhost",
        no_proxy: "127.0.0.1,localhost",
      },
      // The mock resource's certificate is self-signed.
      extra: { upstream: { tls: { verify: false } } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [version, apiVersion] of Object.entries(VERSIONS) as [Version, string | undefined][]) {
      const pk = await seed.createProviderKey({
        display_name: `az-base-${version}-pk`,
        provider: "azure",
        adapter: "azure-openai",
        api_key: SECRET,
        api_base: API_BASE,
        ...(apiVersion ? { api_version: apiVersion } : {}),
      });
      await seed.createModel({
        display_name: `az-base-${version}`,
        provider: "azure",
        model_name: DEPLOYMENT,
        provider_key_id: pk.id,
      });
    }
    // Seeded last: once it authenticates, every resource above is live.
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["*"],
    });
    const client = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await client.listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await proxy?.close();
    await resource?.close();
  });

  test.each(versions)("realtime session (%s)", async (version) => {
    if (!etcdReachable || !app || !resource || !proxy) return;
    const before = resource.handshakes.length;
    const tunnels = proxy.targets.length;
    const c = new WsClient(`${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=az-base-${version}`, {
      headers: { authorization: `Bearer ${CALLER}` },
    });
    const relayed = await new Promise<string>((resolve, reject) => {
      c.on("open", () => c.send(JSON.stringify({ type: "session.update" })));
      c.on("message", (d) => resolve(d.toString()));
      c.on("unexpected-response", (_q, res) => reject(new Error(`upgrade refused: ${res.statusCode}`)));
      c.on("error", (e) => reject(e));
    });
    c.terminate();
    expect(JSON.parse(relayed).type).toBe("session.updated");
    expect(proxy.targets.slice(tunnels)).toEqual([`${RESOURCE_HOST}:443`]);
    expect(resource.handshakes[before]).toBe(
      version === "v1"
        ? `/openai/v1/realtime?model=${DEPLOYMENT}`
        : `/openai/realtime?api-version=${version === "dated" ? DATED : "2024-10-01-preview"}&deployment=${DEPLOYMENT}`,
    );
  });

  for (const route of ["files", "batches"] as const) {
    test.each(versions)(`${route} list (%s)`, async (version) => {
      if (!etcdReachable || !app || !resource || !proxy) return;
      const before = resource.http.length;
      const res = await fetch(`${app.proxyUrl}/v1/${route}?model=az-base-${version}&limit=5`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      const text = await res.text();
      expect(res.status, text).toBe(200);
      expect(resource.http.slice(before)).toEqual([
        version === "v1"
          ? `/openai/v1/${route}?limit=5`
          : `/openai/${route}?api-version=${version === "dated" ? DATED : "2024-10-21"}&limit=5`,
      ]);
      expect(proxy.targets).toContain(`${RESOURCE_HOST}:443`);
    });
  }
});
