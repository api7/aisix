import { randomUUID } from "node:crypto";
import { createServer, request, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  pickFreePort,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockIdp,
  waitConfigPropagation,
  waitForLogLine,
  type McpUpstream,
  type MockIdp,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an MCP resource URL carrying a path prefix, for a gateway behind a
// front proxy that routes one shared domain by path prefix and strips the
// prefix before forwarding. Clients only ever see the prefixed URLs, so the
// challenge has to send them to a metadata URL the front proxy can route.
//
// A small prefix-stripping proxy started here plays the front proxy, and
// every request below goes through it.

const PREFIX = "/team-a/gw";
const SERVER = "prefixed";

interface FrontProxy {
  url: string;
  close(): Promise<void>;
}

/** Forwards `${prefix}/...` to `target` as `/...`; anything else is a 404. */
async function startPrefixStrippingProxy(prefix: string, target: URL): Promise<FrontProxy> {
  const port = await pickFreePort();
  const server: Server = createServer((req, res) => {
    const path = req.url ?? "";
    if (!path.startsWith(`${prefix}/`)) {
      res.writeHead(404).end();
      return;
    }
    const upstream = request(
      {
        hostname: target.hostname,
        port: target.port,
        method: req.method,
        path: path.slice(prefix.length),
        headers: req.headers,
      },
      (upstreamRes) => {
        res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
        upstreamRes.pipe(res);
      },
    );
    upstream.on("error", () => {
      if (res.headersSent) res.destroy();
      else res.writeHead(502).end();
    });
    req.pipe(upstream);
  });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}`,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

const initialize = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-11-25",
    capabilities: {},
    clientInfo: { name: "mcp-resource-url-path-prefix-e2e", version: "0.1" },
  },
};

const mcpPost = (url: string, headers: Record<string, string> = {}): Promise<Response> =>
  fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      ...headers,
    },
    body: JSON.stringify(initialize),
  });

describe("mcp_auth_settings.resource_url with a path prefix", () => {
  let app: SpawnedApp | undefined;
  let idp: MockIdp | undefined;
  let upstream: McpUpstream | undefined;
  let front: FrontProxy | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const settingsId = randomUUID();

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    idp = await startMockIdp();
    upstream = await startMcpUpstream("prefixed");
    app = await spawnApp({ logLevel: "info" });
    front = await startPrefixStrippingProxy(PREFIX, new URL(app.proxyUrl));
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createOidcProvider({
      name: "corp",
      issuer: idp.url,
      audiences: ["aisix-gateway"],
      jwks_uri: idp.jwksUrl,
    });
    await seed.update("mcp_servers", randomUUID(), { name: SERVER, type: "mcp", url: upstream.url });
  }, 60_000);

  afterAll(async () => {
    await front?.close();
    await app?.exit();
    await upstream?.close();
    await idp?.close();
  });

  /**
   * Point the row at `resourceUrl` and wait for the gateway to apply it, read
   * off the metadata document rather than the challenge the tests assert on:
   * a gateway publishing `resourceUrl`, or nothing at all for an invalid one.
   */
  const configure = async (resourceUrl: string, valid: boolean) => {
    await seed!.update("mcp_auth_settings", settingsId, { resource_url: resourceUrl });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/.well-known/oauth-protected-resource/mcp`);
      if (!valid) return res.status === 404;
      return res.ok && ((await res.json()) as { resource: string }).resource === resourceUrl;
    });
  };

  test("the challenge sends a client to a metadata URL under the prefix", async () => {
    if (!etcdReachable) return;
    const resource = `${front!.url}${PREFIX}/mcp`;
    const metadata = `${front!.url}${PREFIX}/.well-known/oauth-protected-resource/mcp`;
    const challenge = `Bearer resource_metadata="${metadata}"`;
    await configure(resource, true);

    for (const endpoint of [resource, `${resource}/${SERVER}`]) {
      const missing = await mcpPost(endpoint);
      expect(missing.status, endpoint).toBe(401);
      expect(missing.headers.get("www-authenticate"), endpoint).toBe(challenge);

      const invalid = await mcpPost(endpoint, { authorization: "Bearer not-a-valid-token" });
      expect(invalid.status, endpoint).toBe(401);
      expect(invalid.headers.get("www-authenticate"), endpoint).toBe(
        `Bearer error="invalid_token", resource_metadata="${metadata}"`,
      );
    }

    const document = await fetch(metadata);
    expect(document.status).toBe(200);
    expect(await document.json()).toEqual({
      resource,
      authorization_servers: [idp!.url],
      bearer_methods_supported: ["header"],
    });
  });

  test("a plain /mcp URL keeps the metadata URL at the root", async () => {
    if (!etcdReachable) return;
    const resource = `${app!.proxyUrl}/mcp`;
    const metadata = `${app!.proxyUrl}/.well-known/oauth-protected-resource/mcp`;
    const challenge = `Bearer resource_metadata="${metadata}"`;
    await configure(resource, true);

    for (const endpoint of [resource, `${resource}/${SERVER}`]) {
      const res = await mcpPost(endpoint);
      expect(res.status, endpoint).toBe(401);
      expect(res.headers.get("www-authenticate"), endpoint).toBe(challenge);
    }

    const document = await fetch(metadata);
    expect(document.status).toBe(200);
    expect(((await document.json()) as { resource: string }).resource).toBe(resource);
  });

  test("a path that only looks like it ends in /mcp keeps discovery dormant, with one warning", async () => {
    if (!etcdReachable) return;
    const probe = `${app!.proxyUrl}/mcp`;
    // Start active, or the first wait for "dormant" would pass before the row lands.
    await configure(probe, true);
    for (const path of [`${PREFIX}/foo-mcp`, `${PREFIX}/mcp/`, `${PREFIX}/mcp/x`]) {
      await configure(`${front!.url}${path}`, false);
      const res = await mcpPost(probe);
      expect(res.status, path).toBe(401);
      expect(res.headers.get("www-authenticate"), path).toBeNull();
      // Back to a valid URL, so the next value's wait cannot pass on this one.
      await configure(probe, true);
    }

    const isWarning = (line: string) => line.includes("OAuth discovery surface stays dormant");
    expect(await waitForLogLine(app!, isWarning, "the dormant-surface warning")).toContain("ending in /mcp");
    // This request's access-log line follows anything the requests above
    // queued, so a repeat of the warning would already be readable below it.
    const barrier = (await mcpPost(probe)).headers.get("x-aisix-request-id");
    await waitForLogLine(app!, (line) => line.includes(`request_id="${barrier}"`), "the barrier request's access log");
    expect(app!.output().split("\n").filter(isWarning)).toHaveLength(1);
  });
});
