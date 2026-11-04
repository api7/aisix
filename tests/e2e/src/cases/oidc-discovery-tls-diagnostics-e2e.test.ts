import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:https";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  agentClaims,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockIdp,
  waitConfigPropagation,
  waitForLogLine,
  type MockIdp,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an identity provider the gateway cannot reach is diagnosable from the
// gateway's log. The issuer here is served over HTTPS with a certificate the
// gateway does not trust, the ordinary private-CA on-prem shape. The operator's
// log has to name the certificate; the caller's answer has to say nothing more
// than it always did.

describe("OIDC discovery failure diagnostics", () => {
  let app: SpawnedApp | undefined;
  let idp: MockIdp | undefined;
  let issuerServer: Server | undefined;
  let pkiDir: string | undefined;
  let issuer = "";
  let ready = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return;

    pkiDir = mkdtempSync(join(tmpdir(), "aisix-oidc-tls-"));
    const dir = pkiDir;
    execFileSync(
      "openssl",
      [
        "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", join(dir, "key.pem"), "-out", join(dir, "cert.pem"),
        "-days", "1", "-subj", "/CN=127.0.0.1",
        "-addext", "subjectAltName=IP:127.0.0.1",
      ],
      { stdio: "ignore" },
    );
    issuerServer = createServer(
      {
        key: readFileSync(join(dir, "key.pem")),
        cert: readFileSync(join(dir, "cert.pem")),
      },
      (_req, res) => res.writeHead(500).end(),
    );
    // A handshake the client aborts is the point of this server, not a fault.
    issuerServer.on("tlsClientError", () => {});
    await new Promise<void>((resolve) => issuerServer!.listen(0, "127.0.0.1", resolve));
    issuer = `https://127.0.0.1:${(issuerServer.address() as AddressInfo).port}`;

    idp = await startMockIdp();
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createOidcProvider({
      name: "untrusted-issuer",
      issuer,
      audiences: ["aisix-gateway"],
    });
    const readyKey = "sk-oidc-tls-ready";
    await seed.createApiKey({
      key_hash: createHash("sha256").update(readyKey).digest("hex"),
      allowed_models: [],
    });
    const proxy = new ProxyClient(app.proxyUrl, readyKey);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
    ready = true;
  });

  afterAll(async () => {
    await app?.exit();
    await idp?.close();
    await new Promise((resolve) => (issuerServer ? issuerServer.close(resolve) : resolve(undefined)));
    if (pkiDir) rmSync(pkiDir, { recursive: true, force: true });
  });

  test("the log names the certificate and the caller's answer stays generic", async (ctx) => {
    if (!ready) {
      ctx.skip();
      return;
    }

    const res = await fetch(`${app!.proxyUrl}/v1/models`, {
      headers: { authorization: `Bearer ${idp!.sign(agentClaims(issuer))}` },
    });

    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({
      error: {
        message: "unable to fetch the identity provider's signing keys",
        type: "api_error",
        code: "jwks_unavailable",
      },
    });

    const line = await waitForLogLine(
      app!,
      (l) => l.includes("cannot resolve the trust provider's JWKS endpoint"),
      "the discovery failure's log line",
    );
    expect(line).toContain(`${issuer}/.well-known/openid-configuration`);
    expect(line.toLowerCase()).toContain("certificate");
  });
});
