import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

// E2E: outbound TLS trust is configurable (#860).
//
// An upstream whose certificate is signed by a private CA is the ordinary
// on-prem shape, and before this the gateway had no way to be told about
// one: no `ca_file`, no verification override, and the only working
// mechanism (`SSL_CERT_FILE`) was undocumented and process-wide. What an
// operator saw was a generic 502.
//
// Each scenario runs a fresh gateway against an HTTPS mock upstream whose
// leaf is signed by a throwaway CA:
//
//   1. no TLS config      → 502, and the error names the certificate as
//                           the cause rather than something generic;
//   2. upstream.tls.ca_file → 200;
//   3. upstream.tls.verify: false → 200 without any CA configured;
//   4. SSL_CERT_FILE      → 200, and public roots stay trusted (the
//                           pre-existing mechanism the issue asks to have
//                           documented, pinned so it cannot regress).

const CALLER_PLAINTEXT = "sk-upstream-tls-e2e";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const MODEL = "private-ca-model";

const REPLY = {
  id: "cmpl-private-ca",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "trusted" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};

interface Pki {
  dir: string;
  caCertPath: string;
  serverKey: string;
  serverCert: string;
}

/**
 * A throwaway CA and a leaf it signs for 127.0.0.1. Deliberately NOT
 * self-signed at the leaf: the gateway must be shown to trust the *issuer*
 * from `ca_file`, which a self-signed leaf would not distinguish from
 * "trusts anything it was handed".
 */
function makePki(): Pki {
  const dir = mkdtempSync(join(tmpdir(), "aisix-upstream-tls-"));
  const p = (name: string) => join(dir, name);
  const openssl = (args: string[]) =>
    execFileSync("openssl", args, { stdio: ["ignore", "pipe", "pipe"] });

  openssl(["genrsa", "-out", p("ca.key"), "2048"]);
  openssl([
    "req", "-x509", "-new", "-key", p("ca.key"),
    "-out", p("ca.crt"), "-days", "1", "-sha256",
    "-subj", "/CN=aisix-e2e-private-ca",
  ]);

  openssl(["genrsa", "-out", p("server.key"), "2048"]);
  openssl([
    "req", "-new", "-key", p("server.key"), "-out", p("server.csr"),
    "-subj", "/CN=127.0.0.1",
  ]);
  writeFileSync(p("ext.cnf"), "subjectAltName=IP:127.0.0.1\n", "utf8");
  openssl([
    "x509", "-req", "-in", p("server.csr"),
    "-CA", p("ca.crt"), "-CAkey", p("ca.key"), "-CAcreateserial",
    "-out", p("server.crt"), "-days", "1", "-sha256",
    "-extfile", p("ext.cnf"),
  ]);

  const read = (name: string) =>
    execFileSync("cat", [p(name)]).toString("utf8");
  return {
    dir,
    caCertPath: p("ca.crt"),
    serverKey: read("server.key"),
    serverCert: read("server.crt"),
  };
}

function opensslAvailable(): boolean {
  try {
    execFileSync("openssl", ["version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

describe("outbound TLS trust (#860)", () => {
  let pki: Pki | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  let haveOpenssl = false;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    haveOpenssl = opensslAvailable();
    if (!etcdReachable || !haveOpenssl) return;

    pki = makePki();
    upstream = await startOpenAiUpstream({
      nonStreamBody: REPLY,
      tls: { key: pki.serverKey, cert: pki.serverCert },
    });
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await upstream?.close();
  });

  /** Boot a gateway with the given `upstream.tls` block, seeded to route to the mock. */
  async function gatewayFor(
    tls: Record<string, unknown> | undefined,
    extraEnv?: Record<string, string>,
  ): Promise<SpawnedApp> {
    const app = await spawnApp({
      ...(tls ? { extra: { upstream: { tls } } } : {}),
      ...(extraEnv ? { extraEnv } : {}),
    });
    apps.push(app);
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL],
    });
    const pk = await seed.createProviderKey({
      display_name: `private-ca-pk-${app.etcdPrefix}`,
      secret: "sk-mock",
      api_base: upstream!.baseUrl,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    return app;
  }

  async function chat(app: SpawnedApp): Promise<{ status: number; body: string }> {
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    return { status: res.status, body: await res.text() };
  }

  /**
   * Poll until the seeded key and model have propagated far enough that
   * the request reaches upstream dispatch — which is the only point
   * where a TLS decision is observable. 401/403/404 all mean the
   * snapshot is still catching up; 200 and 502 are both "dispatch ran",
   * and which one it is is exactly what each test asserts.
   */
  async function settle(app: SpawnedApp): Promise<void> {
    await waitConfigPropagation(async () => {
      try {
        const { status } = await chat(app);
        return status === 200 || status === 502;
      } catch {
        return false;
      }
    });
  }

  test("without any TLS config the private CA is rejected, and the error says so", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    const app = await gatewayFor(undefined);
    await settle(app);

    const { status, body } = await chat(app);
    expect(status).toBe(502);
    // The whole complaint in the issue is that the operator cannot tell
    // a trust problem from any other transport failure.
    expect(body.toLowerCase()).toContain("certificate");
  });

  test("upstream.tls.ca_file makes the private CA trusted", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    const app = await gatewayFor({ ca_file: pki!.caCertPath });
    await settle(app);

    const { status, body } = await chat(app);
    expect(status).toBe(200);
    expect(JSON.parse(body).choices[0].message.content).toBe("trusted");
  });

  test("upstream.tls.verify false accepts the certificate with no CA configured", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    const app = await gatewayFor({ verify: false });
    await settle(app);

    expect((await chat(app)).status).toBe(200);
  });

  test("SSL_CERT_FILE is still honoured", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    // The mechanism the issue asks to have documented. It worked only
    // because reqwest's native-roots feature arrived transitively from
    // `object_store`'s cloud features; that feature is now declared
    // outright, and this pins the behaviour so removing an unrelated
    // dependency cannot silently take it away again.
    //
    // The additive half — that a private CA here does not displace the
    // built-in roots — is not asserted: it needs a real public host, and
    // these specs stay offline. It follows from reqwest loading the
    // webpki set alongside the native one.
    const app = await gatewayFor(undefined, { SSL_CERT_FILE: pki!.caCertPath });
    await settle(app);

    expect((await chat(app)).status).toBe(200);
  });

  test("a ca_file that is not a certificate fails the boot instead of trusting nothing", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    const bogus = join(pki!.dir, "not-a-cert.pem");
    writeFileSync(bogus, "this is not a certificate\n", "utf8");

    await expect(
      spawnApp({ extra: { upstream: { tls: { ca_file: bogus } } } }),
    ).rejects.toThrow();
  });
});
