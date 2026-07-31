import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
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

// E2E: `provider_key.tls` scopes trust to one endpoint (#860).
//
// The deployment-wide `upstream.tls.ca_file` covers a gateway facing one
// private authority. This covers the other half of the issue: a deployment
// facing more than one, where trust has to be declared where the endpoint is
// declared. One gateway, no `upstream.tls` at all, three Provider Keys:
//
//   - key A points at an endpoint behind CA A and carries CA A inline  → 200
//   - key B points at the same endpoint and carries CA B (wrong issuer) → 502
//   - key C points at the same endpoint with `verify: false`           → 200
//
// B is the one that matters: it shows the per-key CA is actually consulted
// rather than every key inheriting a union of everything configured anywhere.

const CALLER_PLAINTEXT = "sk-provider-key-tls-e2e";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const MODEL_TRUSTED = "pk-tls-trusted";
const MODEL_WRONG_CA = "pk-tls-wrong-ca";
const MODEL_NO_VERIFY = "pk-tls-no-verify";

const REPLY = {
  id: "cmpl-pk-tls",
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

interface Ca {
  certPem: string;
  keyPath: string;
  certPath: string;
}

let dir = "";
const openssl = (args: string[]) =>
  execFileSync("openssl", args, { stdio: ["ignore", "pipe", "pipe"] });

function makeCa(name: string): Ca {
  const keyPath = join(dir, `${name}.key`);
  const certPath = join(dir, `${name}.crt`);
  openssl(["genrsa", "-out", keyPath, "2048"]);
  openssl([
    "req", "-x509", "-new", "-key", keyPath, "-out", certPath,
    "-days", "1", "-sha256", "-subj", `/CN=aisix-e2e-${name}`,
  ]);
  return { certPem: readFileSync(certPath, "utf8"), keyPath, certPath };
}

/** A leaf for 127.0.0.1 signed by `ca`. */
function makeLeaf(name: string, ca: Ca): { key: string; cert: string } {
  const keyPath = join(dir, `${name}.key`);
  const csrPath = join(dir, `${name}.csr`);
  const certPath = join(dir, `${name}.crt`);
  const extPath = join(dir, `${name}.ext`);
  openssl(["genrsa", "-out", keyPath, "2048"]);
  openssl(["req", "-new", "-key", keyPath, "-out", csrPath, "-subj", "/CN=127.0.0.1"]);
  writeFileSync(extPath, "subjectAltName=IP:127.0.0.1\n", "utf8");
  openssl([
    "x509", "-req", "-in", csrPath,
    "-CA", ca.certPath, "-CAkey", ca.keyPath, "-CAcreateserial",
    "-out", certPath, "-days", "1", "-sha256", "-extfile", extPath,
  ]);
  return { key: readFileSync(keyPath, "utf8"), cert: readFileSync(certPath, "utf8") };
}

function opensslAvailable(): boolean {
  try {
    execFileSync("openssl", ["version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

describe("provider_key.tls (#860)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  let haveOpenssl = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    haveOpenssl = opensslAvailable();
    if (!etcdReachable || !haveOpenssl) return;

    dir = mkdtempSync(join(tmpdir(), "aisix-pk-tls-"));
    const caA = makeCa("ca-a");
    const caB = makeCa("ca-b");
    const leaf = makeLeaf("server", caA);

    upstream = await startOpenAiUpstream({
      nonStreamBody: REPLY,
      tls: { key: leaf.key, cert: leaf.cert },
    });

    // No `upstream.tls` anywhere: everything below has to come from the
    // Provider Keys themselves.
    app = await spawnApp();
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL_TRUSTED, MODEL_WRONG_CA, MODEL_NO_VERIFY],
    });

    const cases: Array<[string, Record<string, unknown>]> = [
      [MODEL_TRUSTED, { ca_cert: caA.certPem }],
      [MODEL_WRONG_CA, { ca_cert: caB.certPem }],
      [MODEL_NO_VERIFY, { verify: false }],
    ];
    for (const [model, tls] of cases) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-mock",
        api_base: upstream.baseUrl,
        tls,
      });
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function chat(model: string): Promise<{ status: number; body: string }> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
    });
    return { status: res.status, body: await res.text() };
  }

  test("a key carrying the endpoint's own CA reaches it", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      try {
        const { status } = await chat(MODEL_TRUSTED);
        return status === 200 || status === 502;
      } catch {
        return false;
      }
    });

    const { status, body } = await chat(MODEL_TRUSTED);
    expect(status).toBe(200);
    expect(JSON.parse(body).choices[0].message.content).toBe("trusted");
  });

  test("a key carrying a different CA is still rejected", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    // The load-bearing assertion: keys do not pool their trust. A CA
    // configured on one key must not make a second key's endpoint
    // reachable, which is exactly what a single process-wide trust store
    // would have done.
    const { status, body } = await chat(MODEL_WRONG_CA);
    expect(status).toBe(502);
    expect(body.toLowerCase()).toContain("certificate");
  });

  test("a key with verify disabled reaches the endpoint with no CA at all", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    expect((await chat(MODEL_NO_VERIFY)).status).toBe(200);
  });
});
