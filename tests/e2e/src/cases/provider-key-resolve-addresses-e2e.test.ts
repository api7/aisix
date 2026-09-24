import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `provider_key.resolve_addresses` (AISIX-Cloud#1661).
//
// An upstream reached over a private link that has no DNS entry, while the
// vendor still 404s anything that does not carry its own hostname. The key
// names the vendor's hostname in `api_base` and the link's addresses in
// `resolve_addresses`: the gateway dials one and sends the hostname.
//
// Every base URL below names `vendor-1661.invalid`, a name that cannot
// resolve (RFC 2606 reserves `.invalid`), so nothing here can pass by
// accident — reaching the mock at all proves the override was applied, and
// the key without it is the mutation that must fail.

const CALLER_PLAINTEXT = "sk-resolve-addresses-e2e";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** The hostname the vendor requires in `Host`, and that no resolver knows. */
const VENDOR_HOST = "vendor-1661.invalid";
const LOOPBACK = "127.0.0.1";

const MODEL_HTTP = "resolve-http";
const MODEL_UNRESOLVED = "resolve-absent";
const MODEL_EMBEDDINGS = "resolve-embeddings";
const MODEL_HTTPS = "resolve-https";
const MODEL_FAILOVER = "resolve-failover";

/**
 * A loopback address nothing in this spec binds. The mocks listen on
 * `127.0.0.1` only, so a connection to this address on the same port is
 * refused immediately — which is what makes it a usable first entry for
 * the ordering case.
 */
const DEAD_LOOPBACK = "127.0.0.2";

const CHAT_REPLY = {
  id: "cmpl-resolve-address",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "reached over the private link" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};

const EMBEDDING_VECTOR = [0.25, 0.5, 0.75];
const EMBEDDINGS_REPLY = {
  object: "list",
  model: "text-embedding-3-small",
  data: [{ object: "embedding", index: 0, embedding: EMBEDDING_VECTOR }],
  usage: { prompt_tokens: 2, total_tokens: 2 },
};

let dir = "";
const openssl = (args: string[]) =>
  execFileSync("openssl", args, { stdio: ["ignore", "pipe", "pipe"] });

/** A self-signed CA, and a leaf it issues for `VENDOR_HOST`. */
function makeCaAndLeaf(): { caPem: string; key: string; cert: string } {
  const caKey = join(dir, "ca.key");
  const caCert = join(dir, "ca.crt");
  openssl(["genrsa", "-out", caKey, "2048"]);
  openssl([
    "req", "-x509", "-new", "-key", caKey, "-out", caCert,
    "-days", "1", "-sha256", "-subj", "/CN=aisix-e2e-resolve-ca",
  ]);

  const leafKey = join(dir, "leaf.key");
  const leafCsr = join(dir, "leaf.csr");
  const leafCert = join(dir, "leaf.crt");
  const leafExt = join(dir, "leaf.ext");
  openssl(["genrsa", "-out", leafKey, "2048"]);
  openssl(["req", "-new", "-key", leafKey, "-out", leafCsr, "-subj", `/CN=${VENDOR_HOST}`]);
  // DNS, deliberately: no `IP:127.0.0.1`. The certificate is valid for the
  // vendor's NAME and for nothing else, so a gateway that verified against
  // the address it dialled would fail this.
  writeFileSync(leafExt, `subjectAltName=DNS:${VENDOR_HOST}\n`, "utf8");
  openssl([
    "x509", "-req", "-in", leafCsr,
    "-CA", caCert, "-CAkey", caKey, "-CAcreateserial",
    "-out", leafCert, "-days", "1", "-sha256", "-extfile", leafExt,
  ]);
  return {
    caPem: readFileSync(caCert, "utf8"),
    key: readFileSync(leafKey, "utf8"),
    cert: readFileSync(leafCert, "utf8"),
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

/** `http://127.0.0.1:PORT` → `http://vendor-1661.invalid:PORT`. */
function asVendorUrl(baseUrl: string): string {
  return baseUrl.replace(`//${LOOPBACK}:`, `//${VENDOR_HOST}:`);
}

/** The port a mock ended up on, as it appears in a `Host` header. */
function hostHeaderFor(upstream: OpenAiUpstream): string {
  return new URL(upstream.baseUrl).host.replace(LOOPBACK, VENDOR_HOST);
}

describe("provider_key.resolve_addresses (AISIX-Cloud#1661)", () => {
  let app: SpawnedApp | undefined;
  let chatUpstream: OpenAiUpstream | undefined;
  let embeddingsUpstream: OpenAiUpstream | undefined;
  let tlsUpstream: OpenAiUpstream | undefined;
  let failoverUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  let haveOpenssl = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    haveOpenssl = opensslAvailable();
    if (!etcdReachable || !haveOpenssl) return;

    dir = mkdtempSync(join(tmpdir(), "aisix-resolve-address-"));
    const tls = makeCaAndLeaf();

    chatUpstream = await startOpenAiUpstream({ nonStreamBody: CHAT_REPLY });
    embeddingsUpstream = await startOpenAiUpstream({
      nonStreamBody: EMBEDDINGS_REPLY,
    });
    tlsUpstream = await startOpenAiUpstream({
      nonStreamBody: CHAT_REPLY,
      tls: { key: tls.key, cert: tls.cert },
    });
    failoverUpstream = await startOpenAiUpstream({ nonStreamBody: CHAT_REPLY });

    app = await spawnApp();
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);

    const keys: Array<[string, Record<string, unknown>]> = [
      [
        MODEL_HTTP,
        {
          api_base: asVendorUrl(chatUpstream.baseUrl),
          resolve_addresses: [LOOPBACK],
        },
      ],
      // The mutation: same endpoint, same hostname, no override.
      [MODEL_UNRESOLVED, { api_base: asVendorUrl(chatUpstream.baseUrl) }],
      [
        MODEL_EMBEDDINGS,
        {
          api_base: asVendorUrl(embeddingsUpstream.baseUrl),
          resolve_addresses: [LOOPBACK],
        },
      ],
      [
        MODEL_HTTPS,
        {
          api_base: asVendorUrl(tlsUpstream.baseUrl),
          resolve_addresses: [LOOPBACK],
          // `verify` left at its default, so the handshake is checked
          // against the name in the URL.
          tls: { ca_cert: tls.caPem },
        },
      ],
      [
        MODEL_FAILOVER,
        {
          api_base: asVendorUrl(failoverUpstream.baseUrl),
          // First entry refuses, second serves.
          resolve_addresses: [DEAD_LOOPBACK, LOOPBACK],
        },
      ],
    ];
    for (const [model, pkFields] of keys) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-mock",
        ...pkFields,
      });
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name:
          model === MODEL_EMBEDDINGS ? "text-embedding-3-small" : "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        MODEL_HTTP,
        MODEL_UNRESOLVED,
        MODEL_EMBEDDINGS,
        MODEL_HTTPS,
        MODEL_FAILOVER,
      ],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await chatUpstream?.close();
    await embeddingsUpstream?.close();
    await tlsUpstream?.close();
    await failoverUpstream?.close();
  });

  async function post(
    path: string,
    body: unknown,
  ): Promise<{ status: number; body: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    return { status: res.status, body: await res.text() };
  }

  const chat = (model: string) =>
    post("/v1/chat/completions", {
      model,
      messages: [{ role: "user", content: "hi" }],
    });

  test("the request reaches the address and carries the hostname", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    const { status, body } = await chat(MODEL_HTTP);
    expect(status).toBe(200);
    expect(JSON.parse(body).choices[0].message.content).toBe(
      "reached over the private link",
    );
    // The whole point of the feature: the connection went to 127.0.0.1 and
    // the vendor still saw its own name. A Host-header override would have
    // produced the same status with a different name here.
    const seen = chatUpstream!.receivedRequests.at(-1)!;
    expect(seen.headers.host).toBe(hostHeaderFor(chatUpstream!));
  });

  test("the same key without the override cannot resolve the hostname", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    // The mutation check for the test above: identical in every respect
    // except `resolve_addresses`, and it must not reach the upstream at
    // all.
    const before = chatUpstream!.receivedRequests.length;
    const { status } = await chat(MODEL_UNRESOLVED);
    expect(status).not.toBe(200);
    expect(chatUpstream!.receivedRequests.length).toBe(before);
  });

  test("a non-chat surface goes to the address too", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    const { status, body } = await post("/v1/embeddings", {
      model: MODEL_EMBEDDINGS,
      input: "hello",
    });
    expect(status).toBe(200);
    expect(JSON.parse(body).data[0].embedding).toEqual(EMBEDDING_VECTOR);
    const seen = embeddingsUpstream!.receivedRequests.at(-1)!;
    expect(seen.headers.host).toBe(hostHeaderFor(embeddingsUpstream!));
  });

  test("a second address is tried when the first refuses", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    // The list is an answer a resolver could have given, so the connector
    // walks it: the first address refuses the connection and the request
    // still lands on the second. A single-address field could not express
    // this, and an implementation that only ever dialled the first entry
    // would fail here.
    const { status, body } = await chat(MODEL_FAILOVER);
    expect(status).toBe(200);
    expect(JSON.parse(body).choices[0].message.content).toBe(
      "reached over the private link",
    );
    const seen = failoverUpstream!.receivedRequests.at(-1)!;
    expect(seen.headers.host).toBe(hostHeaderFor(failoverUpstream!));
  });

  test("TLS is verified against the hostname, not the address", async (ctx) => {
    if (!etcdReachable || !haveOpenssl) {
      ctx.skip();
      return;
    }
    // The endpoint's certificate names `vendor-1661.invalid` and no IP, so
    // a 200 here can only mean the gateway sent that name as its SNI and
    // verified the certificate against it, while connecting to 127.0.0.1.
    const { status } = await chat(MODEL_HTTPS);
    expect(status).toBe(200);
    const seen = tlsUpstream!.receivedRequests.at(-1)!;
    expect(seen.headers.host).toBe(hostHeaderFor(tlsUpstream!));
  });
});
