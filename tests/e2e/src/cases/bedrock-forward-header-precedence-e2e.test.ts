import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `default_headers` vs `forward_client_headers` on an AWS Bedrock
// provider key.
//
// Every other face merges the two features into a `HeaderMap`, where the
// static entry wins the name and a forwarded value displaces it in a
// CREDENTIAL slot only. Bedrock cannot: its headers ride the AWS SDK's
// pre-signing interceptor, which is first-wins, so the precedence has to
// be settled before the list gets there — in `resolve_extra_headers`, a
// second implementation of the same rule and therefore a place the two
// can silently disagree. `forward-client-headers-e2e` covers the merge
// faces; this covers the flattened one.
//
// Titan embeddings are the cheapest real Bedrock round-trip: one
// SigV4-signed InvokeModel per input, against a mock endpoint that
// records what actually arrived.

const CALLER_PLAINTEXT = "sk-bedrock-hdr-prec-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const MODEL = "bedrock-hdr-prec-model";
// A credential slot the gateway does not itself read on `/v1/*`, so the
// caller's value is unambiguously the forwarded one and not an artifact
// of gateway authentication.
const CREDENTIAL_SLOT = "cookie";
const PLAIN_HEADER = "x-team";

interface Received {
  path: string;
  headers: Record<string, string>;
}

interface MockBedrock {
  baseUrl: string;
  received: Received[];
  close(): Promise<void>;
}

/** Answers Titan's embed shape and records every request's headers. */
async function startMockBedrock(): Promise<MockBedrock> {
  const received: Received[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    req.on("data", () => {});
    req.on("end", () => {
      received.push({
        path: (req.url ?? "/").split("?")[0],
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [
            k,
            Array.isArray(v) ? v.join(",") : (v ?? ""),
          ]),
        ),
      });
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ embedding: [0.1, 0.2], inputTextTokenCount: 4 }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

describe("bedrock header precedence e2e", () => {
  let app: SpawnedApp | undefined;
  let bedrock: MockBedrock | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    bedrock = await startMockBedrock();
    app = await spawnApp({});
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // Both features name the SAME two headers: one credential slot, one
    // ordinary name. The operator's static value is what an upstream
    // would see if the flattened path resolved the collision the way it
    // did before — for both names.
    const pk = await seed.createProviderKey({
      display_name: "bedrock-hdr-prec-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        access_key_id: "AKIA-hdr-prec-e2e",
        secret_access_key: "sk-hdr-prec-e2e",
        region: "us-west-2",
      }),
      api_base: bedrock.baseUrl,
      request: {
        default_headers: {
          [CREDENTIAL_SLOT]: "session=operator-static",
          [PLAIN_HEADER]: "operator-static",
        },
        forward_client_headers: [CREDENTIAL_SLOT, PLAIN_HEADER],
      },
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "bedrock",
      model_name: "amazon.titan-embed-text-v1",
      provider_key_id: pk.id,
    });
    // Seeded last, so it authenticating implies the whole set landed.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await bedrock?.close();
  });

  test("a forwarded credential slot outranks default_headers; an ordinary name does not", async (ctx) => {
    if (!etcdReachable || !app || !bedrock) {
      ctx.skip();
      return;
    }

    const res = await fetch(`${app.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
        [CREDENTIAL_SLOT]: "session=callers-own",
        [PLAIN_HEADER]: "client-claimed",
      },
      body: JSON.stringify({ model: MODEL, input: ["hello"] }),
    });
    expect(res.status, await res.clone().text()).toBe(200);

    const invoke = bedrock.received.find((r) =>
      r.path.includes("/model/amazon.titan-embed-text-v1/invoke"),
    );
    expect(invoke, `no invoke recorded: ${JSON.stringify(bedrock.received)}`).toBeDefined();

    // The slot the operator opted in: the caller's own value, alone.
    expect(invoke!.headers[CREDENTIAL_SLOT]).toBe("session=callers-own");
    // Every other name keeps the static entry — the operator's value is
    // the more specific statement of intent there.
    expect(invoke!.headers[PLAIN_HEADER]).toBe("operator-static");
    // And the signer still owns `authorization` on both paths.
    expect(invoke!.headers.authorization).toContain("AWS4-HMAC-SHA256");
  });
});
