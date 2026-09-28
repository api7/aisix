import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an exporter endpoint URL may carry userinfo, loopback `http://`
// included. It is an opaque URL: the OTLP exporter reaches a receiver that
// requires the Basic credential written there, an object_store exporter
// uploads with its provider's own signature, and a host that is not on the
// loopback allow-list stays refused however the userinfo in front of it is
// written.

const CALLER = "sk-exporter-endpoint-userinfo";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const MODEL = "endpoint-userinfo-model";
const OTLP_USER = "otlp-e2e";
const OTLP_PASS = "otlp-e2e-pw";
const CREDENTIAL_REF = "userinfo-s3";
const BUCKET = "userinfo-bucket";

interface Recorded {
  method: string;
  path: string;
  authorization: string[];
  body: string;
}

interface Receiver {
  port: number;
  seen: Recorded[];
  close(): Promise<void>;
}

/** Every `authorization` header line the request carried, in order. */
function authorizationLines(req: IncomingMessage): string[] {
  const out: string[] = [];
  for (let i = 0; i < req.rawHeaders.length; i += 2) {
    if (req.rawHeaders[i].toLowerCase() === "authorization") out.push(req.rawHeaders[i + 1]);
  }
  return out;
}

async function startReceiver(answer: (r: Recorded) => number): Promise<Receiver> {
  const seen: Recorded[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const r: Recorded = {
        method: req.method ?? "",
        path: decodeURIComponent((req.url ?? "").split("?")[0]),
        authorization: authorizationLines(req),
        body: Buffer.concat(chunks).toString("utf8"),
      };
      seen.push(r);
      res.statusCode = answer(r);
      if (res.statusCode === 200) res.setHeader("etag", '"e2e-etag"');
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    port,
    seen,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

async function waitFor<T>(probe: () => T | undefined, what: string): Promise<T> {
  const deadline = Date.now() + 20_000;
  for (;;) {
    const hit = probe();
    if (hit !== undefined) return hit;
    if (Date.now() >= deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 100));
  }
}

describe("exporter endpoint URLs may carry userinfo", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let otlp: Receiver | undefined;
  let s3: Receiver | undefined;
  let refusedId = "";
  let etcdReachable = false;

  const basic = `Basic ${Buffer.from(`${OTLP_USER}:${OTLP_PASS}`).toString("base64")}`;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    otlp = await startReceiver((r) => (r.authorization.includes(basic) ? 200 : 401));
    s3 = await startReceiver(() => 200);

    const slug = CREDENTIAL_REF.toUpperCase().replace(/[^A-Z0-9]/g, "_");
    app = await spawnApp({
      extraEnv: {
        [`OBJSTORE_CRED_${slug}_AWS_ACCESS_KEY_ID`]: "AKIDE2E",
        [`OBJSTORE_CRED_${slug}_AWS_SECRET_ACCESS_KEY`]: "secret-e2e",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "otlp-basic-auth",
      enabled: true,
      kind: "otlp_http",
      endpoint: `http://${OTLP_USER}:${OTLP_PASS}@127.0.0.1:${otlp.port}/v1/traces`,
    });
    await seed.createObservabilityExporter({
      name: "s3-userinfo-endpoint",
      enabled: true,
      kind: "object_store",
      provider: "s3",
      bucket: BUCKET,
      prefix: "p",
      region: "us-east-1",
      endpoint: `http://ignored-user:ignored-pw@127.0.0.1:${s3.port}`,
      compression: "none",
      credential_ref: CREDENTIAL_REF,
    });
    // Not on the loopback allow-list: the userinfo in front of the host
    // must not smuggle it past the pattern.
    const refused = await seed.createObservabilityExporter({
      name: "otlp-not-allow-listed",
      enabled: true,
      kind: "otlp_http",
      endpoint: "http://localhost@evil.example/v1/traces",
    });
    refusedId = refused.id;

    const pk = await seed.createProviderKey({
      display_name: "endpoint-userinfo-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: [MODEL] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await otlp?.close();
    await s3?.close();
  });

  test("both exporters load and deliver; the non-allow-listed host is rejected", async (ctx) => {
    if (!etcdReachable || !app || !otlp || !s3) return ctx.skip();

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "hi" }] }),
    });
    await res.arrayBuffer();
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");

    // The OTLP receiver answers 401 to anything but the Basic credential
    // written in the endpoint, so an accepted export is the proof.
    const export_ = await waitFor(
      () => otlp!.seen.find((r) => r.body.includes(requestId)),
      "an OTLP export carrying the request",
    );
    expect(export_.authorization).toEqual([basic]);

    // The S3 upload is signed by the provider, never Basic.
    const upload = await waitFor(
      () => s3!.seen.find((r) => r.method === "PUT" && r.body.includes(requestId)),
      "an object upload carrying the request",
    );
    expect(upload.path.startsWith(`/${BUCKET}/p/`)).toBe(true);
    expect(upload.authorization.some((a) => a.startsWith("AWS4-HMAC-SHA256 "))).toBe(true);

    const status = await fetch(`${app.metricsUrl}/status/config`);
    const cfg = (await status.json()) as {
      rejected: Array<{ resource_id: string; resource_kind: string; last_error_kind: string }>;
    };
    const rejection = cfg.rejected.find((r) => r.resource_id === refusedId);
    expect(rejection, JSON.stringify(cfg.rejected)).toBeDefined();
    expect(rejection!.last_error_kind).toBe("schema_failed");
  });
});
