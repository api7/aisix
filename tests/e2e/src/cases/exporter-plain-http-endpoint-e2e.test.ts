import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
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
import { startMockOtlp, type MockOtlp } from "../harness/otlp-mock.js";

// E2E: an `otlp_http` or `object_store` exporter endpoint may be plain
// `http://` on any host, the way an intranet collector or MinIO is usually
// reached. The receivers here listen on 127.0.0.2, a host that is not one of
// the test-fixture names the endpoint pattern used to allow, so a gateway
// that still restricted the host would drop the exporter rows (etcd) or
// refuse the resources file (file mode) and nothing would be delivered.

const HOST = "127.0.0.2";
const CALLER = "sk-exporter-plain-http";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CALLER_ENV = "EXPORTER_PLAIN_HTTP_CALLER";
const MODEL = "plain-http-endpoint-model";
const CREDENTIAL_REF = "plain-http-s3";
const BUCKET = "plain-http-bucket";

interface Upload {
  method: string;
  path: string;
  body: string;
}

interface S3Receiver {
  port: number;
  seen: Upload[];
  close(): Promise<void>;
}

async function startS3Receiver(): Promise<S3Receiver> {
  const seen: Upload[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      seen.push({
        method: req.method ?? "",
        path: decodeURIComponent((req.url ?? "").split("?")[0]),
        body: Buffer.concat(chunks).toString("utf8"),
      });
      res.statusCode = 200;
      res.setHeader("etag", '"e2e-etag"');
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, HOST, resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("s3 receiver: no listen address");
  return {
    port: address.port,
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

async function chat(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "hi" }] }),
  });
  await res.arrayBuffer();
  expect(res.status).toBe(200);
  const requestId = res.headers.get("x-aisix-request-id") ?? "";
  expect(requestId).not.toBe("");
  return requestId;
}

const spanOf = (otlp: MockOtlp, requestId: string) =>
  otlp.spans.find((s) => s.attributes["aisix.request_id"] === requestId);

describe("exporter endpoints accept plain http on any host", () => {
  let upstream: OpenAiUpstream | undefined;
  let etcdApp: SpawnedApp | undefined;
  let fileApp: SpawnedApp | undefined;
  let etcdOtlp: MockOtlp | undefined;
  let fileOtlp: MockOtlp | undefined;
  let s3: S3Receiver | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    etcdOtlp = await startMockOtlp({ host: HOST });
    fileOtlp = await startMockOtlp({ host: HOST });
    s3 = await startS3Receiver();

    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const slug = CREDENTIAL_REF.toUpperCase().replace(/[^A-Z0-9]/g, "_");
    etcdApp = await spawnApp({
      extraEnv: {
        [`OBJSTORE_CRED_${slug}_AWS_ACCESS_KEY_ID`]: "AKIDE2E",
        [`OBJSTORE_CRED_${slug}_AWS_SECRET_ACCESS_KEY`]: "secret-e2e",
      },
    });
    const seed = new SeedClient(etcd, etcdApp.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "otlp-plain-http",
      enabled: true,
      kind: "otlp_http",
      endpoint: etcdOtlp.url,
    });
    await seed.createObservabilityExporter({
      name: "s3-plain-http",
      enabled: true,
      kind: "object_store",
      provider: "s3",
      bucket: BUCKET,
      prefix: "p",
      region: "us-east-1",
      endpoint: `http://${HOST}:${s3.port}`,
      compression: "none",
      credential_ref: CREDENTIAL_REF,
    });
    const pk = await seed.createProviderKey({
      display_name: "plain-http-endpoint-pk",
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
    const proxy = new ProxyClient(etcdApp.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await etcdApp?.exit();
    await fileApp?.exit();
    await upstream?.close();
    await etcdOtlp?.close();
    await fileOtlp?.close();
    await s3?.close();
  });

  test("etcd: OTLP and object_store exporters load and deliver", async (ctx) => {
    if (!etcdReachable || !etcdApp || !etcdOtlp || !s3) return ctx.skip();

    const requestId = await chat(etcdApp);
    await waitFor(() => spanOf(etcdOtlp!, requestId), "an OTLP span carrying the request");
    const upload = await waitFor(
      () => s3!.seen.find((r) => r.method === "PUT" && r.body.includes(requestId)),
      "an object upload carrying the request",
    );
    expect(upload.path.startsWith(`/${BUCKET}/p/`)).toBe(true);
  });

  test("resources file: an OTLP exporter loads and delivers", async () => {
    fileApp = await spawnApp({
      resourcesFile: `
_format_version: "1"
provider_keys:
  - display_name: plain-http-file-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstream!.baseUrl}/v1
models:
  - display_name: ${MODEL}
    provider: openai
    model_name: gpt-4o-mini
    provider_key: plain-http-file-pk
api_keys:
  - display_name: plain-http-file-caller
    key_env: ${CALLER_ENV}
    allowed_models: ["${MODEL}"]
observability_exporters:
  - name: otlp-plain-http-file
    kind: otlp_http
    endpoint: ${fileOtlp!.url}
`,
      extraEnv: { [CALLER_ENV]: CALLER },
    });

    const requestId = await chat(fileApp);
    await waitFor(() => spanOf(fileOtlp!, requestId), "an OTLP span carrying the request");
  });
});
