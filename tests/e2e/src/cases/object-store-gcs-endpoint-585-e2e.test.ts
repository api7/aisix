import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
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

// #585: an `object_store` exporter for provider `gcs` honours `endpoint`, as
// the `s3` and `azure_blob` providers already did. It used to be ignored, so
// the upload went to Google's native host whatever the exporter said — an
// emulator or private base URL could only be reached through a
// `gcs_base_url` inside the service-account JSON.
//
// The service account here deliberately carries NO `gcs_base_url` (and
// disables OAuth, as an emulator account does), so the only thing that can
// send the upload to the local receiver is the exporter's own `endpoint`.

const CALLER_PLAINTEXT = "sk-objstore-gcs-endpoint-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "objstore-gcs-endpoint-model";
const BUCKET = "aisix-gcs-endpoint";
const PREFIX = "gcs-endpoint-e2e";
const CREDENTIAL_REF = "gcs-emulator";
const SERVICE_ACCOUNT = JSON.stringify({
  disable_oauth: true,
  client_email: "emulator@example.iam.gserviceaccount.com",
  private_key_id: "",
  private_key: "",
});

interface Upload {
  method: string;
  path: string;
  body: Buffer;
}

interface GcsReceiver {
  url: string;
  uploads: Upload[];
  close(): Promise<void>;
}

/** Accepts Cloud Storage XML-API object uploads and keeps each one. */
async function startGcsReceiver(): Promise<GcsReceiver> {
  const uploads: Upload[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      uploads.push({
        method: req.method ?? "",
        path: decodeURIComponent((req.url ?? "").split("?")[0]),
        body: Buffer.concat(chunks),
      });
      res.statusCode = 200;
      res.setHeader("etag", '"e2e-etag"');
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}`,
    uploads,
    async close() {
      await new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      );
    },
  };
}

describe("object_store exporter for gcs honours endpoint (#585)", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let gcs: GcsReceiver | undefined;
  let app: SpawnedApp | undefined;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    gcs = await startGcsReceiver();
  });

  afterAll(async () => {
    await app?.exit();
    await gcs?.close();
    await upstream?.close();
  });

  test(
    "the NDJSON upload for a request lands at the configured endpoint",
    async (ctx) => {
      if (!etcdReachable || !upstream || !gcs) {
        ctx.skip();
        return;
      }
      const slug = CREDENTIAL_REF.toUpperCase().replace(/[^A-Z0-9]/g, "_");
      app = await spawnApp({
        extraEnv: { [`OBJSTORE_CRED_${slug}_GCS_SERVICE_ACCOUNT_KEY`]: SERVICE_ACCOUNT },
      });
      const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
      await seed.createObservabilityExporter({
        name: "gcs-endpoint",
        enabled: true,
        kind: "object_store",
        provider: "gcs",
        bucket: BUCKET,
        prefix: PREFIX,
        endpoint: gcs.url,
        compression: "none",
        credential_ref: CREDENTIAL_REF,
      });
      const pk = await seed.createProviderKey({
        display_name: "objstore-gcs-endpoint-pk",
        secret: "sk-mock-objstore-gcs",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: MODEL,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [MODEL] });
      const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "hi" }] }),
      });
      await res.arrayBuffer();
      expect(res.status).toBe(200);
      const requestId = res.headers.get("x-aisix-request-id") ?? "";
      expect(requestId).not.toBe("");

      const deadline = Date.now() + 15_000;
      const mine = () =>
        gcs!.uploads.find((u) => u.body.toString("utf8").includes(requestId));
      while (!mine() && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 100));
      }
      const upload = mine();
      expect(
        upload,
        `no upload carrying ${requestId} reached the endpoint; saw ${gcs.uploads.length} request(s)`,
      ).toBeDefined();
      expect(upload!.method).toBe("PUT");
      expect(upload!.path.startsWith(`/${BUCKET}/${PREFIX}/`)).toBe(true);
      const event = JSON.parse(upload!.body.toString("utf8").trim().split("\n")[0]);
      expect(event.request_id).toBe(requestId);
    },
    60_000,
  );
});
