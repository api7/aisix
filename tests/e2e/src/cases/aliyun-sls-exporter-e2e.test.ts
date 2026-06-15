import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// L2 mock e2e (AISIX-Cloud#687): a dashboard-configured `aliyun_sls`
// exporter makes the real DP deliver request events to SLS over a signed
// PutLogs call. We stand up a mock SLS receiver, register the exporter, drive
// one chat, and assert the DP POSTed a correctly-shaped, signed lz4-protobuf
// PutLogs to the configured logstore.
//
// Scope boundary (deliberate): the mock cannot validate the SLS signature or
// decode the lz4+protobuf body, so this test pins the WIRING and wire SHAPE —
// path, the SLS headers, the lz4 framing, and that the Authorization carries
// the AccessKey the DP resolved from its environment (proving the
// credential_ref → env path, not a key on the kine config). The body's field
// mapping is covered by the Rust round-trip unit test (`sink::sls::tests`),
// and that a real Aliyun endpoint actually accepts the signed request is
// validated by the control-plane full-chain e2e (api7/AISIX-Cloud), not in
// this repo.

const CALLER_PLAINTEXT = "sk-sls-exporter-caller-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-sls-exporter";

// The exporter's credential_ref; the DP resolves it from the env vars the
// harness injects (SLS_CRED_<REF>_AK_{ID,SECRET}, ref upper-cased).
const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const SLS_LOGSTORE = "request-events";
const PUTLOGS_PATH = `/logstores/${SLS_LOGSTORE}/shards/lb`;

interface CapturedPutLogs {
  method: string;
  path: string;
  headers: IncomingMessage["headers"];
  bodyLen: number;
}

interface MockSls {
  /** Base URL the exporter points at (no `<project>.` prefix; sink uses it verbatim). */
  url: string;
  requests: CapturedPutLogs[];
  close(): Promise<void>;
}

async function startMockSls(): Promise<MockSls> {
  const requests: CapturedPutLogs[] = [];
  const server: Server = createServer((req, res) => {
    // Only the byte count matters here (lz4+protobuf decoding lives in the
    // Rust round-trip test), so accumulate length instead of the bytes.
    let bodyLen = 0;
    req.on("data", (c: Buffer) => {
      bodyLen += c.length;
    });
    req.on("end", () => {
      requests.push({
        method: req.method ?? "",
        path: (req.url ?? "").split("?")[0],
        headers: req.headers,
        bodyLen,
      });
      // SLS PutLogs returns 200 with an empty body on success.
      res.statusCode = 200;
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function seedRouting(admin: AdminClient, upstream: OpenAiUpstream) {
  const pk = await admin.createProviderKey({
    display_name: "sls-exporter-pk",
    secret: PROVIDER_SECRET,
    api_base: `${upstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: "sls-exporter-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: ["sls-exporter-model"],
  });
}

async function chat(app: SpawnedApp): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "sls-exporter-model",
      messages: [{ role: "user", content: "hello sls" }],
    }),
  });
}

async function waitForPutLogs(
  sls: MockSls,
  timeoutMs = 10_000,
): Promise<CapturedPutLogs> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = sls.requests.find((r) => r.path === PUTLOGS_PATH);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no PutLogs to ${PUTLOGS_PATH} recorded within ${timeoutMs}ms`);
}

function headerValue(
  headers: IncomingMessage["headers"],
  name: string,
): string {
  const v = headers[name];
  return Array.isArray(v) ? (v[0] ?? "") : (v ?? "");
}

describe("aliyun_sls exporter e2e (#687): DP delivers a signed PutLogs to SLS", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    sls = await startMockSls();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await upstream?.close();
    await sls?.close();
  });

  test(
    "a configured aliyun_sls exporter posts a signed lz4-protobuf PutLogs per chat",
    async (ctx) => {
      if (!etcdReachable || !upstream || !sls) {
        ctx.skip();
        return;
      }
      const app = await spawnApp({
        // The AccessKey rides the DP's own env, never the kine config.
        extraEnv: {
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
        },
      });
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "mock-sls",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: SLS_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
      });
      await seedRouting(admin, upstream);

      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app);
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const res = await chat(app);
      expect(res.status).toBe(200);
      await res.text();

      const put = await waitForPutLogs(sls);

      // Correct logstore + verb.
      expect(put.method).toBe("POST");
      expect(put.path).toBe(PUTLOGS_PATH);
      // SLS PutLogs wire shape: protobuf body, lz4 block compression, API ver.
      expect(headerValue(put.headers, "content-type")).toBe("application/x-protobuf");
      expect(headerValue(put.headers, "x-log-compresstype")).toBe("lz4");
      expect(headerValue(put.headers, "x-log-apiversion")).toBe("0.6.0");
      // Uncompressed size present and non-trivial (a real LogGroup was built).
      expect(Number(headerValue(put.headers, "x-log-bodyrawsize"))).toBeGreaterThan(0);
      expect(put.bodyLen).toBeGreaterThan(0);
      // Signature carries the AccessKey id the DP resolved from its env —
      // proves credential_ref → env resolution wired into signing, and that
      // the secret never appears on the wire.
      const auth = headerValue(put.headers, "authorization");
      expect(auth.startsWith(`LOG ${MOCK_AK_ID}:`)).toBe(true);
      expect(auth.includes(MOCK_AK_SECRET)).toBe(false);
    },
    60_000,
  );
});
