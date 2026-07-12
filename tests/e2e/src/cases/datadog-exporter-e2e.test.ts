import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
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

// L2 mock e2e (api7/ai-gateway#57, AISIX-Cloud#688): a dashboard-configured
// `datadog` exporter makes the real DP deliver request events to Datadog's
// native Logs HTTP intake over a gzip JSON-array POST. We stand up a mock
// Datadog intake (the SLS suite's in-test-receiver pattern), register the
// exporter, drive one chat, and assert the DP POSTed a correctly-shaped,
// API-key-bearing request to `/api/v2/logs`.
//
// Scope boundary (deliberate), mirroring the SLS L2 test: this pins the
// WIRING and the on-the-wire SHAPE the intake observes — the path, the
// `Content-Encoding: gzip` framing, that the `DD-API-KEY` header carries the
// key the DP resolved from its environment (proving the credential_ref → env
// path, not a key on the kine config), and that the gunzipped JSON-array body
// carries the Datadog reserved attributes + the OTel GenAI semconv token
// fields. The full field-mapping matrix is covered by the Rust round-trip unit
// tests (`sink::datadog::tests`); that a real Datadog site accepts the request
// is validated by the control-plane full-chain e2e (api7/AISIX-Cloud), not here.
//
// The harness binds the in-test mock to a free loopback port and points `site`
// at `127.0.0.1:<port>`, exactly as the SLS / OTLP mock-edge tests point their
// endpoint at `http://127.0.0.1:<port>`. The `datadog` `site` validator admits
// a loopback host with an optional `:port` via the same `(:[0-9]+)?` regex
// group as the SLS / OTLP bypasses (api7/ai-gateway#548, fixed in this PR), so
// the Admin API accepts it.

const CALLER_PLAINTEXT = "sk-datadog-exporter-caller-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-datadog-exporter";

// The exporter's credential_ref; the DP resolves it from the env var the
// harness injects (DD_CRED_<REF>_API_KEY, ref upper-cased, non-alnum → `_`).
const CREDENTIAL_REF = "e2e";
// The mock accepts any key — this is the test key the DP must surface in the
// DD-API-KEY header (and NOWHERE else). Not a real Datadog credential.
const DD_API_KEY = "dd-e2e-test-key-7f3a2b";

const DD_SERVICE = "aisix-e2e";
const DD_TAGS = ["team:platform", "tier:e2e"];
const INTAKE_PATH = "/api/v2/logs";

// Unique tokens planted in the request + the mock upstream response, so the
// content-capture assertion can prove which made it into the log body. The DP
// captures the prompt (request body, carrying PROMPT_TOKEN) and the assembled
// response (assistant content, carrying RESPONSE_TOKEN) only under
// `content_mode = full`.
const PROMPT_TOKEN = "dd-prompt-tok-9f3a2b";
const RESPONSE_TOKEN = "dd-response-tok-7c1d8e";

interface CapturedLog {
  method: string;
  path: string;
  headers: IncomingMessage["headers"];
  /** The gunzipped, JSON-parsed log objects (the intake body is a JSON array). */
  logs: unknown[];
  /** The raw decompressed body text, for substring (content-leak) assertions. */
  bodyText: string;
}

interface MockDatadog {
  /** `host:port` the exporter's `site` points at (the sink builds `http://<site>/api/v2/logs`). */
  site: string;
  requests: CapturedLog[];
  close(): Promise<void>;
}

/** Stand up a mock Datadog Logs intake on a free loopback port. */
async function startMockDatadog(): Promise<MockDatadog> {
  const requests: CapturedLog[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "").split("?")[0];
      if (req.method === "POST" && path === INTAKE_PATH) {
        const compressed = Buffer.concat(chunks);
        // The intake advertises `Content-Encoding: gzip`; gunzip before parse.
        // If the body isn't valid gzip JSON the capture stays empty and the
        // poll below times out — a loud failure, never a false green.
        let logs: unknown[] = [];
        let bodyText = "";
        try {
          bodyText = gunzipSync(compressed).toString("utf8");
          const parsed: unknown = JSON.parse(bodyText);
          if (Array.isArray(parsed)) logs = parsed;
        } catch {
          // leave logs empty / bodyText as-is — the assertion side will fail
          // visibly rather than silently pass.
        }
        requests.push({
          method: req.method ?? "",
          path,
          headers: req.headers,
          logs,
          bodyText,
        });
      }
      // Datadog's logs intake answers 202 Accepted on success.
      res.statusCode = 202;
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    site: `127.0.0.1:${port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function seedRouting(admin: AdminClient, upstream: OpenAiUpstream, model: string) {
  const pk = await admin.createProviderKey({
    display_name: `${model}-pk`,
    secret: PROVIDER_SECRET,
    api_base: `${upstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: model,
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [model],
  });
}

async function chat(app: SpawnedApp, model: string, content: string): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content }],
    }),
  });
}

async function waitForIntake(
  dd: MockDatadog,
  timeoutMs = 10_000,
): Promise<CapturedLog> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = dd.requests.find((r) => r.path === INTAKE_PATH && r.logs.length > 0);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no decodable POST to ${INTAKE_PATH} recorded within ${timeoutMs}ms`);
}

function headerValue(headers: IncomingMessage["headers"], name: string): string {
  const v = headers[name];
  return Array.isArray(v) ? (v[0] ?? "") : (v ?? "");
}

/** Narrow one captured log object to a plain record for field assertions. */
function asRecord(log: unknown): Record<string, unknown> {
  expect(log, "log entry must be a JSON object").toBeTypeOf("object");
  return log as Record<string, unknown>;
}

describe("datadog exporter e2e (#688): DP delivers a gzip JSON intake to Datadog", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let dd: MockDatadog | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    // Plant the response token in the mock upstream's assistant content so the
    // content-capture test can search for it in the `full` log body.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "mock-datadog-1",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "mock-model",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `sure, ${RESPONSE_TOKEN} noted` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    dd = await startMockDatadog();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await upstream?.close();
    await dd?.close();
  });

  test(
    "a configured datadog exporter posts a gzip JSON-array intake with the resolved DD-API-KEY",
    async (ctx) => {
      if (!etcdReachable || !upstream || !dd) {
        ctx.skip();
        return;
      }
      const app = await spawnApp({
        // The API key rides the DP's own env, never the kine config.
        extraEnv: {
          [`DD_CRED_${CREDENTIAL_REF.toUpperCase()}_API_KEY`]: DD_API_KEY,
        },
      });
      apps.push(app);
      // Deliberately seeds via the Admin API: deprecation-window coverage.
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "mock-datadog",
        enabled: true,
        kind: "datadog",
        site: dd.site,
        credential_ref: CREDENTIAL_REF,
        service: DD_SERVICE,
        tags: DD_TAGS,
        // Default privacy posture: operational metadata only, never content.
        content_mode: "metadata_only",
      });
      await seedRouting(admin, upstream, "datadog-exporter-model");

      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app, "datadog-exporter-model", "hello datadog");
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const res = await chat(app, "datadog-exporter-model", "hello datadog");
      expect(res.status).toBe(200);
      await res.text();

      const intake = await waitForIntake(dd);

      // Correct intake verb + path.
      expect(intake.method).toBe("POST");
      expect(intake.path).toBe(INTAKE_PATH);
      // Datadog Logs intake wire shape: gzip-compressed JSON.
      expect(headerValue(intake.headers, "content-encoding")).toBe("gzip");
      expect(headerValue(intake.headers, "content-type")).toBe("application/json");
      // The DD-API-KEY header carries EXACTLY the key the DP resolved from its
      // env — proves credential_ref → DD_CRED_<REF>_API_KEY resolution wired
      // into delivery, and that the key was not something else.
      expect(headerValue(intake.headers, "dd-api-key")).toBe(DD_API_KEY);

      // Body is a JSON array of log objects (one per request event).
      expect(Array.isArray(intake.logs)).toBe(true);
      expect(intake.logs.length).toBeGreaterThan(0);
      const log = asRecord(intake.logs[0]);

      // Datadog reserved attributes set by the sink.
      expect(log.ddsource).toBeTypeOf("string");
      expect((log.ddsource as string).length).toBeGreaterThan(0);
      expect(log.service).toBe(DD_SERVICE);
      // ddtags is the configured tags, comma-joined.
      expect(log.ddtags).toBe(DD_TAGS.join(","));

      // OTel GenAI semconv token fields ride the log (same keys an OTLP span
      // would carry). Counts come from the mock upstream's usage block.
      expect(log["gen_ai.usage.input_tokens"]).toBe(5);
      expect(log["gen_ai.usage.output_tokens"]).toBe(3);
      // A model dimension is present under a GenAI / aisix key (the upstream
      // echoes `mock-model`); assert the response model semconv field carries it.
      expect(log["gen_ai.response.model"]).toBe("mock-model");

      // The API key must appear ONLY in the header — never anywhere in the body.
      expect(intake.bodyText.includes(DD_API_KEY)).toBe(false);
      // Metadata-only posture: no captured prompt / response fields at all.
      expect(log["gen_ai.prompt"]).toBeUndefined();
      expect(log["gen_ai.completion"]).toBeUndefined();
    },
    60_000,
  );

  test(
    "content_mode=full ships gen_ai.prompt/completion; metadata_only ships neither",
    async (ctx) => {
      if (!etcdReachable || !upstream || !dd) {
        ctx.skip();
        return;
      }
      // A fresh mock so this test only sees its own request, and the wire-shape
      // test's request can't bleed into the content assertions.
      const ddFull = await startMockDatadog();
      const ddMeta = await startMockDatadog();
      const app = await spawnApp({
        extraEnv: {
          [`DD_CRED_${CREDENTIAL_REF.toUpperCase()}_API_KEY`]: DD_API_KEY,
        },
      });
      apps.push(app);
      try {
        const admin = new AdminClient(app.adminUrl, app.adminKey);
        // Two exporters on the same DP: one captures content, one does not.
        await admin.createObservabilityExporter({
          name: "datadog-full",
          enabled: true,
          kind: "datadog",
          site: ddFull.site,
          credential_ref: CREDENTIAL_REF,
          service: DD_SERVICE,
          content_mode: "full",
        });
        await admin.createObservabilityExporter({
          name: "datadog-meta",
          enabled: true,
          kind: "datadog",
          site: ddMeta.site,
          credential_ref: CREDENTIAL_REF,
          service: DD_SERVICE,
          content_mode: "metadata_only",
        });
        await seedRouting(admin, upstream, "datadog-content-model");

        await waitConfigPropagation(async () => {
          try {
            const r = await chat(
              app,
              "datadog-content-model",
              `please remember the token ${PROMPT_TOKEN}`,
            );
            await r.text();
            return r.status === 200;
          } catch {
            return false;
          }
        });

        const res = await chat(
          app,
          "datadog-content-model",
          `please remember the token ${PROMPT_TOKEN}`,
        );
        expect(res.status).toBe(200);
        await res.text();

        const fullIntake = await waitForIntake(ddFull);
        const metaIntake = await waitForIntake(ddMeta);

        const fullLog = asRecord(fullIntake.logs[0]);
        // Full capture carries both the prompt and the assembled response.
        expect(fullLog["gen_ai.prompt"]).toBeTypeOf("string");
        expect(fullLog["gen_ai.completion"]).toBeTypeOf("string");
        expect(fullIntake.bodyText).toContain(PROMPT_TOKEN);
        expect(fullIntake.bodyText).toContain(RESPONSE_TOKEN);

        // Metadata-only carries NEITHER the prompt nor the response — content
        // gating holds end-to-end through the real binary.
        const metaLog = asRecord(metaIntake.logs[0]);
        expect(metaLog["gen_ai.prompt"]).toBeUndefined();
        expect(metaLog["gen_ai.completion"]).toBeUndefined();
        expect(metaIntake.bodyText).not.toContain(PROMPT_TOKEN);
        expect(metaIntake.bodyText).not.toContain(RESPONSE_TOKEN);
      } finally {
        await ddFull.close();
        await ddMeta.close();
      }
    },
    60_000,
  );
});
