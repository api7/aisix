import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { createServer as createNetServer } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an exporter's delivery error never carries the credentials it was
// configured with. The error text is what the gateway logs on every failed
// attempt and reports as the exporter's `last_error` in the managed-mode
// heartbeat, so a credential in it reaches the log pipeline and the console.
//
// Two ways one used to get there, one exporter each:
//   - the endpoint URL carries `user:pass@`, and a transport failure names
//     the endpoint;
//   - the receiver rejects the export and echoes the credential header back
//     in its error body.
const CALLER = "sk-exporter-error-redaction";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const USERINFO_USER = "otlp-e2e-user";
const USERINFO_PASS = "otlp-e2e-pass";
const HEADER_TOKEN = "otlp-e2e-header-token";
const SECRETS = [USERINFO_USER, USERINFO_PASS, HEADER_TOKEN];

/** A port with nothing listening on it. */
async function closedPort(): Promise<number> {
  const srv = createNetServer();
  await new Promise<void>((resolve) => srv.listen(0, "127.0.0.1", resolve));
  const port = (srv.address() as { port: number }).port;
  await new Promise<void>((resolve) => srv.close(() => resolve()));
  return port;
}

describe("exporter delivery errors carry no configured credential", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let echoing: Server | undefined;
  let unreachablePort = 0;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // A receiver that refuses every export and quotes the credential it was
    // sent — what a verbose collector or auth proxy does on a 401.
    echoing = createServer((req, res) => {
      req.resume();
      req.on("end", () => {
        res.writeHead(401, { "content-type": "text/plain" });
        res.end(`rejected credential ${req.headers.authorization ?? ""}`);
      });
    });
    await new Promise<void>((resolve) => echoing!.listen(0, "127.0.0.1", resolve));
    const echoPort = (echoing.address() as { port: number }).port;
    unreachablePort = await closedPort();

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "otlp-userinfo-unreachable",
      enabled: true,
      kind: "otlp_http",
      endpoint: `https://${USERINFO_USER}:${USERINFO_PASS}@127.0.0.1:${unreachablePort}/v1/traces`,
    });
    await seed.createObservabilityExporter({
      name: "otlp-echoing-receiver",
      enabled: true,
      kind: "otlp_http",
      endpoint: `http://127.0.0.1:${echoPort}/v1/traces`,
      headers: { authorization: `Bearer ${HEADER_TOKEN}` },
    });
    const pk = await seed.createProviderKey({
      display_name: "redaction-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "redaction-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Seeded last: its arrival implies everything above is in the snapshot.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["redaction-model"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await new Promise<void>((resolve) => (echoing ? echoing.close(() => resolve()) : resolve()));
  });

  test("the failed-delivery log lines name the receiver, not its credentials", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !echoing) return ctx.skip();

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: "redaction-model",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    await res.text();

    const unreachable = await waitForLogLine(
      app,
      (l) => l.includes("sink delivery failed") && l.includes("otlp-userinfo-unreachable"),
      "a retried delivery to the unreachable exporter",
      20_000,
    );
    // Still diagnosable: which receiver, and why.
    expect(unreachable).toContain(`https://***@127.0.0.1:${unreachablePort}/v1/traces`);

    const rejected = await waitForLogLine(
      app,
      (l) => l.includes("sink delivery dropped") && l.includes("otlp-echoing-receiver"),
      "a dropped delivery to the echoing receiver",
      20_000,
    );
    expect(rejected).toContain("HTTP 401");

    for (const secret of SECRETS) {
      expect(app.output(), `${secret} reached the gateway output`).not.toContain(secret);
    }
  });
});
