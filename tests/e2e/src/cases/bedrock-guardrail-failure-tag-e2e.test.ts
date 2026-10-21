import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the failure tag a Bedrock guardrail records names the cause the way
// every other remote guardrail kind does.
//
// A non-429 4xx from ApplyGuardrail — a denied credential, a guardrail id
// that does not exist, a request AWS rejects — is something the operator
// has to fix, and other kinds tag it `<kind>_config_error`. Bedrock folded it
// into `bedrock_5xx`, the tag for a server-side or transport failure, so an
// alert on outages fired on a misconfiguration and a dashboard filtering on
// configuration errors never saw Bedrock's.

const CALLER = "sk-bedrock-failure-tag-e2e";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

/** guardrail id → the AWS error ApplyGuardrail answers it with. */
const FAILURES: Record<string, { status: number; type: string }> = {
  deniedgr0001: { status: 403, type: "AccessDeniedException" },
  invalidgr001: { status: 400, type: "ValidationException" },
  missinggr001: { status: 404, type: "ResourceNotFoundException" },
  brokengr0001: { status: 500, type: "InternalServerException" },
};

/** A guardrail whose role the mock STS refuses to let the gateway assume:
 * the call is never signed, so ApplyGuardrail is never reached. */
const DENIED_ROLE_GUARDRAIL = "deniedrole01";

async function startDenyingSts(): Promise<{ url: string; server: Server; calls: () => number }> {
  let calls = 0;
  const server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      calls += 1;
      res.writeHead(403, { "content-type": "text/xml" });
      res.end(
        '<ErrorResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><Error><Type>Sender</Type>' +
          "<Code>AccessDenied</Code><Message>refused by the mock</Message></Error>" +
          "<RequestId>req-sts</RequestId></ErrorResponse>",
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { url: `http://127.0.0.1:${addr.port}`, server, calls: () => calls };
}

async function startFailingBedrock(): Promise<{ url: string; server: Server }> {
  const server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      const id = /\/guardrail\/([^/]+)\//.exec(req.url ?? "")?.[1] ?? "";
      const failure = FAILURES[id] ?? { status: 500, type: "InternalServerException" };
      res.writeHead(failure.status, {
        "content-type": "application/json",
        "x-amzn-errortype": failure.type,
      });
      res.end(JSON.stringify({ message: `refused by the mock (${failure.type})` }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { url: `http://127.0.0.1:${addr.port}`, server };
}

describe("Bedrock guardrail failure tags", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let bedrock: Awaited<ReturnType<typeof startFailingBedrock>> | undefined;
  let sts: Awaited<ReturnType<typeof startDenyingSts>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-bft",
        object: "chat.completion",
        model: "gpt-4o-mini",
        choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });
    bedrock = await startFailingBedrock();
    sts = await startDenyingSts();
    app = await spawnApp({
      extraEnv: {
        AWS_ENDPOINT_URL_STS: sts.url,
        AWS_EC2_METADATA_DISABLED: "true",
        AWS_CONFIG_FILE: "/nonexistent/aisix-e2e-aws-config",
        AWS_SHARED_CREDENTIALS_FILE: "/nonexistent/aisix-e2e-aws-credentials",
      },
      extra: { bedrock_endpoint_url: bedrock.url },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "bft-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const models: string[] = [];
    for (const id of [...Object.keys(FAILURES), DENIED_ROLE_GUARDRAIL]) {
      const model = await seed.createModel({
        display_name: `bft-${id}`,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      const gr = await seed.createGuardrail(
        {
          name: `bft-${id}`,
          enabled: true,
          hook_point: "input",
          fail_open: true,
          kind: "bedrock",
          guardrail_id: id,
          guardrail_version: "DRAFT",
          region: "us-east-1",
          aws_credentials: {
            kind: "static",
            access_key_id: "AKIDFAILURETAG000001",
            secret_access_key: "s",
            ...(id === DENIED_ROLE_GUARDRAIL ? { role_arn: "arn:aws:iam::111122223333:role/Denied" } : {}),
          },
          latency_mode: { kind: "serial" },
        },
        { attach: false },
      );
      await seed.attachGuardrailToModel(gr.id, model.id);
      models.push(`bft-${id}`);
    }
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: models });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      const body = await r.text();
      return r.status === 200 && models.every((m) => body.includes(m));
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    for (const s of [bedrock?.server, sts?.server]) {
      if (s) await new Promise<void>((resolve) => s.close(() => resolve()));
    }
  });

  async function bypassReasonOf(id: string): Promise<Record<string, number>> {
    const before = await scrapeMetrics(app!.metricsUrl);
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: `bft-${id}`, messages: [{ role: "user", content: "hello" }] }),
    });
    expect(res.status, await res.text()).toBe(200);
    const after = await scrapeMetrics(app!.metricsUrl);
    const out: Record<string, number> = {};
    for (const reason of ["bedrock_config_error", "bedrock_5xx"]) {
      out[reason] = metricDelta(before, after, "aisix_guardrail_bypasses_total", { reason });
    }
    return out;
  }

  for (const [id, { status, type }] of Object.entries(FAILURES)) {
    const expected = status >= 500 ? "bedrock_5xx" : "bedrock_config_error";
    test(`a ${status} ${type} fails open as ${expected}`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const deltas = await bypassReasonOf(id);
      expect(deltas).toEqual({
        bedrock_config_error: expected === "bedrock_config_error" ? 1 : 0,
        bedrock_5xx: expected === "bedrock_5xx" ? 1 : 0,
      });
    });
  }

  test("credentials STS refuses to issue fail open as bedrock_config_error", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = sts!.calls();
    const deltas = await bypassReasonOf(DENIED_ROLE_GUARDRAIL);
    expect(sts!.calls()).toBeGreaterThan(before);
    expect(deltas).toEqual({ bedrock_config_error: 1, bedrock_5xx: 0 });
  });
});
