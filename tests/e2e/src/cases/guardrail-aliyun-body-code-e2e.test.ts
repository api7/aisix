import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: Aliyun's RPC API reports quota throttling (Code 588 EXCEED_QUOTA)
// and its own request timeout (Code 581 TIMEOUT) as HTTP 200 with the code
// in the body — never as HTTP 429 or a dropped connection. Both Aliyun
// guardrail kinds must file those under the same bypass reasons an HTTP 429
// or an elapsed `timeout_ms` gets, so an operator filtering on
// `aliyun_throttled` sees the throttling instead of a generic `aliyun_5xx`.
//
// Error-code tables: MultiModalGuard
// https://help.aliyun.com/zh/document_detail/2937221.html and
// TextModerationPlus https://help.aliyun.com/zh/document_detail/2671445.html.

const KEY = "sk-aliyun-body-code-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const LOGSTORE = "aliyun-body-code";

const QUOTA_MARKER = "bodycode588marker";
const TIMEOUT_MARKER = "bodycode581marker";

/**
 * green-cip stand-in serving both actions: HTTP 200 always, with the body
 * `Code` picked by a marker in the scanned content.
 */
async function startAliyunMock(): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let content = "";
      try {
        const sp = JSON.parse(new URLSearchParams(raw).get("ServiceParameters") ?? "{}");
        content = typeof sp.content === "string" ? sp.content : "";
      } catch {
        // leave default
      }
      const code = content.includes(QUOTA_MARKER)
        ? 588
        : content.includes(TIMEOUT_MARKER)
          ? 581
          : 200;
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.setHeader("x-acs-request-id", `ALIYUN-${code}`);
      res.end(
        JSON.stringify(
          code === 200
            ? {
                Code: 200,
                Message: "OK",
                RequestId: "ALIYUN-200",
                Data: { RiskLevel: "none", Suggestion: "pass", Result: [] },
              }
            : { Code: code, Message: "mock failure", RequestId: `ALIYUN-${code}` },
        ),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  };
}

const KINDS = [
  { kind: "aliyun_text_moderation", extra: { risk_level_threshold: "high" } },
  { kind: "aliyun_ai_guardrail", extra: {} },
] as const;

const CASES = [
  { code: 588, marker: QUOTA_MARKER, tag: "aliyun_throttled", meaning: "EXCEED_QUOTA" },
  { code: 581, marker: TIMEOUT_MARKER, tag: "aliyun_timeout", meaning: "TIMEOUT" },
] as const;

const modelFor = (kind: string, code: number) => `${kind}-code-${code}`;

describe("aliyun guardrails classify body Code 588 / 581", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let aliyun: Awaited<ReturnType<typeof startAliyunMock>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    aliyun = await startAliyunMock();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-body-code",
        object: "chat.completion",
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
      },
    });
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });

    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-aliyun-body-code",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    const pk = await seed.createProviderKey({
      display_name: "aliyun-body-code-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    for (const { kind, extra } of KINDS) {
      const guardrail = await seed.createGuardrail(
        {
          name: `${kind}-body-code`,
          enabled: true,
          hook_point: "input",
          fail_open: true,
          kind,
          region: "cn-shanghai",
          endpoint: aliyun.baseUrl,
          access_key_id: "LTAI_E2E",
          access_key_secret: "e2e-secret",
          ...extra,
        },
        { attach: false },
      );
      // One model per (kind, code) so each usage event is found by its
      // requested model, a field independent of the one under test.
      for (const { code } of CASES) {
        const model = await seed.createModel({
          display_name: modelFor(kind, code),
          provider: "openai",
          model_name: "gpt-4o-mini",
          provider_key_id: pk.id,
        });
        await seed.attachGuardrailToModel(guardrail.id, model.id);
      }
    }

    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
    await aliyun?.close();
  });

  for (const { kind } of KINDS) {
    for (const { code, marker, tag, meaning } of CASES) {
      test(`${kind}: HTTP 200 + Code ${code} bypasses as ${tag}`, async (ctx) => {
        if (!etcdReachable || !app || !sls) return ctx.skip();
        const model = modelFor(kind, code);

        const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
          body: JSON.stringify({
            model,
            messages: [{ role: "user", content: `please ${marker} now` }],
          }),
        });
        // fail_open: the guardrail could not decide, so the request passes.
        expect(res.status).toBe(200);

        const log = await waitForSlsLog(
          sls,
          LOGSTORE,
          (entry) => entry.get("requested_model") === model,
          `${model} usage event`,
        );
        expect(log.get("guardrail_bypassed_reason") ?? "").toBe(tag);

        // The log line names Aliyun's own code, which is what separates an
        // Aliyun-side 581 from our `timeout_ms` elapsing under the same tag.
        await waitForLogLine(
          app,
          (line) =>
            line.includes(`${kind}-body-code`) &&
            line.includes(`Code ${code} ${meaning}`),
          `${kind} log line naming Code ${code}`,
        );
      });
    }
  }
});
