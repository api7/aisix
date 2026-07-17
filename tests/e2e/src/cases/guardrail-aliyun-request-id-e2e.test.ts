import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the Aliyun guardrail must preserve the upstream call's own
// diagnostics and make them joinable to the gateway request the caller
// holds (AISIX-Cloud#1060).
//
// The contract under test is a triage journey, not a field: a caller gets a
// 422 plus an `x-aisix-request-id`, hands that id to an operator, and the
// operator must be able to reach the matching record in the Aliyun console.
// That only works if ONE log line carries both ids — so these tests assert
// co-location on a single line, not the mere presence of each.
//
// Separate from `guardrail-aliyun-e2e.test.ts` because it needs the DP at
// `RUST_LOG=info` (a block logs at info; the shared harness default is warn).
//
// Kept as an E2E rather than a unit test because the correlation is produced
// by the request-scoped tracing span in aisix-proxy, while the Aliyun id is
// produced in aisix-guardrails: only a real binary serving a real request
// exercises the seam between them. The streaming case additionally covers
// the SSE generator, which hyper polls after the middleware has returned.

const CALLER_PLAINTEXT = "sk-aliyun-reqid-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const RISKY_MARKER = "aliyunreqidmarker";

// Fixed per service, so a log line proves WHICH upstream call it came from
// rather than just that some id was copied through.
const INPUT_REQUEST_ID = "ALIYUN-INPUT-REQ-0001";
const OUTPUT_REQUEST_ID = "ALIYUN-OUTPUT-REQ-0002";

// Matched content the real API echoes back in `RiskWords` / `RiskPositions`.
// The mock returns them exactly as green-cip does so the no-leak assertion
// has something real to catch (#153).
const RISK_WORDS = "riskwordleakcanary";

interface AliyunMock {
  baseUrl: string;
  close(): Promise<void>;
}

/**
 * Mock green-cip that mirrors the live endpoint's diagnostic surface: the
 * `x-acs-request-id` response header alongside the body's `RequestId`, a
 * multi-entry `Result` array, and the `RiskWords`/`RiskPositions` fields
 * that echo the offending text.
 */
async function startAliyunMock(): Promise<AliyunMock> {
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const params = new URLSearchParams(raw);
      const service = params.get("Service") ?? "";
      let content = "";
      try {
        const sp = JSON.parse(params.get("ServiceParameters") ?? "{}");
        content = typeof sp.content === "string" ? sp.content : "";
      } catch {
        // leave default
      }
      const isInput = service === "llm_query_moderation";
      const requestId = isInput ? INPUT_REQUEST_ID : OUTPUT_REQUEST_ID;
      const risky = content.includes(RISKY_MARKER);

      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.setHeader("x-acs-request-id", requestId);
      res.end(
        JSON.stringify({
          Code: 200,
          Message: "OK",
          RequestId: requestId,
          Data: {
            RiskLevel: risky ? "high" : "none",
            Result: risky
              ? [
                  {
                    Label: "inappropriate_oral",
                    Confidence: 100.0,
                    RiskWords: RISK_WORDS,
                    RiskPositions: [
                      { StartPos: 0, EndPos: 3, RiskWord: RISK_WORDS },
                    ],
                  },
                  { Label: "violent_incidents", Confidence: 100.0 },
                ]
              : [{ Label: "nonLabel" }],
          },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

/**
 * Poll the DP's captured output for a line satisfying `pred`. Log delivery
 * to the harness lags the HTTP response (the child's stderr is piped), so a
 * bare read right after the request is racy.
 */
async function waitForLogLine(
  app: SpawnedApp,
  pred: (line: string) => boolean,
  what: string,
): Promise<string> {
  const deadline = Date.now() + 5_000;
  let last = "";
  while (Date.now() < deadline) {
    last = app.output();
    const hit = last.split("\n").find(pred);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`timed out waiting for ${what}; DP output was:\n${last}`);
}

describe("aliyun guardrail e2e: upstream RequestId is preserved and correlatable", () => {
  let app: SpawnedApp | undefined;
  let benignUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let aliyun: AliyunMock | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    aliyun = await startAliyunMock();

    benignUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-clean",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a safe and clean reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 },
      },
    });

    // Streamed response carrying the marker — exercises the output hook from
    // inside the SSE generator, which is polled after the request-id
    // middleware has returned.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-reqid","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        `{"id":"strm-reqid","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"streamed ${RISKY_MARKER} payload"},"finish_reason":null}]}`,
        '{"id":"strm-reqid","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 50,
    });

    // A block logs at info; the harness default is warn.
    app = await spawnApp({ extraEnv: { RUST_LOG: "info" } });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const benignPk = await seed.createProviderKey({
      display_name: "aliyun-reqid-pk",
      secret: "sk-mock",
      api_base: `${benignUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aliyun-reqid-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: benignPk.id,
    });

    const streamPk = await seed.createProviderKey({
      display_name: "aliyun-reqid-stream-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aliyun-reqid-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["aliyun-reqid-e2e", "aliyun-reqid-stream-e2e"],
    });

    await seed.createGuardrail({
      name: "aliyun-reqid-guard",
      enabled: true,
      hook_point: "both",
      fail_open: false,
      kind: "aliyun_text_moderation",
      region: "cn-shanghai",
      endpoint: aliyun.baseUrl,
      access_key_id: "LTAI_E2E",
      access_key_secret: "e2e-secret",
      risk_level_threshold: "high",
      stream_processing_mode: "window",
      window_size: 16,
      window_overlap_size: 4,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await benignUpstream?.close();
    await streamUpstream?.close();
    await aliyun?.close();
  });

  test("input block: one log line joins x-aisix-request-id to Aliyun's RequestId", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const probe = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "aliyun-reqid-e2e",
          messages: [{ role: "user", content: `probe ${RISKY_MARKER}` }],
        }),
      });
      return probe.status === 422;
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "aliyun-reqid-e2e",
        messages: [{ role: "user", content: `please do ${RISKY_MARKER} now` }],
      }),
    });
    expect(res.status).toBe(422);

    // This is the only id the caller ever sees, and the only thing they can
    // quote to an operator.
    const gatewayRequestId = res.headers.get("x-aisix-request-id");
    expect(gatewayRequestId, "the 422 must carry x-aisix-request-id").toBeTruthy();

    const body = (await res.json()) as { error?: { type?: unknown } };
    expect(body.error?.type).toBe("content_filter");
    // The upstream id stays out of the caller's envelope by design: they get
    // the gateway id and nothing about who moderated them.
    expect(JSON.stringify(body)).not.toContain(INPUT_REQUEST_ID);
    expect(JSON.stringify(body)).not.toContain(RISK_WORDS);

    // The whole point: a SINGLE line carrying both ids. Two lines each
    // holding one id would not let an operator join them.
    const line = await waitForLogLine(
      app,
      (l) =>
        l.includes(`request_id=${gatewayRequestId}`) &&
        l.includes(`aliyun_request_id=${INPUT_REQUEST_ID}`),
      "the input-block line joining the gateway and Aliyun request ids",
    );

    // The two ids must stay distinguishable — that is exactly the confusion
    // this issue set out to remove.
    expect(line).toContain(`aliyun_risk_level=high`);
    expect(line).toContain(`aliyun_labels=inappropriate_oral,violent_incidents`);
    expect(line).toContain(`aliyun_code=200`);

    // #153: Aliyun echoes the offending text back; it must not reach a log.
    expect(app.output()).not.toContain(RISK_WORDS);
    expect(app.output()).not.toContain("RiskPositions");
  });

  test("streamed output block: the SSE generator keeps the request span", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "aliyun-reqid-stream-e2e",
        messages: [{ role: "user", content: "tell me something" }],
        stream: true,
      }),
    });
    expect(res.status).toBe(200);

    const gatewayRequestId = res.headers.get("x-aisix-request-id");
    expect(gatewayRequestId).toBeTruthy();

    const wire = await res.text();
    expect(wire).toContain("event: error");
    expect(wire).not.toContain(RISK_WORDS);

    // The output check runs inside the SSE generator, which hyper polls
    // after `ensure_request_id` has already returned. Without the span being
    // re-attached to the body stream this line would carry the Aliyun id but
    // no `request_id`, and the caller's id would dead-end.
    await waitForLogLine(
      app,
      (l) =>
        l.includes(`request_id=${gatewayRequestId}`) &&
        l.includes(`aliyun_request_id=${OUTPUT_REQUEST_ID}`),
      "the streamed-output block line joining the gateway and Aliyun request ids",
    );
  });
});
