import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: the `aliyun_text_moderation` guardrail (#603) moderates chat
// input and output against Aliyun's content-safety guardrail
// (`TextModerationPlus`). We stand up a mock green-cip endpoint that
// grades any text containing RISKY_MARKER as RiskLevel "high" and
// everything else "none", point the guardrail's `endpoint` override at
// it, and verify the full DP journey end-to-end with a real `aisix`
// binary + etcd + mock upstream. No control plane involved.
//
// References:
// - Aliyun TextModerationPlus (llm_query_moderation / llm_response_moderation)
//   <https://help.aliyun.com/zh/document_detail/2671445.html>
// - OpenAI / Azure `error.type: "content_filter"` envelope convention.

const CALLER_PLAINTEXT = "sk-aliyun-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Letters survive percent-encoding inside the signed form body, so a
// plain-letter marker is detectable in the raw request the mock receives.
const RISKY_MARKER = "aliyunriskymarker";

interface AliyunMockRequest {
  service: string;
  sessionId?: string;
  content: string;
  raw: string;
}

interface AliyunMock {
  baseUrl: string;
  requests: AliyunMockRequest[];
  close(): Promise<void>;
}

// Minimal mock of the green-cip TextModerationPlus RPC endpoint. Parses
// the form-urlencoded body, extracts Service + the ServiceParameters JSON
// (content + sessionId), and returns "high" when the content carries the
// marker. It does NOT verify the signature — signature correctness is
// pinned by a known-vector unit test in the dispatcher crate.
async function startAliyunMock(): Promise<AliyunMock> {
  const requests: AliyunMockRequest[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const params = new URLSearchParams(raw);
      const service = params.get("Service") ?? "";
      let content = "";
      let sessionId: string | undefined;
      try {
        const sp = JSON.parse(params.get("ServiceParameters") ?? "{}");
        content = typeof sp.content === "string" ? sp.content : "";
        sessionId = typeof sp.sessionId === "string" ? sp.sessionId : undefined;
      } catch {
        // leave defaults
      }
      requests.push({ service, sessionId, content, raw });

      const risky = content.includes(RISKY_MARKER);
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          Code: 200,
          Data: {
            RiskLevel: risky ? "high" : "none",
            Result: [{ Label: risky ? "violent_content" : "nonLabel" }],
          },
          RequestId: "mock-req-1",
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

describe("aliyun guardrail e2e: TextModerationPlus blocks risky input/output", () => {
  let app: SpawnedApp | undefined;
  let benignUpstream: OpenAiUpstream | undefined;
  let riskyOutputUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let aliyun: AliyunMock | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    aliyun = await startAliyunMock();

    // Clean upstream for the input-side cases (its output is benign so the
    // output hook always passes for these models).
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

    // Upstream whose RESPONSE carries the risky marker — the input is
    // innocent, so this exercises the output hook.
    riskyOutputUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-risky-out",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `here it is: ${RISKY_MARKER}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 6, total_tokens: 11 },
      },
    });

    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        '{"id":"strm-risky","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        `{"id":"strm-risky","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"streamed ${RISKY_MARKER} payload"},"finish_reason":null}]}`,
        '{"id":"strm-risky","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 50,
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const benignPk = await seed.createProviderKey({
      display_name: "aliyun-e2e-pk",
      secret: "sk-mock",
      api_base: `${benignUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aliyun-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: benignPk.id,
    });

    const riskyOutPk = await seed.createProviderKey({
      display_name: "aliyun-out-e2e-pk",
      secret: "sk-mock",
      api_base: `${riskyOutputUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aliyun-out-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: riskyOutPk.id,
    });

    const streamPk = await seed.createProviderKey({
      display_name: "aliyun-stream-e2e-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aliyun-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["aliyun-e2e", "aliyun-out-e2e", "aliyun-stream-e2e"],
    });

    // One env-wide guardrail covering input + output. Small window so the
    // streaming case triggers a windowed output call (and reuses the
    // stream's sessionId across windows). `endpoint` points at the mock.
    await seed.createGuardrail({
      name: "aliyun-e2e-guard",
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
    await riskyOutputUpstream?.close();
    await streamUpstream?.close();
    await aliyun?.close();
  });

  test("risky input → 422 content_filter, upstream never called", async (ctx) => {
    if (!etcdReachable || !app || !benignUpstream) {
      ctx.skip();
      return;
    }
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Gate on the guardrail being live: poll with a risky prompt until 422.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "aliyun-e2e",
          messages: [{ role: "user", content: `probe ${RISKY_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    // Benign request passes and hits the upstream.
    const okBefore = benignUpstream.receivedRequests.length;
    const clean = await client.chat.completions.create({
      model: "aliyun-e2e",
      messages: [{ role: "user", content: "what is a safe and clean topic" }],
    });
    expect(clean.choices[0]?.message.role).toBe("assistant");
    expect(benignUpstream.receivedRequests.length).toBe(okBefore + 1);

    // Risky input is blocked BEFORE the upstream is called.
    const upstreamBefore = benignUpstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "aliyun-e2e",
        messages: [{ role: "user", content: `please do ${RISKY_MARKER} now` }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    // The matched content must not leak back to the caller (#153).
    expect(JSON.stringify(caught.error ?? {})).not.toContain(RISKY_MARKER);
    expect(benignUpstream.receivedRequests.length).toBe(upstreamBefore);
  });

  test("risky model output → 422 content_filter after upstream call", async (ctx) => {
    if (!etcdReachable || !app || !riskyOutputUpstream) {
      ctx.skip();
      return;
    }
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const upstreamBefore = riskyOutputUpstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "aliyun-out-e2e",
        messages: [{ role: "user", content: "an innocent question" }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    expect(JSON.stringify(caught.error ?? {})).not.toContain(RISKY_MARKER);
    // Output hook runs AFTER the upstream → the upstream IS hit.
    expect(riskyOutputUpstream.receivedRequests.length).toBe(upstreamBefore + 1);

    // The dispatcher called Aliyun's output service with a sessionId
    // (derived from the response id) so windows of one stream correlate.
    const outCall = aliyun!.requests.find(
      (r) => r.service === "llm_response_moderation" && r.content.includes(RISKY_MARKER),
    );
    expect(outCall, "expected an llm_response_moderation call").toBeDefined();
    expect(outCall?.sessionId, "output call must carry a sessionId").toBeTruthy();
  });

  test("streaming risky output → SSE error event, no [DONE] (windowed)", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream) {
      ctx.skip();
      return;
    }
    const outBefore = aliyun!.requests.filter(
      (r) => r.service === "llm_response_moderation",
    ).length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "aliyun-stream-e2e",
        messages: [{ role: "user", content: "tell me something" }],
        stream: true,
      }),
    });

    expect(res.status).toBe(200);
    const wire = await res.text();
    expect(wire).toContain("event: error");
    expect(wire).not.toContain("data: [DONE]");

    const errEventIdx = wire.indexOf("event: error\n");
    const afterErr = wire.slice(errEventIdx + "event: error\n".length);
    const dataLine = afterErr
      .split("\n")
      .find((l: string) => l.startsWith("data: "));
    expect(dataLine).toBeDefined();
    const parsed = JSON.parse(dataLine!.slice("data: ".length)) as {
      error?: { type?: unknown };
    };
    expect(parsed.error?.type).toBe("content_filter");

    // Every windowed output call for this stream must carry one stable
    // sessionId (the upstream's request id, "strm-risky"), proving the
    // chunks of a single response correlate at Aliyun.
    const streamOutCalls = aliyun!.requests
      .filter((r) => r.service === "llm_response_moderation")
      .slice(outBefore);
    expect(streamOutCalls.length).toBeGreaterThan(0);
    const sessionIds = new Set(streamOutCalls.map((r) => r.sessionId));
    expect(sessionIds.size).toBe(1);
    expect([...sessionIds][0]).toBe("strm-risky");
  });
});
