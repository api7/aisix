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

// E2E: the `aliyun_ai_guardrail` guardrail (AISIX-Cloud#1070) moderates
// chat input and output against Aliyun's AI Guardrails product
// (`MultiModalGuard`) — a different product from TextModerationPlus,
// with a Suggestion-driven verdict computed by the Aliyun-side policy.
// We stand up a mock green-cip endpoint that answers Suggestion "block"
// for text containing RISKY_MARKER, "watch" for WATCH_MARKER, and
// "pass" otherwise, point the guardrail's `endpoint` override at it,
// and verify the full DP journey end-to-end with a real `aisix` binary
// + etcd + mock upstream. No control plane involved.
//
// References:
// - Aliyun MultiModalGuard <https://help.aliyun.com/zh/document_detail/2932956.html>
// - OpenAI / Azure `error.type: "content_filter"` envelope convention.

const CALLER_PLAINTEXT = "sk-aliyun-aig-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Letters survive percent-encoding inside the signed form body, so
// plain-letter markers are detectable in the raw request the mock receives.
const RISKY_MARKER = "aigblockmarker";
const WATCH_MARKER = "aigwatchmarker";

interface AigMockRequest {
  action: string;
  service: string;
  sessionId?: string;
  chatId?: string;
  content: string;
  raw: string;
}

interface AigMock {
  baseUrl: string;
  requests: AigMockRequest[];
  close(): Promise<void>;
}

// Minimal mock of the green-cip MultiModalGuard RPC endpoint. Parses the
// form-urlencoded body, extracts Action + Service + the ServiceParameters
// JSON (content / sessionId / chatId), and answers the documented
// response shape with an overall Data.Suggestion. It does NOT verify the
// signature — signature correctness is pinned by a known-vector unit
// test in the dispatcher crate.
async function startAigMock(): Promise<AigMock> {
  const requests: AigMockRequest[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const params = new URLSearchParams(raw);
      const action = params.get("Action") ?? "";
      const service = params.get("Service") ?? "";
      let content = "";
      let sessionId: string | undefined;
      let chatId: string | undefined;
      try {
        const sp = JSON.parse(params.get("ServiceParameters") ?? "{}");
        content = typeof sp.content === "string" ? sp.content : "";
        sessionId = typeof sp.sessionId === "string" ? sp.sessionId : undefined;
        chatId = typeof sp.chatId === "string" ? sp.chatId : undefined;
      } catch {
        // leave defaults
      }
      requests.push({ action, service, sessionId, chatId, content, raw });

      const suggestion = content.includes(RISKY_MARKER)
        ? "block"
        : content.includes(WATCH_MARKER)
          ? "watch"
          : "pass";
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          Code: 200,
          Message: "OK",
          RequestId: "mock-aig-req-1",
          Data: {
            Detail: [
              {
                Type: "contentModeration",
                Level: suggestion === "block" ? "high" : "none",
                Suggestion: suggestion,
                Result: [
                  {
                    Label:
                      suggestion === "block" ? "violent_incidents" : "nonLabel",
                    Level: suggestion === "block" ? "high" : "none",
                  },
                ],
              },
            ],
            Suggestion: suggestion,
          },
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

describe("aliyun AI guardrail e2e: MultiModalGuard suggestion drives the verdict", () => {
  let app: SpawnedApp | undefined;
  let benignUpstream: OpenAiUpstream | undefined;
  let riskyOutputUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let aig: AigMock | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    aig = await startAigMock();

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
        '{"id":"strm-aig","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        `{"id":"strm-aig","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"streamed ${RISKY_MARKER} payload"},"finish_reason":null}]}`,
        '{"id":"strm-aig","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "[DONE]",
      ],
      eventDelayMs: 50,
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const benignPk = await seed.createProviderKey({
      display_name: "aig-e2e-pk",
      secret: "sk-mock",
      api_base: `${benignUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aig-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: benignPk.id,
    });

    const riskyOutPk = await seed.createProviderKey({
      display_name: "aig-out-e2e-pk",
      secret: "sk-mock",
      api_base: `${riskyOutputUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aig-out-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: riskyOutPk.id,
    });

    const streamPk = await seed.createProviderKey({
      display_name: "aig-stream-e2e-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aig-stream-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["aig-e2e", "aig-out-e2e", "aig-stream-e2e"],
    });

    // One env-wide guardrail covering input + output. Small window so the
    // streaming case triggers windowed output calls (which must reuse the
    // stream's sessionId/chatId). `endpoint` points at the mock.
    await seed.createGuardrail({
      name: "aig-e2e-guard",
      enabled: true,
      hook_point: "both",
      fail_open: false,
      kind: "aliyun_ai_guardrail",
      region: "cn-shanghai",
      endpoint: aig.baseUrl,
      access_key_id: "LTAI_E2E",
      access_key_secret: "e2e-secret",
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
    await aig?.close();
  });

  test("suggestion=block on input → 422 content_filter, upstream never called", async (ctx) => {
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
          model: "aig-e2e",
          messages: [{ role: "user", content: `probe ${RISKY_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    // Benign request passes and hits the upstream via the Pro input service.
    const okBefore = benignUpstream.receivedRequests.length;
    const clean = await client.chat.completions.create({
      model: "aig-e2e",
      messages: [{ role: "user", content: "what is a safe and clean topic" }],
    });
    expect(clean.choices[0]?.message.role).toBe("assistant");
    expect(benignUpstream.receivedRequests.length).toBe(okBefore + 1);

    // Risky input is blocked BEFORE the upstream is called.
    const upstreamBefore = benignUpstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "aig-e2e",
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

    // The dispatcher spoke MultiModalGuard, not TextModerationPlus, and
    // used the Pro input service by default.
    const inputCall = aig!.requests.find(
      (r) => r.content.includes(RISKY_MARKER) && !r.service.startsWith("response"),
    );
    expect(inputCall, "expected an input-side MultiModalGuard call").toBeDefined();
    expect(inputCall?.action).toBe("MultiModalGuard");
    expect(inputCall?.service).toBe("query_security_check_pro");
  });

  test("suggestion=watch releases the request (only block blocks)", async (ctx) => {
    if (!etcdReachable || !app || !benignUpstream) {
      ctx.skip();
      return;
    }
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    const okBefore = benignUpstream.receivedRequests.length;
    const watched = await client.chat.completions.create({
      model: "aig-e2e",
      messages: [{ role: "user", content: `note ${WATCH_MARKER} here` }],
    });
    expect(watched.choices[0]?.message.role).toBe("assistant");
    expect(benignUpstream.receivedRequests.length).toBe(okBefore + 1);
  });

  test("suggestion=block on output → 422 after upstream call, session+chat ids sent", async (ctx) => {
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
        model: "aig-out-e2e",
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

    // The output call used the Pro response service and carried BOTH
    // correlation ids (sessionId for content concatenation, chatId for
    // the console's Q/A-round record).
    const outCall = aig!.requests.find(
      (r) =>
        r.service === "response_security_check_pro" &&
        r.content.includes(RISKY_MARKER),
    );
    expect(outCall, "expected a response_security_check_pro call").toBeDefined();
    expect(outCall?.sessionId, "output call must carry a sessionId").toBeTruthy();
    expect(outCall?.chatId, "output call must carry a chatId").toBeTruthy();
    expect(outCall?.chatId).toBe(outCall?.sessionId);
  });

  test("streaming risky output → SSE error event, stable session/chat ids across windows", async (ctx) => {
    if (!etcdReachable || !app || !streamUpstream) {
      ctx.skip();
      return;
    }
    const outBefore = aig!.requests.filter(
      (r) => r.service === "response_security_check_pro",
    ).length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "aig-stream-e2e",
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
    // sessionId AND chatId (the upstream's request id, "strm-aig"),
    // proving the windows of a single response correlate into one Aliyun
    // console record.
    const streamOutCalls = aig!.requests
      .filter((r) => r.service === "response_security_check_pro")
      .slice(outBefore);
    expect(streamOutCalls.length).toBeGreaterThan(0);
    const sessionIds = new Set(streamOutCalls.map((r) => r.sessionId));
    const chatIds = new Set(streamOutCalls.map((r) => r.chatId));
    expect(sessionIds.size).toBe(1);
    expect(chatIds.size).toBe(1);
    expect([...sessionIds][0]).toBe("strm-aig");
    expect([...chatIds][0]).toBe("strm-aig");
  });
});

describe("aliyun AI guardrail e2e: basic service level uses non-Pro codes", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let aig: AigMock | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    aig = await startAigMock();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-basic",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "clean" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "aig-basic-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "aig-basic-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["aig-basic-e2e"],
    });
    await seed.createGuardrail({
      name: "aig-basic-guard",
      enabled: true,
      hook_point: "both",
      fail_open: false,
      kind: "aliyun_ai_guardrail",
      region: "cn-shanghai",
      endpoint: aig.baseUrl,
      access_key_id: "LTAI_E2E",
      access_key_secret: "e2e-secret",
      service_level: "basic",
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await aig?.close();
  });

  test("basic tier calls query_security_check / response_security_check", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "aig-basic-e2e",
          messages: [{ role: "user", content: `probe ${RISKY_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    const services = new Set(aig!.requests.map((r) => r.service));
    expect(services.has("query_security_check")).toBe(true);
    expect(services.has("query_security_check_pro")).toBe(false);

    // A clean call flows through and moderates the output with the basic
    // response service.
    const clean = await client.chat.completions.create({
      model: "aig-basic-e2e",
      messages: [{ role: "user", content: "hello there" }],
    });
    expect(clean.choices[0]?.message.role).toBe("assistant");
    expect(
      aig!.requests.some((r) => r.service === "response_security_check"),
    ).toBe(true);
  });
});
