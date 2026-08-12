import { createHash } from "node:crypto";
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

// E2E: the provider's own response id must be reachable from the plain
// application log, joined to the gateway request id the caller holds
// (AISIX-Cloud#1289).
//
// The contract is a triage journey, not a field: a caller reports a bad
// answer and quotes their `x-aisix-request-id`; the operator must be able to
// turn that into the id the provider's own console indexes by. That only
// works if ONE line carries both, so every assertion here is co-location on a
// single line rather than the mere presence of each id somewhere in the
// output.
//
// Kept as an E2E rather than a unit test because the two ids are produced in
// different places — the gateway id by the request-scoped tracing span in
// aisix-proxy, the provider id by the response/stream readers — and only a
// real binary serving a real request exercises the seam. The streamed case
// additionally covers the SSE generator, which hyper polls long after the
// access-log line for that request was already written: that is precisely the
// case the one-line-per-request access log structurally cannot carry, and the
// reason the per-attempt `provider call completed` line exists.

const CALLER_PLAINTEXT = "sk-provider-request-id-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Fixed per upstream, so a matching log line proves WHICH provider call it
// came from rather than merely that some id was printed.
const NONSTREAM_ID = "chatcmpl-e2e-nonstream-1289";
const STREAM_ID = "chatcmpl-e2e-stream-1289";
const RESPONSES_ID = "resp_e2e_1289";

/**
 * Poll the DP's captured output for a line satisfying `pred`. Log delivery to
 * the harness lags the HTTP response (the child's stderr is piped), so a bare
 * read right after the request is racy.
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

async function call(
  app: SpawnedApp,
  path: string,
  body: unknown,
): Promise<{ status: number; requestId: string; text: string }> {
  const res = await fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const requestId = res.headers.get("x-aisix-request-id") ?? "";
  // Drain: a streamed body must run to completion before the end-of-stream
  // telemetry (and its log line) exists at all.
  const text = await res.text();
  return { status: res.status, requestId, text };
}

describe("provider_request_id reaches the access log and the plain log", () => {
  let app: SpawnedApp | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    nonStreamUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: NONSTREAM_ID,
        object: "chat.completion",
        created: 1_700_000_000,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hi" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
      },
    });

    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        `{"id":"${STREAM_ID}","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
        `{"id":"${STREAM_ID}","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
        `{"id":"${STREAM_ID}","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`,
        "[DONE]",
      ],
      eventDelayMs: 2,
    });

    // `/v1/responses` is one of the endpoints that recorded no id before
    // #1289 — it must now behave like chat does.
    responsesUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: RESPONSES_ID,
        object: "response",
        model: "gpt-4o-mini",
        output: [
          {
            type: "message",
            id: "msg_e2e",
            role: "assistant",
            content: [{ type: "output_text", text: "hi" }],
          },
        ],
        usage: { input_tokens: 5, output_tokens: 1, total_tokens: 6 },
      },
    });

    // The access log and the provider-call line are INFO; the shared harness
    // default is warn.
    app = await spawnApp({ extraEnv: { RUST_LOG: "info" } });
    seed = new SeedClient(etcd, app.etcdPrefix);

    for (const [name, upstream] of [
      ["pri-nonstream", nonStreamUpstream],
      ["pri-stream", streamUpstream],
      ["pri-responses", responsesUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: `${name}-e2e`,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["pri-nonstream-e2e", "pri-stream-e2e", "pri-responses-e2e"],
    });

    // The caller key is seeded last, so it authenticating implies every
    // resource above is in the snapshot. Gating on a chat call instead would
    // exercise the behavior under test and turn an assertion failure into a
    // timeout.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await nonStreamUpstream?.close();
    await streamUpstream?.close();
    await responsesUpstream?.close();
  });

  test("non-streaming: the access-log line joins both ids", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await call(app, "/v1/chat/completions", {
      model: "pri-nonstream-e2e",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    expect(res.requestId, "the 200 must carry x-aisix-request-id").toBeTruthy();

    const line = await waitForLogLine(
      app,
      (l) =>
        l.includes("proxy request completed") &&
        l.includes(`request_id="${res.requestId}"`),
      "the access-log line for this request",
    );
    // The gap #1289 set out to close: before it this line had `request_id`
    // and nothing that could be taken to the provider.
    expect(line).toContain(`provider_request_id="${NONSTREAM_ID}"`);
    // Two distinct ids, neither standing in for the other.
    expect(res.requestId).not.toBe(NONSTREAM_ID);
  });

  test("streaming: the provider-call line carries the id the access log cannot", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await call(app, "/v1/chat/completions", {
      model: "pri-stream-e2e",
      stream: true,
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(200);
    expect(res.text).toContain("[DONE]");

    // The whole point of the per-attempt line: the id only exists once the
    // first upstream frame lands, by which time the access-log line for this
    // request has already been written.
    const line = await waitForLogLine(
      app,
      (l) =>
        l.includes("provider call completed") &&
        l.includes(`request_id="${res.requestId}"`),
      "the provider-call line for this streamed request",
    );
    expect(line).toContain(`provider_request_id="${STREAM_ID}"`);
    // `request_id` + `attempt_index` is what identifies an individual
    // provider call across a retried / failed-over request.
    expect(line).toContain("attempt_index=");
  });

  test("/v1/responses records the provider id too", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await call(app, "/v1/responses", {
      model: "pri-responses-e2e",
      input: "hello",
    });
    expect(res.status).toBe(200);

    const line = await waitForLogLine(
      app,
      (l) =>
        l.includes("provider call completed") &&
        l.includes(`request_id="${res.requestId}"`),
      "the provider-call line for /v1/responses",
    );
    expect(line).toContain(`provider_request_id="${RESPONSES_ID}"`);
  });
});
