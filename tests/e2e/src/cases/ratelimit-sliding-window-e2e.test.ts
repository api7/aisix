import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { harnessRequest } from "../harness/http.js";

// E2E: `window_type: sliding` on an API key's inline rate limit.
//
// A sliding minute window counts usage in 10-second sub-buckets aligned to
// the epoch and sums the current one with the six before it. Two things a
// caller can observe follow from that, and both are what this spec pins:
//
// - Usage admitted just before the top of a minute still counts after it,
//   so a burst across the boundary is refused where a fixed window would
//   hand out a fresh allowance.
// - A refusal's wait is when enough of that usage leaves the window — the
//   end of its sub-bucket plus a minute — and retrying then is admitted.
//
// The streamed /v1/messages case covers the post-stream token commit,
// which records the tokens after the response has ended and must still
// record them in the sliding window the key's limit counts against.

const RPM_CALLER = "sk-sw-rpm";
const TPM_CALLER = "sk-sw-tpm";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const MODEL = "sw-model";

const STEP = 10;
const WINDOW = 60;

const STREAM = [
  JSON.stringify({
    id: "chatcmpl-sw",
    object: "chat.completion.chunk",
    created: 0,
    model: "up-sw",
    choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-sw",
    object: "chat.completion.chunk",
    created: 0,
    model: "up-sw",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  }),
  JSON.stringify({
    id: "chatcmpl-sw",
    object: "chat.completion.chunk",
    created: 0,
    model: "up-sw",
    choices: [],
    usage: { prompt_tokens: 8, completion_tokens: 8, total_tokens: 16 },
  }),
  "[DONE]",
];

interface RawResponse {
  status: number;
  headers: Record<string, string>;
}

async function raw(
  url: string,
  headers: Record<string, string>,
  body: unknown,
): Promise<RawResponse> {
  const res = await harnessRequest(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  await res.body.text();
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(res.headers)) {
    out[k.toLowerCase()] = Array.isArray(v) ? v.join(", ") : String(v ?? "");
  }
  return { status: res.statusCode, headers: out };
}

const nowSecs = () => Math.floor(Date.now() / 1000);

/** Sleep until the wall clock reads `target` (epoch seconds) or later. */
async function sleepUntil(target: number): Promise<void> {
  const ms = target * 1000 - Date.now();
  if (ms > 0) await new Promise((r) => setTimeout(r, ms + 50));
}

/**
 * Assert a refusal's standard headers, with the wait bounded by when the
 * usage recorded in sub-bucket `bucketStart` leaves the window, measured
 * from either side of the request (a second can tick during it).
 */
function expectRefusal(
  res: RawResponse,
  expected: { scope: string; limit: number; bucketStart: number; before: number; after: number },
): number {
  expect(res.status).toBe(429);
  expect(res.headers["x-ratelimit-scope"]).toBe(expected.scope);
  expect(res.headers["x-ratelimit-limit"]).toBe(String(expected.limit));
  expect(res.headers["x-ratelimit-remaining"]).toBe("0");
  const reset = Number.parseInt(res.headers["x-ratelimit-reset"] ?? "", 10);
  expect(res.headers["retry-after"]).toBe(String(reset));
  const leaves = expected.bucketStart + WINDOW + STEP;
  expect(reset).toBeGreaterThanOrEqual(leaves - expected.after);
  expect(reset).toBeLessThanOrEqual(leaves - expected.before);
  return reset;
}

describe("rate limit e2e: sliding window", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    streamUpstream = await startOpenAiUpstream({ streamEvents: STREAM, eventDelayMs: 2 });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "sw-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const streamPk = await seed.createProviderKey({
      display_name: "sw-stream-pk",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "sw-stream",
      provider: "openai",
      model_name: "up-sw",
      provider_key_id: streamPk.id,
    });
    // Caller keys last: a key authenticating implies everything above
    // has reached the gateway.
    await seed.createApiKey({
      key_hash: hash(RPM_CALLER),
      allowed_models: [MODEL],
      rate_limit: { rpm: 2, window_type: "sliding" },
    });
    await seed.createApiKey({
      key_hash: hash(TPM_CALLER),
      allowed_models: ["sw-stream"],
      rate_limit: { tpm: 10, window_type: "sliding" },
    });
    for (const key of [RPM_CALLER, TPM_CALLER]) {
      const probe = new ProxyClient(app.proxyUrl, key);
      await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
  });

  const chat = (key: string) =>
    raw(
      `${app!.proxyUrl}/v1/chat/completions`,
      { authorization: `Bearer ${key}` },
      { model: MODEL, messages: [{ role: "user", content: "hi" }] },
    );

  test("usage from before the top of the minute still counts after it", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Land both admitted requests in the minute's last sub-bucket, early
    // enough in it that they cannot straddle into the next minute.
    let minute = nowSecs() - (nowSecs() % WINDOW);
    if (nowSecs() - minute > WINDOW - STEP) minute += WINDOW;
    const lastBucket = minute + WINDOW - STEP;
    await sleepUntil(lastBucket);

    const first = await chat(RPM_CALLER);
    expect(first.status).toBe(200);
    expect(first.headers["x-ratelimit-limit-requests"]).toBe("2");
    expect(first.headers["x-ratelimit-remaining-requests"]).toBe("1");
    const second = await chat(RPM_CALLER);
    expect(second.status).toBe(200);
    expect(second.headers["x-ratelimit-remaining-requests"]).toBe("0");
    // The reset on a 200 is when the window is empty again: the end of
    // the newest sub-bucket plus a minute, so more than a minute away.
    const okReset = Number.parseInt(
      (second.headers["x-ratelimit-reset-requests"] ?? "").replace(/s$/, ""),
      10,
    );
    expect(okReset).toBeGreaterThan(WINDOW);
    expect(okReset).toBeLessThanOrEqual(WINDOW + STEP);

    let before = nowSecs();
    const refused = await chat(RPM_CALLER);
    expectRefusal(refused, {
      scope: "rpm",
      limit: 2,
      bucketStart: lastBucket,
      before,
      after: nowSecs(),
    });

    // Past the top of the minute a fixed window would admit again.
    await sleepUntil(minute + WINDOW + 1);
    before = nowSecs();
    const acrossBoundary = await chat(RPM_CALLER);
    const after = nowSecs();
    const reset = expectRefusal(acrossBoundary, {
      scope: "rpm",
      limit: 2,
      bucketStart: lastBucket,
      before,
      after,
    });

    // Retrying when told is admitted. The gateway read its clock somewhere
    // inside the request, so count the wait from its end.
    await sleepUntil(after + reset);
    expect((await chat(RPM_CALLER)).status).toBe(200);
  }, 150_000);

  test("streamed tokens count in the sliding token window", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Keep the stream and the follow-up inside one sub-bucket so the wait
    // can be bounded by it.
    if (STEP - (nowSecs() % STEP) < 5) await sleepUntil(nowSecs() - (nowSecs() % STEP) + STEP);
    const bucketStart = nowSecs() - (nowSecs() % STEP);

    const messages = () =>
      raw(
        `${app!.proxyUrl}/v1/messages`,
        { "x-api-key": TPM_CALLER },
        {
          model: "sw-stream",
          max_tokens: 64,
          stream: true,
          messages: [{ role: "user", content: "hi" }],
        },
      );
    expect((await messages()).status).toBe(200);

    // The post-stream commit lands a tick after the client finishes
    // reading, so the first follow-up can race ahead of it.
    let refused: RawResponse | undefined;
    let before = 0;
    let after = 0;
    const deadline = Date.now() + 3_000;
    while (Date.now() < deadline) {
      before = nowSecs();
      const res = await messages();
      after = nowSecs();
      if (res.status === 429) {
        refused = res;
        break;
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(refused, "the streamed 16 tokens must exhaust tpm 10").toBeDefined();
    expectRefusal(refused!, { scope: "tpm", limit: 10, bucketStart, before, after });
  }, 30_000);
});
