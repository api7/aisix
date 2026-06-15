import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: routing failover edge cases. The existing fallback-e2e covers
// the happy path (5xx → fallback → 200). Two real user pain points
// that prior coverage didn't pin:
//
//   1. Non-retryable client errors (400 bad request) MUST NOT trigger
//      fallback. The caller's bug should surface immediately, not get
//      silently re-routed to a different upstream that might 200 on a
//      different prompt — that would mask request bugs and is a real
//      production debugging trap.
//
//   2. When all targets fail, the caller must see a clean error
//      envelope. Hangs or generic 500s break SDK error-handling
//      assumptions.
//
// Reference: OpenAI Chat Completions API spec
// (https://platform.openai.com/docs/api-reference/chat/create) for
// the request/response shape; the gateway's fallback policy is its
// own contract.

const CALLER_PLAINTEXT = "sk-fb-edges-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("fallback edge cases e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  // Each test sets up its own pair/triple of mock upstreams so the
  // request-count assertions stay isolated across cases.
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    // Wildcard ApiKey lets the same caller reach every per-test
    // virtual model without re-issuing keys.
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("non-retryable 4xx (400) does NOT trigger fallback — caller sees the 4xx", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // First target returns a 400 with an OpenAI-shape error envelope.
    // The point of fallback is to insulate callers from upstream
    // *infrastructure* failures (timeouts, 5xx, connection resets) —
    // not from the caller's own bad input. A 400 means "your request
    // is malformed"; falling over to a different upstream and getting
    // a 200 there would tell the caller "you know what, try again, it
    // worked the second time" — which is misleading.
    const badRequestUpstream = await startOpenAiUpstream({
      status: 400,
      errorBody: {
        error: {
          message: "Invalid 'model' parameter",
          type: "invalid_request_error",
          code: "invalid_model",
        },
      },
    });
    const goodUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-good",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "should NOT be reached" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreams.push(badRequestUpstream, goodUpstream);

    const badPk = await admin.createProviderKey({
      display_name: "fb-edges-400-pk",
      secret: "sk-mock",
      api_base: `${badRequestUpstream.baseUrl}/v1`,
    });
    const goodPk = await admin.createProviderKey({
      display_name: "fb-edges-400-good-pk",
      secret: "sk-mock",
      api_base: `${goodUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "fb-edges-400-bad",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: badPk.id,
    });
    await admin.createModel({
      display_name: "fb-edges-400-good",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: goodPk.id,
    });
    await admin.createModel({
      display_name: "fb-edges-400-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "fb-edges-400-bad" },
          { model: "fb-edges-400-good" },
        ],
      },
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "fb-edges-400-good",
          messages: [{ role: "user", content: "ready-probe-good" }],
        });
        return true;
      } catch {
        return false;
      }
    });
    // Second propagation gate on the virtual model itself. Without
    // this, a slow CI that loaded `fb-edges-400-good` first but
    // hadn't yet registered the routing block could still emit a
    // 400 here — but from "unknown virtual model", not from the bad
    // target's 400 the test is meant to verify. The "good upstream
    // count == 0" assertion would then trivially pass on a broken
    // gateway. The expected outcome here is APIError 400 (the bad
    // target propagating up); 200 means the bad target wasn't
    // tried, which is the wrong config state.
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "fb-edges-400-virtual",
          messages: [{ role: "user", content: "ready-probe-virtual" }],
        });
        return false; // 200 means bad target wasn't reached
      } catch (e) {
        return e instanceof APIError && e.status === 400;
      }
    });

    const badBaseline = badRequestUpstream.receivedRequests.length;
    const goodBaseline = goodUpstream.receivedRequests.length;

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "fb-edges-400-virtual",
        messages: [{ role: "user", content: "trigger 400" }],
      });
    } catch (e) {
      caught = e;
    }

    // The caller MUST see the 400 propagated, not a 200 from the
    // fallback target. If the gateway silently re-routed, this
    // assertion would fail because the call would have succeeded.
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    expect(caught.status).toBe(400);
    // Disambiguate "bad target's 400 propagated" from "unknown
    // virtual model 400" — same envelope-discrimination pattern
    // allowed-models-e2e uses. The bad target's mock body declares
    // type `invalid_request_error`; a "model not found" 400 would
    // contain that phrase in the message.
    const errMsg = (caught.error as { message?: unknown })?.message;
    expect(typeof errMsg).toBe("string");
    expect((errMsg as string).toLowerCase()).not.toContain("not found");

    // Bad target was called exactly once (the dispatch attempt).
    // Good target was NOT called: a 4xx is not a retryable signal,
    // so dispatch must short-circuit instead of walking the list.
    expect(badRequestUpstream.receivedRequests.length - badBaseline).toBe(1);
    expect(goodUpstream.receivedRequests.length - goodBaseline).toBe(0);
  });

  test("all targets fail → caller sees a clean error envelope, not hang", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Both targets return retryable 5xx — fallback walks the list,
    // exhausts options, must surface a clean error to the caller.
    // A regression that hung indefinitely or returned a non-OpenAI-
    // shape generic 500 would break SDK error handling.
    const bad1 = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "bad1 down", type: "server_error" } },
    });
    const bad2 = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "bad2 down", type: "server_error" } },
    });
    upstreams.push(bad1, bad2);

    const pk1 = await admin.createProviderKey({
      display_name: "fb-edges-allfail-1-pk",
      secret: "sk-mock",
      api_base: `${bad1.baseUrl}/v1`,
    });
    const pk2 = await admin.createProviderKey({
      display_name: "fb-edges-allfail-2-pk",
      secret: "sk-mock",
      api_base: `${bad2.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "fb-edges-allfail-1",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk1.id,
    });
    await admin.createModel({
      display_name: "fb-edges-allfail-2",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk2.id,
    });
    await admin.createModel({
      display_name: "fb-edges-allfail-virtual",
      routing: {
        strategy: "failover",
        targets: [
          { model: "fb-edges-allfail-1" },
          { model: "fb-edges-allfail-2" },
        ],
      },
    });

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      // 30s leaves headroom for dispatch + retry walk + envelope
      // render even on a constrained CI runner; small enough that
      // a true gateway hang still surfaces as a test failure within
      // a reasonable bound rather than running indefinitely.
      timeout: 30_000,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      // Probe via the virtual model directly. Both real targets are
      // down; the only positive outcome the test cares about is the
      // routing block being loaded so the dispatcher can walk both
      // targets. A 5xx from the SDK proves snapshot is loaded; a
      // 4xx (e.g. unknown model) means routing config hasn't landed
      // yet.
      try {
        await client.chat.completions.create({
          model: "fb-edges-allfail-virtual",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return false; // unexpected 200
      } catch (e) {
        return e instanceof APIError && (e.status ?? 0) >= 500;
      }
    });

    const bad1Baseline = bad1.receivedRequests.length;
    const bad2Baseline = bad2.receivedRequests.length;

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "fb-edges-allfail-virtual",
        messages: [{ role: "user", content: "all should fail" }],
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    // 5xx is the correct family: dispatch was attempted, all upstreams
    // failed, the gateway gives up and reports a server-side error.
    // Anything in the 4xx family would mislead callers into thinking
    // their request was malformed.
    expect(caught.status).toBeGreaterThanOrEqual(500);
    expect(caught.status).toBeLessThan(600);

    // OpenAI-shape error envelope: `error.type` and `error.message`
    // both populated. A regression that surfaced a Rust panic
    // wrapped in a generic Axum 500 would have empty / missing
    // fields here.
    expect(typeof caught.error).toBe("object");
    const errType = (caught.error as { type?: unknown })?.type;
    const errMsg = (caught.error as { message?: unknown })?.message;
    expect(typeof errType).toBe("string");
    expect((errType as string).length).toBeGreaterThan(0);
    expect(typeof errMsg).toBe("string");
    expect((errMsg as string).length).toBeGreaterThan(0);

    // Both targets were tried exactly once each (the failover walk).
    // A regression that bailed after the first failure (or hammered
    // the same target twice) would inflate / deflate these counts.
    expect(bad1.receivedRequests.length - bad1Baseline).toBe(1);
    expect(bad2.receivedRequests.length - bad2Baseline).toBe(1);
  });
});
