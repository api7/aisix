import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: Anthropic `/v1/messages/count_tokens` through the DP (#418).
//
// The Anthropic SDK exposes this as `anthropic.messages.countTokens(...)`
// — the documented, billing-relevant endpoint callers use to size a
// prompt before a paid `/v1/messages` call. Before #418 the route was
// unregistered and the DP returned a bare 404. This test drives the
// endpoint the way a real Anthropic-SDK / Claude-Code caller does (raw
// HTTP with the `x-api-key` + `anthropic-version` auth shape, since there
// is no Anthropic SDK in this harness) and asserts the externally
// observable contract:
//
//   - the caller gets 200 with `{"input_tokens": <number>}`;
//   - the gateway forwarded to the Anthropic upstream's
//     `/v1/messages/count_tokens` sub-route (NOT `/v1/messages`);
//   - it rewrote the `model` alias to the upstream id;
//   - it spoke the Anthropic auth shape (`x-api-key` +
//     `anthropic-version`), not `Authorization: Bearer`.
//
// The mock-upstream harness is path-agnostic, so feeding it the
// count_tokens response body lets it stand in for Anthropic's
// `/v1/messages/count_tokens`. `receivedRequests` confirms the path and
// request shape the gateway actually sent.
//
// Reference:
// - Anthropic Count Message Tokens API:
//   <https://platform.claude.com/docs/en/api/messages-count-tokens>
//   (`POST /v1/messages/count_tokens` → `{"input_tokens": <int>}`).

const CALLER_PLAINTEXT = "sk-ct-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const UPSTREAM_MODEL_ID = "claude-haiku-4-5-20251001";
const MODEL_ALIAS = "ct-e2e";

describe("anthropic count_tokens e2e: /v1/messages/count_tokens through the DP (#418)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      // Anthropic's documented count_tokens response shape.
      nonStreamBody: { input_tokens: 42 },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    // Anthropic bridge appends the path to the bare host (no `/v1`).
    const pk = await seed.createProviderKey({
      display_name: "ct-e2e-pk",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: MODEL_ALIAS,
      provider: "anthropic",
      model_name: UPSTREAM_MODEL_ID,
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL_ALIAS],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("counts tokens against the Anthropic upstream and returns input_tokens", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const countTokens = (model: string) =>
      fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": CALLER_PLAINTEXT,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "hello" }],
        }),
      });

    // Wait for config propagation by probing the route itself — this
    // also proves the route is registered (a 404 would never become ok).
    await waitConfigPropagation(async () => {
      try {
        const r = await countTokens(MODEL_ALIAS);
        return r.ok;
      } catch {
        return false;
      }
    });

    // Baseline-isolate so the assertions below match THIS call, not the
    // readiness probe (which also lands on /v1/messages/count_tokens).
    const baseline = upstream.receivedRequests.length;

    const res = await countTokens(MODEL_ALIAS);

    // Caller-visible contract: 200 + the documented count_tokens body.
    expect(res.status).toBe(200);
    const body = (await res.json()) as { input_tokens?: unknown };
    expect(typeof body.input_tokens).toBe("number");
    expect(body.input_tokens).toBe(42);

    // Request-side wire shape. Pin the sub-route explicitly: a regression
    // that routed count_tokens to `/v1/messages` (or any other path)
    // would still 200 against the path-agnostic mock without this.
    const ctReq = upstream.receivedRequests
      .slice(baseline)
      .find((r) => r.path === "/v1/messages/count_tokens");
    expect(ctReq).toBeDefined();

    // model alias rewritten to the upstream id.
    const sentBody = JSON.parse(ctReq!.body) as {
      model?: string;
      messages?: Array<{ role?: string; content?: unknown }>;
    };
    expect(sentBody.model).toBe(UPSTREAM_MODEL_ID);
    expect(sentBody.messages?.[0]?.role).toBe("user");

    // Anthropic auth shape forwarded to upstream — not Bearer. A
    // regression that forwarded the OpenAI auth shape would 401 in
    // production but pass against the permissive mock without this.
    expect(ctReq!.headers["x-api-key"]).toBe("sk-ant-mock");
    expect(ctReq!.headers["anthropic-version"]).toBeDefined();
    expect(ctReq!.headers["authorization"]).toBeUndefined();
  });

  test("count_tokens on an unknown model returns 404, not a bare route 404", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // The route exists; an unknown model must surface the gateway's
    // model-not-found path. This guards against a future regression that
    // unregisters the route (which would 404 every model identically).
    const res = await fetch(`${app.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "no-such-model",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(404);
    // Anthropic-shape error envelope so the Claude SDK can parse it.
    const body = (await res.json()) as {
      type?: string;
      error?: { type?: string };
    };
    expect(body.type).toBe("error");
    expect(body.error?.type).toBe("not_found_error");
  });
});
