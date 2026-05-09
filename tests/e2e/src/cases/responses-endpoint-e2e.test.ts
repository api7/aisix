import { createHash } from "node:crypto";
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

// E2E: /v1/responses end-to-end. The OpenAI Responses API
// (introduced 2024) is the recommended endpoint for new
// integrations and is rapidly displacing /v1/chat/completions in
// new code. Prior to this file, the gateway had **zero** e2e
// coverage on /v1/responses.
//
// Two user journeys pinned, both derived from the gateway's own
// published contract in `docs/api-proxy.md` §4.6:
//
//   > Native OpenAI Responses API. OpenAI Models only — non-OpenAI
//   > providers return 400.
//
//   1. Happy path — POST /v1/responses with an OpenAI-provider
//      Model. Gateway dispatches to upstream's /v1/responses
//      (NOT /v1/chat/completions), caller receives the upstream's
//      Responses-shape body byte-for-byte, with the configured
//      Model's display name translated to upstream model_name.
//
//   2. Provider mismatch — POST /v1/responses with an Anthropic-
//      provider Model. Gateway must return 400 per the published
//      contract; upstream must NOT be hit (the entire point of
//      the restriction is OpenAI-Responses-shape doesn't translate
//      to Anthropic Messages today).
//
// References:
// - OpenAI Responses API spec
//   <https://platform.openai.com/docs/api-reference/responses>
// - Gateway's own /v1/responses contract: `docs/api-proxy.md` §4.6
// - OpenAI error envelope spec
//   <https://platform.openai.com/docs/guides/error-codes/api-errors>

const CALLER_PLAINTEXT = "sk-resp-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Distinctive content the upstream emits, so a regression that
// silently substituted a generic body would surface here.
const UPSTREAM_REPLY_TEXT = "Hello from /v1/responses!";

describe("responses endpoint e2e: /v1/responses dispatch + provider mismatch", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("OpenAI provider: caller receives upstream Responses body byte-for-byte", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Mock upstream returns an OpenAI Responses-shape body. Note
    // this is a different envelope from /v1/chat/completions:
    // top-level `output` array of content-block messages, not
    // `choices[].message`. Per OpenAI Responses API spec.
    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_e2e_01",
        object: "response",
        created_at: Math.floor(Date.now() / 1000),
        status: "completed",
        model: "gpt-4o-mini",
        output: [
          {
            id: "msg_e2e_01",
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: UPSTREAM_REPLY_TEXT }],
          },
        ],
        usage: {
          input_tokens: 5,
          output_tokens: 6,
          total_tokens: 11,
        },
      },
    });
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "resp-openai-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "resp-openai",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    // Readiness gate: /v1/responses propagation. A 200 response
    // body shape-checked for `object: "response"` so a half-
    // propagated 200-with-malformed-body doesn't falsely report
    // ready.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/responses`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "resp-openai",
            input: "ready-probe",
          }),
        });
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { object?: unknown };
        return j.object === "response";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "resp-openai",
        input: "Say hello",
      }),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      id?: unknown;
      object?: unknown;
      status?: unknown;
      output?: Array<{
        type?: unknown;
        role?: unknown;
        content?: Array<{ type?: unknown; text?: unknown }>;
      }>;
      usage?: { input_tokens?: unknown; output_tokens?: unknown; total_tokens?: unknown };
    };

    // OpenAI Responses response envelope shape: distinct from chat
    // completions. A regression that mis-routed via the chat path
    // would return `object: "chat.completion"` here.
    expect(body.object).toBe("response");
    expect(body.status).toBe("completed");
    expect(typeof body.id).toBe("string");
    expect(body.output).toHaveLength(1);
    expect(body.output?.[0]?.type).toBe("message");
    expect(body.output?.[0]?.role).toBe("assistant");
    expect(body.output?.[0]?.content?.[0]?.type).toBe("output_text");
    // Reply text round-trips byte-for-byte.
    expect(body.output?.[0]?.content?.[0]?.text).toBe(UPSTREAM_REPLY_TEXT);
    // Usage counters: Responses uses `input_tokens` /
    // `output_tokens` / `total_tokens` (different field names from
    // chat completions, which uses `prompt_tokens` /
    // `completion_tokens`). A regression that translated through
    // chat-completions field names would mismatch here.
    expect(body.usage?.input_tokens).toBe(5);
    expect(body.usage?.output_tokens).toBe(6);
    expect(body.usage?.total_tokens).toBe(11);

    // Dispatch contract: gateway hit `/v1/responses` exactly once
    // (NOT `/v1/chat/completions`). Mis-routing through the chat
    // path is the regression mode this assertion catches.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/responses");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
    expect(testCalls[0]?.headers["authorization"]).toBe("Bearer sk-mock");

    // Wire-shape contract per gateway docs §4.6 ("Native OpenAI
    // Responses API"): body is OpenAI-Responses-shape with display
    // name → upstream model_name translation, and caller's input
    // reaches upstream verbatim.
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      model?: string;
      input?: unknown;
    };
    expect(sentBody.model).toBe("gpt-4o-mini");
    expect(sentBody.input).toBe("Say hello");
  });

  test("non-OpenAI provider (anthropic): caller sees 400, upstream untouched (per docs §4.6)", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Set up a Model with provider=anthropic. Per gateway docs
    // §4.6, /v1/responses is OpenAI-only and non-OpenAI providers
    // return 400. The mock upstream is registered but should
    // never be hit.
    const upstream = await startOpenAiUpstream();
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "resp-anthropic-pk",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "resp-anthropic",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll until the gateway returns the
    // documented 400, not a model-not-found 400 from snapshot lag.
    // Disambiguate by checking the error envelope is fully formed
    // (a snapshot-lag 400 might have a different message style).
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/responses`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "resp-anthropic",
            input: "ready-probe",
          }),
        });
        if (r.status !== 400) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as {
          error?: { type?: unknown; message?: unknown };
        };
        // Per OpenAI error envelope spec, error.type is a non-empty
        // string discriminator. Both "model not found" and "wrong
        // provider" 400s would have this; we're just gating on the
        // gateway being up enough to return SOME 400 with a body.
        return typeof j.error?.type === "string";
      } catch {
        return false;
      }
    });

    const upstreamHitsBefore = upstream.receivedRequests.length;

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "resp-anthropic",
        input: "Say hello",
      }),
    });

    // Per docs §4.6: non-OpenAI providers return 400. Status family
    // 5xx would mean the gateway crashed (it should refuse cleanly,
    // not panic).
    expect(res.status).toBe(400);

    const body = (await res.json()) as {
      error?: { type?: unknown; message?: unknown };
    };
    // OpenAI error envelope per spec: type and message non-empty.
    expect(typeof body.error?.type).toBe("string");
    expect((body.error?.type as string).length).toBeGreaterThan(0);
    expect(typeof body.error?.message).toBe("string");
    expect((body.error?.message as string).length).toBeGreaterThan(0);

    // Hard contract: upstream must never be hit when the gateway
    // refuses for provider mismatch — otherwise the gateway is
    // billing the caller's quota on a request it claims to reject.
    expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
  });
});
