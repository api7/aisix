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
    // `id` round-trips byte-for-byte. A regression that re-issued
    // ids during gateway-side normalization would silently break
    // SDK paginators and webhook callbacks that key off response id.
    expect(body.id).toBe("resp_e2e_01");
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

  // All three non-OpenAI providers per docs §6 (anthropic, gemini,
  // deepseek). Per docs §4.6, /v1/responses on any of them must
  // return 400. Parametrizing across all three catches a regression
  // that special-cased one provider but mis-handled others — gemini
  // and deepseek's bridges *do* speak OpenAI wire shape upstream, so
  // a regression that "just dispatched anyway" would actually return
  // 200 from the upstream-compat layer, billing the caller and
  // silently violating the published contract.
  const NON_OPENAI_PROVIDERS = [
    {
      provider: "anthropic" as const,
      modelName: "claude-3-5-haiku-20241022",
      secret: "sk-ant-mock",
      // Anthropic's documented endpoint is `https://api.anthropic.com/v1/messages`
      // → api_base is the bare host.
      apiBaseSuffix: "" as const,
    },
    {
      provider: "gemini" as const,
      modelName: "gemini-2.0-flash",
      secret: "sk-mock",
      apiBaseSuffix: "/v1" as const,
    },
    {
      provider: "deepseek" as const,
      modelName: "deepseek-chat",
      secret: "sk-mock",
      apiBaseSuffix: "/v1" as const,
    },
  ];

  for (const tc of NON_OPENAI_PROVIDERS) {
    test(`non-OpenAI provider (${tc.provider}): caller sees 400 invalid_request_error, upstream untouched (per docs §4.6)`, async (ctx) => {
      if (!etcdReachable || !app || !admin) {
        ctx.skip();
        return;
      }

      const upstream = await startOpenAiUpstream();
      upstreams.push(upstream);

      const pk = await admin.createProviderKey({
        display_name: `resp-${tc.provider}-pk`,
        secret: tc.secret,
        api_base: `${upstream.baseUrl}${tc.apiBaseSuffix}`,
      });
      const modelDisplayName = `resp-${tc.provider}`;
      await admin.createModel({
        display_name: modelDisplayName,
        provider: tc.provider,
        model_name: tc.modelName,
        provider_key_id: pk.id,
      });

      const headers = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };

      // Readiness gate: poll until the gateway returns the
      // documented 400 with `error.type: "invalid_request_error"`
      // per docs §2 status→type table. A 404 here would be the
      // snapshot-lag "model not found" case (model_not_found is
      // mapped to 404, NOT 400, per the docs), so probing on
      // 400 + invalid_request_error specifically gates on the
      // gateway resolving the model AND refusing per §4.6.
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/responses`, {
            method: "POST",
            headers,
            body: JSON.stringify({
              model: modelDisplayName,
              input: "ready-probe",
            }),
          });
          if (r.status !== 400) {
            await r.text();
            return false;
          }
          const j = (await r.json()) as {
            error?: { type?: unknown };
          };
          return j.error?.type === "invalid_request_error";
        } catch {
          return false;
        }
      });

      const upstreamHitsBefore = upstream.receivedRequests.length;

      const res = await fetch(`${app.proxyUrl}/v1/responses`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: modelDisplayName,
          input: "Say hello",
        }),
      });

      // Per docs §4.6: non-OpenAI providers return 400. Status
      // family 5xx would mean the gateway crashed (it should
      // refuse cleanly, not panic).
      expect(res.status).toBe(400);

      const body = (await res.json()) as {
        error?: { type?: unknown; message?: unknown };
      };
      // Per docs §2 status→type table: 400 → invalid_request_error.
      // Pinning the exact value catches a regression where the
      // gateway's refusal vocabulary drifts from the published
      // contract (e.g. emits "service_unavailable" or
      // "model_not_found"). Same convention body-edges-e2e and
      // error-envelope-normalization-e2e use.
      expect(body.error?.type).toBe("invalid_request_error");
      expect(typeof body.error?.message).toBe("string");
      expect((body.error?.message as string).length).toBeGreaterThan(0);

      // Hard contract: upstream must never be hit when the gateway
      // refuses for provider mismatch — otherwise the gateway is
      // billing the caller's quota on a request it claims to reject.
      // Critical for gemini and deepseek specifically, whose bridges
      // DO speak OpenAI wire shape upstream — a regression that
      // dispatched anyway would silently 200 from the upstream-
      // compat layer.
      expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
    });
  }
});
