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
//   1. Happy path (OpenAI) — POST /v1/responses with an OpenAI-provider
//      Model. Gateway dispatches to upstream's /v1/responses
//      (NOT /v1/chat/completions), caller receives the upstream's
//      Responses-shape body byte-for-byte, with the configured
//      Model's display name translated to upstream model_name.
//
//   2. Cross-provider (#825) — POST /v1/responses with a non-OpenAI
//      Model. The gateway no longer rejects with 400; it bridges the
//      Responses request through the canonical ChatFormat to the
//      provider's native endpoint and re-encodes the reply into the
//      Responses shape. (Anthropic streaming/tool coverage lives in
//      responses-cross-provider-e2e; here we pin the OpenAI-compatible
//      deepseek bridge as the representative non-OpenAI case.)
//
// References:
// - OpenAI Responses API spec
//   <https://platform.openai.com/docs/api-reference/responses>
// - Gateway's own /v1/responses contract: `docs/api-proxy.md` §4.6

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

  // #825: a non-OpenAI provider on /v1/responses is now bridged, not
  // rejected. deepseek is the representative case — its bridge speaks the
  // OpenAI chat wire shape upstream, so the default mock body (a
  // chat.completion) round-trips cleanly. The gateway must translate the
  // Responses request into a chat completion, hit the provider's native
  // /chat/completions endpoint (NOT /v1/responses), and re-encode the
  // reply back into the Responses shape. (Richer Anthropic streaming/tool
  // coverage lives in responses-cross-provider-e2e.)
  test("non-OpenAI provider (deepseek): /v1/responses is bridged to a Responses-shape 200", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-ds-01",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "deepseek-chat",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "bridged reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 8, completion_tokens: 4, total_tokens: 12 },
      },
    });
    upstreams.push(upstream);

    const pk = await admin.createProviderKey({
      display_name: "resp-deepseek-pk",
      provider: "deepseek",
      adapter: "openai",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "resp-deepseek",
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: pk.id,
    });

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    // Readiness gate: poll until the bridged path returns a Responses-shape
    // 200 (no longer the pre-#825 400).
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/responses`, {
          method: "POST",
          headers,
          body: JSON.stringify({ model: "resp-deepseek", input: "ready-probe" }),
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
      body: JSON.stringify({ model: "resp-deepseek", input: "Say hello" }),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      object?: unknown;
      status?: unknown;
      output?: Array<{ type?: unknown; content?: Array<{ text?: unknown }> }>;
    };
    expect(body.object).toBe("response");
    expect(body.status).toBe("completed");
    expect(body.output?.[0]?.type).toBe("message");
    expect(body.output?.[0]?.content?.[0]?.text).toBe("bridged reply");

    // The bridge dispatched to the provider's chat endpoint, not
    // /v1/responses (the verbatim path is OpenAI-only).
    const calls = upstream.receivedRequests.slice(baseline);
    expect(calls.some((r) => r.path.endsWith("/chat/completions"))).toBe(true);
    expect(calls.some((r) => r.path === "/v1/responses")).toBe(false);
  });
});
