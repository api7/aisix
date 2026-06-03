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

// E2E: provider_key.api_base tolerates an operator pasting the full
// upstream URL (endpoint suffix included). The bridge strips the known
// suffix before appending its own path, so the request still lands at
// the canonical endpoint instead of `…/chat/completions/chat/completions`.
//
// Two scenarios per provider category:
//   - OpenAI-compat bridge: api_base ending in `/chat/completions` is
//     stripped; final upstream request path is `/chat/completions`.
//   - Anthropic bridge: api_base ending in `/v1/messages` is stripped;
//     final upstream request path is `/v1/messages`.
//
// The unit suites in `crates/aisix-provider-openai` and
// `crates/aisix-provider-anthropic` cover the canonical-host /v1
// normalization (e.g. pasting `https://api.openai.com` without `/v1`).
// That path can only be exercised against the real OpenAI host, so the
// e2e here narrows to the host-agnostic suffix-stripping contract.
//
// References:
//   - OpenAI Chat Completions API spec
//     <https://platform.openai.com/docs/api-reference/chat/create>
//   - Anthropic Messages API spec
//     <https://docs.anthropic.com/en/api/messages>

const OPENAI_CALLER_PLAINTEXT = "sk-api-base-openai-e2e";
const OPENAI_CALLER_KEY_HASH = createHash("sha256")
  .update(OPENAI_CALLER_PLAINTEXT)
  .digest("hex");
const ANTHROPIC_CALLER_PLAINTEXT = "sk-api-base-anthropic-e2e";
const ANTHROPIC_CALLER_KEY_HASH = createHash("sha256")
  .update(ANTHROPIC_CALLER_PLAINTEXT)
  .digest("hex");

describe("api_base tolerance e2e: endpoint suffix is stripped before dispatch", () => {
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
      key_hash: OPENAI_CALLER_KEY_HASH,
      allowed_models: ["openai-suffix-tolerance"],
    });
    await admin.createApiKey({
      key_hash: ANTHROPIC_CALLER_KEY_HASH,
      allowed_models: ["anthropic-suffix-tolerance"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("OpenAI bridge strips a `/chat/completions` suffix from api_base", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // The mock returns 200 on every path, so a regression that
    // forgot to strip would still produce 200. Path assertion below
    // pins what we actually care about: the upstream URL the bridge
    // constructed.
    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-api-base-openai",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "tolerance ok" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    upstreams.push(upstream);

    // Operator pastes the full upstream URL into api_base. Without
    // suffix stripping, the bridge would dispatch to
    //   `${baseUrl}/chat/completions/chat/completions`
    // and `upstream.receivedRequests[*].path` would echo that.
    const pk = await admin.createProviderKey({
      display_name: "openai-suffix-tolerance-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/chat/completions`,
    });
    await admin.createModel({
      display_name: "openai-suffix-tolerance",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    await waitConfigPropagation(async () => {
      try {
        const res = await fetch(`${app?.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${OPENAI_CALLER_PLAINTEXT}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: "openai-suffix-tolerance",
            messages: [{ role: "user", content: "ready" }],
          }),
        });
        await res.text();
        return res.status === 200;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${OPENAI_CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "openai-suffix-tolerance",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    await res.text();

    // The exactly-one new upstream request landed at the canonical
    // endpoint path. A regression that doubled the suffix would record
    // `/chat/completions/chat/completions` instead.
    const newRequests = upstream.receivedRequests.slice(baseline);
    expect(newRequests).toHaveLength(1);
    expect(newRequests[0]?.path).toBe("/chat/completions");
  });

  test("Anthropic bridge strips a `/v1/messages` suffix from api_base", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_api_base_anthropic",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "anthropic tolerance ok" }],
        model: "claude-3-5-haiku-20241022",
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      },
    });
    upstreams.push(upstream);

    // Operator pastes the full Anthropic upstream URL. Without the
    // strip, the bridge would dispatch to
    //   `${baseUrl}/v1/messages/v1/messages`.
    const pk = await admin.createProviderKey({
      display_name: "anthropic-suffix-tolerance-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: `${upstream.baseUrl}/v1/messages`,
    });
    await admin.createModel({
      display_name: "anthropic-suffix-tolerance",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });

    await waitConfigPropagation(async () => {
      try {
        const res = await fetch(`${app?.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${ANTHROPIC_CALLER_PLAINTEXT}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: "anthropic-suffix-tolerance",
            messages: [{ role: "user", content: "ready" }],
          }),
        });
        await res.text();
        return res.status === 200;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${ANTHROPIC_CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "anthropic-suffix-tolerance",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBe(200);
    await res.text();

    const newRequests = upstream.receivedRequests.slice(baseline);
    expect(newRequests).toHaveLength(1);
    expect(newRequests[0]?.path).toBe("/v1/messages");
  });
});
