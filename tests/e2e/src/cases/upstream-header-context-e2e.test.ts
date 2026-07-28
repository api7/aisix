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

// E2E: the two operator-facing outbound-header features on
// `ProviderKey.request`, exercised through a real gateway against a real
// upstream (AISIX-Cloud#1112 + #1167).
//
//   1. `default_headers` values carrying `${...}` request-context
//      variables — rendered per request, and DROPPED (not blanked) when a
//      variable has no value for that request.
//   2. `forward_client_headers` — the operator's allowlist of inbound
//      client headers to relay upstream, exact names and `x-*` globs.
//
// The security contracts are pinned alongside the happy paths, because
// they are the reason the default is "forward nothing": the caller's own
// `Authorization` / `x-api-key`, the gateway's `x-aisix-*` namespace, and
// the transport headers must never reach the provider — not even when the
// operator writes `"*"` in the allowlist. Precedence is asserted in the
// same direction the pipeline enforces it: gateway > operator > client.
//
// Coverage spans three dispatch families that build their upstream headers
// in different code: the OpenAI Bridge (`/v1/chat/completions`, streaming
// and not) and the Anthropic native passthrough (`/v1/messages`), which
// the issue calls out as the pair most likely to drift.
//
// Reference:
//   - LiteLLM's equivalent: `forward_client_headers_to_llm_api`
//     <https://docs.litellm.ai/docs/proxy/forward_client_headers>
//   - Anthropic beta headers (the canonical forwarding case)
//     <https://docs.anthropic.com/en/api/beta-headers>

const CALLER_PLAINTEXT = "sk-hdrctx-caller-PLAINTEXT-MARKER";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// A second caller on the same ProviderKey, with no team — proves an
// unresolvable `${request.api_key.team_id}` drops just that header.
const TEAMLESS_PLAINTEXT = "sk-hdrctx-teamless-PLAINTEXT-MARKER";
const TEAMLESS_KEY_HASH = createHash("sha256")
  .update(TEAMLESS_PLAINTEXT)
  .digest("hex");

const PROVIDER_SECRET = "sk-mock-provider-secret";
const TEAM_ID = "team-acme-42";
const USER_ID = "user-7";
const KEY_NAME = "acme-prod-key";

const ANTHROPIC_BODY = {
  id: "msg_01",
  type: "message",
  role: "assistant",
  content: [{ type: "text", text: "hi" }],
  model: "claude-3-5-haiku-20241022",
  stop_reason: "end_turn",
  usage: { input_tokens: 5, output_tokens: 4 },
};

describe("upstream header context e2e: ${...} variables + client-header allowlist", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let anthropicUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  let openaiModelId = "";
  let openaiPkId = "";
  let anthropicModelId = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Two mocks: the OpenAI-shaped one answers the Bridge path, the
    // Anthropic-shaped one answers the native `/v1/messages` passthrough.
    // A single mock cannot serve both — its canned body is path-agnostic,
    // so an Anthropic body on the chat route decodes as a 502.
    upstream = await startOpenAiUpstream();
    anthropicUpstream = await startOpenAiUpstream({
      nonStreamBody: ANTHROPIC_BODY,
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    // One ProviderKey carrying both features: templated static headers
    // plus a client-header allowlist mixing an exact name and a glob.
    const request = {
      default_headers: {
        "x-tenant-id": "${request.api_key.team_id}",
        "x-caller-key": "${request.api_key.name}",
        "x-corp-model": "${model.name}",
        "x-provider-key": "${provider_key.name}",
        "x-model-id": "${model.id}",
        "x-pk-id": "${provider_key.id}",
        "x-corp-request-id": "${request.id}",
        "x-corp-static": "static-value",
        // Same name as a header the client also sends — operator wins.
        "x-overlap": "from-operator",
      },
      forward_client_headers: [
        "anthropic-beta",
        "x-trace-*",
        "x-overlap",
      ],
    };

    const openaiPk = await seed.createProviderKey({
      display_name: "hdrctx-openai-pk",
      secret: PROVIDER_SECRET,
      api_base: `${upstream.baseUrl}/v1`,
      request,
    });
    const openaiModel = await seed.createModel({
      display_name: "hdrctx-openai-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: openaiPk.id,
    });
    openaiModelId = openaiModel.id;
    openaiPkId = openaiPk.id;

    const anthropicPk = await seed.createProviderKey({
      display_name: "hdrctx-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: PROVIDER_SECRET,
      api_base: anthropicUpstream.baseUrl,
      request,
    });
    const anthropicModel = await seed.createModel({
      display_name: "hdrctx-anthropic-model",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: anthropicPk.id,
    });
    anthropicModelId = anthropicModel.id;

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      display_name: KEY_NAME,
      team_id: TEAM_ID,
      user_id: USER_ID,
      allowed_models: ["hdrctx-openai-model", "hdrctx-anthropic-model"],
    });
    await seed.createApiKey({
      key_hash: TEAMLESS_KEY_HASH,
      display_name: "teamless-key",
      allowed_models: ["hdrctx-openai-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await anthropicUpstream?.close();
  });

  /** POST a chat request as `plaintext`, returning the upstream's view of it. */
  async function chat(
    plaintext: string,
    headers: Record<string, string>,
    extraBody: Record<string, unknown> = {},
  ) {
    const baseline = upstream!.receivedRequests.length;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${plaintext}`,
        "content-type": "application/json",
        ...headers,
      },
      body: JSON.stringify({
        model: "hdrctx-openai-model",
        messages: [{ role: "user", content: "hello" }],
        ...extraBody,
      }),
    });
    await res.text();
    const sent = upstream!.receivedRequests.slice(baseline);
    return { status: res.status, sent };
  }

  test(
    "default_headers render request-context variables; unresolved ones are dropped",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      await waitConfigPropagation(async () => {
        const { status } = await chat(CALLER_PLAINTEXT, {});
        return status === 200;
      });

      const { status, sent } = await chat(CALLER_PLAINTEXT, {});
      expect(status).toBe(200);
      expect(sent).toHaveLength(1);
      const h = sent[0]!.headers;

      expect(h["x-tenant-id"]).toBe(TEAM_ID);
      expect(h["x-caller-key"]).toBe(KEY_NAME);
      expect(h["x-provider-key"]).toBe("hdrctx-openai-pk");
      expect(h["x-corp-static"]).toBe("static-value");
      // `${model.name}` is the gateway-facing display name, not the
      // upstream model id — the operator configured the former.
      expect(h["x-corp-model"]).toBe("hdrctx-openai-model");

      // A key with no team must not send `x-tenant-id: ""` — a blank
      // header reads to the upstream as "the tenant is the empty
      // string", which is a different and wrong claim.
      const teamless = await chat(TEAMLESS_PLAINTEXT, {});
      expect(teamless.status).toBe(200);
      const th = teamless.sent[0]!.headers;
      expect(th["x-tenant-id"]).toBeUndefined();
      expect(th["x-caller-key"]).toBe("teamless-key");
    },
    30_000,
  );

  test(
    "allowlisted client headers reach upstream; unlisted ones do not",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      const { status, sent } = await chat(CALLER_PLAINTEXT, {
        "anthropic-beta": "tools-2024-05-16",
        "x-trace-id": "trace-abc",
        "x-trace-parent": "parent-def",
        // Not named by any allowlist entry.
        "x-not-allowlisted": "nope",
      });
      expect(status).toBe(200);
      const h = sent[0]!.headers;

      expect(h["anthropic-beta"]).toBe("tools-2024-05-16");
      expect(h["x-trace-id"]).toBe("trace-abc");
      expect(h["x-trace-parent"]).toBe("parent-def");
      expect(h["x-not-allowlisted"]).toBeUndefined();
    },
    30_000,
  );

  test(
    "operator default_headers win over a forwarded client header of the same name",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      const { status, sent } = await chat(CALLER_PLAINTEXT, {
        "x-overlap": "from-client",
      });
      expect(status).toBe(200);
      // Single-valued, and the operator's value — not a two-valued
      // `from-operator, from-client` list.
      expect(sent[0]!.headers["x-overlap"]).toBe("from-operator");
    },
    30_000,
  );

  test(
    "auth, gateway-owned and transport headers are never forwarded",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      const { status, sent } = await chat(CALLER_PLAINTEXT, {
        "x-api-key": CALLER_PLAINTEXT,
        cookie: "session=secret",
        "x-aisix-request-id": "spoofed-by-caller",
        "x-stainless-lang": "js",
      });
      expect(status).toBe(200);
      const req = sent[0]!;

      // The upstream sees the ProviderKey's credential, and the caller's
      // plaintext appears nowhere in the request.
      expect(req.headers.authorization).toBe(`Bearer ${PROVIDER_SECRET}`);
      for (const [name, value] of Object.entries(req.headers)) {
        expect(value, `caller plaintext leaked in "${name}"`).not.toContain(
          CALLER_PLAINTEXT,
        );
      }
      expect(req.body).not.toContain(CALLER_PLAINTEXT);
      expect(req.headers.cookie).toBeUndefined();
      expect(req.headers["x-stainless-lang"]).toBeUndefined();
      // The gateway stamps its own correlation id; the caller's forged
      // value must not be what the upstream sees.
      expect(req.headers["x-aisix-request-id"]).not.toBe("spoofed-by-caller");
    },
    30_000,
  );

  test(
    "streaming dispatch sends the same headers as non-streaming",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      const { status, sent } = await chat(
        CALLER_PLAINTEXT,
        { "x-trace-id": "trace-stream" },
        { stream: true },
      );
      expect(status).toBe(200);
      const h = sent[0]!.headers;
      expect(h["x-tenant-id"]).toBe(TEAM_ID);
      expect(h["x-caller-key"]).toBe(KEY_NAME);
      expect(h["x-trace-id"]).toBe("trace-stream");
      expect(h.authorization).toBe(`Bearer ${PROVIDER_SECRET}`);
    },
    30_000,
  );

  test(
    "the Anthropic native passthrough applies the same pipeline",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      const baseline = anthropicUpstream!.receivedRequests.length;
      const res = await fetch(`${app.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
          "anthropic-beta": "prompt-caching-2024-07-31",
          "x-not-allowlisted": "nope",
        },
        body: JSON.stringify({
          model: "hdrctx-anthropic-model",
          max_tokens: 16,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      await res.text();
      expect(res.status).toBe(200);

      const sent = anthropicUpstream!.receivedRequests.slice(baseline);
      expect(sent).toHaveLength(1);
      const h = sent[0]!.headers;
      expect(h["anthropic-beta"]).toBe("prompt-caching-2024-07-31");
      expect(h["x-tenant-id"]).toBe(TEAM_ID);
      expect(h["x-provider-key"]).toBe("hdrctx-anthropic-pk");
      expect(h["x-model-id"]).toBe(anthropicModelId);
      expect(h["x-not-allowlisted"]).toBeUndefined();
      // Anthropic auth shape stays the gateway's own.
      expect(h["x-api-key"]).toBe(PROVIDER_SECRET);
    },
    30_000,
  );

  test(
    "id variables resolve to the resource ids, and request.id to this request",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      // `${model.id}` / `${provider_key.id}` are the etcd resource ids;
      // assert against the ids the seed returned, so a regression that
      // resolved them off the wrong entry is caught rather than merely
      // producing some non-empty string.
      const baseline = upstream.receivedRequests.length;
      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "hdrctx-openai-model",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      await res.text();
      expect(res.status).toBe(200);
      const h = upstream.receivedRequests.slice(baseline)[0]!.headers;

      expect(h["x-model-id"]).toBe(openaiModelId);
      expect(h["x-pk-id"]).toBe(openaiPkId);
      // `${request.id}` is this request's correlation id — the same one
      // the gateway echoes to the caller and stamps upstream.
      expect(h["x-corp-request-id"]).toBe(h["x-aisix-request-id"]);
      expect(h["x-corp-request-id"]).toBeTruthy();
    },
    30_000,
  );
});
