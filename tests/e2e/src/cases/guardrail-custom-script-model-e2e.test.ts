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

// E2E: a `custom` guardrail script sees `ctx.model` on the input hook of
// every LLM endpoint. On `/v1/chat/completions`, `/v1/responses` and
// `/v1/messages` a custom script runs only through the segment pass, which
// used to hand it no model at all — so a policy that branches on the model
// silently never fired there.
//
// `ctx.model` is the model the request addressed: the name in the request
// body, not the upstream model it is served by. The aliases below differ
// from their `model_name` so the two cannot be confused.

const CALLER = "sk-custom-script-model-e2e";
const HASH = createHash("sha256").update(CALLER).digest("hex");

const BLOCKED = "csm-restricted";
const ALLOWED = "csm-open";
// Anthropic-backed pair: `/v1/messages/count_tokens` serves only those.
const CT_BLOCKED = "csm-ct-restricted";
const CT_ALLOWED = "csm-ct-open";
// Routing parents. The script restricts ROUTER_BLOCKED, whose only target
// is the open model, and does not restrict ROUTER_ALLOWED, whose only
// target is the restricted one — so a script that saw the dispatched
// target instead of the addressed name gets both backwards.
const ROUTER_BLOCKED = "csm-router-restricted";
const ROUTER_ALLOWED = "csm-router-open";

const RESTRICTED = [BLOCKED, CT_BLOCKED, ROUTER_BLOCKED];

const SCRIPT = `
export async function checkInput(ctx) {
  if (${JSON.stringify(RESTRICTED)}.indexOf(ctx.model) !== -1) {
    return { action: "block", reason_code: "MODEL-RESTRICTED" };
  }
  return { action: "none" };
}
`;

describe("custom guardrail script: ctx.model on the segment pass", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "csm-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const gr = await seed.createGuardrail(
      {
        name: "csm-model-policy",
        enabled: true,
        hook_point: "input",
        fail_open: false,
        kind: "custom",
        script: SCRIPT,
        timeout_ms: 5000,
      },
      { attach: false },
    );
    for (const alias of [BLOCKED, ALLOWED]) {
      const model = await seed.createModel({
        display_name: alias,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await seed.attachGuardrailToModel(gr.id, model.id);
    }

    const anthropicPk = await seed.createProviderKey({
      display_name: "csm-anthropic-pk",
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    for (const alias of [CT_BLOCKED, CT_ALLOWED]) {
      const model = await seed.createModel({
        display_name: alias,
        provider: "anthropic",
        model_name: "claude-3-5-haiku-20241022",
        provider_key_id: anthropicPk.id,
      });
      await seed.attachGuardrailToModel(gr.id, model.id);
    }

    for (const [alias, target] of [
      [ROUTER_BLOCKED, ALLOWED],
      [ROUTER_ALLOWED, BLOCKED],
    ]) {
      const model = await seed.createModel({
        display_name: alias,
        routing: { strategy: "failover", targets: [{ model: target }] },
      });
      await seed.attachGuardrailToModel(gr.id, model.id);
    }

    // Seeded last: this key authenticating implies everything above it is
    // in the gateway's snapshot.
    await seed.createApiKey({
      key_hash: HASH,
      allowed_models: [
        BLOCKED,
        ALLOWED,
        CT_BLOCKED,
        CT_ALLOWED,
        ROUTER_BLOCKED,
        ROUTER_ALLOWED,
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const post = (path: string, body: unknown): Promise<Response> =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
      },
      body: JSON.stringify(body),
    });

  const send = {
    chat: (model: string) =>
      post("/v1/chat/completions", {
        model,
        messages: [{ role: "user", content: "hello" }],
      }),
    responses: (model: string) =>
      post("/v1/responses", { model, input: "hello" }),
    messages: (model: string) =>
      post("/v1/messages", {
        model,
        max_tokens: 16,
        messages: [{ role: "user", content: "hello" }],
      }),
    completions: (model: string) =>
      post("/v1/completions", { model, prompt: "hello", max_tokens: 16 }),
  };

  // The caller key is seeded after the guardrail and both models, so the
  // key authenticating means the policy is loaded too. Gated on the open
  // model so the wait never depends on the behaviour under test.
  async function waitForPolicy(): Promise<void> {
    await waitConfigPropagation(async () => {
      const res = await send.chat(ALLOWED);
      await res.text();
      return res.status === 200;
    });
  }

  for (const endpoint of ["chat", "responses", "messages", "completions"] as const) {
    test(`${endpoint}: a script branching on ctx.model blocks only the addressed model`, async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      await waitForPolicy();

      const before = upstream.receivedRequests.length;
      const blocked = await send[endpoint](BLOCKED);
      const blockedBody = await blocked.text();
      expect(blocked.status, blockedBody).toBe(422);
      expect(
        upstream.receivedRequests.length - before,
        "a blocked request never reaches the upstream",
      ).toBe(0);

      const allowed = await send[endpoint](ALLOWED);
      const allowedBody = await allowed.text();
      expect(
        allowed.status,
        `the same script lets the other model through: ${allowedBody}`,
      ).not.toBe(422);
    });
  }

  test("count_tokens: a script branching on ctx.model blocks only the addressed model", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitForPolicy();
    const count = (model: string) =>
      post("/v1/messages/count_tokens", {
        model,
        messages: [{ role: "user", content: "hello" }],
      });

    const before = upstream.receivedRequests.length;
    const blocked = await count(CT_BLOCKED);
    const blockedBody = await blocked.text();
    expect(blocked.status, blockedBody).toBe(422);
    expect(upstream.receivedRequests.length - before).toBe(0);

    const allowed = await count(CT_ALLOWED);
    const allowedBody = await allowed.text();
    expect(allowed.status, allowedBody).not.toBe(422);
    expect(
      upstream.receivedRequests.length - before,
      "the open model's count reached the upstream, so the 422 above was the policy",
    ).toBe(1);
  });

  test("routing: the script sees the routing model the caller addressed, not the dispatched target", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitForPolicy();

    const blocked = await send.chat(ROUTER_BLOCKED);
    const blockedBody = await blocked.text();
    expect(
      blocked.status,
      `the restricted routing parent is refused although its target is open: ${blockedBody}`,
    ).toBe(422);

    const allowed = await send.chat(ROUTER_ALLOWED);
    const allowedBody = await allowed.text();
    expect(
      allowed.status,
      `the open routing parent is served although its target is restricted: ${allowedBody}`,
    ).toBe(200);
  });
});
