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

// E2E for the deployment-wide request-timeout default
// (`upstream.timeout_ms` / `upstream.stream_timeout_ms`).
//
// Before this knob, a Model without `timeout` had NO deadline at all: an
// upstream that accepted the connection and then went silent forever held
// the request open indefinitely. Now every model inherits a deployment
// backstop, resolved model → group → `upstream.timeout_ms`:
//   - a model with neither its own `timeout` nor a group one times out at
//     the deployment default (non-streaming AND the streaming budget);
//   - `timeout: 0` on the model opts it out of the backstop entirely;
//   - an explicit model `timeout` beats the deployment default in both
//     directions;
//   - a group's `timeout` applies to members that don't set their own.
//
// The gateway surfaces an elapsed upstream deadline as 504.

const CALLER_PLAINTEXT = "sk-timeout-defaults-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Deployment default under test; upstream stalls are comfortably longer.
const DEFAULT_MS = 1500;
const STALL_MS = 8000;
const SLOW_OK_MS = 3000;

function reply(content: string): unknown {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

function chunk(content: string): string {
  return JSON.stringify({
    id: "evt",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });
}

async function callChat(
  app: SpawnedApp,
  model: string,
  stream = false,
): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: "hi" }],
      ...(stream ? { stream: true } : {}),
    }),
  });
}

describe("deployment-wide upstream timeout default", () => {
  // App with a short deployment default — the subject under test.
  let app: SpawnedApp | undefined;
  // App with a long deployment default — proves the group-level fallback
  // (its 1.5s cut can only come from the group's own `timeout`).
  let grouped: SpawnedApp | undefined;
  let etcdReachable = false;

  let stalling: OpenAiUpstream | undefined;
  let slowOk: OpenAiUpstream | undefined;
  let streamStall: OpenAiUpstream | undefined;
  let groupStalling: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    stalling = await startOpenAiUpstream({
      responseDelayMs: STALL_MS,
      nonStreamBody: reply("stalled"),
    });
    slowOk = await startOpenAiUpstream({
      responseDelayMs: SLOW_OK_MS,
      nonStreamBody: reply("slow-ok"),
    });
    streamStall = await startOpenAiUpstream({
      // First chunk immediately, then a stall far past the default.
      eventDelayMs: STALL_MS,
      streamEvents: [chunk("first "), chunk("never "), "[DONE]"],
    });
    groupStalling = await startOpenAiUpstream({
      responseDelayMs: STALL_MS,
      nonStreamBody: reply("group-stalled"),
    });

    app = await spawnApp({ extra: { upstream: { timeout_ms: DEFAULT_MS } } });
    grouped = await spawnApp({ extra: { upstream: { timeout_ms: 60_000 } } });

    {
      const seed = new SeedClient(etcd, app.etcdPrefix);
      const pk = async (name: string, u: OpenAiUpstream) =>
        (
          await seed.createProviderKey({
            display_name: name,
            secret: "sk-mock",
            api_base: `${u.baseUrl}/v1`,
          })
        ).id;
      const stallingPk = await pk("td-stalling-pk", stalling);
      const slowOkPk = await pk("td-slow-ok-pk", slowOk);
      const streamStallPk = await pk("td-stream-stall-pk", streamStall);
      // No `timeout` anywhere → inherits the deployment default.
      await seed.createModel({
        display_name: "td-defaulted",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: stallingPk,
        cooldown: { enabled: false },
      });
      // Same silent upstream, `timeout: 0` → opted out of the backstop.
      await seed.createModel({
        display_name: "td-opted-out",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: slowOkPk,
        timeout: 0,
        cooldown: { enabled: false },
      });
      // Explicit model `timeout` above the deployment default wins.
      await seed.createModel({
        display_name: "td-explicit",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: slowOkPk,
        timeout: SLOW_OK_MS + 2000,
        cooldown: { enabled: false },
      });
      // Streaming: no `stream_timeout`/`timeout` → the deployment default
      // is the streaming budget too.
      await seed.createModel({
        display_name: "td-stream-defaulted",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: streamStallPk,
        cooldown: { enabled: false },
      });
      await seed.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: [
          "td-defaulted",
          "td-opted-out",
          "td-explicit",
          "td-stream-defaulted",
        ],
      });
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app!.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        });
        if (!res.ok) return false;
        const body = (await res.json()) as { data?: Array<{ id: string }> };
        return !!body.data?.some((m) => m.id === "td-stream-defaulted");
      });
    }

    {
      const seed = new SeedClient(etcd, grouped.etcdPrefix);
      const pkId = (
        await seed.createProviderKey({
          display_name: "td-group-pk",
          secret: "sk-mock",
          api_base: `${groupStalling.baseUrl}/v1`,
        })
      ).id;
      await seed.createModel({
        display_name: "td-member",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pkId,
        cooldown: { enabled: false },
      });
      // The group's own `timeout` — the member sets none, and the
      // deployment default here is 60s, so a fast 504 can only come from
      // this level.
      await seed.createModel({
        display_name: "td-group",
        timeout: DEFAULT_MS,
        routing: { strategy: "failover", targets: [{ model: "td-member" }] },
      });
      await seed.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["td-group"],
      });
      // Routing models are not listed on /v1/models — gate on a probe call
      // instead (the pattern timeout-fallback-e2e uses). A 504 means the
      // virtual model and its member are both loaded; before that the
      // gateway answers 404.
      await waitConfigPropagation(async () => {
        const res = await callChat(grouped!, "td-group");
        if (res.status !== 504) {
          await res.text();
          return false;
        }
        await res.text();
        return true;
      });
    }
  });

  afterAll(async () => {
    await Promise.all([app?.exit(), grouped?.exit()]);
    await Promise.all([
      stalling?.close(),
      slowOk?.close(),
      streamStall?.close(),
      groupStalling?.close(),
    ]);
  });

  test("a model with no timeout inherits the deployment default", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const started = Date.now();
    const res = await callChat(app, "td-defaulted");
    const elapsed = Date.now() - started;
    expect(res.status).toBe(504);
    // Cut by the 1.5s default, well before the 8s upstream stall.
    expect(elapsed).toBeGreaterThanOrEqual(DEFAULT_MS - 200);
    expect(elapsed).toBeLessThan(STALL_MS - 1500);
    const body = (await res.json()) as { error?: { message?: string } };
    expect(body.error?.message ?? "").toContain("timed out");
  }, 30_000);

  test("timeout: 0 opts a model out of the deployment default", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await callChat(app, "td-opted-out");
    // 3s upstream, 1.5s deployment default: only the opt-out lets this
    // complete.
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    expect(body.choices[0]?.message.content).toBe("slow-ok");
  }, 30_000);

  test("an explicit model timeout beats the deployment default", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await callChat(app, "td-explicit");
    expect(res.status).toBe(200);
  }, 30_000);

  test("streaming inherits the deployment default as its chunk budget", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const started = Date.now();
    const res = await callChat(app, "td-stream-defaulted", true);
    expect(res.status).toBe(200);
    const text = await res.text();
    const elapsed = Date.now() - started;
    // The first chunk arrived; the 8s mid-stream stall was cut by the
    // 1.5s default rather than waiting out the upstream.
    expect(text).toContain("first");
    expect(text).not.toContain("never");
    expect(elapsed).toBeLessThan(STALL_MS - 1500);
  }, 30_000);

  test("a group's timeout applies to members without their own", async (ctx) => {
    if (!etcdReachable || !grouped) return ctx.skip();
    const started = Date.now();
    const res = await callChat(grouped, "td-group");
    const elapsed = Date.now() - started;
    expect(res.status).toBe(504);
    // The deployment default in this app is 60s; only the group's own
    // 1.5s `timeout` can cut the call this early.
    expect(elapsed).toBeGreaterThanOrEqual(DEFAULT_MS - 200);
    expect(elapsed).toBeLessThan(STALL_MS - 1500);
  }, 30_000);
});
