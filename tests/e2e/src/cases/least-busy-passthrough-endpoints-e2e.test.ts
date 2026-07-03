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

// least_busy on the passthrough endpoints. The in-flight counter used
// to be fed by /v1/chat/completions only: /v1/messages and
// /v1/responses dispatched without ever raising it, so a least_busy
// group serving Anthropic-SDK or Codex traffic saw all-zero counts and
// silently degraded to declaration order (failover) — same
// silent-degradation class as AISIX-Cloud#954. These tests mirror
// least-busy-routing-e2e.test.ts (chat) for both endpoints: occupy the
// declared-first slow target with an un-awaited request, then assert
// the next request diverts to the idle one.

const CALLER_PLAINTEXT = "sk-least-busy-passthrough-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function chatBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
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

function responsesShapeBody(text: string) {
  return {
    id: `resp_${text}`,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status: "completed",
    model: "gpt-4o-mini",
    output: [
      {
        id: `msg_${text}`,
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text }],
      },
    ],
    usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
  };
}

describe("least-busy routing via passthrough endpoints e2e", () => {
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

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!admin) throw new Error("admin client not initialized");
    const providerKey = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
    });
  }

  async function postMessages(model: string, content: string) {
    const res = await fetch(`${app?.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 64,
        messages: [{ role: "user", content }],
      }),
    });
    const body = (await res.json()) as {
      content?: Array<{ text?: string }>;
    };
    return { status: res.status, text: body.content?.[0]?.text ?? "" };
  }

  async function postResponses(model: string, input: string) {
    const res = await fetch(`${app?.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, input }),
    });
    const body = (await res.json()) as { object?: string };
    return { status: res.status, object: body.object ?? "" };
  }

  test("/v1/messages routes away from an in-flight target", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // Declared-first target A is slow (stays in flight); B is fast.
    // openai-provider targets exercise the cross-provider bridge path.
    const slow = await startOpenAiUpstream({
      responseDelayMs: 800,
      nonStreamBody: chatBody("msg-a-served"),
    });
    const fast = await startOpenAiUpstream({
      nonStreamBody: chatBody("msg-b-served"),
    });
    upstreams.push(slow, fast);

    await createOpenAiModel("msg-busy-a", slow);
    await createOpenAiModel("msg-busy-b", fast);
    await admin.createModel({
      display_name: "msg-busy-virtual",
      routing: {
        strategy: "least_busy",
        targets: [{ model: "msg-busy-a" }, { model: "msg-busy-b" }],
      },
    });

    // Gate on the group dispatching through the DP snapshot; the warmup
    // completes before we proceed, so its in-flight count is released.
    await waitConfigPropagation(async () => {
      const r = await postMessages("msg-busy-virtual", "warmup");
      return r.status === 200;
    });

    const slowBase = slow.receivedRequests.length;
    const fastBase = fast.receivedRequests.length;

    // Occupy A (both idle → declaration order) and do NOT await it.
    // Wait until the slow upstream has actually RECEIVED it — the
    // in-flight guard is acquired before the upstream send, so
    // arrival implies the count is raised (a fixed sleep can race on
    // slow CI).
    const inflight = postMessages("msg-busy-virtual", "occupy-a");
    await waitConfigPropagation(
      async () => slow.receivedRequests.length - slowBase >= 1,
    );

    // A has 1 in-flight, B has 0 → least_busy must divert to B. Before
    // the fix /v1/messages never raised the count, so this landed on A.
    const diverted = await postMessages("msg-busy-virtual", "divert-to-b");
    expect(diverted.status).toBe(200);
    expect(diverted.text).toBe("msg-b-served");

    const first = await inflight;
    expect(first.status).toBe(200);
    expect(first.text).toBe("msg-a-served");

    expect(slow.receivedRequests.length - slowBase).toBe(1);
    expect(fast.receivedRequests.length - fastBase).toBe(1);
  });

  test("/v1/responses routes away from an in-flight target", async (ctx) => {
    if (!etcdReachable || !app || !admin) {
      ctx.skip();
      return;
    }

    // openai-provider targets exercise the Responses passthrough path.
    const slow = await startOpenAiUpstream({
      responseDelayMs: 800,
      nonStreamBody: responsesShapeBody("resp-a-served"),
    });
    const fast = await startOpenAiUpstream({
      nonStreamBody: responsesShapeBody("resp-b-served"),
    });
    upstreams.push(slow, fast);

    await createOpenAiModel("resp-busy-a", slow);
    await createOpenAiModel("resp-busy-b", fast);
    await admin.createModel({
      display_name: "resp-busy-virtual",
      routing: {
        strategy: "least_busy",
        targets: [{ model: "resp-busy-a" }, { model: "resp-busy-b" }],
      },
    });

    await waitConfigPropagation(async () => {
      const r = await postResponses("resp-busy-virtual", "warmup");
      return r.status === 200 && r.object === "response";
    });

    const slowBase = slow.receivedRequests.length;
    const fastBase = fast.receivedRequests.length;

    // Same deterministic gate as the messages test above.
    const inflight = postResponses("resp-busy-virtual", "occupy-a");
    await waitConfigPropagation(
      async () => slow.receivedRequests.length - slowBase >= 1,
    );

    const diverted = await postResponses("resp-busy-virtual", "divert-to-b");
    expect(diverted.status).toBe(200);

    const first = await inflight;
    expect(first.status).toBe(200);

    // Attribution via per-upstream request counters: the diverted
    // request must have reached the idle target, not queued behind A.
    expect(slow.receivedRequests.length - slowBase).toBe(1);
    expect(fast.receivedRequests.length - fastBase).toBe(1);
  });
});
