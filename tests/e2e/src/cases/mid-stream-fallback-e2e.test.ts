import { createHash } from "node:crypto";
import OpenAI from "openai";
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

const CALLER_PLAINTEXT = "sk-mid-stream-fallback-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const chunk = (content: string, finish: string | null = null) =>
  JSON.stringify({
    id: "up-1",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, delta: { content }, finish_reason: finish },
    ],
  });

// AISIX-Cloud#1222: `routing.stream_failure: continue` — recover a
// streaming response INSIDE the committed 200 after the upstream fails
// mid-generation. Covers the transports the Rust integration tests
// cannot simulate with wiremock: a real mid-body connection drop and a
// real inter-chunk stall, plus the client-cancel non-trigger.
describe("mid-stream fallback e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function createTarget(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
    });
  }

  async function createGroup(
    name: string,
    targets: string[],
    streamFailure: Record<string, unknown>,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    await seed.createModel({
      display_name: name,
      routing: {
        strategy: "failover",
        targets: targets.map((t) => ({ model: t })),
        stream_failure: streamFailure,
      },
      ...extra,
    });
  }

  // Watch events apply in revision order: once a canary key written
  // AFTER the resources authenticates, everything before it is live.
  async function waitSeedApplied(label: string): Promise<void> {
    const canary = `sk-canary-${label}-${Date.now()}`;
    await seed!.createApiKey({
      key_hash: createHash("sha256").update(canary).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${canary}` },
      });
      return res.status === 200;
    });
  }

  function sdk(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app?.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  test("connection drop mid-stream continues on the fallback target in the same stream", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Primary streams two content chunks then destroys the socket —
    // a real transport break, no error frame at all. The inter-event
    // delay lets each write reach the gateway before the RST (a reset
    // discards any data still sitting in the receiver's buffer, which
    // would turn this into a pre-first-chunk failure instead).
    const primary = await startOpenAiUpstream({
      streamEvents: [
        chunk("Once "),
        chunk("upon "),
        chunk("NEVER-SENT"),
      ],
      eventDelayMs: 200,
      disconnectAfterEvents: 2,
    });
    upstreams.push(primary);
    const secondary = await startOpenAiUpstream({
      streamEvents: [chunk("a time."), chunk("", "stop"), "[DONE]"],
    });
    upstreams.push(secondary);

    await createTarget("msf-drop-primary", primary);
    await createTarget("msf-drop-secondary", secondary);
    await createGroup(
      "msf-drop-group",
      ["msf-drop-primary", "msf-drop-secondary"],
      { mode: "continue" },
    );
    await waitSeedApplied("msf-drop");

    const collected: string[] = [];
    let sawFinish = false;
    let surfacedError = false;
    const stream = await sdk().chat.completions.create({
      model: "msf-drop-group",
      messages: [{ role: "user", content: "tell me a story" }],
      stream: true,
    });
    try {
      for await (const c of stream) {
        const delta = c.choices[0]?.delta;
        if (delta?.content) collected.push(delta.content);
        if (c.choices[0]?.finish_reason) sawFinish = true;
      }
    } catch {
      surfacedError = true;
    }

    // The client saw primary content, then the fallback's
    // continuation, then a clean completion — one logical answer.
    expect(collected.join("")).toBe("Once upon a time.");
    expect(sawFinish).toBe(true);
    expect(surfacedError).toBe(false);

    // The fallback received the original messages plus the
    // continuation instruction and the partial as an assistant
    // message (LiteLLM's mid-stream fallback shape).
    const calls = secondary.receivedRequests.filter((r) =>
      r.path.endsWith("/chat/completions"),
    );
    expect(calls.length).toBe(1);
    const body = JSON.parse(calls[0].body) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(body.messages.length).toBe(3);
    expect(body.messages[0].role).toBe("user");
    expect(body.messages[1].role).toBe("system");
    expect(body.messages[1].content).toContain(
      "do not repeat any of its content",
    );
    expect(body.messages[2].role).toBe("assistant");
    expect(body.messages[2].content).toBe("Once upon ");
  }, 30_000);

  test("inter-chunk stall past stream_timeout falls back when read_timeout is a trigger", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Primary sends one chunk fast, then stalls far past the group's
    // stream_timeout before the next one.
    const primary = await startOpenAiUpstream({
      streamEvents: [chunk("The answer "), chunk("NEVER-ARRIVES")],
      eventDelayMs: 5_000,
    });
    upstreams.push(primary);
    const secondary = await startOpenAiUpstream({
      streamEvents: [chunk("is 42."), chunk("", "stop"), "[DONE]"],
    });
    upstreams.push(secondary);

    await createTarget("msf-stall-primary", primary);
    await createTarget("msf-stall-secondary", secondary);
    await createGroup(
      "msf-stall-group",
      ["msf-stall-primary", "msf-stall-secondary"],
      { mode: "continue", on: ["read_timeout", "transport_error"] },
      // Group-level per-chunk budget: 1.5s gaps time out (#809 —
      // stream_timeout is per chunk, not whole-response).
      { stream_timeout: 1_500 },
    );
    await waitSeedApplied("msf-stall");

    const collected: string[] = [];
    let sawFinish = false;
    const stream = await sdk().chat.completions.create({
      model: "msf-stall-group",
      messages: [{ role: "user", content: "what is the answer" }],
      stream: true,
    });
    for await (const c of stream) {
      const delta = c.choices[0]?.delta;
      if (delta?.content) collected.push(delta.content);
      if (c.choices[0]?.finish_reason) sawFinish = true;
    }

    expect(collected.join("")).toBe("The answer is 42.");
    expect(sawFinish).toBe(true);
    const calls = secondary.receivedRequests.filter((r) =>
      r.path.endsWith("/chat/completions"),
    );
    expect(calls.length).toBe(1);
    const body = JSON.parse(calls[0].body) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(body.messages[2]?.content).toBe("The answer ");
  }, 30_000);

  test("client cancel mid-stream never dispatches the fallback target", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    // Primary drips chunks slowly enough for the client to abort
    // between them; the eventual disconnect after the abort must NOT
    // start a fallback request (no ghost upstream traffic, #1094).
    const primary = await startOpenAiUpstream({
      streamEvents: [
        chunk("drip "),
        chunk("drip "),
        chunk("drip "),
        chunk("", "stop"),
        "[DONE]",
      ],
      eventDelayMs: 500,
    });
    upstreams.push(primary);
    const secondary = await startOpenAiUpstream({
      streamEvents: [chunk("ghost"), chunk("", "stop"), "[DONE]"],
    });
    upstreams.push(secondary);

    await createTarget("msf-cancel-primary", primary);
    await createTarget("msf-cancel-secondary", secondary);
    await createGroup(
      "msf-cancel-group",
      ["msf-cancel-primary", "msf-cancel-secondary"],
      { mode: "continue" },
      { stream_timeout: 1_000 },
    );
    await waitSeedApplied("msf-cancel");

    const stream = await sdk().chat.completions.create({
      model: "msf-cancel-group",
      messages: [{ role: "user", content: "drip feed" }],
      stream: true,
    });
    // Take the first content chunk, then abandon the stream.
    for await (const c of stream) {
      if (c.choices[0]?.delta?.content) break;
    }
    stream.controller.abort();

    // Give the gateway ample time to (wrongly) fire a fallback if the
    // cancel path were broken — including the 1s read-timeout window.
    await new Promise((r) => setTimeout(r, 3_000));
    const calls = secondary.receivedRequests.filter((r) =>
      r.path.endsWith("/chat/completions"),
    );
    expect(calls.length).toBe(0);
  }, 30_000);
});
