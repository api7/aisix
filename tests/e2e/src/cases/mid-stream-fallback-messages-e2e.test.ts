import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type OpenAiUpstreamOptions,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER_PLAINTEXT = "sk-mid-stream-messages-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// Anthropic-wire SSE frames (verbatim, incl. framing) for the
// passthrough upstream mocks.
const frame = (event: string, data: Record<string, unknown>) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

const MESSAGE_START = frame("message_start", {
  type: "message_start",
  message: {
    id: "msg_up1",
    type: "message",
    role: "assistant",
    content: [],
    model: "claude-3-5-haiku-20241022",
    stop_reason: null,
    usage: { input_tokens: 5, output_tokens: 1 },
  },
});
const BLOCK_START = frame("content_block_start", {
  type: "content_block_start",
  index: 0,
  content_block: { type: "text", text: "" },
});
const delta = (text: string) =>
  frame("content_block_delta", {
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text },
  });
const RECOVERY_FRAMES = [
  frame("message_start", {
    type: "message_start",
    message: {
      id: "msg_up2",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 6, output_tokens: 0 },
    },
  }),
  BLOCK_START,
  delta(" a time."),
  frame("content_block_stop", { type: "content_block_stop", index: 0 }),
  frame("message_delta", {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: 7 },
  }),
  frame("message_stop", { type: "message_stop" }),
];

// AISIX-Cloud#1222 phase 2: mid-stream fallback on /v1/messages.
// Covers the transports the Rust wiremock suite cannot simulate — a
// real mid-body connection drop and a real inter-frame stall on the
// Anthropic PASSTHROUGH leg (byte-verbatim, resumed-encoder splice),
// plus the cross-protocol recovery (Anthropic-wire head, OpenAI-wire
// fallback) and the client-cancel non-trigger.
describe("mid-stream fallback /v1/messages e2e", () => {
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

  async function anthropicUpstream(
    opts: OpenAiUpstreamOptions,
  ): Promise<OpenAiUpstream> {
    const upstream = await startOpenAiUpstream(opts);
    upstreams.push(upstream);
    return upstream;
  }

  async function createAnthropicTarget(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: displayName,
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: providerKey.id,
    });
  }

  async function createOpenAiTarget(
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

  async function streamMessages(
    model: string,
    signal?: AbortSignal,
  ): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 128,
        stream: true,
        messages: [{ role: "user", content: "tell me a story" }],
      }),
      signal,
    });
    expect(res.status).toBe(200);
    return res;
  }

  async function readAll(res: Response): Promise<string> {
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let wire = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      wire += decoder.decode(value, { stream: true });
    }
    return wire;
  }

  test(
    "passthrough: connection drop mid-stream continues the client envelope on the fallback target",
    { timeout: 30_000 },
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      // eventDelayMs keeps a real inter-frame gap so the delivered
      // frames are flushed (and ACKed) before the RST — without it the
      // kernel discards receiver-buffered data on destroy() and the
      // failure degrades into a pre-first-frame abort.
      const primary = await anthropicUpstream({
        rawStreamFrames: [MESSAGE_START, BLOCK_START, delta("Once upon")],
        eventDelayMs: 200,
        disconnectAfterEvents: 3,
      });
      const secondary = await anthropicUpstream({
        rawStreamFrames: RECOVERY_FRAMES,
      });
      await createAnthropicTarget("msf-msg-drop-a", primary);
      await createAnthropicTarget("msf-msg-drop-b", secondary);
      await createGroup(
        "msf-msg-drop",
        ["msf-msg-drop-a", "msf-msg-drop-b"],
        { mode: "continue" },
      );
      await waitSeedApplied("msf-msg-drop");

      const wire = await readAll(await streamMessages("msf-msg-drop"));
      // One committed envelope across the switch, text block continues.
      expect(wire.match(/event: message_start/g)).toHaveLength(1);
      expect(wire.match(/event: content_block_start/g)).toHaveLength(1);
      expect(wire).toContain("Once upon");
      expect(wire).toContain(" a time.");
      expect(wire).not.toContain("event: error");
      expect(wire.match(/event: message_stop/g)).toHaveLength(1);

      // The fallback saw the continuation: instruction + the partial as
      // the trailing (prefill) assistant message on the Anthropic wire.
      const cont = secondary.receivedRequests.find((r) =>
        r.path.includes("/messages"),
      );
      expect(cont).toBeDefined();
      expect(cont!.body).toContain("interrupted mid-stream");
      const body = JSON.parse(cont!.body) as {
        messages: Array<{ role: string }>;
      };
      expect(body.messages.at(-1)?.role).toBe("assistant");
    },
  );

  test(
    "passthrough: inter-frame stall (read timeout) fails over inside the stream",
    { timeout: 30_000 },
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      // Head flows quickly, then the upstream hangs without closing —
      // the armed combinator owns the read timeout and classifies the
      // stall as `read_timeout`.
      const primary = await anthropicUpstream({
        rawStreamFrames: [
          MESSAGE_START,
          BLOCK_START,
          delta("Once upon"),
          delta(" (never sent)"),
        ],
        eventDelayMs: 100,
        stallAfterEvents: 3,
      });
      const secondary = await anthropicUpstream({
        rawStreamFrames: RECOVERY_FRAMES,
      });
      await createAnthropicTarget("msf-msg-stall-a", primary);
      await createAnthropicTarget("msf-msg-stall-b", secondary);
      await createGroup(
        "msf-msg-stall",
        ["msf-msg-stall-a", "msf-msg-stall-b"],
        { mode: "continue", on: ["read_timeout"] },
        { stream_timeout: 1500 },
      );
      await waitSeedApplied("msf-msg-stall");

      const wire = await readAll(await streamMessages("msf-msg-stall"));
      expect(wire.match(/event: message_start/g)).toHaveLength(1);
      expect(wire).toContain("Once upon");
      expect(wire).toContain(" a time.");
      expect(wire).not.toContain("event: error");
      expect(wire.match(/event: message_stop/g)).toHaveLength(1);
    },
  );

  test(
    "cross-protocol: anthropic-wire head resumes on an OpenAI-protocol fallback",
    { timeout: 30_000 },
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const primary = await anthropicUpstream({
        rawStreamFrames: [
          MESSAGE_START,
          BLOCK_START,
          delta("Once upon"),
          frame("error", {
            type: "error",
            error: { type: "overloaded_error", message: "Overloaded" },
          }),
        ],
        eventDelayMs: 100,
      });
      const chunk = (content: string, finish: string | null = null) =>
        JSON.stringify({
          id: "up-2",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content }, finish_reason: finish }],
        });
      const secondary = await startOpenAiUpstream({
        streamEvents: [chunk(" a time."), chunk("", "stop"), "[DONE]"],
      });
      upstreams.push(secondary);
      await createAnthropicTarget("msf-msg-xp-a", primary);
      await createOpenAiTarget("msf-msg-xp-b", secondary);
      await createGroup("msf-msg-xp", ["msf-msg-xp-a", "msf-msg-xp-b"], {
        mode: "continue",
      });
      await waitSeedApplied("msf-msg-xp");

      const wire = await readAll(await streamMessages("msf-msg-xp"));
      expect(wire.match(/event: message_start/g)).toHaveLength(1);
      expect(wire).toContain("Once upon");
      expect(wire).toContain(" a time.");
      expect(wire).not.toContain("event: error");
      expect(wire.match(/event: message_stop/g)).toHaveLength(1);

      // OpenAI-wire continuation body: instruction + partial assistant.
      const cont = secondary.receivedRequests.find((r) =>
        r.path.includes("/chat/completions"),
      );
      expect(cont).toBeDefined();
      const body = JSON.parse(cont!.body) as {
        messages: Array<{ role: string; content: string }>;
      };
      const last = body.messages.at(-1);
      expect(last?.role).toBe("assistant");
      expect(last?.content).toContain("Once upon");
      expect(
        body.messages.some(
          (m) =>
            m.role === "system" &&
            m.content.includes("do not repeat any of its content"),
        ),
      ).toBe(true);
    },
  );

  test(
    "client cancel mid-stream never dispatches the fallback target",
    { timeout: 30_000 },
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const primary = await anthropicUpstream({
        rawStreamFrames: [
          MESSAGE_START,
          BLOCK_START,
          delta("Once upon"),
          delta(" a slow"),
          delta(" stream"),
        ],
        eventDelayMs: 500,
      });
      const secondary = await anthropicUpstream({
        rawStreamFrames: RECOVERY_FRAMES,
      });
      await createAnthropicTarget("msf-msg-cancel-a", primary);
      await createAnthropicTarget("msf-msg-cancel-b", secondary);
      await createGroup(
        "msf-msg-cancel",
        ["msf-msg-cancel-a", "msf-msg-cancel-b"],
        { mode: "continue" },
      );
      await waitSeedApplied("msf-msg-cancel");

      const abort = new AbortController();
      const res = await streamMessages("msf-msg-cancel", abort.signal);
      const reader = res.body!.getReader();
      // Read a couple of chunks, then walk away mid-stream.
      await reader.read();
      await reader.read();
      abort.abort();
      // Give any (incorrect) fallback dispatch time to happen.
      await new Promise((r) => setTimeout(r, 2_000));
      expect(
        secondary.receivedRequests.filter((r) => r.path.includes("/messages")),
      ).toHaveLength(0);
    },
  );
});
