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

const CALLER_PLAINTEXT = "sk-mid-stream-responses-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// OpenAI Responses-wire SSE frames (verbatim, incl. framing) for the
// verbatim upstream mocks.
const frame = (event: string, data: Record<string, unknown>) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

let seqCounter = 0;
const seq = () => seqCounter++;
const headFrames = () => {
  seqCounter = 0;
  return [
    frame("response.created", {
      type: "response.created",
      sequence_number: seq(),
      response: {
        id: "resp_up1",
        object: "response",
        created_at: 170,
        status: "in_progress",
        model: "gpt-4o-mini",
        output: [],
      },
    }),
    frame("response.output_item.added", {
      type: "response.output_item.added",
      sequence_number: seq(),
      output_index: 0,
      item: {
        type: "message",
        id: "msg_item1",
        status: "in_progress",
        role: "assistant",
        content: [],
      },
    }),
    frame("response.content_part.added", {
      type: "response.content_part.added",
      sequence_number: seq(),
      item_id: "msg_item1",
      output_index: 0,
      content_index: 0,
      part: { type: "output_text", text: "", annotations: [] },
    }),
    frame("response.output_text.delta", {
      type: "response.output_text.delta",
      sequence_number: seq(),
      item_id: "msg_item1",
      output_index: 0,
      content_index: 0,
      delta: "Once upon",
    }),
  ];
};

const chatChunk = (content: string, finish: string | null = null) =>
  JSON.stringify({
    id: "up-2",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content }, finish_reason: finish }],
  });

// AISIX-Cloud#1222 phase 2: mid-stream fallback on /v1/responses.
// Covers the transports the Rust wiremock suite cannot simulate — a
// real mid-body connection drop and a real inter-frame stall on the
// verbatim (OpenAI-provider) leg, resumed onto a fallback through the
// chat bridge with the client's envelope intact — plus the
// client-cancel non-trigger.
describe("mid-stream fallback /v1/responses e2e", () => {
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

  async function upstream(
    opts: OpenAiUpstreamOptions,
  ): Promise<OpenAiUpstream> {
    const up = await startOpenAiUpstream(opts);
    upstreams.push(up);
    return up;
  }

  async function createOpenAiTarget(
    displayName: string,
    up: OpenAiUpstream,
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const providerKey = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${up.baseUrl}/v1`,
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

  async function streamResponses(
    model: string,
    signal?: AbortSignal,
  ): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        input: "tell me a story",
        stream: true,
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
    "verbatim: connection drop mid-stream continues the client envelope on the fallback target",
    { timeout: 30_000 },
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      // eventDelayMs keeps a real inter-frame gap so the delivered
      // frames are flushed (and ACKed) before the RST.
      const primary = await upstream({
        rawStreamFrames: headFrames(),
        eventDelayMs: 200,
        disconnectAfterEvents: 4,
      });
      const secondary = await upstream({
        streamEvents: [chatChunk(" a time."), chatChunk("", "stop"), "[DONE]"],
      });
      await createOpenAiTarget("msf-resp-drop-a", primary);
      await createOpenAiTarget("msf-resp-drop-b", secondary);
      await createGroup(
        "msf-resp-drop",
        ["msf-resp-drop-a", "msf-resp-drop-b"],
        { mode: "continue" },
      );
      await waitSeedApplied("msf-resp-drop");

      const wire = await readAll(await streamResponses("msf-resp-drop"));
      expect(wire.match(/event: response\.created/g)).toHaveLength(1);
      expect(wire.match(/event: response\.output_item\.added/g)).toHaveLength(
        1,
      );
      expect(wire).toContain("resp_up1");
      expect(wire).toContain("Once upon");
      expect(wire).toContain(" a time.");
      expect(wire).not.toContain("event: error");
      expect(wire.match(/event: response\.completed/g)).toHaveLength(1);
      expect(wire).toContain("Once upon a time.");

      // The fallback saw the continuation on the chat wire.
      const cont = secondary.receivedRequests.find((r) =>
        r.path.includes("/chat/completions"),
      );
      expect(cont).toBeDefined();
      expect(cont!.body).toContain("interrupted mid-stream");
      const body = JSON.parse(cont!.body) as {
        messages: Array<{ role: string; content: string }>;
      };
      expect(body.messages.at(-1)?.role).toBe("assistant");
      expect(body.messages.at(-1)?.content).toBe("Once upon");
    },
  );

  test(
    "verbatim: inter-frame stall (read timeout) fails over inside the stream",
    { timeout: 30_000 },
    async (ctx) => {
      if (!etcdReachable || !app || !seed) {
        ctx.skip();
        return;
      }
      const primary = await upstream({
        rawStreamFrames: [
          ...headFrames(),
          frame("response.output_text.delta", {
            type: "response.output_text.delta",
            sequence_number: 99,
            item_id: "msg_item1",
            output_index: 0,
            content_index: 0,
            delta: " (never sent)",
          }),
        ],
        eventDelayMs: 100,
        stallAfterEvents: 4,
      });
      const secondary = await upstream({
        streamEvents: [chatChunk(" a time."), chatChunk("", "stop"), "[DONE]"],
      });
      await createOpenAiTarget("msf-resp-stall-a", primary);
      await createOpenAiTarget("msf-resp-stall-b", secondary);
      await createGroup(
        "msf-resp-stall",
        ["msf-resp-stall-a", "msf-resp-stall-b"],
        { mode: "continue", on: ["read_timeout"] },
        { stream_timeout: 1500 },
      );
      await waitSeedApplied("msf-resp-stall");

      const wire = await readAll(await streamResponses("msf-resp-stall"));
      expect(wire.match(/event: response\.created/g)).toHaveLength(1);
      expect(wire).toContain("Once upon");
      expect(wire).toContain(" a time.");
      expect(wire).not.toContain("event: error");
      expect(wire.match(/event: response\.completed/g)).toHaveLength(1);
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
      const primary = await upstream({
        rawStreamFrames: [
          ...headFrames(),
          frame("response.output_text.delta", {
            type: "response.output_text.delta",
            sequence_number: 4,
            item_id: "msg_item1",
            output_index: 0,
            content_index: 0,
            delta: " a slow stream",
          }),
        ],
        eventDelayMs: 500,
      });
      const secondary = await upstream({
        streamEvents: [chatChunk(" a time."), chatChunk("", "stop"), "[DONE]"],
      });
      await createOpenAiTarget("msf-resp-cancel-a", primary);
      await createOpenAiTarget("msf-resp-cancel-b", secondary);
      await createGroup(
        "msf-resp-cancel",
        ["msf-resp-cancel-a", "msf-resp-cancel-b"],
        { mode: "continue" },
      );
      await waitSeedApplied("msf-resp-cancel");

      const abort = new AbortController();
      const res = await streamResponses("msf-resp-cancel", abort.signal);
      const reader = res.body!.getReader();
      await reader.read();
      await reader.read();
      abort.abort();
      await new Promise((r) => setTimeout(r, 2_000));
      expect(
        secondary.receivedRequests.filter((r) =>
          r.path.includes("/chat/completions"),
        ),
      ).toHaveLength(0);
    },
  );
});
