import { createHash, randomUUID } from "node:crypto";
import { WebSocket as WsClient, WebSocketServer, type WebSocket as WsSocket } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startA2aUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type A2aUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a `kind: pii` mask rule on a surface that has no channel to write a
// mask back is reported, not silent.
//
// A passthrough route relays the caller's body and the provider's reply as
// bytes, and so do a job's body, a realtime frame, an A2A message and a video
// prompt. A mask rule that matches there cannot rewrite
// anything: the content is forwarded unmodified. Before this, the enforcing
// row left no trace at all — the event read exactly like a request with no
// PII in it — and the monitor row previewed `would_mask`, promising a
// redaction that enforcing it would never perform.
//
// Two env-scoped rows with the same email rule govern every surface here, one
// enforcing and one in monitor mode, so each request answers both halves:
//   - enforce: `guardrail_enforced_hits` carries `mask_unsupported` with the
//     detector counts, `guardrail_blocked` stays false, and
//     `redacted_entity_counts` stays empty — nothing was redacted;
//   - monitor: `guardrail_monitor_hits` carries `would_mask_unsupported`,
//     never `would_mask`.
// Each surface's row is found by a field independent of the ones under test.

const KEY = "sk-pii-mask-unsupported-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "pii-mask-unsupported";

const ENFORCE = "pmu-enforce";
const MONITOR = "pmu-monitor";
const ROUTE = "pmu-tunnel";
const STREAM_ROUTE = "pmu-stream-tunnel";
const AGENT = "pmu-a2a";

// Two addresses in every caller-authored body, one in every provider reply,
// so a count that merely says "something matched" cannot pass.
const IN_A = "alice@example.com";
const IN_B = "bob@example.com";
const OUT = "carol@example.com";
const PROMPT = `write to ${IN_A} and ${IN_B}`;
const REPLY = `ask ${OUT}`;

const chatReply = {
  id: "chatcmpl-pmu",
  object: "chat.completion",
  model: "gpt-4o",
  choices: [{ index: 0, message: { role: "assistant", content: REPLY }, finish_reason: "stop" }],
  usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
};
const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-pmu",
    object: "chat.completion.chunk",
    model: "gpt-4o",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });

interface Hit {
  guardrail_name: string;
  hook: string;
  action: string;
  counts?: Record<string, number>;
}

interface RealtimeUpstream {
  port: number;
  close(): Promise<void>;
}

/** Answers every client frame with a `response.done` that carries PII. */
async function startRealtimeUpstream(): Promise<RealtimeUpstream> {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.on("message", () => {
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: {
            output: [{ type: "message", content: [{ type: "text", text: REPLY }] }],
            usage: { input_tokens: 2, output_tokens: 1 },
          },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    close: () =>
      new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("a pii mask rule where content cannot be rewritten is reported, not silent", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let a2a: A2aUpstream | undefined;
  let realtime: RealtimeUpstream | undefined;
  let chatUp: OpenAiUpstream | undefined;
  let streamUp: OpenAiUpstream | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    chatUp = await startOpenAiUpstream({ nonStreamBody: chatReply });
    streamUp = await startOpenAiUpstream({
      streamEvents: [
        chatChunk({ role: "assistant" }),
        chatChunk({ content: REPLY }),
        chatChunk({}, "stop"),
        "[DONE]",
      ],
    });
    const videoUp = await startOpenAiUpstream({
      nonStreamBody: { output: { task_id: "task-pmu", task_status: "PENDING" }, request_id: "req-pmu" },
    });
    upstreams.push(chatUp, streamUp, videoUp);
    realtime = await startRealtimeUpstream();
    a2a = await startA2aUpstream({ wireShape: "0.3" });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "pmu-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const pk = async (name: string, up: OpenAiUpstream, extra: Record<string, unknown> = {}) =>
      seed.createProviderKey({
        display_name: name,
        secret: "sk-mock",
        api_base: `${up.baseUrl}/v1`,
        ...extra,
      });
    const chatPk = await pk("pmu-chat-pk", chatUp);
    const streamPk = await pk("pmu-stream-pk", streamUp);
    const videoPk = await seed.createProviderKey({
      display_name: "pmu-video-pk",
      secret: "sk-mock",
      api_base: videoUp.baseUrl,
      provider: "alibaba",
    });
    const realtimePk = await seed.createProviderKey({
      display_name: "pmu-realtime-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${realtime.port}/v1`,
    });
    const model = (display_name: string, provider_key_id: string, provider: string, model_name: string) =>
      seed.createModel({ display_name, provider, model_name, provider_key_id });
    await model("pmu-batches", chatPk.id, "openai", "gpt-4o");
    await model("pmu-video", videoPk.id, "alibaba", "wan-mock");
    await model("pmu-realtime", realtimePk.id, "openai", "gpt-realtime-mock");
    await seed.createPassthroughRoute({
      name: ROUTE,
      path_prefix: "/passthrough/pmu",
      target_url: `${chatUp.baseUrl}/v1`,
      provider_key_id: chatPk.id,
    });
    await seed.createPassthroughRoute({
      name: STREAM_ROUTE,
      path_prefix: "/passthrough/pmu-stream",
      target_url: `${streamUp.baseUrl}/v1`,
      provider_key_id: streamPk.id,
    });
    await seed.update("a2a_agents", randomUUID(), {
      name: AGENT,
      url: a2a.url,
      protocol_version: "0.3",
      auth_type: "none",
      enabled: true,
    });
    for (const [name, mode] of [
      [ENFORCE, "block"],
      [MONITOR, "monitor"],
    ] as const) {
      await seed.createGuardrail({
        name,
        enabled: true,
        hook_point: "both",
        kind: "pii",
        enforcement_mode: mode,
        detectors: [{ type: "email", action: "mask" }],
      });
    }

    // Caller key LAST: it authenticating implies every row above is live.
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: ["*"],
      allowed_routes: ["*"],
      allowed_agents: ["*"],
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await realtime?.close();
    await a2a?.close();
    await sls?.close();
  });

  const postJson = async (path: string, body: unknown) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
      body: JSON.stringify(body),
    });
    return { status: res.status, text: await res.text() };
  };

  const hitsOf = (log: Map<string, string>, field: string) =>
    JSON.parse(log.get(field) ?? "[]") as Hit[];

  /**
   * The row reports both rows' matches on each hook in `hooks`, with the
   * detector counts, as unsupported masks — and nothing as redacted.
   */
  const expectReported = (
    log: Map<string, string>,
    what: string,
    hooks: Array<{ hook: string; email: number }>,
  ) => {
    expect(log.get("guardrail_blocked") ?? "false", `${what}: refused`).not.toBe("true");
    expect(log.get("redacted_entity_counts"), `${what}: reports a redaction nobody made`).toBeUndefined();
    const enforced = hitsOf(log, "guardrail_enforced_hits").map(({ guardrail_name, hook, action, counts }) => ({
      guardrail_name,
      hook,
      action,
      counts,
    }));
    const monitor = hitsOf(log, "guardrail_monitor_hits").map(({ guardrail_name, hook, action, counts }) => ({
      guardrail_name,
      hook,
      action,
      counts,
    }));
    const sortByHook = <T extends { hook: string }>(xs: T[]) => [...xs].sort((a, b) => a.hook.localeCompare(b.hook));
    expect(sortByHook(enforced), `${what}: enforced hits`).toEqual(
      sortByHook(
        hooks.map(({ hook, email }) => ({
          guardrail_name: ENFORCE,
          hook,
          action: "mask_unsupported",
          counts: { email },
        })),
      ),
    );
    expect(sortByHook(monitor), `${what}: monitor hits`).toEqual(
      sortByHook(
        hooks.map(({ hook, email }) => ({
          guardrail_name: MONITOR,
          hook,
          action: "would_mask_unsupported",
          counts: { email },
        })),
      ),
    );
  };

  const BOTH = [
    { hook: "input", email: 2 },
    { hook: "output", email: 1 },
  ];

  test("passthrough route: the body and the reply pass unmodified and the match is reported", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const before = chatUp!.receivedRequests.length;
    const res = await postJson("/passthrough/pmu/chat/completions", {
      model: "gpt-4o",
      messages: [{ role: "user", content: PROMPT }],
    });
    expect(res.status).toBe(200);
    expect(res.text, "the reply reached the caller as the provider sent it").toContain(OUT);
    const sent = chatUp!.receivedRequests.slice(before).map((r) => r.body).join("\n");
    expect(sent, "the body reached the provider as the caller sent it").toContain(IN_A);
    expect(sent).toContain(IN_B);
    const log = await waitForSlsLog(sls, LOGSTORE, (l) => l.get("passthrough_route_name") === ROUTE, "passthrough");
    expectReported(log, "passthrough", BOTH);
  });

  test("streamed passthrough route: the held stream is released unmodified and the match is reported", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await postJson("/passthrough/pmu-stream/chat/completions", {
      model: "gpt-4o",
      stream: true,
      messages: [{ role: "user", content: PROMPT }],
    });
    expect(res.status).toBe(200);
    expect(res.text).toContain(OUT);
    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("passthrough_route_name") === STREAM_ROUTE,
      "streamed passthrough",
    );
    expectReported(log, "streamed passthrough", BOTH);
  });

  test("/v1/batches: the job body and its reply are reported", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await postJson("/v1/batches", {
      model: "pmu-batches",
      input_file_id: "file-pmu-in",
      endpoint: "/v1/chat/completions",
      completion_window: "24h",
      metadata: { note: PROMPT },
    });
    expect(res.status).toBe(200);
    const log = await waitForSlsLog(sls, LOGSTORE, (l) => l.get("requested_model") === "pmu-batches", "batches");
    expectReported(log, "/v1/batches", BOTH);
  });

  test("/v1/videos: the prompt is reported", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await postJson("/v1/videos", { model: "pmu-video", prompt: PROMPT });
    expect(res.status).toBe(200);
    const log = await waitForSlsLog(sls, LOGSTORE, (l) => l.get("requested_model") === "pmu-video", "videos");
    expectReported(log, "/v1/videos", [{ hook: "input", email: 2 }]);
  });

  test("/a2a: the message is reported", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await postJson(`/a2a/${AGENT}`, {
      jsonrpc: "2.0",
      id: "pmu",
      method: "message/send",
      params: {
        message: {
          role: "user",
          parts: [{ kind: "text", text: PROMPT }],
          messageId: "pmu-message",
        },
      },
    });
    expect(res.status).toBe(200);
    const sent = a2a!.requests.map((r) => JSON.stringify(r.body ?? {})).join("\n");
    expect(sent, "the message reached the agent as the caller sent it").toContain(IN_A);
    const log = await waitForSlsLog(sls, LOGSTORE, (l) => l.get("a2a_agent_name") === AGENT, "a2a");
    expectReported(log, "/a2a", [{ hook: "input", email: 2 }]);
  });

  test("/v1/realtime: both directions' frames are reported", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const ws = new WsClient(`${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=pmu-realtime`, {
      headers: { authorization: `Bearer ${KEY}` },
    });
    await new Promise<void>((resolve, reject) => {
      ws.once("open", () => resolve());
      ws.once("error", reject);
    });
    const reply = new Promise<string>((resolve) => ws.once("message", (m) => resolve(m.toString())));
    ws.send(JSON.stringify({ type: "session.update", session: { instructions: PROMPT } }));
    expect(await reply).toContain(OUT);
    ws.close();
    const log = await waitForSlsLog(sls, LOGSTORE, (l) => l.get("requested_model") === "pmu-realtime", "realtime");
    expectReported(log, "/v1/realtime", BOTH);
  });
});
