import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an Azure text-moderation OUTPUT guardrail in its default `window`
// streaming mode, on the routes that hold a streamed response whole —
// /v1/messages and /v1/responses, each native and through the Chat bridge.
//
// A block-capable output guardrail has to judge the content before any of it
// reaches the caller, so on each route:
//   - a stream the provider flags is refused with none of it on the wire —
//     not even the clean text the upstream sent ahead of the flagged part;
//   - a clean stream still arrives whole, after the scan.
// Held content is capped by the row's own `max_buffer_bytes`, and what
// happens past it is the row's own `on_buffer_exceeded`:
//   - `fail_closed` refuses the stream and names the row whose cap it
//     outgrew (`blocked_buffer_exceeded`);
//   - `fail_open` releases it unscanned and records the skipped scan as an
//     `output_buffer_exceeded` bypass.

const CALLER = "sk-window-stream-hold-e2e";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "window-stream-hold-events";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const LEAD = "Here is the first part of the answer. ";
const FLAGGED = "wndholdflaggedphrase";
const CLEAN_TAIL = "Nothing else to add.";

const BLOCK_ROW = "window-azure-block";
const CLOSED_ROW = "window-azure-cap-closed";
const OPEN_ROW = "window-azure-cap-open";

// 30 pieces of 100 bytes: three times the rows' cap, far under the window.
const CAP = 1_000;
const BIG = Array.from({ length: 30 }, (_, i) => `${String(i).padStart(2, "0")}${"w".repeat(98)}`);

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-window-hold",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
const chatEvents = (pieces: string[]) => [
  chatChunk({ role: "assistant" }),
  ...pieces.map((content) => chatChunk({ content })),
  chatChunk({}, "stop"),
  "[DONE]",
];

const anthropicEvents = (pieces: string[]) => [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_window_hold",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ...pieces.map((text) =>
    JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } }),
  ),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 40 } }),
  JSON.stringify({ type: "message_stop" }),
];

const responsesEvents = (pieces: string[]) => [
  JSON.stringify({
    type: "response.created",
    response: { id: "resp_window_hold", object: "response", status: "in_progress", model: "gpt-4o-mini", output: [] },
  }),
  ...pieces.map((delta) =>
    JSON.stringify({
      type: "response.output_text.delta",
      item_id: "msg_window_hold",
      output_index: 0,
      content_index: 0,
      delta,
    }),
  ),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_window_hold",
      object: "response",
      status: "completed",
      model: "gpt-4o-mini",
      output: [],
      usage: { input_tokens: 5, output_tokens: 40, total_tokens: 45 },
    },
  }),
];

// Azure Content Safety `text:analyze`: severity 6 in Violence for text that
// carries the flagged phrase, 0 otherwise.
async function startMockAzure(): Promise<{ url: string; close: () => Promise<void> }> {
  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const text = (JSON.parse(body || "{}") as { text?: string }).text ?? "";
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          categoriesAnalysis: [
            { category: "Hate", severity: 0 },
            { category: "Violence", severity: text.includes(FLAGGED) ? 6 : 0 },
          ],
          blocklistsMatch: [],
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

type Variant = "block" | "clean" | "closed" | "open";
const piecesFor = (v: Variant) =>
  v === "block" ? [LEAD, FLAGGED, CLEAN_TAIL] : v === "clean" ? [LEAD, CLEAN_TAIL] : BIG;

interface EnforcedHit {
  guardrail_name: string;
  hook: string;
  action: string;
}

describe("a window-mode output guardrail holds a streamed response it cannot release by window", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let azure: { url: string; close: () => Promise<void> } | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  const routes = [
    { route: "/v1/messages (native)", slug: "msg-native", provider: "anthropic", events: anthropicEvents, pk: {} },
    { route: "/v1/messages (bridged)", slug: "msg-bridge", provider: "openai", events: chatEvents, pk: {} },
    { route: "/v1/responses (native)", slug: "resp-native", provider: "openai", events: responsesEvents, pk: {} },
    // `apis: {}`: no `/v1/responses` on this endpoint, so the route reaches
    // it through the chat bridge.
    { route: "/v1/responses (bridged)", slug: "resp-bridge", provider: "openai", events: chatEvents, pk: { apis: {} } },
  ] as const;
  const modelName = (slug: string, v: Variant) => `wnd-${slug}-${v}`;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    azure = await startMockAzure();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-window-hold",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // No `stream_processing_mode`: the kind's default, `window`.
    const row = (name: string, extra: Record<string, unknown> = {}) =>
      seed.createGuardrail(
        {
          name,
          enabled: true,
          kind: "azure_content_safety_text_moderation",
          hook_point: "output",
          endpoint: azure!.url,
          api_key: "azure-mock-key",
          ...extra,
        },
        { attach: false },
      );
    const rows: Record<Variant, { id: string }> = {
      block: await row(BLOCK_ROW),
      clean: await row(`${BLOCK_ROW}-clean`),
      closed: await row(CLOSED_ROW, { max_buffer_bytes: CAP, on_buffer_exceeded: "fail_closed" }),
      open: await row(OPEN_ROW, { max_buffer_bytes: CAP, on_buffer_exceeded: "fail_open" }),
    };

    for (const r of routes) {
      for (const v of ["block", "clean", "closed", "open"] as const) {
        const upstream = await startOpenAiUpstream({ streamEvents: r.events(piecesFor(v)) });
        upstreams.push(upstream);
        const pk = await seed.createProviderKey({
          display_name: `${modelName(r.slug, v)}-pk`,
          secret: "sk-mock",
          api_base: r.provider === "openai" ? `${upstream.baseUrl}/v1` : upstream.baseUrl,
          ...(r.provider === "anthropic" ? { provider: "anthropic", adapter: "anthropic" } : {}),
          ...r.pk,
        });
        const m = await seed.createModel({
          display_name: modelName(r.slug, v),
          provider: r.provider,
          model_name: r.provider === "openai" ? "gpt-4o-mini" : "claude-3-5-haiku-20241022",
          provider_key_id: pk.id,
        });
        await seed.attachGuardrailToModel(rows[v].id, m.id);
      }
    }

    // Caller key LAST: it authenticating implies every row above is live.
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await azure?.close();
    await sls?.close();
  });

  const send = async (slug: string, model: string) => {
    const path = slug.startsWith("msg") ? "/v1/messages" : "/v1/responses";
    const body = slug.startsWith("msg")
      ? { model, max_tokens: 256, messages: [{ role: "user", content: "go" }] }
      : { model, input: "go" };
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({ ...body, stream: true }),
    });
    return res.text();
  };
  const byModel = (model: string) => (l: Map<string, string>) => l.get("requested_model") === model;

  for (const r of routes) {
    test(`${r.route}: a flagged stream is refused with none of it sent`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const model = modelName(r.slug, "block");
      const body = await send(r.slug, model);
      expect(body, "the clean text ahead of the flagged part went out before the verdict").not.toContain(LEAD);
      expect(body).not.toContain(FLAGGED);
      const log = await waitForSlsLog(sls, LOGSTORE, byModel(model), r.route);
      expect(log.get("guardrail_blocked"), `${r.route}: the refusal is not recorded`).toBe("true");
    });

    test(`${r.route}: a clean stream arrives whole after the scan`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const body = await send(r.slug, modelName(r.slug, "clean"));
      expect(body).toContain(LEAD);
      expect(body).toContain(CLEAN_TAIL);
    });

    test(`${r.route}: past the row's cap under fail_closed, the stream is refused and the row named`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const model = modelName(r.slug, "closed");
      const body = await send(r.slug, model);
      expect(body).toContain("output_buffer_exceeded");
      expect(body).not.toContain(BIG[29]);
      const log = await waitForSlsLog(sls, LOGSTORE, byModel(model), r.route);
      expect(log.get("guardrail_blocked")).toBe("true");
      const hits = JSON.parse(log.get("guardrail_enforced_hits") ?? "[]") as EnforcedHit[];
      expect(
        hits.map(({ guardrail_name, hook, action }) => ({ guardrail_name, hook, action })),
        `${r.route}: the refusal names no row, or the wrong one`,
      ).toEqual([{ guardrail_name: CLOSED_ROW, hook: "output", action: "blocked_buffer_exceeded" }]);
    });

    test(`${r.route}: past the row's cap under fail_open, the stream is released and the bypass recorded`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const model = modelName(r.slug, "open");
      const body = await send(r.slug, model);
      expect(body, "fail_open releases what was held").toContain(BIG[29]);
      expect(body).not.toContain("output_buffer_exceeded");
      const log = await waitForSlsLog(sls, LOGSTORE, byModel(model), r.route);
      expect(log.get("guardrail_blocked") ?? "false").not.toBe("true");
      expect(log.get("guardrail_bypassed_reason"), `${r.route}: the unscanned release is not recorded`).toBe(
        "output_buffer_exceeded",
      );
    });
  }
});
