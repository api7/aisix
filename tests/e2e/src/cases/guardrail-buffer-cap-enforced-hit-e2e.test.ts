import { createHash, randomUUID } from "node:crypto";
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

// E2E: a streamed response refused because it outgrew the output
// guardrail's hold-back cap under `on_buffer_exceeded: fail_closed` names
// the row whose cap it outgrew (#1029).
//
// The event already said `guardrail_blocked: true`. What an operator could
// not tell from it was WHICH row refused and WHY — a policy match calls for
// tuning the policy, a response that outgrew the scan buffer calls for
// raising the cap. The exported `guardrail_enforced_hits` entry answers
// both: the row, on `output`, with `action: "blocked_buffer_exceeded"`.
//
// Every route that holds a stream back is driven: chat, /v1/messages native
// and bridged, /v1/responses native and bridged, and a passthrough route.
// Each is governed by TWO output rows, and the one with the looser cap runs
// first in chain order — so naming the chain's first member, or the larger
// cap, names the wrong row. A third case covers a kind that has no
// `max_buffer_bytes` of its own and holds back under the default cap.
//
// The same trip under `on_buffer_exceeded: fail_open` refuses nothing: what
// was held goes out unscanned. That is a bypass, and each path records it as
// `guardrail_bypassed_reason: "output_buffer_exceeded"` — without it the row
// looks exactly like a response the guardrail screened.

const KEY = "sk-buffer-cap-enforced-hit-e2e";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");
const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "buffer-cap-enforced-hit";

const LOOSE = "cap-loose-runs-first";
const TIGHT = "cap-tight-runs-second";
const DEFAULT_CAP_ROW = "cap-default-keyword";
const ROUTE = "cap-passthrough";
const OPEN_ROW = "cap-fail-open";
const OPEN_ROUTE = "cap-open-passthrough";

// 30 pieces of 100 bytes: three times the tight cap, far under the loose one.
const TIGHT_CAP = 1_000;
const LOOSE_CAP = 100_000;
const PIECES = Array.from({ length: 30 }, (_, i) => `${String(i).padStart(2, "0")}${"t".repeat(98)}`);
// Past the 256 KiB default cap a kind without `max_buffer_bytes` holds under.
const BIG_PIECES = Array.from({ length: 300 }, (_, i) => `${String(i).padStart(3, "0")}${"b".repeat(997)}`);

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-cap",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
const chatStream = (pieces: string[]) => [
  chatChunk({ role: "assistant" }),
  ...pieces.map((p) => chatChunk({ content: p })),
  chatChunk({}, "stop"),
  "[DONE]",
];

const ANTHROPIC_STREAM = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_cap",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ...PIECES.map((p) =>
    JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: p } }),
  ),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 40 } }),
  JSON.stringify({ type: "message_stop" }),
];

const RESPONSES_STREAM = [
  JSON.stringify({
    type: "response.created",
    response: { id: "resp_cap", object: "response", status: "in_progress", model: "gpt-4o-mini", output: [] },
  }),
  ...PIECES.map((p) =>
    JSON.stringify({
      type: "response.output_text.delta",
      item_id: "msg_cap",
      output_index: 0,
      content_index: 0,
      delta: p,
    }),
  ),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_cap",
      object: "response",
      status: "completed",
      model: "gpt-4o-mini",
      output: [],
      usage: { input_tokens: 5, output_tokens: 40, total_tokens: 45 },
    },
  }),
];

interface EnforcedHit {
  guardrail_name: string;
  hook: string;
  action: string;
  error_type?: string;
  counts?: Record<string, number>;
}

describe("a stream refused by the hold-back cap names the row whose cap it outgrew (#1029)", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-buffer-cap",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const pii = (name: string, cap: number, onExceeded = "fail_closed") =>
      seed.createGuardrail(
        {
          name,
          enabled: true,
          hook_point: "output",
          kind: "pii",
          detectors: [{ type: "email", action: "mask" }],
          max_buffer_bytes: cap,
          on_buffer_exceeded: onExceeded,
        },
        { attach: false },
      );
    const loose = await pii(LOOSE, LOOSE_CAP);
    const tight = await pii(TIGHT, TIGHT_CAP);
    const open = await pii(OPEN_ROW, TIGHT_CAP, "fail_open");
    const keyword = await seed.createGuardrail(
      {
        name: DEFAULT_CAP_ROW,
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "never-in-this-stream" }],
      },
      { attach: false },
    );
    // Priority decides chain order: the loose row runs first.
    const attachBoth = async (scope_type: string, scope_id: string) => {
      await seed.update("guardrail_attachments", randomUUID(), {
        guardrail_id: loose.id,
        scope_type,
        scope_id,
        priority: 200,
      });
      await seed.update("guardrail_attachments", randomUUID(), {
        guardrail_id: tight.id,
        scope_type,
        scope_id,
        priority: 100,
      });
    };

    const model = async (
      display: string,
      provider: "openai" | "anthropic",
      streamEvents: string[],
      pkExtra: Record<string, unknown> = {},
    ) => {
      const upstream = await startOpenAiUpstream({ streamEvents });
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: provider === "openai" ? `${upstream.baseUrl}/v1` : upstream.baseUrl,
        ...(provider === "anthropic" ? { provider: "anthropic", adapter: "anthropic" } : {}),
        ...pkExtra,
      });
      const m = await seed.createModel({
        display_name: display,
        provider,
        model_name: provider === "openai" ? "gpt-4o-mini" : "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
      return { upstream, pk, model: m };
    };

    for (const [display, provider, events, extra] of [
      ["cap-chat", "openai", chatStream(PIECES), {}],
      ["cap-msg-native", "anthropic", ANTHROPIC_STREAM, {}],
      ["cap-msg-bridge", "openai", chatStream(PIECES), {}],
      ["cap-resp-native", "openai", RESPONSES_STREAM, {}],
      // `apis: {}`: no `/v1/responses` on this endpoint, so the route
      // reaches it through the chat bridge.
      ["cap-resp-bridge", "openai", chatStream(PIECES), { apis: {} }],
    ] as const) {
      const { model: m } = await model(display, provider, [...events], extra);
      await attachBoth("model", m.id);
      const { model: o } = await model(`${display}-open`, provider, [...events], extra);
      await seed.attachGuardrailToModel(open.id, o.id);
    }
    const big = await model("cap-default", "openai", chatStream(BIG_PIECES));
    await seed.attachGuardrailToModel(keyword.id, big.model.id);

    const routeUp = await model("cap-route-backing", "openai", chatStream(PIECES));
    const route = await seed.createPassthroughRoute({
      name: ROUTE,
      path_prefix: "/passthrough/cap",
      target_url: `${routeUp.upstream.baseUrl}/v1`,
      provider_key_id: routeUp.pk.id,
    });
    await attachBoth("passthrough_route", route.id);
    const openRoute = await seed.createPassthroughRoute({
      name: OPEN_ROUTE,
      path_prefix: "/passthrough/cap-open",
      target_url: `${routeUp.upstream.baseUrl}/v1`,
      provider_key_id: routeUp.pk.id,
    });
    await seed.update("guardrail_attachments", randomUUID(), {
      guardrail_id: open.id,
      scope_type: "passthrough_route",
      scope_id: openRoute.id,
      priority: 100,
    });

    // Caller key LAST: it authenticating implies every row above is live.
    await seed.createApiKey({
      key_hash: sha256(KEY),
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
    const proxy = new ProxyClient(app.proxyUrl, KEY);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 90_000);

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await sls?.close();
  });

  const post = async (path: string, body: Record<string, unknown>) => {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${KEY}`,
        "x-api-key": KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({ ...body, stream: true }),
    });
    return res.text();
  };
  const chat = (model: string) => post("/v1/chat/completions", { model, messages: [{ role: "user", content: "go" }] });

  const expectCapHit = async (what: string, pred: (log: Map<string, string>) => boolean, row: string) => {
    const log = await waitForSlsLog(sls!, LOGSTORE, (l) => pred(l) && l.get("guardrail_blocked") === "true", what);
    const hits = JSON.parse(log.get("guardrail_enforced_hits") ?? "[]") as EnforcedHit[];
    expect(hits, `${what}: the refusal names no row, or the wrong one`).toEqual([
      { guardrail_name: row, hook: "output", action: "blocked_buffer_exceeded" },
    ]);
  };
  // Gated on the route's own row, not on the field under test, so a missing
  // tag fails the assertion rather than the poll.
  const expectBypass = async (what: string, pred: (log: Map<string, string>) => boolean) => {
    const log = await waitForSlsLog(sls!, LOGSTORE, pred, what);
    expect(log.get("guardrail_blocked") ?? "false", `${what}: fail_open refused nothing`).not.toBe("true");
    expect(log.get("guardrail_bypassed_reason"), `${what}: the unscanned release is not recorded`).toBe(
      "output_buffer_exceeded",
    );
    expect(JSON.parse(log.get("guardrail_enforced_hits") ?? "[]")).toEqual([]);
  };
  const byModel = (model: string) => (l: Map<string, string>) => l.get("requested_model") === model;

  const msg = (model: string) =>
    post("/v1/messages", { model, max_tokens: 256, messages: [{ role: "user", content: "go" }] });
  const resp = (model: string) => post("/v1/responses", { model, input: "go" });
  const cases: Array<{ route: string; model: string; send: (model: string) => Promise<string> }> = [
    { route: "/v1/chat/completions", model: "cap-chat", send: chat },
    { route: "/v1/messages (native)", model: "cap-msg-native", send: msg },
    { route: "/v1/messages (bridged)", model: "cap-msg-bridge", send: msg },
    { route: "/v1/responses (native)", model: "cap-resp-native", send: resp },
    { route: "/v1/responses (bridged)", model: "cap-resp-bridge", send: resp },
  ];

  for (const c of cases) {
    test(`${c.route}: names the row with the stricter cap, not the chain's first`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const body = await c.send(c.model);
      expect(body).toContain("output_buffer_exceeded");
      expect(body).not.toContain(PIECES[29]);
      await expectCapHit(c.route, byModel(c.model), TIGHT);
    });

    test(`${c.route}: a fail_open trip records the skipped output scan as a bypass`, async (ctx) => {
      if (!etcdReachable || !app || !sls) return ctx.skip();
      const model = `${c.model}-open`;
      const body = await c.send(model);
      expect(body, "fail_open releases what was held").toContain(PIECES[29]);
      await expectBypass(c.route, byModel(model));
    });
  }

  test("passthrough route: a fail_open trip records the skipped output scan as a bypass", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const body = await post("/passthrough/cap-open/chat/completions", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "go" }],
    });
    expect(body, "fail_open releases what was held").toContain(PIECES[29]);
    await expectBypass("passthrough", (l) => l.get("passthrough_route_name") === OPEN_ROUTE);
  });

  test("passthrough route: names the row with the stricter cap, not the chain's first", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const body = await post("/passthrough/cap/chat/completions", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "go" }],
    });
    expect(body).toContain("output_buffer_exceeded");
    expect(body).not.toContain(PIECES[29]);
    await expectCapHit("passthrough", (l) => l.get("passthrough_route_name") === ROUTE, TIGHT);
  });

  test("a kind with no max_buffer_bytes of its own is named when the default cap trips", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const body = await chat("cap-default");
    expect(body).toContain("output_buffer_exceeded");
    await expectCapHit("default cap", byModel("cap-default"), DEFAULT_CAP_ROW);
  });
});
