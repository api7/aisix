import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (AISIX-Cloud#726): on the Anthropic surface a guardrail refusal must be
// distinguishable from a malformed request. `error.type` there is the
// Anthropic SDK's closed literal for a 422, `invalid_request_error`, which a
// malformed body gets too — so the refusal is named by `error.code`:
// `content_filter` for a policy block, `guardrail_unavailable` for a
// fail-closed row that could not evaluate the content (the code the OpenAI
// surface carries for that case). Both the 422 body and the streaming
// `event: error` frame carry it, on `/v1/messages` and
// `/v1/messages/count_tokens`.

const CALLER = "sk-msg-gr-code-726-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const MARKER = "gr726blockmarker";

const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_726",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: `the reply mentions ${MARKER} here` },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 9 } }),
  JSON.stringify({ type: "message_stop" }),
];

/** model name → the guardrail its traffic runs through. */
const CASES: Record<string, Record<string, unknown>> = {
  // Refuses the marker: a working policy.
  "g726-policy": {
    kind: "custom",
    timeout_ms: 5000,
    hook_point: "input",
    fail_open: false,
    script: `export function checkInput(ctx) {
  return ctx.text.includes("${MARKER}") ? { action: "block" } : { action: "none" };
}`,
  },
  // Returns an action the gateway does not know: a script fault, refused
  // because the row is fail-closed.
  "g726-broken": {
    kind: "custom",
    timeout_ms: 5000,
    hook_point: "input",
    fail_open: false,
    script: `export function checkInput() { return { action: "permit" }; }`,
  },
  // A keyword row holds the whole stream back until it scans clean.
  "g726-stream-policy": {
    kind: "keyword",
    hook_point: "output",
    patterns: [{ kind: "literal", value: MARKER }],
  },
  "g726-stream-broken": {
    kind: "custom",
    timeout_ms: 5000,
    hook_point: "output",
    output_fail_open: false,
    script: `export function checkOutput() { return { action: "permit" }; }`,
  },
};

type AnthropicError = { type?: string; error?: { type?: string; code?: string; message?: string } };

describe("Anthropic-surface guardrail refusals carry error.code (AISIX-Cloud#726)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS, eventDelayMs: 2 });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "g726-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    for (const [model, guardrail] of Object.entries(CASES)) {
      const m = await seed.createModel({
        display_name: model,
        provider: "anthropic",
        model_name: "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
      const g = await seed.createGuardrail(
        { name: `${model}-guard`, enabled: true, ...guardrail },
        { attach: false },
      );
      await seed.attachGuardrailToModel(g.id, m.id);
    }
    // Seeded last: its key authenticating implies the whole seed is live.
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: Object.keys(CASES) });
    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, CALLER).listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const post = (path: string, body: Record<string, unknown> | string) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER,
        "anthropic-version": "2023-06-01",
      },
      body: typeof body === "string" ? body : JSON.stringify(body),
    });

  const request = (model: string, content: string, extra: Record<string, unknown> = {}) => ({
    model,
    max_tokens: 64,
    messages: [{ role: "user", content }],
    ...extra,
  });

  const refusal = async (res: Response): Promise<AnthropicError> => {
    expect(res.status).toBe(422);
    const json = (await res.json()) as AnthropicError;
    expect(json.type).toBe("error");
    expect(json.error?.type).toBe("invalid_request_error");
    return json;
  };

  const lastErrorFrame = async (res: Response, heldBack = false): Promise<AnthropicError> => {
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body, "stream must end with an SSE error event").toContain("event: error");
    if (heldBack) expect(body, "the refused content stays off the wire").not.toContain(MARKER);
    const frame = body.slice(body.lastIndexOf("event: error"));
    return JSON.parse(
      frame.slice(frame.indexOf("data: ") + "data: ".length, frame.indexOf("\n\n")),
    ) as AnthropicError;
  };

  for (const path of ["/v1/messages", "/v1/messages/count_tokens"]) {
    test(`${path}: a policy block is code content_filter`, async (ctx) => {
      if (!etcdReachable) ctx.skip();
      const json = await refusal(await post(path, request("g726-policy", `say ${MARKER}`)));
      expect(json.error?.code).toBe("content_filter");
      expect(json.error?.message).toContain("blocked by content policy");
    });

    test(`${path}: a fail-closed refusal is code guardrail_unavailable`, async (ctx) => {
      if (!etcdReachable) ctx.skip();
      const json = await refusal(await post(path, request("g726-broken", "an ordinary question")));
      expect(json.error?.code).toBe("guardrail_unavailable");
      expect(json.error?.message).toContain("could not evaluate");
    });

    test(`${path}: a malformed request carries no guardrail code`, async (ctx) => {
      if (!etcdReachable) ctx.skip();
      const res = await post(path, "{ not json");
      expect(res.status).toBe(400);
      const json = (await res.json()) as AnthropicError;
      expect(json.error?.type).toBe("invalid_request_error");
      expect(json.error?.code).toBeUndefined();
    });
  }

  test("streaming /v1/messages: a policy block's error frame is code content_filter", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    const res = await post("/v1/messages", request("g726-stream-policy", "go", { stream: true }));
    const frame = await lastErrorFrame(res, true);
    expect(frame.type).toBe("error");
    expect(frame.error?.type).toBe("invalid_request_error");
    expect(frame.error?.code).toBe("content_filter");
  });

  test("streaming /v1/messages: a fail-closed refusal's error frame is code guardrail_unavailable", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    const frame = await lastErrorFrame(
      await post("/v1/messages", request("g726-stream-broken", "go", { stream: true })),
    );
    expect(frame.type).toBe("error");
    expect(frame.error?.type).toBe("invalid_request_error");
    expect(frame.error?.code).toBe("guardrail_unavailable");
  });
});
