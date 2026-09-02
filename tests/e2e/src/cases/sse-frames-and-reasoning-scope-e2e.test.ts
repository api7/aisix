import { createHash, randomUUID } from "node:crypto";
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
import { startMockSls, waitForSlsLog, type MockSls } from "../harness/sls-mock.js";

// E2E for the SSE frame-reading and reasoning-scope contract (#1103 / #1104).
//
// Four things are pinned here, all of them invisible to a response-content
// assertion and all of them silent when broken:
//
//  1. A frame's payload is ALL of its `data:` lines joined, so a terminal
//     `response.completed` written over several lines must still be read as
//     the provider's own usage. Reading the first line alone did not produce
//     "no usage" — a local token estimator filled the counters in — so the
//     assertion has to name the exact numbers AND the absence of the
//     `usage_estimated` flag.
//  2. Reasoning replayed by the CALLER is request text: it is scanned, and a
//     mask that reports a hit there has to actually rewrite the dispatched
//     body. Where the block is signed by the provider (Anthropic `thinking`),
//     a rewrite would invalidate the signature, so the request is refused
//     instead of forwarded with the match still in it.
//  3. Reasoning the MODEL generates stays out of the output-guardrail scope,
//     on every `/v1/messages` and `/v1/responses` shape — neither scanned nor
//     rewritten.
//  4. A 200 that is not SSE at all, from an upstream that ignored
//     `stream: true`, takes the buffered scan+mask path rather than the SSE
//     hold-back, which used to release it unscanned with `\n\n` appended.

const CALLER = "sk-sse-reasoning-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER).digest("hex");
const HEADERS = {
  "content-type": "application/json",
  authorization: `Bearer ${CALLER}`,
};

// Distinct from any PII pattern: this is the literal a BLOCK rule fires on.
const BLOCKED = "ABSOLUTELYFORBIDDENWORD";
// Masked by the `email` detector, so its presence or absence in a body says
// whether the mask reached that slot.
const EMAIL = "leak@example.com";
const MASKED = "[EMAIL_REDACTED]";

const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "sse-reasoning-events";
const CREDENTIAL_REF = "mock";

// The provider's own counters on a terminal frame written over three
// `data:` lines. Deliberately far from anything the local estimator could
// produce for these prompts, so the numbers alone identify which side the
// usage came from.
const PROVIDER_PROMPT_TOKENS = 4321;
const PROVIDER_COMPLETION_TOKENS = 765;

describe("sse frames + reasoning scope e2e (#1103/#1104)", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let seed: SeedClient | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  /** Attach a guardrail to one model only, the way cp-api projects it. */
  async function attachToModel(guardrailId: string, modelId: string): Promise<void> {
    await etcd!.put(
      `${app!.etcdPrefix}/guardrail_attachments/${randomUUID()}`,
      JSON.stringify({
        guardrail_id: guardrailId,
        env_id: randomUUID(),
        scope_type: "model",
        scope_id: modelId,
        priority: 0,
        enabled: true,
      }),
    );
  }

  /**
   * Seed one scenario: a provider key, the model under test, its guardrail
   * attachments, and — written LAST — a sentinel model the readiness gate
   * waits on.
   *
   * The sentinel is what makes the gate sound. etcd watch events apply in
   * revision order, so seeing the sentinel in `GET /v1/models` proves every
   * earlier write landed, attachments included. Gating on the model under
   * test instead would prove nothing about the attachment written after it,
   * and gating on the behaviour under test would turn an assertion failure
   * into a 30s timeout.
   */
  async function seedScenario(
    display: string,
    upstream: OpenAiUpstream,
    protocol: "openai" | "anthropic",
    guardrailIds: string[] = [],
  ): Promise<void> {
    const pk = await seed!.createProviderKey(
      protocol === "anthropic"
        ? {
            display_name: `${display}-pk`,
            secret: "sk-mock",
            api_base: upstream.baseUrl,
            provider: "anthropic",
            adapter: "anthropic",
          }
        : {
            display_name: `${display}-pk`,
            secret: "sk-mock",
            api_base: `${upstream.baseUrl}/v1`,
          },
    );
    const model = await seed!.createModel({
      display_name: display,
      provider: protocol,
      model_name: "upstream-model-x",
      provider_key_id: pk.id,
    });
    for (const id of guardrailIds) await attachToModel(id, model.id);
    await seed!.createModel({
      display_name: `${display}-gate`,
      provider: protocol,
      model_name: "upstream-model-x",
      provider_key_id: pk.id,
    });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: HEADERS.authorization },
      });
      if (r.status !== 200) {
        await r.text();
        return false;
      }
      const j = (await r.json()) as { data?: Array<{ id?: string }> };
      return !!j.data?.some((m) => m.id === `${display}-gate`);
    });
  }

  async function upstreamWith(
    opts: Parameters<typeof startOpenAiUpstream>[0],
  ): Promise<OpenAiUpstream> {
    const u = await startOpenAiUpstream(opts);
    upstreams.push(u);
    return u;
  }

  let maskGuardrailId = "";
  let blockGuardrailId = "";

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "sse-reasoning-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // Two guardrails, each attached per model below so one file can hold
    // both a guarded and an unguarded relay.
    const mask = await seed.createGuardrail(
      {
        name: "sse-reasoning-mask",
        enabled: true,
        hook_point: "both",
        kind: "pii",
        detectors: [{ type: "email", action: "mask" }],
      },
      { attach: false },
    );
    maskGuardrailId = mask.id;

    const block = await seed.createGuardrail(
      {
        name: "sse-reasoning-block",
        enabled: true,
        hook_point: "both",
        kind: "keyword",
        patterns: [{ kind: "literal", value: BLOCKED }],
      },
      { attach: false },
    );
    blockGuardrailId = block.id;

    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    for (const u of upstreams) await u.close();
  });

  // ── 1. whole-frame reads ───────────────────────────────────────────

  test("/v1/responses live relay: a terminal frame split over several data lines still bills the provider's own counters", async (ctx) => {
    if (!etcdReachable || !app || !seed || !sls) return void ctx.skip();

    // ONE frame, ONE JSON document, three `data:` lines — a shape the SSE
    // spec allows and some relays emit. No output guardrail on this model,
    // so the verbatim live relay reads the stream, which is the path that
    // used to stop at the first `data:` line.
    const upstream = await upstreamWith({
      rawStreamFrames: [
        `event: response.created\ndata: {"type":"response.created","response":{"id":"resp_split","object":"response","status":"in_progress","model":"upstream-model-x"}}\n\n`,
        `event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"m","delta":"hello"}\n\n`,
        `event: response.completed\ndata: {"type":"response.completed",\ndata: "response":{"id":"resp_split","object":"response","status":"completed","model":"upstream-model-x",\ndata: "usage":{"input_tokens":${PROVIDER_PROMPT_TOKENS},"output_tokens":${PROVIDER_COMPLETION_TOKENS},"total_tokens":${PROVIDER_PROMPT_TOKENS + PROVIDER_COMPLETION_TOKENS}}}}\n\n`,
      ],
    });
    await seedScenario("split-frame-usage", upstream, "openai");

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "split-frame-usage",
        input: "hi",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    // The whole document reaches the caller, not just its first line.
    const text = await res.text();
    expect(text).toContain(`"input_tokens":${PROVIDER_PROMPT_TOKENS}`);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("requested_model") === "split-frame-usage",
      "usage row for the split terminal frame",
      15_000,
    );
    // The trap: on the broken read the counters are NOT absent — the local
    // estimator fills them — so "usage exists" passes either way. Only the
    // provider's exact numbers, plus the absence of the estimated flag,
    // tell the two apart.
    expect(log.get("prompt_tokens")).toBe(String(PROVIDER_PROMPT_TOKENS));
    expect(log.get("completion_tokens")).toBe(String(PROVIDER_COMPLETION_TOKENS));
    expect(log.get("usage_estimated")).not.toBe("true");
    // The provider's own call id rides the same frame.
    expect(log.get("provider_request_id")).toBe("resp_split");
  });

  test("/v1/responses live relay: a CRLF-framed stream with no [DONE] sentinel bills the same", async (ctx) => {
    if (!etcdReachable || !app || !seed || !sls) return void ctx.skip();

    // Framing varies per endpoint, not per vendor: on one host
    // `/v1/audio/transcriptions` is pure CRLF while `/v1/responses` is pure
    // LF, and the Responses API sends no `[DONE]`. A comment-only frame
    // (zero `data:` lines) leads, as some relays emit while the model
    // thinks.
    const upstream = await upstreamWith({
      rawStreamFrames: [
        `: keep-alive\r\n\r\n`,
        `data: {"type":"response.created","response":{"id":"resp_crlf","object":"response","status":"in_progress","model":"upstream-model-x"}}\r\n\r\n`,
        `data: {"type":"response.completed","response":{"id":"resp_crlf","object":"response","status":"completed","model":"upstream-model-x","usage":{"input_tokens":${PROVIDER_PROMPT_TOKENS},"output_tokens":${PROVIDER_COMPLETION_TOKENS},"total_tokens":${PROVIDER_PROMPT_TOKENS + PROVIDER_COMPLETION_TOKENS}}}}\r\n\r\n`,
      ],
    });
    await seedScenario("crlf-frame-usage", upstream, "openai");

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: "crlf-frame-usage", input: "hi", stream: true }),
    });
    expect(res.status).toBe(200);
    await res.text();

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("requested_model") === "crlf-frame-usage",
      "usage row for the CRLF stream",
      15_000,
    );
    expect(log.get("prompt_tokens")).toBe(String(PROVIDER_PROMPT_TOKENS));
    expect(log.get("completion_tokens")).toBe(String(PROVIDER_COMPLETION_TOKENS));
    expect(log.get("usage_estimated")).not.toBe("true");
  });

  // ── 2. request-side reasoning is scanned AND masked ────────────────

  test("/v1/responses: a mask hit inside a replayed reasoning item rewrites the DISPATCHED body", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "resp_reason_req",
        object: "response",
        status: "completed",
        model: "upstream-model-x",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "ok" }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
      },
    });
    await seedScenario("reasoning-request-mask", upstream, "openai", [maskGuardrailId]);

    const before = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "reasoning-request-mask",
        input: [
          { role: "user", content: [{ type: "input_text", text: "carry on" }] },
          {
            type: "reasoning",
            id: "rs_replayed",
            summary: [{ type: "summary_text", text: `summary says ${EMAIL}` }],
            content: [{ type: "reasoning_text", text: `content says ${EMAIL}` }],
          },
        ],
      }),
    });
    expect(res.status).toBe(200);

    // The point of the test: a Mask verdict that REPORTS a hit but leaves
    // the dispatched body untouched is the fail-open this closes, so the
    // assertion is on what the upstream received, never on the verdict.
    expect(upstream.receivedRequests.length).toBe(before + 1);
    const dispatched = upstream.receivedRequests.at(-1)!.body;
    expect(dispatched).not.toContain(EMAIL);
    expect(dispatched).toContain("summary says " + MASKED);
    expect(dispatched).toContain("content says " + MASKED);
  });

  test("/v1/messages: a blocked literal inside a replayed thinking block is refused", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "msg_thinking_block",
        type: "message",
        role: "assistant",
        model: "upstream-model-x",
        content: [{ type: "text", text: "ok" }],
        usage: { input_tokens: 3, output_tokens: 1 },
      },
    });
    await seedScenario("thinking-request-block", upstream, "anthropic", [blockGuardrailId]);

    const before = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "thinking-request-block",
        max_tokens: 16,
        messages: [
          { role: "user", content: "what did you decide" },
          {
            role: "assistant",
            content: [
              {
                type: "thinking",
                thinking: `I will now ${BLOCKED} quietly`,
                signature: "sig-abc",
              },
              { type: "text", text: "nothing to see" },
            ],
          },
          { role: "user", content: "go on" },
        ],
      }),
    });
    // The scan parse keeps thinking text the dispatch parse drops, so the
    // block fires. Pre-fix the payload was invisible and the turn reached
    // the upstream.
    expect(res.status).toBe(422);
    expect(await res.text()).not.toContain(BLOCKED);
    expect(upstream.receivedRequests.length).toBe(before);
  });

  test("/v1/messages: a mask hit inside a signed thinking block is forwarded unchanged", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "msg_thinking_mask",
        type: "message",
        role: "assistant",
        model: "upstream-model-x",
        content: [{ type: "text", text: "ok" }],
        usage: { input_tokens: 3, output_tokens: 1 },
      },
    });
    await seedScenario("thinking-request-mask", upstream, "anthropic", [maskGuardrailId]);

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "thinking-request-mask",
        max_tokens: 16,
        messages: [
          { role: "user", content: "what did you decide" },
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: `write to ${EMAIL}`, signature: "sig-abc" },
              { type: "text", text: `and also ${EMAIL}` },
            ],
          },
          { role: "user", content: "go on" },
        ],
      }),
    });

    // The deliberate exception to "no write-back channel means block": the
    // signed block cannot be rewritten without invalidating its signature,
    // and refusing the replay protects nothing, because the gateway itself
    // emitted that block unmasked on the previous turn (generated reasoning
    // is out of the output scope). So it is forwarded as-is.
    expect(res.status).toBe(200);
    await res.text();

    const dispatched = JSON.parse(upstream.receivedRequests.at(-1)!.body) as {
      messages: Array<{ content: Array<Record<string, unknown>> }>;
    };
    const assistant = dispatched.messages[1].content;
    expect(assistant[0]).toEqual({
      type: "thinking",
      thinking: `write to ${EMAIL}`,
      signature: "sig-abc",
    });
    // …and the ordinary text block beside it in the SAME message is still
    // masked, so this pins an exception for the signed block rather than an
    // exemption for the whole request.
    expect(assistant[1]).toEqual({ type: "text", text: `and also ${MASKED}` });
  });

  test("/v1/messages: a block rule still fires on thinking content", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "msg_thinking_blockrule",
        type: "message",
        role: "assistant",
        model: "upstream-model-x",
        content: [{ type: "text", text: "ok" }],
        usage: { input_tokens: 3, output_tokens: 1 },
      },
    });
    await seedScenario("thinking-mask-vs-block", upstream, "anthropic", [blockGuardrailId]);

    const before = upstream.receivedRequests.length;
    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "thinking-mask-vs-block",
        max_tokens: 16,
        messages: [
          {
            role: "assistant",
            content: [
              {
                type: "thinking",
                thinking: `I will ${BLOCKED} quietly`,
                signature: "sig-abc",
              },
            ],
          },
          { role: "user", content: "go on" },
        ],
      }),
    });
    // The mask exception is for the MASK only. A block rule still reaches
    // thinking text, or the exemption would have opened a channel any
    // forbidden content could travel through.
    expect(res.status).toBe(422);
    expect(await res.text()).not.toContain(BLOCKED);
    expect(upstream.receivedRequests.length).toBe(before);
  });

  // ── 3. generated reasoning stays out of the OUTPUT scope ───────────

  test("/v1/messages buffered: generated thinking is neither scanned nor masked", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "msg_out_thinking",
        type: "message",
        role: "assistant",
        model: "upstream-model-x",
        content: [
          {
            type: "thinking",
            thinking: `internally I considered ${BLOCKED} and ${EMAIL}`,
            signature: "sig-out",
          },
          { type: "text", text: `you can reach us at ${EMAIL}` },
          {
            type: "tool_use",
            id: "t1",
            name: "lookup",
            input: { q: "argument text" },
          },
        ],
        usage: { input_tokens: 3, output_tokens: 9 },
      },
    });
    await seedScenario("thinking-output-scope", upstream, "anthropic", [
      maskGuardrailId,
      blockGuardrailId,
    ]);

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "thinking-output-scope",
        max_tokens: 16,
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    // NOT blocked: the raw-content dump the output scan takes for tool-use
    // arguments used to sweep the thinking text in with them, so a block
    // rule matching only inside reasoning refused a response that is out of
    // the output scope.
    expect(res.status).toBe(200);
    const body = await res.text();
    const parsed = JSON.parse(body) as {
      content: Array<Record<string, unknown>>;
    };

    // Not rewritten: the signed block reaches the caller byte-for-byte,
    // literal email and all.
    expect(parsed.content[0]).toEqual({
      type: "thinking",
      thinking: `internally I considered ${BLOCKED} and ${EMAIL}`,
      signature: "sig-out",
    });
    // …while everything the output scan and mask DO cover still works:
    // the text block is masked.
    expect(parsed.content[1]).toEqual({
      type: "text",
      text: `you can reach us at ${MASKED}`,
    });
    expect(body).toContain("argument text");
  });

  /// The raw-array dump this change filters is the one #448 added so
  /// tool-use arguments reach the output scan. Dropping the reasoning
  /// blocks must not drop that: a blocked literal inside `tool_use.input`
  /// still has to refuse the response. Asserting the argument text merely
  /// ARRIVES proves nothing — it is part of the upstream body and is
  /// delivered verbatim whether or not the dump still feeds the scan.
  test("/v1/messages buffered: tool-use arguments are still scanned after the reasoning filter", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "msg_out_tooluse",
        type: "message",
        role: "assistant",
        model: "upstream-model-x",
        content: [
          {
            type: "thinking",
            thinking: "nothing interesting in here",
            signature: "sig-out",
          },
          { type: "text", text: "calling the tool" },
          {
            type: "tool_use",
            id: "t1",
            name: "lookup",
            input: { q: `argument carrying ${BLOCKED}` },
          },
        ],
        usage: { input_tokens: 3, output_tokens: 9 },
      },
    });
    await seedScenario("tooluse-output-scan", upstream, "anthropic", [blockGuardrailId]);

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "tooluse-output-scan",
        max_tokens: 16,
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    expect(res.status).toBe(422);
    const body = await res.text();
    expect(body).not.toContain(BLOCKED);
    expect(body).not.toContain("argument carrying");
  });

  test("/v1/messages streaming: generated thinking deltas are neither scanned nor masked", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      rawStreamFrames: [
        `event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream_thinking","type":"message","role":"assistant","model":"upstream-model-x","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}\n\n`,
        `event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n\n`,
        `event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"considering ${BLOCKED} and ${EMAIL}"}}\n\n`,
        `event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n`,
        `event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n`,
        `event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"reach us at ${EMAIL}"}}\n\n`,
        `event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n`,
        `event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}\n\n`,
        `event: message_stop\ndata: {"type":"message_stop"}\n\n`,
      ],
    });
    await seedScenario("thinking-stream-output-scope", upstream, "anthropic", [
      maskGuardrailId,
      blockGuardrailId,
    ]);

    const r = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "thinking-stream-output-scope",
        max_tokens: 16,
        stream: true,
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    expect(r.status).toBe(200);
    const text = await r.text();

    // The block rule never sees the thinking delta, so the stream is not
    // terminated with an error frame. Assert the shape this endpoint
    // ACTUALLY emits: since the terminal frame moved to
    // `invalid_request_error`, a `content_filter` assertion here could never
    // go red no matter what the guardrail did.
    expect(text).not.toContain("event: error");
    expect(text).not.toContain('"type":"invalid_request_error"');
    // …the thinking delta reaches the caller unrewritten…
    expect(text).toContain(`considering ${BLOCKED} and ${EMAIL}`);
    // …and the text delta beside it is masked, which is what proves the
    // output mask ran at all.
    expect(text).toContain(`reach us at ${MASKED}`);
  });

  test("/v1/responses buffered: generated reasoning is neither scanned nor masked", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    // BLOCKED sits in BOTH reasoning slots. `content[]` is the one that
    // matters: the output walk reads `content` off every item regardless of
    // type, so without an explicit skip a reasoning item's text reached the
    // output scan on this surface alone.
    const reasoningItem = `{"type":"reasoning","id":"rs_out","summary":[{"type":"summary_text","text":"considering ${BLOCKED} and ${EMAIL}"}],"content":[{"type":"reasoning_text","text":"also ${BLOCKED} and ${EMAIL}"}]}`;
    const messageItem = `{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"reach us at ${EMAIL}"}]}`;
    const upstream = await upstreamWith({
      rawStreamFrames: [
        `event: response.created\ndata: {"type":"response.created","response":{"id":"resp_out_reasoning","object":"response","status":"in_progress","model":"upstream-model-x"}}\n\n`,
        `event: response.reasoning_summary_text.delta\ndata: {"type":"response.reasoning_summary_text.delta","item_id":"rs_out","delta":"considering ${BLOCKED} and ${EMAIL}"}\n\n`,
        `event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"m","delta":"reach us at ${EMAIL}"}\n\n`,
        `event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_out_reasoning","object":"response","status":"completed","model":"upstream-model-x","output":[${reasoningItem},${messageItem}],"usage":{"input_tokens":3,"output_tokens":9,"total_tokens":12}}}\n\n`,
      ],
    });
    await seedScenario("reasoning-output-scope", upstream, "openai", [
      maskGuardrailId,
      blockGuardrailId,
    ]);

    const r = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "reasoning-output-scope",
        input: "hello",
        stream: true,
      }),
    });
    expect(r.status).toBe(200);
    const text = await r.text();

    // Not blocked by the literal that appears only inside reasoning. This
    // surface refuses with an HTTP 422 rather than an in-band frame, so the
    // `r.status` assertion above is the real check — a `content_filter`
    // substring assertion here could never go red, because `responses.rs`
    // has no error-frame producer at all on the verbatim hold-back path.
    expect(text).not.toContain("guardrail");
    // …the reasoning summary and content survive the output mask…
    expect(text).toContain(`considering ${BLOCKED} and ${EMAIL}`);
    expect(text).toContain(`also ${BLOCKED} and ${EMAIL}`);
    // …and the assistant text beside them is masked.
    expect(text).toContain(`reach us at ${MASKED}`);
  });

  // ── 4. a non-SSE body answering a streaming request ────────────────

  test("/v1/responses: a JSON body answering stream:true takes the buffered scan+mask path", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    // The mock replies with a JSON document regardless of the request's
    // `stream` flag — an upstream that ignores it, which is what this arm
    // exists for.
    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "resp_not_sse",
        object: "response",
        status: "completed",
        model: "upstream-model-x",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: `reach us at ${EMAIL}` }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 5, total_tokens: 8 },
      },
    });
    await seedScenario("responses-not-sse", upstream, "openai", [maskGuardrailId]);

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "responses-not-sse",
        input: "hello",
        stream: true,
      }),
    });
    const text = await res.text();

    expect(res.status).toBe(200);
    // Scanned and masked, as the non-streaming body it actually is. On the
    // request-flag branch it entered the SSE hold-back, where a body with
    // no frames was seen by nothing.
    expect(text).toContain(MASKED);
    expect(text).not.toContain(EMAIL);
    // …and released as itself: no `\n\n` frame terminator appended to a
    // document that is not SSE, and not relabelled as a stream.
    expect(text.endsWith("\n\n")).toBe(false);
    expect(() => JSON.parse(text)).not.toThrow();
    expect(res.headers.get("content-type") ?? "").toContain("application/json");
  });

  test("/v1/messages: a JSON body answering stream:true takes the buffered scan+mask path", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await upstreamWith({
      nonStreamBody: {
        id: "msg_not_sse",
        type: "message",
        role: "assistant",
        model: "upstream-model-x",
        content: [{ type: "text", text: `reach us at ${EMAIL}` }],
        usage: { input_tokens: 3, output_tokens: 5 },
      },
    });
    await seedScenario("messages-not-sse", upstream, "anthropic", [maskGuardrailId]);

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "messages-not-sse",
        max_tokens: 16,
        stream: true,
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    const text = await res.text();

    expect(res.status).toBe(200);
    expect(text).toContain(MASKED);
    expect(text).not.toContain(EMAIL);
    expect(text.endsWith("\n\n")).toBe(false);
    expect(() => JSON.parse(text)).not.toThrow();
  });
});
