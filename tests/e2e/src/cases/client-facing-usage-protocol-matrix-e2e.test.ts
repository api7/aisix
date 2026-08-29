import { createHash } from "node:crypto";
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

// E2E for AISIX-Cloud#1447: an OpenAI-protocol client in front of an
// Anthropic-protocol upstream.
//
// A gateway that speaks several protocols stores usage in whichever
// accounting shape the UPSTREAM reported, and the two shapes disagree on
// exactly one thing — whether the cache counters live inside the input
// count or beside it:
//
//   OpenAI shape     prompt_tokens INCLUDES prompt_tokens_details.cached
//   Anthropic shape  input_tokens EXCLUDES cache_creation / cache_read
//
// Copying those stored fields onto an OpenAI-protocol response therefore
// answers the client in a foreign accounting. Pre-fix, this upstream's
// `input_tokens: 40` reached an OpenAI client verbatim while the total
// had folded all 140 input tokens in, so `/v1/chat/completions` reported
// `40 + 10 = 150` and dropped the cache detail entirely, and
// `/v1/responses` reported `cached_tokens: 70` against
// `input_tokens: 40` — a subset larger than its own superset.
//
// The rule this pins, on both exits of every cell:
//
//   client      → the caller's OWN protocol's accounting
//   UsageEvent  → the upstream's raw counters, untouched
//
// so a call bills identically whichever protocol addressed it, while no
// client is ever handed another protocol's numbers.
//
// The mirror direction (`/v1/messages` over an OpenAI upstream) is
// AISIX-Cloud#1405 and has its own spec —
// `messages-openai-cache-tokens-e2e.test.ts`.
//
// References:
// - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
// - https://platform.openai.com/docs/api-reference/chat/object
// - https://platform.openai.com/docs/api-reference/responses/object

const CALLER_PLAINTEXT = "sk-usage-matrix-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "e2e";
const MOCK_AK_ID = "mock-ak-id";
const MOCK_AK_SECRET = "mock-ak-secret";
const SLS_PROJECT = "usage-matrix-proj";
const LOGSTORE = "usage-matrix-store";

// One call, as the Anthropic upstream reports it. The model read
// 40 + 30 + 70 = 140 input tokens and 150 in total; no client may be
// told otherwise, whichever protocol it asks in.
const UNCACHED_IN = 40;
const CACHE_WRITE = 30;
const CACHE_READ = 70;
const OUT = 10;
const TOTAL_IN = UNCACHED_IN + CACHE_WRITE + CACHE_READ;
const TOTAL = TOTAL_IN + OUT;

// The same call under OpenAI accounting, for the non-conversion cell:
// one prompt count that already contains the cache read.
const OAI_PROMPT = 120;
const OAI_CACHED = 70;
const OAI_OUT = 9;

const ANTHROPIC_MODEL = "usage-matrix-anthropic";
const ANTHROPIC_STREAM_MODEL = "usage-matrix-anthropic-stream";
const OPENAI_MODEL = "usage-matrix-openai";

const ANTHROPIC_BODY = {
  id: "msg_usage_matrix",
  type: "message",
  role: "assistant",
  model: "claude-3-5-haiku-20241022",
  content: [{ type: "text", text: "ok" }],
  stop_reason: "end_turn",
  usage: {
    input_tokens: UNCACHED_IN,
    output_tokens: OUT,
    cache_creation_input_tokens: CACHE_WRITE,
    cache_read_input_tokens: CACHE_READ,
  },
};

// Anthropic's real streaming split: `message_start` carries the input
// side (cache counters included), `message_delta` closes with the output.
const ANTHROPIC_STREAM_FRAMES = [
  `event: message_start\ndata: ${JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_usage_matrix",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: {
        input_tokens: UNCACHED_IN,
        cache_creation_input_tokens: CACHE_WRITE,
        cache_read_input_tokens: CACHE_READ,
        output_tokens: 1,
      },
    },
  })}\n\n`,
  `event: content_block_delta\ndata: ${JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "ok" },
  })}\n\n`,
  `event: message_delta\ndata: ${JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: OUT },
  })}\n\n`,
  `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
];

const OPENAI_BODY = {
  id: "chatcmpl-usage-matrix",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: {
    prompt_tokens: OAI_PROMPT,
    completion_tokens: OAI_OUT,
    total_tokens: OAI_PROMPT + OAI_OUT,
    prompt_tokens_details: { cached_tokens: OAI_CACHED },
  },
};

interface Called {
  status: number;
  requestId: string;
  text: string;
}

/** The usage block a `response.completed` / final chunk carries. */
function usageFromSse(text: string, pick: (frame: Record<string, unknown>) => unknown): Record<string, number> {
  let found: Record<string, number> | undefined;
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6).trim();
    if (data === "[DONE]") continue;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(data) as Record<string, unknown>;
    } catch {
      continue;
    }
    const candidate = pick(parsed);
    if (candidate && typeof candidate === "object") {
      found = candidate as Record<string, number>;
    }
  }
  if (!found) throw new Error(`no usage frame in stream:\n${text}`);
  return found;
}

describe("client-facing usage is the CALLER's protocol, not the upstream's (AISIX-Cloud#1447)", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let anthropicUpstream: OpenAiUpstream | undefined;
  let anthropicStreamUpstream: OpenAiUpstream | undefined;
  let openaiUpstream: OpenAiUpstream | undefined;

  async function call(path: string, body: unknown): Promise<Called> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify(body),
    });
    return {
      status: res.status,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
      text: await res.text(),
    };
  }

  /** The UsageEvent must carry the UPSTREAM's own counters, every cell. */
  async function expectAnthropicShapedEvent(requestId: string): Promise<void> {
    const log = await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
    expect(log.get("prompt_tokens")).toBe(String(UNCACHED_IN));
    expect(log.get("completion_tokens")).toBe(String(OUT));
    expect(log.get("cache_creation_tokens")).toBe(String(CACHE_WRITE));
    expect(log.get("cache_read_tokens")).toBe(String(CACHE_READ));
    // The Anthropic shape has no OpenAI-style subset; a value here would
    // mean the client-facing conversion had leaked back into billing.
    expect(log.get("cached_prompt_tokens") ?? "0").toBe("0");
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    anthropicUpstream = await startOpenAiUpstream({ nonStreamBody: ANTHROPIC_BODY });
    anthropicStreamUpstream = await startOpenAiUpstream({
      rawStreamFrames: ANTHROPIC_STREAM_FRAMES,
    });
    openaiUpstream = await startOpenAiUpstream({ nonStreamBody: OPENAI_BODY });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "usage-matrix-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    // The Anthropic bridge appends `/v1/messages` to the api_base, so it
    // gets the bare host; the OpenAI bridge expects the `/v1` suffix.
    for (const [model, upstream] of [
      [ANTHROPIC_MODEL, anthropicUpstream],
      [ANTHROPIC_STREAM_MODEL, anthropicStreamUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-ant-mock",
        api_base: upstream.baseUrl,
        provider: "anthropic",
        adapter: "anthropic",
      });
      await seed.createModel({
        display_name: model,
        provider: "anthropic",
        model_name: "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
    }
    const oaiPk = await seed.createProviderKey({
      display_name: "usage-matrix-openai-pk",
      secret: "sk-openai-mock",
      api_base: `${openaiUpstream.baseUrl}/v1`,
      provider: "openai",
      adapter: "openai",
    });
    await seed.createModel({
      display_name: OPENAI_MODEL,
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: oaiPk.id,
    });

    // Caller key last: one etcd watch applies in revision order, so the
    // moment it authenticates the whole seed set is in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [ANTHROPIC_MODEL, ANTHROPIC_STREAM_MODEL, OPENAI_MODEL],
    });

    // Gate on something the tests do NOT assert on, so a usage regression
    // fails as an assertion rather than a propagation timeout.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return [ANTHROPIC_MODEL, ANTHROPIC_STREAM_MODEL, OPENAI_MODEL].every((m) =>
        data.some((row) => row.id === m),
      );
    });
  });

  afterAll(async () => {
    await app?.exit();
    await anthropicUpstream?.close();
    await anthropicStreamUpstream?.close();
    await openaiUpstream?.close();
    await sls?.close();
  });

  test("chat/completions over an Anthropic upstream: full input, cache hit, decomposable total", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: ANTHROPIC_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(200);
    const usage = (JSON.parse(res.text) as { usage: Record<string, unknown> }).usage;

    expect(usage.prompt_tokens).toBe(TOTAL_IN);
    expect(usage.completion_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(TOTAL);
    expect((usage.prompt_tokens_details as { cached_tokens: number }).cached_tokens).toBe(CACHE_READ);

    await expectAnthropicShapedEvent(res.requestId);
  });

  test("chat/completions streaming over an Anthropic upstream reports the same numbers", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: ANTHROPIC_STREAM_MODEL,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
      stream_options: { include_usage: true },
    });
    expect(res.status).toBe(200);
    const usage = usageFromSse(res.text, (f) => f.usage);

    expect(usage.prompt_tokens).toBe(TOTAL_IN);
    expect(usage.completion_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(TOTAL);
    // The cache detail has to survive the stream too — dropping it only
    // on the streaming exit is how #1405 shipped half-fixed.
    expect(
      (usage as unknown as { prompt_tokens_details: { cached_tokens: number } })
        .prompt_tokens_details.cached_tokens,
    ).toBe(CACHE_READ);

    await expectAnthropicShapedEvent(res.requestId);
  });

  test("responses over an Anthropic upstream keeps cached_tokens a subset of input_tokens", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/responses", { model: ANTHROPIC_MODEL, input: "hi" });
    expect(res.status).toBe(200);
    const usage = (JSON.parse(res.text) as { usage: Record<string, unknown> }).usage;

    expect(usage.input_tokens).toBe(TOTAL_IN);
    expect(usage.output_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(TOTAL);
    const cached = (usage.input_tokens_details as { cached_tokens: number }).cached_tokens;
    expect(cached).toBe(CACHE_READ);
    expect(cached).toBeLessThanOrEqual(usage.input_tokens as number);

    await expectAnthropicShapedEvent(res.requestId);
  });

  test("responses streaming over an Anthropic upstream reports the same numbers", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/responses", {
      model: ANTHROPIC_STREAM_MODEL,
      input: "hi",
      stream: true,
    });
    expect(res.status).toBe(200);
    const usage = usageFromSse(res.text, (f) => (f.response as { usage?: unknown } | undefined)?.usage);

    expect(usage.input_tokens).toBe(TOTAL_IN);
    expect(usage.output_tokens).toBe(OUT);
    expect(usage.total_tokens).toBe(TOTAL);
    const cached = (usage as unknown as { input_tokens_details: { cached_tokens: number } })
      .input_tokens_details.cached_tokens;
    expect(cached).toBe(CACHE_READ);
    expect(cached).toBeLessThanOrEqual(usage.input_tokens);

    await expectAnthropicShapedEvent(res.requestId);
  });

  test("no protocol conversion: an OpenAI upstream still passes through untouched", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await call("/v1/chat/completions", {
      model: OPENAI_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(200);
    const usage = (JSON.parse(res.text) as { usage: Record<string, unknown> }).usage;

    // Already OpenAI accounting: the projection must be the identity
    // here, never adding the cache hit on top of a prompt that contains it.
    expect(usage.prompt_tokens).toBe(OAI_PROMPT);
    expect(usage.completion_tokens).toBe(OAI_OUT);
    expect(usage.total_tokens).toBe(OAI_PROMPT + OAI_OUT);
    expect((usage.prompt_tokens_details as { cached_tokens: number }).cached_tokens).toBe(OAI_CACHED);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => l.get("request_id") === res.requestId,
      `usage row for ${res.requestId}`,
      15_000,
    );
    expect(log.get("prompt_tokens")).toBe(String(OAI_PROMPT));
    expect(log.get("cached_prompt_tokens")).toBe(String(OAI_CACHED));
    expect(log.get("cache_read_tokens") ?? "0").toBe("0");
  });
});
