import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1405: `/v1/messages` in front of an
// OpenAI-compatible upstream that reports a prompt-cache hit.
//
// Pre-fix the hit was dropped on BOTH exits of the cross-protocol
// bridge — the Anthropic `usage` handed to the client, and the
// UsageEvent that drives Logs / billing — so an operator saw the full
// 68k prompt billed at the uncached rate and had no cache detail to
// reconcile against the provider's own bill. Worse, the absence read
// as "the provider never cached", which is unfalsifiable from the
// dashboard.
//
// The two exits carry the SAME facts in DIFFERENT shapes, and that is
// the whole point of the fix:
//
//   client (Anthropic semantics: input excludes cache)
//     input_tokens            = P - C
//     cache_read_input_tokens = C
//
//   UsageEvent (upstream's own OpenAI semantics: cached ⊂ prompt)
//     prompt_tokens           = P
//     cached_prompt_tokens    = C
//
// Keeping the UsageEvent in the upstream's shape is what makes the same
// upstream call bill identically whichever inbound protocol addressed
// it, and it is the shape cp-api's pricing split expects (it charges
// `prompt - cached` at the prompt rate and `cached` at the cache-read
// rate, and rejects an event whose `cached` exceeds its `prompt`).
//
// The UsageEvent is observed through a mock Datadog intake — the DP's
// only e2e-reachable surface that carries the full event, since the
// OTLP span attributes stop at input/output tokens and the cache
// counters have no Prometheus metric of their own yet (#1404).
//
// Reporter's numbers: MiniMax M3 behind an OpenAI-compatible provider,
// addressed by the Claude CLI.
// https://platform.minimax.io/docs/api-reference/text-prompt-caching

const CALLER_PLAINTEXT = "sk-messages-cache-tokens-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const PROMPT_TOKENS = 68_274;
const CACHED_TOKENS = 60_000;
const COMPLETION_TOKENS = 497;
const UNCACHED_TOKENS = PROMPT_TOKENS - CACHED_TOKENS;

const CREDENTIAL_REF = "e2e";
const DD_API_KEY = "dd-cache-tokens-test-key";
const INTAKE_PATH = "/api/v2/logs";

const NON_STREAM_MODEL = "cache-tokens-nonstream";
const STREAM_MODEL = "cache-tokens-stream";

const USAGE = {
  prompt_tokens: PROMPT_TOKENS,
  completion_tokens: COMPLETION_TOKENS,
  total_tokens: PROMPT_TOKENS + COMPLETION_TOKENS,
  prompt_tokens_details: { cached_tokens: CACHED_TOKENS },
};

const NON_STREAM_BODY = {
  id: "chatcmpl-cache-test",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "MiniMax-M3",
  choices: [
    { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
  ],
  usage: USAGE,
};

function chunk(json: Record<string, unknown>): string {
  return JSON.stringify({
    id: "chatcmpl-cache-test",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "MiniMax-M3",
    ...json,
  });
}

// OpenAI's real `include_usage` shape: the usage-only frame lands AFTER
// the stop chunk, so the cache hit is only knowable at stream close.
const STREAM_EVENTS = [
  chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }] }),
  chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  chunk({ choices: [], usage: USAGE }),
  "[DONE]",
];

interface CapturedLog {
  path: string;
  logs: Record<string, unknown>[];
  /** Set when the intake body did not decode as gzipped JSON. */
  decodeError?: string;
}

interface MockDatadog {
  site: string;
  requests: CapturedLog[];
  close(): Promise<void>;
}

/** Mock Datadog Logs intake — the DP's e2e window onto the full UsageEvent. */
async function startMockDatadog(): Promise<MockDatadog> {
  const requests: CapturedLog[] = [];
  const server: Server = createServer((req: IncomingMessage, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "").split("?")[0];
      if (req.method === "POST" && path === INTAKE_PATH) {
        let logs: Record<string, unknown>[] = [];
        let decodeError: string | undefined;
        try {
          const parsed: unknown = JSON.parse(gunzipSync(Buffer.concat(chunks)).toString("utf8"));
          if (Array.isArray(parsed)) {
            logs = parsed as Record<string, unknown>[];
          } else {
            decodeError = `intake body decoded to ${typeof parsed}, expected a JSON array`;
          }
        } catch (err) {
          // Keep the reason. Swallowing it would turn "the DP changed its
          // intake encoding" into an unrelated poll timeout below.
          decodeError = `intake body is not gzipped JSON: ${String(err)}`;
        }
        requests.push({ path, logs, decodeError });
      }
      res.statusCode = 202;
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    site: `127.0.0.1:${port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

/**
 * Poll the intake for a UsageEvent of one model. `/v1/messages` does not
 * emit the `x-aisix-call-id` response header the chat handler does, so the
 * streaming and non-streaming legs use a model each and match on that —
 * every event for a given model came from the same canned upstream, so any
 * of them carries the counts under test.
 */
async function waitForUsageEvent(
  dd: MockDatadog,
  model: string,
  timeoutMs = 15_000,
): Promise<Record<string, unknown>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    for (const req of dd.requests) {
      // A body the mock could not decode is a harness/wire failure, not a
      // slow event — fail on it as itself instead of waiting out the poll.
      if (req.decodeError) throw new Error(req.decodeError);
      const hit = req.logs.find((l) => l["aisix.requested_model"] === model);
      if (hit) return hit;
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no usage event for model ${model} within ${timeoutMs}ms`);
}

async function postMessages(
  app: SpawnedApp,
  model: string,
  stream: boolean,
): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/messages`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": CALLER_PLAINTEXT,
      "user-agent": "claude-cli/2.1.118 (external, cli)",
    },
    body: JSON.stringify({
      model,
      max_tokens: 200,
      stream,
      messages: [{ role: "user", content: "issue 1405 regression" }],
    }),
  });
}

/** The closing `message_delta` — the only frame that can carry final usage. */
function closingUsage(sse: string): Record<string, unknown> {
  for (const line of sse.split("\n")) {
    const data = line.startsWith("data: ") ? line.slice(6) : undefined;
    if (!data) continue;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(data) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (parsed.type === "message_delta") {
      return (parsed.usage ?? {}) as Record<string, unknown>;
    }
  }
  throw new Error(`no message_delta in stream:\n${sse}`);
}

describe("/v1/messages keeps an OpenAI upstream's prompt-cache hit (AISIX-Cloud#1405)", () => {
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let dd: MockDatadog | undefined;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    nonStreamUpstream = await startOpenAiUpstream({ nonStreamBody: NON_STREAM_BODY });
    streamUpstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS });
    dd = await startMockDatadog();

    app = await spawnApp({
      admin: false,
      extraEnv: { [`DD_CRED_${CREDENTIAL_REF.toUpperCase()}_API_KEY`]: DD_API_KEY },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "cache-tokens-datadog",
      enabled: true,
      kind: "datadog",
      site: dd.site,
      credential_ref: CREDENTIAL_REF,
      service: "aisix-e2e",
      content_mode: "metadata_only",
    });

    for (const [model, upstream] of [
      [NON_STREAM_MODEL, nonStreamUpstream],
      [STREAM_MODEL, streamUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-mock-minimax",
        api_base: `${upstream.baseUrl}/v1`,
        provider: "openai",
        adapter: "openai",
      });
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "MiniMax-M3",
        provider_key_id: pk.id,
      });
    }
    // Caller key last, so gating on it authenticating implies the whole
    // seed set has landed (tests/e2e/AGENTS.md).
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [NON_STREAM_MODEL, STREAM_MODEL],
    });

    // Gate on an independent, non-throwing condition: driving `/v1/messages`
    // here would let a usage regression surface as a propagation timeout
    // instead of as the assertion that actually broke.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return [NON_STREAM_MODEL, STREAM_MODEL].every((m) =>
        data.some((row) => row.id === m),
      );
    });
  });

  afterAll(async () => {
    await app?.exit();
    await nonStreamUpstream?.close();
    await streamUpstream?.close();
    await dd?.close();
  });

  test("non-streaming: cache read reaches the Anthropic client and the usage event", async (ctx) => {
    if (!etcdReachable || !app || !dd) {
      ctx.skip();
      return;
    }
    const res = await postMessages(app, NON_STREAM_MODEL, false);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { usage?: Record<string, unknown> };
    const usage = body.usage ?? {};

    // Anthropic semantics: input_tokens is the NON-cached input.
    expect(usage.input_tokens).toBe(UNCACHED_TOKENS);
    expect(usage.cache_read_input_tokens).toBe(CACHED_TOKENS);
    expect(usage.output_tokens).toBe(COMPLETION_TOKENS);
    // An OpenAI upstream reports no cache WRITE — never fabricate one.
    expect(usage.cache_creation_input_tokens).toBeUndefined();

    // UsageEvent keeps the upstream's own OpenAI shape.
    const event = await waitForUsageEvent(dd, NON_STREAM_MODEL);
    expect(event["gen_ai.usage.input_tokens"]).toBe(PROMPT_TOKENS);
    expect(event["aisix.cached_prompt_tokens"]).toBe(CACHED_TOKENS);
    expect(event["gen_ai.usage.output_tokens"]).toBe(COMPLETION_TOKENS);
    // The hit is a SUBSET of prompt_tokens, so the Anthropic-shape
    // additive counters stay absent and the total is not double-counted.
    expect(event["aisix.cache_read_tokens"]).toBeUndefined();
    expect(event["aisix.cache_creation_tokens"]).toBeUndefined();
    expect(event["aisix.inbound_protocol"]).toBe("anthropic");
  });

  test("streaming: cache read rides the closing message_delta and the usage event", async (ctx) => {
    if (!etcdReachable || !app || !dd) {
      ctx.skip();
      return;
    }
    const res = await postMessages(app, STREAM_MODEL, true);
    expect(res.status).toBe(200);
    const usage = closingUsage(await res.text());

    expect(usage.input_tokens).toBe(UNCACHED_TOKENS);
    expect(usage.cache_read_input_tokens).toBe(CACHED_TOKENS);
    expect(usage.output_tokens).toBe(COMPLETION_TOKENS);
    expect(usage.cache_creation_input_tokens).toBeUndefined();

    const event = await waitForUsageEvent(dd, STREAM_MODEL);
    expect(event["gen_ai.usage.input_tokens"]).toBe(PROMPT_TOKENS);
    expect(event["aisix.cached_prompt_tokens"]).toBe(CACHED_TOKENS);
    expect(event["gen_ai.usage.output_tokens"]).toBe(COMPLETION_TOKENS);
    expect(event["aisix.cache_read_tokens"]).toBeUndefined();
  });
});
