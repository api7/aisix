import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
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

// E2E: a guardrail leaves an `aisix_guardrail_latency_seconds` sample only
// on the hook it is attached to. An input-only row has no `phase="output"`
// series, an output-only row no `phase="input"` one, and a `both` row has
// both — on non-streaming and streaming requests, across the chat,
// messages and responses surfaces. A row that does not apply at a phase
// did no work there, so a zero-length `allowed` sample on that phase would
// inflate its per-phase counts and drag its latency distribution to zero.

const CALLER = "sk-guardrail-phase-applicability-e2e";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const MODEL = "guardrail-phase-applicability";
const MODEL_STREAM = "guardrail-phase-applicability-stream";

const IN_KW = "phase-kw-input-only";
const OUT_KW = "phase-kw-output-only";
const BOTH_KW = "phase-kw-both";
const IN_MOD = "phase-moderation-input-only";

async function startModerationMock(): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const server: Server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "modr-mock",
          model: "omni-moderation-latest",
          results: [{ flagged: false, categories: {}, category_scores: {} }],
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  };
}

function chunk(delta: Record<string, unknown>, extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    id: "chatcmpl-phase",
    object: "chat.completion.chunk",
    model: "deepseek-chat",
    choices: [{ index: 0, delta, ...extra }],
  });
}

/** `aisix_guardrail_latency_seconds_count` summed over series matching every label. */
function latencyCount(scrape: string, labels: Record<string, string>): number {
  let sum = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith("aisix_guardrail_latency_seconds_count{")) continue;
    if (!Object.entries(labels).every(([k, v]) => line.includes(`${k}="${v}"`))) continue;
    const v = Number(line.split("}").at(-1)?.trim());
    if (!Number.isNaN(v)) sum += v;
  }
  return sum;
}

describe("guardrail phase applicability e2e: samples only on attached hooks", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let moderation: Awaited<ReturnType<typeof startModerationMock>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    moderation = await startModerationMock();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-phase",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "deepseek-chat",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a clean reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    // The mock answers in one shape per instance, so streaming requests get
    // their own upstream and model.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        chunk({ role: "assistant" }),
        chunk({ content: "a clean reply" }),
        chunk({}, { finish_reason: "stop" }),
        "[DONE]",
      ],
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    for (const [model, up] of [
      [MODEL, upstream],
      [MODEL_STREAM, streamUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        provider: "deepseek",
        adapter: "openai",
        secret: "sk-mock",
        api_base: `${up.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: model,
        provider: "deepseek",
        model_name: "deepseek-chat",
        provider_key_id: pk.id,
      });
    }
    const never = [{ kind: "literal", value: "never-matches-phase-marker" }];
    await seed.createGuardrail({
      name: IN_KW,
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: never,
    });
    await seed.createGuardrail({
      name: OUT_KW,
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: never,
    });
    await seed.createGuardrail({
      name: BOTH_KW,
      enabled: true,
      hook_point: "both",
      kind: "keyword",
      patterns: never,
    });
    await seed.createGuardrail({
      name: IN_MOD,
      enabled: true,
      hook_point: "input",
      fail_open: false,
      kind: "openai_moderation",
      api_key: "sk-moderation-key",
      endpoint: moderation.baseUrl,
    });
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: [MODEL, MODEL_STREAM] });

    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
    await moderation?.close();
  });

  const scrape = async (): Promise<string> => {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    return res.text();
  };

  const cases = [
    {
      name: "chat, non-streaming",
      path: "/v1/chat/completions",
      body: { model: MODEL, messages: [{ role: "user", content: "hello" }] },
    },
    {
      name: "chat, streaming",
      path: "/v1/chat/completions",
      body: { model: MODEL_STREAM, stream: true, messages: [{ role: "user", content: "hello" }] },
    },
    {
      name: "messages, streaming",
      path: "/v1/messages",
      body: {
        model: MODEL_STREAM,
        stream: true,
        max_tokens: 16,
        messages: [{ role: "user", content: "hello" }],
      },
    },
    {
      name: "responses, non-streaming",
      path: "/v1/responses",
      body: { model: MODEL, input: "hello" },
    },
  ];

  for (const c of cases) {
    test(`${c.name}: each guardrail records only the phases it is attached to`, async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      const before = await scrape();
      const res = await fetch(`${app.proxyUrl}${c.path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER}`,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify(c.body),
      });
      const text = await res.text();
      expect(res.status, text).toBe(200);
      expect(text).toContain("a clean reply");
      const after = await scrape();

      const grew = (guardrail: string, phase: string) =>
        latencyCount(after, { guardrail, phase }) >
        latencyCount(before, { guardrail, phase });
      const absent = (guardrail: string, phase: string) =>
        latencyCount(after, { guardrail, phase }) === 0;

      for (const g of [IN_KW, IN_MOD]) {
        expect(grew(g, "input"), `${g} input`).toBe(true);
        expect(absent(g, "output"), `${g} has no output series`).toBe(true);
      }
      expect(grew(OUT_KW, "output"), `${OUT_KW} output`).toBe(true);
      expect(absent(OUT_KW, "input"), `${OUT_KW} has no input series`).toBe(true);
      expect(grew(BOTH_KW, "input"), `${BOTH_KW} input`).toBe(true);
      expect(grew(BOTH_KW, "output"), `${BOTH_KW} output`).toBe(true);
    });
  }
});
