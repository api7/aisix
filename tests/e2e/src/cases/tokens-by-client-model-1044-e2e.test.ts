import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1044: aisix_llm_tokens_by_client_total gains a `model`
// label (the requested logical model, same value as the aisix_llm_* families'
// `model`), so "which models is each client spending tokens on" is answerable.
//
// Pinned here:
//   1. One client_type calling TWO models on /v1/chat/completions
//      (non-streaming model A, streaming model B) produces two independent
//      series per token_type, each with the correct per-model counts.
//   2. /v1/responses records the by-client series at all. Codex — an
//      allowlisted client_type — talks to /v1/responses, yet the endpoint
//      recorded nothing before this fix, so codex traffic was invisible in
//      the metric.

const CALLER = "sk-1044-client-model-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const MODEL_A = "cm1044-chat-a";
const MODEL_B = "cm1044-chat-b";
const RESPONSES_MODEL = "cm1044-responses";

// claude-cli/* normalises to "claude-code"; codex/* to "codex".
const CHAT_UA = "claude-cli/1.2.3";
const CHAT_CLIENT_TYPE = "claude-code";
const RESPONSES_UA = "codex/0.9.0";
const RESPONSES_CLIENT_TYPE = "codex";

const USAGE_A = { prompt_tokens: 11, completion_tokens: 13, total_tokens: 24 };
const USAGE_B = { prompt_tokens: 7, completion_tokens: 5, total_tokens: 12 };
const USAGE_RESP = { input_tokens: 5, output_tokens: 6, total_tokens: 11 };

/** Value of the by-client series for the given label combo, or undefined. */
function seriesValue(
  text: string,
  clientType: string,
  model: string,
  tokenType: string,
): number | undefined {
  for (const line of text.split("\n")) {
    if (
      line.startsWith("aisix_llm_tokens_by_client_total{") &&
      line.includes(`client_type="${clientType}"`) &&
      line.includes(`model="${model}"`) &&
      line.includes(`token_type="${tokenType}"`)
    ) {
      return Number(line.trim().split(/\s+/).pop());
    }
  }
  return undefined;
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

/** Poll the scrape until `probe` extracts a value (stream emits race the scrape). */
async function pollSeries(
  app: SpawnedApp,
  probe: (text: string) => boolean,
): Promise<string> {
  let text = "";
  for (let i = 0; i < 60; i++) {
    text = await scrape(app);
    if (probe(text)) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  return text;
}

describe("aisix_llm_tokens_by_client_total model label (AISIX-Cloud#1044)", () => {
  let app: SpawnedApp | undefined;
  let nonStreamUpstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let responsesUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    nonStreamUpstream = await startOpenAiUpstream({
      nonStreamBody: chatBody(USAGE_A),
    });
    streamUpstream = await startOpenAiUpstream({
      streamEvents: streamEvents(USAGE_B),
      eventDelayMs: 2,
    });
    responsesUpstream = await startOpenAiUpstream({
      nonStreamBody: responsesBody(USAGE_RESP),
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pkA = await seed.createProviderKey({
      display_name: "cm1044-pk-a",
      secret: "sk-mock",
      api_base: `${nonStreamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL_A,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkA.id,
    });
    const pkB = await seed.createProviderKey({
      display_name: "cm1044-pk-b",
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL_B,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkB.id,
    });
    const pkR = await seed.createProviderKey({
      display_name: "cm1044-pk-resp",
      secret: "sk-mock",
      api_base: `${responsesUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: RESPONSES_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pkR.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [MODEL_A, MODEL_B, RESPONSES_MODEL],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await nonStreamUpstream?.close();
    await streamUpstream?.close();
    await responsesUpstream?.close();
  });

  test("one client_type on two chat models yields two per-model series (non-streaming + streaming)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return [MODEL_A, MODEL_B, RESPONSES_MODEL].every((m) =>
        data.some((d) => d.id === m),
      );
    });

    // Same client UA, two different models: A non-streaming, B streaming —
    // exercising both chat recording paths.
    const resA = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "user-agent": CHAT_UA,
      },
      body: JSON.stringify({
        model: MODEL_A,
        messages: [{ role: "user", content: "model a" }],
      }),
    });
    expect(resA.status).toBe(200);

    const resB = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "user-agent": CHAT_UA,
      },
      body: JSON.stringify({
        model: MODEL_B,
        messages: [{ role: "user", content: "model b" }],
        stream: true,
      }),
    });
    expect(resB.status).toBe(200);
    await resB.text(); // drain the SSE so on_complete fires

    const text = await pollSeries(
      app,
      (t) =>
        seriesValue(t, CHAT_CLIENT_TYPE, MODEL_A, "total") !== undefined &&
        seriesValue(t, CHAT_CLIENT_TYPE, MODEL_B, "total") !== undefined,
    );

    // Two independent series under one client_type, correct per-model counts.
    expect(seriesValue(text, CHAT_CLIENT_TYPE, MODEL_A, "input")).toBe(
      USAGE_A.prompt_tokens,
    );
    expect(seriesValue(text, CHAT_CLIENT_TYPE, MODEL_A, "output")).toBe(
      USAGE_A.completion_tokens,
    );
    expect(seriesValue(text, CHAT_CLIENT_TYPE, MODEL_A, "total")).toBe(
      USAGE_A.total_tokens,
    );
    expect(seriesValue(text, CHAT_CLIENT_TYPE, MODEL_B, "input")).toBe(
      USAGE_B.prompt_tokens,
    );
    expect(seriesValue(text, CHAT_CLIENT_TYPE, MODEL_B, "output")).toBe(
      USAGE_B.completion_tokens,
    );
    expect(seriesValue(text, CHAT_CLIENT_TYPE, MODEL_B, "total")).toBe(
      USAGE_B.total_tokens,
    );

    // Every by-client series now carries the model label.
    for (const line of text.split("\n")) {
      if (line.startsWith("aisix_llm_tokens_by_client_total{")) {
        expect(line).toMatch(/model="/);
      }
    }
  });

  test("/v1/responses records the by-client series (codex was invisible before)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER}`,
        "content-type": "application/json",
        "user-agent": RESPONSES_UA,
      },
      body: JSON.stringify({
        model: RESPONSES_MODEL,
        input: "hello from codex",
      }),
    });
    expect(res.status).toBe(200);

    const text = await pollSeries(
      app,
      (t) =>
        seriesValue(t, RESPONSES_CLIENT_TYPE, RESPONSES_MODEL, "total") !==
        undefined,
    );

    expect(
      seriesValue(text, RESPONSES_CLIENT_TYPE, RESPONSES_MODEL, "input"),
    ).toBe(USAGE_RESP.input_tokens);
    expect(
      seriesValue(text, RESPONSES_CLIENT_TYPE, RESPONSES_MODEL, "output"),
    ).toBe(USAGE_RESP.output_tokens);
    expect(
      seriesValue(text, RESPONSES_CLIENT_TYPE, RESPONSES_MODEL, "total"),
    ).toBe(USAGE_RESP.total_tokens);
  });
});

function chatBody(usage: {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}) {
  return {
    id: "chatcmpl-1044",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "hello" },
        finish_reason: "stop",
      },
    ],
    usage,
  };
}

function streamEvents(usage: {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}): string[] {
  return [
    JSON.stringify({
      id: "chatcmpl-1044-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { role: "assistant" } }],
    }),
    JSON.stringify({
      id: "chatcmpl-1044-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: { content: "hello" } }],
    }),
    JSON.stringify({
      id: "chatcmpl-1044-stream",
      object: "chat.completion.chunk",
      model: "gpt-4o-mini",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage,
    }),
    "[DONE]",
  ];
}

function responsesBody(usage: {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}) {
  return {
    id: "resp_1044",
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status: "completed",
    model: "gpt-4o-mini",
    output: [
      {
        id: "msg_1044",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "hello from responses" }],
      },
    ],
    usage,
  };
}
