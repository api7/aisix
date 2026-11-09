import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
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

// The built-in bank_card detector recognises a card however it is printed, a
// grouped number only under a card's prefix, and the built-in detectors recognise
// a value written straight after Chinese text.

const MASK_CALLER = "sk-pii-card-mask";
const BLOCK_CALLER = "sk-pii-card-block";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "pii-card";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const MASKED = "[BANK_CARD_REDACTED]";

const chunk = (content: string) =>
  JSON.stringify({
    id: "strm-card",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });

describe("bank_card recognises grouped numbers and values touching CJK text", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let leaky: OpenAiUpstream | undefined;
  const streams: OpenAiUpstream[] = [];

  /** A streaming upstream that splits `parts` across SSE chunks. */
  async function splitStream(seed: SeedClient, model: string, parts: string[]) {
    const stream = await startOpenAiUpstream({
      streamEvents: [
        ...parts.map(chunk),
        JSON.stringify({
          id: "strm-card",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        }),
        "[DONE]",
      ],
      eventDelayMs: 5,
    });
    streams.push(stream);
    const pk = await seed.createProviderKey({
      display_name: `${model}-pk`,
      secret: "sk-mock",
      api_base: `${stream.baseUrl}/v1`,
    });
    const created = await seed.createModel({
      display_name: model,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    return created.id;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    const answering = (content: string) => ({
      nonStreamBody: {
        id: "cmpl-card",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });
    upstream = await startOpenAiUpstream(answering("noted"));
    leaky = await startOpenAiUpstream(answering("the card on file is 6222 0212 3456 7890 128."));
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: LOGSTORE,
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "pii-card",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const pk = await seed.createProviderKey({
      display_name: "card-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const leakyPk = await seed.createProviderKey({
      display_name: "card-leaky-pk",
      secret: "sk-mock",
      api_base: `${leaky.baseUrl}/v1`,
    });
    const modelIds: Record<string, string> = {};
    for (const [name, providerKey] of [
      ["card-mask", pk],
      ["card-block", pk],
      ["card-mask-output", leakyPk],
      ["card-block-output", leakyPk],
    ] as const) {
      const created = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: providerKey.id,
      });
      modelIds[name] = created.id;
    }
    modelIds["card-mask-split"] = await splitStream(seed, "card-mask-split", [
      "the card is 4111 11",
      "11 1111 1111 on file",
    ]);
    modelIds["card-block-split"] = await splitStream(seed, "card-block-split", [
      "the card is 4111-1111-",
      "1111-1111 on file",
    ]);

    modelIds["card-mask-split-19"] = await splitStream(seed, "card-mask-split-19", [
      "the card is 6222 0212 3456 7890 ",
      "128 on file",
    ]);
    modelIds["card-block-split-19"] = await splitStream(seed, "card-block-split-19", [
      "the card is 6222-0212-3456-7890-",
      "128 on file",
    ]);

    const mask = await seed.createGuardrail(
      {
        name: "card-mask",
        enabled: true,
        hook_point: "both",
        kind: "pii",
        detectors: [{ type: "bank_card", action: "mask" }],
      },
      { attach: false },
    );
    const block = await seed.createGuardrail(
      {
        name: "card-block",
        enabled: true,
        hook_point: "both",
        kind: "pii",
        detectors: [{ type: "bank_card", action: "block" }],
      },
      { attach: false },
    );
    for (const [name, id] of Object.entries(modelIds)) {
      await seed.attachGuardrailToModel(name.includes("block") ? block.id : mask.id, id);
    }

    await seed.createApiKey({ key_hash: hash(MASK_CALLER), allowed_models: ["*"] });
    await seed.createApiKey({ key_hash: hash(BLOCK_CALLER), allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${BLOCK_CALLER}` },
      });
      await res.arrayBuffer();
      return res.status === 200;
    });
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await upstream?.close();
    await leaky?.close();
    for (const stream of streams) await stream.close();
  });

  const chat = (model: string, content: string, stream = false) =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${model.includes("block") ? BLOCK_CALLER : MASK_CALLER}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content }], stream }),
    });

  const upstreamCalls = () => upstream!.receivedRequests.length;
  const lastPrompt = () => {
    const body = upstream!.receivedRequests.at(-1)!.body;
    return (JSON.parse(body) as { messages: Array<{ content: string }> }).messages[0].content;
  };

  const rowOf = (res: Response) => {
    const requestId = res.headers.get("x-aisix-request-id");
    return waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, `row ${requestId}`);
  };

  const SENTENCE = (card: string) => `My credit card number is ${card}, please check my bill.`;

  const MASKED_SAMPLES = [
    ["spaces", SENTENCE("4111 1111 1111 1111"), SENTENCE(MASKED)],
    ["hyphens", SENTENCE("4111-1111-1111-1111"), SENTENCE(MASKED)],
    ["no separator", SENTENCE("4111111111111111"), SENTENCE(MASKED)],
    ["a 19-digit card in 4-4-4-4-3", "card 6222 0212 3456 7890 128 end", `card ${MASKED} end`],
    ["a leading year stays", "ref 2026 4111 1111 1111 1111", `ref 2026 ${MASKED}`],
    ["a 16-digit card and three digits mask whole", "num 4111 1111 1111 1111 123 end", `num ${MASKED} end`],
    ["four trailing digits stay", "num 4111 1111 1111 1111 1234 end", `num ${MASKED} 1234 end`],
    ["touching Chinese", "卡號4111111111111111。", `卡號${MASKED}。`],
    ["grouped, touching Chinese", "卡號4111 1111 1111 1111。", `卡號${MASKED}。`],
    ["four-six-five", "amex 3782 822463 10005 on file", `amex ${MASKED} on file`],
    ["four-six-four", "diners 3056-930902-5904 on file", `diners ${MASKED} on file`],
    ["an unbroken reference number passing Luhn", "ref 2026110700123451", `ref ${MASKED}`],
  ] as const;

  const UNTOUCHED_SAMPLES = [
    "2026110700123450",
    "2026 1107 0012 3450",
    "2026 1107 0012 3451",
    "1234 5678 9012 3452",
    // Known limit: a leading number with a card's prefix hides the card behind it.
    "ref 4026 4111 1111 1111 1111",
  ] as const;

  test.for(MASKED_SAMPLES)("mask on input: %s", async ([, sent, seen], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await chat("card-mask", sent);
    await res.text();
    expect(res.status).toBe(200);
    expect(lastPrompt()).toBe(seen);
    const row = await rowOf(res);
    expect(JSON.parse(row.get("redacted_entity_counts") ?? "{}")).toEqual({ bank_card: 1 });
  });

  test.for(UNTOUCHED_SAMPLES)("left alone by mask: %s", async (number, ctx) => {
    if (!etcdReachable) return ctx.skip();
    const sent = `order ${number} shipped`;
    const res = await chat("card-mask", sent);
    await res.text();
    expect(res.status).toBe(200);
    expect(lastPrompt()).toBe(sent);
    const row = await rowOf(res);
    expect(row.get("redacted_entity_counts")).toBeUndefined();
  });

  test.for(MASKED_SAMPLES)("block on input: %s", async ([, sent], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const before = upstreamCalls();
    const res = await chat("card-block", sent);
    const body = await res.text();
    expect(res.status).toBe(422);
    expect(body).not.toContain("1111");
    expect(upstreamCalls()).toBe(before);
    const row = await rowOf(res);
    expect(row.get("guardrail_blocked")).toBe("true");
    const hits = JSON.parse(row.get("guardrail_enforced_hits") ?? "[]") as Array<{ guardrail_name?: string }>;
    expect(hits.map((h) => h.guardrail_name)).toContain("card-block");
  });

  test.for(UNTOUCHED_SAMPLES)("left alone by block: %s", async (number, ctx) => {
    if (!etcdReachable) return ctx.skip();
    const before = upstreamCalls();
    const res = await chat("card-block", `order ${number} shipped`);
    await res.text();
    expect(res.status).toBe(200);
    expect(upstreamCalls()).toBe(before + 1);
  });

  test("mask on output, non-streaming", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await chat("card-mask-output", "what is on file?");
    const body = (await res.json()) as { choices: Array<{ message: { content: string } }> };
    expect(res.status).toBe(200);
    expect(body.choices[0].message.content).toBe(`the card on file is ${MASKED}.`);
  });

  test("block on output, non-streaming", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await chat("card-block-output", "what is on file?");
    const body = await res.text();
    expect(res.status).toBe(422);
    expect(body).not.toContain("0212");
    expect((await rowOf(res)).get("guardrail_blocked")).toBe("true");
  });

  const streamedText = (wire: string) =>
    wire
      .split("\n")
      .filter((l) => l.startsWith("data: {"))
      .map(
        (l) =>
          (JSON.parse(l.slice(6)) as { choices: Array<{ delta?: { content?: string } }> }).choices[0]
            ?.delta?.content ?? "",
      )
      .join("");

  test.for([
    ["4-4-4-4", "card-mask-split", "1111"],
    ["4-4-4-4-3", "card-mask-split-19", "0212"],
  ] as const)("mask on a stream that splits the card across chunks: %s", async ([, model, digits], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await chat(model, "what is on file?", true);
    const wire = await res.text();
    expect(res.status).toBe(200);
    expect(streamedText(wire)).toBe(`the card is ${MASKED} on file`);
    expect(wire).not.toContain(digits);
  });

  test.for([
    ["4-4-4-4", "card-block-split", "1111"],
    ["4-4-4-4-3", "card-block-split-19", "0212"],
  ] as const)("block on a stream that splits the card across chunks: %s", async ([, model, digits], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const res = await chat(model, "what is on file?", true);
    const wire = await res.text();
    expect(wire).not.toContain(digits);
    const row = await rowOf(res);
    expect(row.get("guardrail_blocked")).toBe("true");
    expect(row.get("error_class")).toBe("content_filter");
  });
});
