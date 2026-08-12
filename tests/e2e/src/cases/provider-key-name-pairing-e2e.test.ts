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

// Issue #941 moved `provider_key_name` resolution out of the metric
// emitters and into a caller-resolved `ResolvedPk`. The contract that has
// to survive that is per-SERIES: the readable name must ride on the same
// sample as the id it was read off, on EVERY metric family and EVERY
// endpoint — not merely appear somewhere in the scrape.
//
// A whole-scrape `toContain('provider_key_name="…"')` cannot see the
// difference: one correct chat emit satisfies it while `/v1/messages`,
// `/v1/embeddings` or the streaming TTFT series report `"unknown"`. That
// is the shape the repo's lockstep rule exists to catch, so this spec
// drives one endpoint from each family that reaches a different emitter.
const CALLER_PLAINTEXT = "sk-pk-pairing-941";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// The mock upstream decides SSE-vs-JSON from how it was constructed, not
// from the request, so the streaming path needs its own upstream — and
// therefore its own ProviderKey, whose pair is asserted separately.
const PK_NAME = "pairing-941-pk";
const PK_NAME_STREAM = "pairing-941-pk-stream";
// …and embeddings needs an embeddings-shaped reply, which the same canned
// mock cannot also serve.
const PK_NAME_EMBED = "pairing-941-pk-embed";
const MODEL = "pairing941-model";
const STREAM_MODEL = "pairing941-stream";
const EMBED_MODEL = "pairing941-embed";

describe("provider_key_name pairs with provider_key_id on every series (#941)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let embedUpstream: OpenAiUpstream | undefined;
  let pkId = "";
  let streamPkId = "";
  let embedPkId = "";
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-941",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hi" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      },
    });

    streamUpstream = await startOpenAiUpstream({
      // A small inter-event delay keeps the stream genuinely incremental so
      // TTFT is recorded (> 0) and its series exists to assert on.
      eventDelayMs: 2,
      streamEvents: [
        JSON.stringify({
          id: "chatcmpl-941",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [
            { index: 0, delta: { content: "hi" }, finish_reason: null },
          ],
        }),
        JSON.stringify({
          id: "chatcmpl-941",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
        }),
        "[DONE]",
      ],
    });

    embedUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        model: "gpt-4o-mini",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
        usage: { prompt_tokens: 3, total_tokens: 3 },
      },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: PK_NAME,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    pkId = pk.id;
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });

    const streamPk = await seed.createProviderKey({
      display_name: PK_NAME_STREAM,
      secret: "sk-mock",
      api_base: `${streamUpstream.baseUrl}/v1`,
    });
    streamPkId = streamPk.id;
    await seed.createModel({
      display_name: STREAM_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: streamPk.id,
    });

    const embedPk = await seed.createProviderKey({
      display_name: PK_NAME_EMBED,
      secret: "sk-mock",
      api_base: `${embedUpstream.baseUrl}/v1`,
    });
    embedPkId = embedPk.id;
    await seed.createModel({
      display_name: EMBED_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: embedPk.id,
    });

    // Seeded last so authenticating with it implies the whole set landed.
    await seed.createApiKey({
      display_name: "pairing-941-caller",
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await streamUpstream?.close();
    await embedUpstream?.close();
  });

  test("the id and the readable name ride the same sample, family by family", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.listModels();
      return probe.status === 200;
    });

    // One request per emitter family. Chat non-streaming and chat
    // streaming take different code paths to `record_usage`; `/v1/messages`
    // and `/v1/embeddings` reach their own emitters, and embeddings is a
    // single-attempt endpoint (no `begin_in_flight`), which is where the
    // last regression in this family hid.
    const chat = await post(app, "/v1/chat/completions", {
      model: MODEL,
      messages: [{ role: "user", content: "ping" }],
    });
    expect(chat.status).toBe(200);

    const stream = await post(app, "/v1/chat/completions", {
      model: STREAM_MODEL,
      messages: [{ role: "user", content: "ping" }],
      stream: true,
    });
    expect(stream.status).toBe(200);
    await stream.text(); // drain: the emit fires at end-of-stream

    const messages = await post(app, "/v1/messages", {
      model: MODEL,
      max_tokens: 16,
      messages: [{ role: "user", content: "ping" }],
    });
    expect(messages.status).toBe(200);

    const embeddings = await post(app, "/v1/embeddings", {
      model: EMBED_MODEL,
      input: "ping",
    });
    expect(embeddings.status).toBe(200);

    // Async emits (the usage channel, the stream's on_complete) settle
    // after the response returns.
    await new Promise((r) => setTimeout(r, 1_500));
    const text = await scrape(app);

    // Every family below is fed by a DIFFERENT emitter. Asserting the pair
    // within one series is the point: `{provider_key_id="X", …,
    // provider_key_name="unknown"}` fails here and passes a whole-scrape
    // substring check.
    for (const metric of [
      "aisix_llm_requests_total",
      "aisix_proxy_requests_total",
      "aisix_llm_input_tokens_total",
    ]) {
      expect(
        pairedSeries(text, metric, pkId, PK_NAME),
        `${metric} carries no sample pairing provider_key_id=${pkId} with provider_key_name=${PK_NAME}`,
      ).not.toHaveLength(0);
    }

    // The streaming TTFT label is the one #941 moved from a request-start
    // resolution to the stream-end one, so it gets its own assertion.
    expect(
      pairedSeries(
        text,
        "aisix_llm_time_to_first_token_seconds_count",
        streamPkId,
        PK_NAME_STREAM,
      ),
      "the streaming TTFT series carries no id/name pair",
    ).not.toHaveLength(0);

    // …and per endpoint, so a family that is correct on chat but broken on
    // the Anthropic-protocol or single-attempt surfaces is caught.
    for (const [endpoint, id, name] of [
      ["/v1/chat/completions", pkId, PK_NAME],
      ["/v1/messages", pkId, PK_NAME],
      ["/v1/embeddings", embedPkId, PK_NAME_EMBED],
    ] as const) {
      const paired = pairedSeries(
        text,
        "aisix_proxy_requests_total",
        id,
        name,
      ).filter((line) => line.includes(`endpoint="${endpoint}"`));
      expect(
        paired,
        `${endpoint} reports no request sample with the id/name pair`,
      ).not.toHaveLength(0);
    }

    // The pair is 1:1 — a resolved id must never be reported next to the
    // unresolved-name sentinel anywhere in the scrape.
    const mismatched = text
      .split("\n")
      .filter(
        (line) =>
          (line.includes(`provider_key_id="${pkId}"`) ||
            line.includes(`provider_key_id="${streamPkId}"`) ||
            line.includes(`provider_key_id="${embedPkId}"`)) &&
          line.includes('provider_key_name="unknown"'),
      );
    expect(mismatched, "a resolved key reported an unknown name").toEqual([]);
  }, 60_000);

  function pairedSeries(
    text: string,
    metric: string,
    id: string,
    name: string,
  ): string[] {
    return text
      .split("\n")
      .filter(
        (line) =>
          line.startsWith(`${metric}{`) &&
          line.includes(`provider_key_id="${id}"`) &&
          line.includes(`provider_key_name="${name}"`),
      );
  }
});

async function post(
  app: SpawnedApp,
  path: string,
  body: unknown,
): Promise<Response> {
  return fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}
