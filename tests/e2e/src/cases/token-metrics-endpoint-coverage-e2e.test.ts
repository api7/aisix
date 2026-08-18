import { createHash } from "node:crypto";
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

// The follow-up half of AISIX-Cloud#1234. Where that fixed the request
// counters, the token families had the same gap in three different shapes:
//
//   aisix_llm_{input,output,total}_tokens_total  chat + messages
//   aisix_llm_spend_micro_usd_total              chat + messages
//   aisix_llm_tokens_by_client_total             chat + messages + responses
//   aisix_tokens_consumed_total (legacy)         chat ALONE
//
// So a gateway that billed a customer for /v1/embeddings reported none of
// those tokens. These specs pin the per-endpoint coverage.

const CALLER_PLAINTEXT = "sk-token-metrics-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const RESP_MODEL = "tokmetrics-resp";
const EMBED_MODEL = "tokmetrics-embed";

describe("token metrics endpoint coverage e2e", () => {
  let app: SpawnedApp | undefined;
  let respUpstream: OpenAiUpstream | undefined;
  let embedUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    respUpstream = await startOpenAiUpstream({ nonStreamBody: responsesBody() });
    embedUpstream = await startOpenAiUpstream({ nonStreamBody: embeddingBody() });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    for (const [name, up] of [
      [RESP_MODEL, respUpstream],
      [EMBED_MODEL, embedUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${up.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [RESP_MODEL, EMBED_MODEL],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await respUpstream?.close();
    await embedUpstream?.close();
  });

  test("/v1/responses reports its tokens on every token family", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const r = await post(app!, "/v1/responses", {
        model: RESP_MODEL,
        input: "ready",
      });
      return r.status === 200;
    });

    // The propagation probe above also billed tokens, so measure the DELTA
    // one request adds rather than the absolute counter.
    const before = await scrape(app);

    expect(
      (await post(app, "/v1/responses", { model: RESP_MODEL, input: "hi" }))
        .status,
    ).toBe(200);

    const text = await scrape(app);

    // The per-key families — absent for this endpoint before the fix.
    for (const metric of [
      "aisix_llm_input_tokens_total",
      "aisix_llm_output_tokens_total",
      "aisix_llm_total_tokens_total",
    ]) {
      const lines = seriesFor(text, metric, "/v1/responses");
      expect(lines, `${metric} missing for /v1/responses`).not.toHaveLength(0);
      expect(lines.join("\n")).toContain(`model="${RESP_MODEL}"`);
      expect(lines.join("\n")).toContain('provider="openai"');
    }

    // The upstream reported 11 in / 13 out; the counters must carry the real
    // values, not merely exist.
    expect(delta(before, text, "aisix_llm_input_tokens_total", "/v1/responses")).toBe(11);
    expect(delta(before, text, "aisix_llm_output_tokens_total", "/v1/responses")).toBe(13);
    expect(delta(before, text, "aisix_llm_total_tokens_total", "/v1/responses")).toBe(24);

    // The legacy series, which was chat-only.
    expect(
      text
        .split("\n")
        .filter(
          (l) =>
            l.startsWith("aisix_tokens_consumed_total{") &&
            l.includes(`model="${RESP_MODEL}"`),
        ),
    ).not.toHaveLength(0);
  }, 30_000);

  test("/v1/embeddings reports its tokens too", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const r = await post(app!, "/v1/embeddings", {
        model: EMBED_MODEL,
        input: "ready",
      });
      return r.status === 200;
    });

    expect(
      (await post(app, "/v1/embeddings", { model: EMBED_MODEL, input: "hi" }))
        .status,
    ).toBe(200);

    const text = await scrape(app);

    const input = seriesFor(text, "aisix_llm_input_tokens_total", "/v1/embeddings");
    expect(input, "embeddings reported no input tokens").not.toHaveLength(0);
    expect(input.join("\n")).toContain(`model="${EMBED_MODEL}"`);

    // Embeddings produce no completion tokens, so that family must stay
    // absent rather than report a zero series.
    expect(
      seriesFor(text, "aisix_llm_output_tokens_total", "/v1/embeddings"),
    ).toHaveLength(0);

    // The by-client family reaches this endpoint now as well.
    expect(
      text
        .split("\n")
        .filter(
          (l) =>
            l.startsWith("aisix_llm_tokens_by_client_total{") &&
            l.includes(`model="${EMBED_MODEL}"`),
        ),
    ).not.toHaveLength(0);
  }, 30_000);

  test("token labels line up with the request families", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const text = await scrape(app);
    // Both families are keyed on the same Caller/Upstream pair, so a join on
    // endpoint+model+provider+api_key_id has to match — that is the whole
    // point of sharing the label structs.
    const req = seriesFor(text, "aisix_llm_requests_total", "/v1/responses")[0];
    const tok = seriesFor(text, "aisix_llm_input_tokens_total", "/v1/responses")[0];
    expect(req).toBeDefined();
    expect(tok).toBeDefined();
    for (const label of ["provider", "model", "upstream_model", "api_key_id", "team_id", "user_id"]) {
      expect(labelOf(tok, label), `${label} differs between the families`).toBe(
        labelOf(req, label),
      );
    }
  }, 30_000);
});

async function post(
  app: SpawnedApp,
  path: string,
  body: unknown,
): Promise<{ status: number }> {
  const res = await fetch(`${app.proxyUrl}${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
  await res.text();
  return { status: res.status };
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

function seriesFor(
  scrapeText: string,
  metric: string,
  endpoint: string,
): string[] {
  return scrapeText
    .split("\n")
    .filter(
      (line) =>
        line.startsWith(`${metric}{`) &&
        line.includes(`endpoint="${endpoint}"`),
    );
}

function valueOf(line: string | undefined): number {
  if (!line) return 0;
  return Number(line.split("}").at(-1)?.trim());
}

/** How much one request added to `metric` on `endpoint`. */
function delta(
  before: string,
  after: string,
  metric: string,
  endpoint: string,
): number {
  return (
    valueOf(seriesFor(after, metric, endpoint)[0]) -
    valueOf(seriesFor(before, metric, endpoint)[0])
  );
}

function labelOf(line: string, label: string): string | undefined {
  return new RegExp(`${label}="([^"]*)"`).exec(line)?.[1];
}

function responsesBody() {
  return {
    id: "resp_tokmetrics",
    object: "response",
    created_at: 0,
    status: "completed",
    model: "gpt-4o-mini",
    output: [
      {
        id: "msg_tokmetrics",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "hello" }],
      },
    ],
    usage: { input_tokens: 11, output_tokens: 13, total_tokens: 24 },
  };
}

function embeddingBody() {
  return {
    object: "list",
    model: "gpt-4o-mini",
    data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
    usage: { prompt_tokens: 7, total_tokens: 7 },
  };
}
