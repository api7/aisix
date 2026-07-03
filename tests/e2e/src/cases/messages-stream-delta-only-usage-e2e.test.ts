import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: AISIX-Cloud#952 — streaming /v1/messages against a relay backend
// that ships NO usage block on message_start (id/model present) and
// reports cumulative input/cache counts only on the terminal
// message_delta. Pre-fix the DP emitted prompt_tokens=0 for these
// streams (cp-api stored NULL, the dashboard displayed 0 and the
// request billed as zero input).
//
// Mirrors the wire shape observed in the POC: message_start carries a
// `gen_*` id and the model but no usage; the final message_delta
// carries the full cumulative usage.

const CALLER_PLAINTEXT = "sk-anth-952-delta-usage-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const INPUT_TOKENS = 11504;
const OUTPUT_TOKENS = 136;
const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "gen_01E2E952",
      role: "assistant",
      content: [],
      model: "mco-5",
      stop_reason: null,
      // no usage block — the #952 relay-backend shape
    },
  }),
  JSON.stringify({
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  }),
  JSON.stringify({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text: "hello there" },
  }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { input_tokens: INPUT_TOKENS, output_tokens: OUTPUT_TOKENS },
  }),
  JSON.stringify({ type: "message_stop" }),
];

describe("anthropic /v1/messages stream with usage only on message_delta (#952)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      streamEvents: STREAM_EVENTS,
      eventDelayMs: 2,
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "anth-952-delta-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "anth-952-delta",
      provider: "anthropic",
      model_name: "mco-5",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["anth-952-delta"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("input tokens reported only on message_delta land in telemetry", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Config propagation: retry until routed.
    const deadline = Date.now() + 10_000;
    let res: Response | undefined;
    while (Date.now() < deadline) {
      res = await fetch(`${app.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": CALLER_PLAINTEXT,
        },
        body: JSON.stringify({
          model: "anth-952-delta",
          max_tokens: 200,
          stream: true,
          messages: [{ role: "user", content: "repro 952" }],
        }),
      });
      if (res.status === 200) break;
      await new Promise((r) => setTimeout(r, 200));
    }
    expect(res!.status).toBe(200);

    // Bytes still pass through verbatim.
    const body = await res!.text();
    expect(body).toContain("gen_01E2E952");
    expect(body).toContain("message_stop");

    // Poll the DP's own /metrics until the token counters appear —
    // pre-fix input stays 0 while output records, so assert both.
    const scrapeDeadline = Date.now() + 5_000;
    let inTokens = 0;
    let outTokens = 0;
    while (Date.now() < scrapeDeadline) {
      const scrape = await fetch(`${app.metricsUrl}/metrics`).then((r) =>
        r.text(),
      );
      inTokens = sumMetric(scrape, "aisix_llm_input_tokens_total", "/v1/messages");
      outTokens = sumMetric(
        scrape,
        "aisix_llm_output_tokens_total",
        "/v1/messages",
      );
      if (inTokens > 0 && outTokens > 0) break;
      await new Promise((r) => setTimeout(r, 100));
    }

    expect(
      inTokens,
      "input_tokens reported only on message_delta must be harvested (#952)",
    ).toBeGreaterThanOrEqual(INPUT_TOKENS);
    expect(outTokens).toBeGreaterThanOrEqual(OUTPUT_TOKENS);
  });
});

function sumMetric(scrape: string, metric: string, endpoint: string): number {
  let total = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith(`${metric}{`)) continue;
    if (!line.includes(`endpoint="${endpoint}"`)) continue;
    const valueStr = line.split("}").at(-1)?.trim() ?? "";
    const v = Number.parseFloat(valueStr);
    if (!Number.isNaN(v)) total += v;
  }
  return total;
}
