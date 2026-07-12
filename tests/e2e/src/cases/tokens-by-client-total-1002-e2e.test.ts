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

// E2E for AISIX-Cloud#1002: the aisix_llm_tokens_by_client_total series gained
// a token_type="total" slice that is CACHE-INCLUSIVE. Anthropic reports
// cache_creation_input_tokens / cache_read_input_tokens as counters SEPARATE
// from input_tokens, so the pre-existing input+output slices undercount cached
// traffic. The new total slice = input + output + cache, matching the
// aisix_llm_total_tokens_total total (#679) and the CP display total (#906).
//
// The native /v1/messages path was the concrete gap: it committed the
// cache-inclusive total to TPM (#995) and reported it on aisix_llm_total_tokens
// yet still fed only prompt+completion to the by-client metric.

const CALLER = "sk-1002-msg-caller";
const MODEL = "msg-client-total";
// A recognised SDK User-Agent so the DP normalises it to a bounded client_type
// label (claude-cli/* -> "claude-code").
const USER_AGENT = "claude-cli/1.2.3";
const CLIENT_TYPE = "claude-code";

// input+output = 4; with cache = 4 + 5 + 3 = 12. The "total" slice must read
// 12 (not 4), proving the two separate cache counters are folded in.
const USAGE = {
  input_tokens: 2,
  output_tokens: 2,
  cache_creation_input_tokens: 5,
  cache_read_input_tokens: 3,
};
const INPUT = USAGE.input_tokens;
const OUTPUT = USAGE.output_tokens;
const TOTAL =
  USAGE.input_tokens +
  USAGE.output_tokens +
  USAGE.cache_creation_input_tokens +
  USAGE.cache_read_input_tokens;

const hash = (s: string) => createHash("sha256").update(s).digest("hex");

function anthropicMessageBody(usage: Record<string, number>) {
  return {
    id: "msg_1002",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: "hello from cache" }],
    model: "claude-3-5-haiku-20241022",
    stop_reason: "end_turn",
    usage,
  };
}

/** Value of the by-client series for a given token_type, or undefined. */
function seriesValue(text: string, tokenType: string): number | undefined {
  for (const line of text.split("\n")) {
    if (
      line.startsWith("aisix_llm_tokens_by_client_total{") &&
      line.includes(`client_type="${CLIENT_TYPE}"`) &&
      line.includes(`token_type="${tokenType}"`)
    ) {
      return Number(line.trim().split(/\s+/).pop());
    }
  }
  return undefined;
}

describe("aisix_llm_tokens_by_client_total token_type=total is cache-inclusive (AISIX-Cloud#1002)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: anthropicMessageBody(USAGE),
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: `${MODEL}-pk`,
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant-mock",
      api_base: upstream.baseUrl,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: [MODEL],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("native /v1/messages emits input/output/total, with total folding in cache tokens", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // listModels leaves the token budget intact and confirms propagation.
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === MODEL);
    });

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER,
        "user-agent": USER_AGENT,
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 200,
        messages: [{ role: "user", content: "count my cache tokens" }],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { usage?: Record<string, number> };
    expect(body.usage?.cache_read_input_tokens).toBe(USAGE.cache_read_input_tokens);

    // The metric is recorded on the usage-emit path; poll briefly in case the
    // scrape races the record.
    let total: number | undefined;
    let input: number | undefined;
    let output: number | undefined;
    for (let i = 0; i < 60; i++) {
      const text = await scrape(app);
      total = seriesValue(text, "total");
      input = seriesValue(text, "input");
      output = seriesValue(text, "output");
      if (total !== undefined && input !== undefined && output !== undefined) {
        break;
      }
      await new Promise((r) => setTimeout(r, 50));
    }

    expect(input).toBe(INPUT);
    expect(output).toBe(OUTPUT);
    // The heart of #1002: total includes the two cache counters, so it exceeds
    // input+output (12 vs 4). Pre-fix there was no token_type="total" series.
    expect(total).toBe(TOTAL);
    expect(total).toBeGreaterThan((input ?? 0) + (output ?? 0));
  });
});

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}
