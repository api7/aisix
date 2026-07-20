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

// E2E for AISIX-Cloud#1045: client_type coverage of common AI coding
// clients + operator-defined `observability.metrics.client_type_rules`.
//
// Pinned here:
//   1. Real User-Agents of newly supported built-in clients produce their
//      own stable client_type series on /metrics (acceptance: ≥2 new
//      clients exercised end-to-end).
//   2. Operator rules classify an in-house UA the built-ins would bucket
//      as "other" — and outrank a generic built-in match (an axios-embedded
//      UA that would land in "node").
//   3. Unmatched UAs still fall back to the built-in table ("other"), so
//      custom rules never change baseline behaviour for other traffic.

const CALLER = "sk-1045-client-type-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const MODEL = "ct1045-chat";
const USAGE = { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 };

/** Real captured/source-verified UAs of newly supported built-in clients. */
const BUILTIN_CASES: Array<{ ua: string; clientType: string }> = [
  { ua: "RooCode/3.54.0", clientType: "roo-code" },
  { ua: "Kilo-Code/5.16.2", clientType: "kilocode" },
  { ua: "ZooCode/3.71.100268", clientType: "zoo-code" },
  { ua: "GitHubCopilotChat/0.44.0", clientType: "github-copilot" },
  // Also pins product-before-SDK precedence: this UA embeds the
  // `ai-sdk/provider-utils` needle of the vercel-ai-sdk bucket.
  {
    ua: "opencode/1.18.3 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
    clientType: "opencode",
  },
];

const CUSTOM_RULES = [
  { pattern: "^internal-agent/", client: "internal-agent" },
  // Deliberately overlaps the built-in `axios` → "node" bucket to pin
  // custom-before-builtin precedence.
  { pattern: "billing-batcher", client: "billing-batcher" },
];

function seriesValue(
  text: string,
  clientType: string,
  tokenType: string,
): number | undefined {
  for (const line of text.split("\n")) {
    if (
      line.startsWith("aisix_llm_tokens_by_client_total{") &&
      line.includes(`client_type="${clientType}"`) &&
      line.includes(`model="${MODEL}"`) &&
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

/** Poll the scrape until `probe` passes (metric emits race the scrape). */
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

async function chatWithUa(app: SpawnedApp, ua: string): Promise<void> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER}`,
      "content-type": "application/json",
      "user-agent": ua,
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  expect(res.status).toBe(200);
}

describe("client_type built-ins + operator rules (AISIX-Cloud#1045)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-1045",
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
        usage: USAGE,
      },
    });

    app = await spawnApp({ clientTypeRules: CUSTOM_RULES });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "ct1045-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [MODEL],
    });

    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((d) => d.id === MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("new built-in clients: real UAs produce their own client_type series", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    for (const { ua } of BUILTIN_CASES) {
      await chatWithUa(app, ua);
    }
    const text = await pollSeries(app, (t) =>
      BUILTIN_CASES.every(
        (c) => seriesValue(t, c.clientType, "total") !== undefined,
      ),
    );
    for (const { clientType } of BUILTIN_CASES) {
      expect(seriesValue(text, clientType, "input")).toBe(USAGE.prompt_tokens);
      expect(seriesValue(text, clientType, "output")).toBe(
        USAGE.completion_tokens,
      );
      expect(seriesValue(text, clientType, "total")).toBe(USAGE.total_tokens);
    }
  });

  test("operator rules classify in-house UAs and outrank generic built-ins", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Would be "other" without the rule.
    await chatWithUa(app, "internal-agent/2.1");
    // Embeds axios — the built-in table alone would bucket it as "node".
    await chatWithUa(app, "billing-batcher/3.0 axios/1.6.0");

    const text = await pollSeries(
      app,
      (t) =>
        seriesValue(t, "internal-agent", "total") !== undefined &&
        seriesValue(t, "billing-batcher", "total") !== undefined,
    );
    expect(seriesValue(text, "internal-agent", "total")).toBe(
      USAGE.total_tokens,
    );
    expect(seriesValue(text, "billing-batcher", "total")).toBe(
      USAGE.total_tokens,
    );
    // The raw UA never becomes a label value.
    expect(text).not.toContain('client_type="internal-agent/2.1"');
  });

  test("unmatched UAs keep built-in behaviour (other/unknown intact)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await chatWithUa(app, "SomeRandomBespokeClient/9.9");
    const text = await pollSeries(
      app,
      (t) => seriesValue(t, "other", "total") !== undefined,
    );
    expect(seriesValue(text, "other", "total")).toBe(USAGE.total_tokens);
  });
});
