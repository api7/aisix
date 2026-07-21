import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the per-execution guardrail latency histogram
// `aisix_guardrail_latency_seconds` (AISIX-Cloud#1076). Every guardrail
// consulted on a request must record one observation labelled with the
// row name, kind, phase (input/output), enforced result, and — for
// fail-open bypasses — the bounded error tag. Covered results: allowed,
// blocked (local keyword, remote moderation, output hook), bypassed
// (moderation 5xx + fail_open), and monitor-mode would_block. The series
// must render as a real bucketed histogram (`_bucket{le=…}`), not a
// summary, so operators can compute P50/P95/P99.

const CALLER = "sk-guardrail-metrics-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const KW_BLOCK_MARKER = "kwblockmarker";
const KW_MONITOR_MARKER = "kwmonitormarker";
const OUTPUT_LEAK_MARKER = "outputleakmarker";
const RISKY_MARKER = "moderationriskymarker";
const ERROR_MARKER = "moderationfivehundredmarker";

const MODEL_CLEAN = "guardrail-metrics-clean";
const MODEL_LEAKY = "guardrail-metrics-leaky";

interface ModerationMock {
  baseUrl: string;
  close(): Promise<void>;
}

/** Flags input containing RISKY_MARKER; 500s on ERROR_MARKER. */
async function startModerationMock(): Promise<ModerationMock> {
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let input = "";
      try {
        const body = JSON.parse(raw);
        input = typeof body.input === "string" ? body.input : "";
      } catch {
        // leave defaults
      }
      if (input.includes(ERROR_MARKER)) {
        res.statusCode = 500;
        res.end("mock moderation outage");
        return;
      }
      const risky = input.includes(RISKY_MARKER);
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "modr-mock",
          model: "omni-moderation-latest",
          results: [
            {
              flagged: risky,
              categories: { violence: risky },
              category_scores: { violence: risky ? 0.97 : 0.01 },
            },
          ],
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
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

function chatCompletionBody(id: string, content: string) {
  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
  };
}

/**
 * Sum the `aisix_guardrail_latency_seconds_count` values of every series
 * matching all given label pairs (labels appear in arbitrary order in the
 * exposition). 0 when no series matches yet.
 */
function guardrailCount(scrape: string, labels: Record<string, string>): number {
  let sum = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith("aisix_guardrail_latency_seconds_count{")) continue;
    if (
      !Object.entries(labels).every(([k, v]) => line.includes(`${k}="${v}"`))
    ) {
      continue;
    }
    const v = parseFloat(line.split("}").at(-1)?.trim() ?? "");
    if (!Number.isNaN(v)) sum += v;
  }
  return sum;
}

describe("guardrail latency metrics e2e: per-execution histogram", () => {
  let app: SpawnedApp | undefined;
  let cleanUpstream: OpenAiUpstream | undefined;
  let leakyUpstream: OpenAiUpstream | undefined;
  let moderation: ModerationMock | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    moderation = await startModerationMock();
    cleanUpstream = await startOpenAiUpstream({
      nonStreamBody: chatCompletionBody("cmpl-clean", "a safe and clean reply"),
    });
    leakyUpstream = await startOpenAiUpstream({
      nonStreamBody: chatCompletionBody(
        "cmpl-leaky",
        `a reply leaking ${OUTPUT_LEAK_MARKER} content`,
      ),
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    for (const [model, upstream] of [
      [MODEL_CLEAN, cleanUpstream],
      [MODEL_LEAKY, leakyUpstream],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: [MODEL_CLEAN, MODEL_LEAKY],
    });

    // Env-wide guardrails, distinct kinds/phases/modes. The blocking input
    // keyword row is written LAST: etcd watch events apply in order, so
    // once the propagation probe sees ITS 422, every row above is live too.
    await seed.createGuardrail({
      name: "metrics-moderation",
      enabled: true,
      hook_point: "input",
      fail_open: true,
      kind: "openai_moderation",
      api_key: "sk-moderation-key",
      endpoint: moderation.baseUrl,
    });
    await seed.createGuardrail({
      name: "metrics-kw-monitor",
      enabled: true,
      enforcement_mode: "monitor",
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: KW_MONITOR_MARKER }],
    });
    await seed.createGuardrail({
      name: "metrics-kw-output",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: OUTPUT_LEAK_MARKER }],
    });
    await seed.createGuardrail({
      name: "metrics-kw-input",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: KW_BLOCK_MARKER }],
    });

    await waitConfigPropagation(async () => {
      try {
        await client().chat.completions.create({
          model: MODEL_CLEAN,
          messages: [{ role: "user", content: `probe ${KW_BLOCK_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await cleanUpstream?.close();
    await leakyUpstream?.close();
    await moderation?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  const scrape = async (): Promise<string> => {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    return res.text();
  };

  const expect422 = async (model: string, content: string) => {
    let caught: unknown;
    try {
      await client().chat.completions.create({
        model,
        messages: [{ role: "user", content }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
  };

  test("clean request records result=allowed for every consulted guardrail, as a real bucketed histogram", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    const resp = await client().chat.completions.create({
      model: MODEL_CLEAN,
      messages: [{ role: "user", content: "a perfectly clean question" }],
    });
    expect(resp.choices[0]?.message?.content).toContain("clean reply");

    const after = await scrape();
    for (const guardrail of [
      "metrics-moderation",
      "metrics-kw-monitor",
      "metrics-kw-input",
    ]) {
      const labels = {
        guardrail,
        phase: "input",
        result: "allowed",
        error_type: "none",
      };
      expect(
        guardrailCount(after, labels),
        `${guardrail} input allowed`,
      ).toBeGreaterThan(guardrailCount(before, labels));
    }
    // The output-hook keyword row runs on the clean reply too.
    expect(
      guardrailCount(after, {
        guardrail: "metrics-kw-output",
        phase: "output",
        result: "allowed",
      }),
    ).toBeGreaterThan(0);

    // Kind label distinguishes local vs remote; env_id follows the SLO
    // histogram convention; `_bucket` + sub-millisecond edge prove the
    // series is a quantile-aggregatable histogram, not a summary.
    expect(after).toMatch(
      /aisix_guardrail_latency_seconds_count\{[^}]*guardrail="metrics-moderation"[^}]*kind="openai_moderation"/,
    );
    expect(after).toMatch(
      /aisix_guardrail_latency_seconds_count\{[^}]*guardrail="metrics-kw-input"[^}]*kind="keyword"/,
    );
    expect(after).toMatch(
      /aisix_guardrail_latency_seconds_bucket\{[^}]*le="0.001"/,
    );
    expect(after).toMatch(/aisix_guardrail_latency_seconds_bucket\{[^}]*env_id="/);
  });

  test("local keyword block records result=blocked on the input phase", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    await expect422(MODEL_CLEAN, `please echo ${KW_BLOCK_MARKER}`);
    const after = await scrape();
    const labels = {
      guardrail: "metrics-kw-input",
      kind: "keyword",
      phase: "input",
      result: "blocked",
    };
    expect(guardrailCount(after, labels)).toBeGreaterThan(
      guardrailCount(before, labels),
    );
  });

  test("remote moderation block records result=blocked for the remote kind", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    await expect422(MODEL_CLEAN, `please describe ${RISKY_MARKER}`);
    const after = await scrape();
    const labels = {
      guardrail: "metrics-moderation",
      kind: "openai_moderation",
      phase: "input",
      result: "blocked",
    };
    expect(guardrailCount(after, labels)).toBeGreaterThan(
      guardrailCount(before, labels),
    );
  });

  test("fail-open on provider 5xx records result=bypassed with the bounded error tag", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    const resp = await client().chat.completions.create({
      model: MODEL_CLEAN,
      messages: [{ role: "user", content: `outage probe ${ERROR_MARKER}` }],
    });
    expect(resp.choices[0]?.message?.content).toContain("clean reply");
    const after = await scrape();
    const labels = {
      guardrail: "metrics-moderation",
      result: "bypassed",
      error_type: "openai_moderation_5xx",
    };
    expect(guardrailCount(after, labels)).toBeGreaterThan(
      guardrailCount(before, labels),
    );
  });

  test("monitor-mode hit records result=would_block while the request succeeds", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    const resp = await client().chat.completions.create({
      model: MODEL_CLEAN,
      messages: [{ role: "user", content: `staged rule ${KW_MONITOR_MARKER}` }],
    });
    expect(resp.choices[0]?.message?.content).toContain("clean reply");
    const after = await scrape();
    const labels = {
      guardrail: "metrics-kw-monitor",
      result: "would_block",
    };
    expect(guardrailCount(after, labels)).toBeGreaterThan(
      guardrailCount(before, labels),
    );
  });

  test("output-hook block records result=blocked on the output phase", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const before = await scrape();
    await expect422(MODEL_LEAKY, "a clean question to a leaky model");
    const after = await scrape();
    const labels = {
      guardrail: "metrics-kw-output",
      kind: "keyword",
      phase: "output",
      result: "blocked",
    };
    expect(guardrailCount(after, labels)).toBeGreaterThan(
      guardrailCount(before, labels),
    );
  });
});
