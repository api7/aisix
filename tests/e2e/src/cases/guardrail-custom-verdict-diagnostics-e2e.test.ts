import { createHash, randomUUID } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: a kind=custom script that MISBEHAVES must not be mistakable for a
// content policy that is WORKING.
//
// The reported defect: a script returning `{action: "allow"}` — the obvious
// word for "let it through", but not one of the accepted actions — fell into
// the unknown-action arm, which is a script FAILURE, and the release's
// fail-closed default turned every failure into a block. The caller got
// `422 request blocked by content policy (guardrail '<name>')`: byte for byte
// the response a correctly-firing policy produces. So the operator's whole
// traffic was refused, and nothing the caller, the logs, or a dashboard
// showed said "your script is broken" rather than "your policy is busy".
//
// Fixed on four surfaces, all asserted here against a real aisix binary
// running real scripts:
//   1. `allow` is an accepted synonym of `none` — the reported script works.
//   2. A script fault still refuses (fail-closed is right: a hook that did
//      not produce a verdict has not screened anything), but the caller's
//      message says the guardrail could not evaluate the request, and the
//      envelope carries `error.code = "guardrail_unavailable"` — while a
//      genuine content block keeps its old message and carries no code.
//   3. The gateway logs name the offending action AND the accepted
//      vocabulary, so the fix is readable off one log line.
//   4. The latency histogram separates the failure modes by `error_type`
//      (`custom_unknown_action` / `custom_no_verdict`), so a dashboard
//      stops counting a broken script as policy volume.

const CALLER = "sk-custom-verdict-diag-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const BLOCK_MARKER = "customverdictblockmarker";

/** model name → the script its guardrail runs. */
const CASES = {
  "cvd-allow": `
export function checkInput() {
  return { action: "allow" };
}`,
  "cvd-none": `
export function checkInput(ctx) {
  if (ctx.text.includes("${BLOCK_MARKER}")) {
    return { action: "block", reason_code: "R-1" };
  }
  return { action: "none" };
}`,
  "cvd-unknown": `
export function checkInput() {
  return { action: "permit" };
}`,
  "cvd-noreturn": `
export function checkInput() {
  // falls off the end — decides nothing
}`,
  "cvd-empty": `
export function checkInput() {
  return {};
}`,
} as const;

type CaseModel = keyof typeof CASES;

function guardrailCount(scrape: string, labels: Record<string, string>): number {
  let sum = 0;
  for (const line of scrape.split("\n")) {
    if (!line.startsWith("aisix_guardrail_latency_seconds_count{")) continue;
    if (!Object.entries(labels).every(([k, v]) => line.includes(`${k}="${v}"`))) {
      continue;
    }
    const v = parseFloat(line.split("}").at(-1)?.trim() ?? "");
    if (!Number.isNaN(v)) sum += v;
  }
  return sum;
}

describe("custom guardrail e2e: a broken script is distinguishable from an enforcing policy", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-clean",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a safe and clean reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "cvd-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    // One model per script, each with its own guardrail attached by model
    // scope — the scripts must not shadow each other, and a model scope is
    // the guardrail's only scope.
    for (const [model, script] of Object.entries(CASES)) {
      const m = await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      const guardrail = await seed.createGuardrail({
        name: `${model}-guard`,
        enabled: true,
        hook_point: "input",
        // The release default, and the setting under which the defect
        // showed up: a row that refuses what it could not check.
        fail_open: false,
        kind: "custom",
        script,
        timeout_ms: 5000,
      }, { attach: false });
      await etcd.put(
        `${app.etcdPrefix}/guardrail_attachments/${randomUUID()}`,
        JSON.stringify({
          guardrail_id: guardrail.id,
          env_id: randomUUID(),
          scope_type: "model",
          scope_id: m.id,
          priority: 0,
          enabled: true,
        }),
      );
    }

    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: Object.keys(CASES),
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  const send = (model: CaseModel, content: string) =>
    client().chat.completions.create({
      model,
      messages: [{ role: "user", content }],
    });

  /** Send and return the error body of the expected 422. */
  const expect422 = async (model: CaseModel, content: string) => {
    let caught: unknown;
    try {
      await send(model, content);
    } catch (e) {
      caught = e;
    }
    expect(caught, `${model} should have been refused`).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    return caught.error as { message?: string; type?: string; code?: string };
  };

  const scrape = async () => {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    return res.text();
  };

  // Gate on the whole seed being live: the enforcing row must actually
  // refuse its marker before any case can be read as a real answer.
  const ensureSeedLive = () =>
    waitConfigPropagation(async () => {
      try {
        await send("cvd-none", `propagation probe ${BLOCK_MARKER}`);
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

  test("`allow` is accepted as a synonym of `none` and the request goes through", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const before = upstream!.receivedRequests.length;
    const reply = await send("cvd-allow", "what is the capital of France");
    expect(reply.choices[0]?.message?.content).toBe("a safe and clean reply");
    expect(upstream!.receivedRequests.length).toBe(before + 1);
  });

  test("a real policy block keeps the content-policy message and carries no error code", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const err = await expect422("cvd-none", `please help with ${BLOCK_MARKER}`);
    expect(err.type).toBe("content_filter");
    expect(err.message).toBe(
      "request blocked by content policy (guardrail 'cvd-none-guard')",
    );
    expect(err.code).toBeUndefined();
  });

  test("an unknown action refuses, but says the guardrail could not evaluate the request", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const before = upstream!.receivedRequests.length;
    const err = await expect422("cvd-unknown", "a perfectly ordinary question");
    // Still fail-closed: unscreened content must not reach the upstream.
    expect(upstream!.receivedRequests.length).toBe(before);
    // ...but the caller is told this was not a content decision.
    expect(err.message).not.toContain("blocked by content policy");
    expect(err.message).toContain("could not evaluate");
    expect(err.message).toContain("cvd-unknown-guard");
    expect(err.message).toContain("custom_unknown_action");
    expect(err.code).toBe("guardrail_unavailable");
  });

  test("a hook that returns nothing is reported as its own failure mode, not as a typo'd action", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    for (const model of ["cvd-noreturn", "cvd-empty"] as const) {
      const err = await expect422(model, "a perfectly ordinary question");
      expect(err.message, model).toContain("could not evaluate");
      expect(err.message, model).toContain("custom_no_verdict");
      expect(err.code, model).toBe("guardrail_unavailable");
    }
  });

  test("the logs name the offending action and the accepted vocabulary", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    await expect422("cvd-unknown", "a perfectly ordinary question");
    const log = app!.output();
    expect(log).toContain("custom guardrail returned an unknown action");
    expect(log).toContain("none | allow | block | mask");
    expect(log).toContain("permit");
  });

  test("the latency histogram separates the two script faults from a policy block", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const before = await scrape();
    await expect422("cvd-unknown", "a perfectly ordinary question");
    await expect422("cvd-noreturn", "a perfectly ordinary question");
    await expect422("cvd-none", `please help with ${BLOCK_MARKER}`);
    const after = await scrape();

    for (const [guardrail, errorType] of [
      ["cvd-unknown-guard", "custom_unknown_action"],
      ["cvd-noreturn-guard", "custom_no_verdict"],
      // A content decision is tagged `none` — the label that separates
      // "your script is broken" from "your policy fired".
      ["cvd-none-guard", "none"],
    ] as const) {
      const labels = { guardrail, result: "blocked", error_type: errorType };
      expect(
        guardrailCount(after, labels),
        `${guardrail} error_type=${errorType}`,
      ).toBeGreaterThan(guardrailCount(before, labels));
    }
  });
});
