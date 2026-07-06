import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the `openai_moderation` guardrail (#52) moderates chat input
// against the OpenAI Moderation API (`POST /moderations`). We stand up a
// mock moderation endpoint that flags any text containing RISKY_MARKER
// (violence, score 0.97) and scores MILD_MARKER at 0.4 without flagging;
// the guardrail's `endpoint` override points at the mock. Covered:
// flagged → 422; enforcement_mode=monitor → 200; per-category threshold
// mode overriding the API's flagged boolean; fail-open on 5xx.
//
// References:
// - OpenAI Moderation <https://platform.openai.com/docs/guides/moderation>
// - LiteLLM `openai_moderation` (behavior baseline): block on the API's
//   `flagged` boolean. The threshold mode is our superset knob.

const CALLER = "sk-moderation-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const RISKY_MARKER = "moderationriskymarker";
const MILD_MARKER = "moderationmildmarker";
const ERROR_MARKER = "moderationfivehundredmarker";

interface ModerationMock {
  baseUrl: string;
  requests: Array<{ auth: string | undefined; model: string; input: string }>;
  close(): Promise<void>;
}

async function startModerationMock(): Promise<ModerationMock> {
  const requests: ModerationMock["requests"] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let input = "";
      let model = "";
      try {
        const body = JSON.parse(raw);
        input = typeof body.input === "string" ? body.input : "";
        model = typeof body.model === "string" ? body.model : "";
      } catch {
        // leave defaults
      }
      requests.push({ auth: req.headers.authorization, model, input });

      if (input.includes(ERROR_MARKER)) {
        res.statusCode = 500;
        res.end("mock moderation outage");
        return;
      }

      const risky = input.includes(RISKY_MARKER);
      const mild = input.includes(MILD_MARKER);
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "modr-mock",
          model: "omni-moderation-latest",
          results: [
            {
              flagged: risky,
              categories: { violence: risky, harassment: false },
              category_scores: {
                violence: risky ? 0.97 : mild ? 0.4 : 0.01,
                harassment: 0.01,
              },
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
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

describe("openai moderation guardrail e2e: flagged blocks, monitor allows, thresholds", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let moderation: ModerationMock | undefined;
  let admin: AdminClient | undefined;
  let guardrailId: string | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    moderation = await startModerationMock();

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
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "moderation-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "moderation-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["moderation-e2e"],
    });

    // One env-wide guardrail on the input hook, block mode. fail_open=true
    // so the 5xx case exercises the bypass path.
    const created = await admin!.json<{ id: string }>(
      "POST",
      "/admin/v1/guardrails",
      {
        name: "moderation-e2e-guard",
        enabled: true,
        hook_point: "input",
        fail_open: true,
        kind: "openai_moderation",
        api_key: "sk-moderation-key",
        endpoint: moderation.baseUrl,
      },
    );
    guardrailId = created.id;
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await moderation?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  const expect422 = async (content: string) => {
    let caught: unknown;
    try {
      await client().chat.completions.create({
        model: "moderation-e2e",
        messages: [{ role: "user", content }],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    return caught;
  };

  // Poll with a flagged probe until the block-mode guardrail is live, so
  // each block-mode test stands alone (no hidden dependency on execution
  // order). The monitor/threshold tests establish their own baseline with
  // a full PUT + their own propagation probe instead.
  const ensureGuardrailLive = () =>
    waitConfigPropagation(async () => {
      try {
        await client().chat.completions.create({
          model: "moderation-e2e",
          messages: [{ role: "user", content: `probe ${RISKY_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

  test("flagged content → 422 content_filter, upstream never called", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !moderation) {
      ctx.skip();
      return;
    }

    await ensureGuardrailLive();

    const upstreamBefore = upstream.receivedRequests.length;
    const err = await expect422(`please describe ${RISKY_MARKER} violence`);
    // The matched content must not leak back to the caller (#153).
    expect(JSON.stringify(err.error ?? {})).not.toContain(RISKY_MARKER);
    expect(upstream.receivedRequests.length).toBe(upstreamBefore);

    // The mock saw the guardrail's key and the default model.
    const guardReq = moderation.requests.at(-1);
    expect(guardReq?.auth).toBe("Bearer sk-moderation-key");
    expect(guardReq?.model).toBe("omni-moderation-latest");
  });

  test("clean prompt → 200 via upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const before = upstream.receivedRequests.length;
    const res = await client().chat.completions.create({
      model: "moderation-e2e",
      messages: [{ role: "user", content: "what is a safe topic" }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("moderation 5xx with fail_open=true → request passes (bypass)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const before = upstream.receivedRequests.length;
    const res = await client().chat.completions.create({
      model: "moderation-e2e",
      messages: [{ role: "user", content: `hello ${ERROR_MARKER}` }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("enforcement_mode=monitor → flagged content passes through", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !admin || !guardrailId) {
      ctx.skip();
      return;
    }
    await admin.json("PUT", `/admin/v1/guardrails/${guardrailId}`, {
      name: "moderation-e2e-guard",
      enabled: true,
      hook_point: "input",
      fail_open: true,
      enforcement_mode: "monitor",
      kind: "openai_moderation",
      api_key: "sk-moderation-key",
      endpoint: moderation!.baseUrl,
    });

    // Monitor mode observes without blocking: the risky probe flips to 200.
    await waitConfigPropagation(async () => {
      try {
        const r = await client().chat.completions.create({
          model: "moderation-e2e",
          messages: [{ role: "user", content: `probe ${RISKY_MARKER}` }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });
  });

  test("category threshold mode enforces the configured category only", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !admin || !guardrailId) {
      ctx.skip();
      return;
    }
    // violence>=0.3 blocks even though the mock does NOT set flagged for
    // MILD_MARKER (score 0.4) — the threshold overrides the API boolean.
    await admin.json("PUT", `/admin/v1/guardrails/${guardrailId}`, {
      name: "moderation-e2e-guard",
      enabled: true,
      hook_point: "input",
      fail_open: true,
      enforcement_mode: "block",
      kind: "openai_moderation",
      api_key: "sk-moderation-key",
      endpoint: moderation!.baseUrl,
      category_thresholds: { violence: 0.3 },
    });

    await waitConfigPropagation(async () => {
      try {
        await client().chat.completions.create({
          model: "moderation-e2e",
          messages: [{ role: "user", content: `probe ${MILD_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

    // Clean content stays under the threshold (violence 0.01) → 200.
    const res = await client().chat.completions.create({
      model: "moderation-e2e",
      messages: [{ role: "user", content: "a calm question" }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
  });
});
