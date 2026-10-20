import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a guardrail refusal's gateway log line names the guardrail that
// refused, under `guardrail_name`, on the request and the response side and
// across inbound protocols — not only inside the free-text reason.

const CALLER = "sk-guardrail-block-log-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const FORBIDDEN = "blocklogforbidden";

describe("guardrail block log lines name the guardrail", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let proxy: ProxyClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-block-log",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `here is ${FORBIDDEN}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "block-log-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const model = await seed.createModel({
      display_name: "block-log-chat",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const keyword = async (name: string, hook: string) => {
      const g = await seed.createGuardrail(
        {
          name,
          enabled: true,
          hook_point: hook,
          kind: "keyword",
          patterns: [{ kind: "literal", value: FORBIDDEN }],
        },
        { attach: false },
      );
      await seed.attachGuardrailToModel(g.id, model.id);
    };
    await keyword("block-log-input-guard", "input");
    await keyword("block-log-output-guard", "output");

    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["block-log-chat"] });
    proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy!.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const blockLine = (message: string, guard: string) =>
    waitForLogLine(
      app!,
      (l) => l.includes(message) && l.includes(`guardrail_name=${guard}`),
      `the ${guard} block line`,
    );

  test("chat: an input refusal names the input guardrail", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await proxy!.chat({
      model: "block-log-chat",
      messages: [{ role: "user", content: `say ${FORBIDDEN}` }],
    });
    expect(res.status).toBe(422);
    const line = await blockLine("guardrail blocked request", "block-log-input-guard");
    expect(line).toContain(`guardrail_hook="input"`);
  });

  test("chat: an output refusal names the output guardrail", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await proxy!.chat({
      model: "block-log-chat",
      messages: [{ role: "user", content: "hello" }],
    });
    expect(res.status).toBe(422);
    const line = await blockLine("guardrail blocked response", "block-log-output-guard");
    expect(line).toContain(`guardrail_hook="output"`);
  });

  test("messages: an output refusal names the output guardrail", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": CALLER },
      body: JSON.stringify({
        model: "block-log-chat",
        max_tokens: 64,
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    expect(res.status).toBe(422);
    await blockLine("guardrail blocked /v1/messages response", "block-log-output-guard");
  });
});
