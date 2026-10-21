import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a chat completion relays the upstream's usage breakdown as reported.
//
// An OpenAI upstream reports every `*_tokens_details` counter it knows,
// zeros included — a prompt-cache hit beside `audio_tokens: 0`, a
// Predicted Outputs count beside `rejected_prediction_tokens: 0`. The caller
// gets each counter the upstream reported, with its value, and none it did
// not. A stream's `include_usage` frame carries the usage with an empty
// `choices` array, as the upstream sends it.

const CALLER = "sk-chat-usage-details-e2e";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const REPORTED_USAGE = {
  prompt_tokens: 1300,
  completion_tokens: 50,
  total_tokens: 1350,
  prompt_tokens_details: { cached_tokens: 1024, audio_tokens: 0, text_tokens: 1200, image_tokens: 100 },
  completion_tokens_details: {
    reasoning_tokens: 0,
    audio_tokens: 0,
    text_tokens: 50,
    accepted_prediction_tokens: 12,
    rejected_prediction_tokens: 0,
  },
};
const BARE_USAGE = { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 };
type Kind = "reported" | "bare";
const modelName = (kind: Kind, stream: boolean) => `details-${kind}${stream ? "-stream" : ""}`;

const completion = (usage: Record<string, unknown>) => ({
  id: "chatcmpl-details",
  object: "chat.completion",
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage,
});

/** The mock streams whenever it has stream events, so each mode gets its own. */
function upstreamReporting(usage: Record<string, unknown>, stream: boolean) {
  if (!stream) return startOpenAiUpstream({ nonStreamBody: completion(usage) });
  const chunk = (body: Record<string, unknown>) =>
    JSON.stringify({ id: "chatcmpl-details", object: "chat.completion.chunk", model: "gpt-4o-mini", ...body });
  return startOpenAiUpstream({
    nonStreamBody: completion(usage),
    streamEvents: [
      chunk({ choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }] }),
      chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
      chunk({ choices: [], usage }),
      "[DONE]",
    ],
  });
}

describe("chat completion usage details reach the caller as the upstream reported them", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const models: string[] = [];
    for (const [kind, usage] of [
      ["reported", REPORTED_USAGE],
      ["bare", BARE_USAGE],
    ] as const) {
      for (const stream of [false, true]) {
        const name = modelName(kind, stream);
        const up = await upstreamReporting(usage, stream);
        upstreams.push(up);
        models.push(name);
        const pk = await seed.createProviderKey({ display_name: `${name}-pk`, secret: "sk-mock", api_base: `${up.baseUrl}/v1` });
        await seed.createModel({ display_name: name, provider: "openai", model_name: "gpt-4o-mini", provider_key_id: pk.id });
      }
    }
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: models });
    // The caller key is seeded last, so its authenticating implies every model is loaded.
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    for (const up of upstreams) await up.close();
  });

  async function chat(model: string, stream: boolean): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "hi" }],
        ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    return text;
  }

  function frames(text: string): Array<Record<string, any>> {
    return text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice(6)));
  }

  test("non-streamed: every reported detail counter reaches the caller, zeros included", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const usage = JSON.parse(await chat(modelName("reported", false), false)).usage;
    expect(usage).toEqual(REPORTED_USAGE);
  });

  test("streamed: the usage frame has empty choices and every reported detail counter", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const text = await chat(modelName("reported", true), true);
    const usageFrames = frames(text).filter((f) => f.usage != null);
    expect(usageFrames, text).toHaveLength(1);
    expect(usageFrames[0]!.choices, text).toEqual([]);
    expect(usageFrames[0]!.usage).toEqual(REPORTED_USAGE);
    // The content and the finish reason still arrive on their own frames.
    const content = frames(text).flatMap((f) => f.choices.map((c: any) => c.delta?.content ?? "")).join("");
    expect(content).toBe("hi");
    expect(frames(text).some((f) => f.choices[0]?.finish_reason === "stop"), text).toBe(true);
  });

  for (const stream of [false, true]) {
    const mode = stream ? "streamed" : "non-streamed";
    test(`${mode}: no details object is invented when the upstream reported none`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const text = await chat(modelName("bare", stream), stream);
      const usage = stream ? frames(text).find((f) => f.usage != null)?.usage : JSON.parse(text).usage;
      expect(usage, text).toEqual(BARE_USAGE);
    });
  }
});
