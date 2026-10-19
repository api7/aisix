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
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// E2E: an exported span names the model entry the caller addressed by id
// (`aisix.requested_model_id`) beside the entry that served the attempt
// (`aisix.model_id`), so a trace backend can join a routing group's traffic
// across a rename of the group.
//
// - A routing-group request: every attempt span — the failed primary and the
//   fallback that served — carries the GROUP's id, while `aisix.model_id`
//   names that attempt's own target. Covered non-streamed and streamed.
// - A direct-model request: both attributes carry the model's own id.

const CALLER = "sk-otlp-requested-model-id-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const GROUP = "rmid-group";
const PRIMARY = "rmid-primary";
const SECONDARY = "rmid-secondary";
const STREAM_GROUP = "rmid-stream-group";
const STREAM_TARGET = "rmid-stream-target";
const DIRECT = "rmid-direct";

const KIND_SERVER = 2;

const OK_BODY = {
  id: "chatcmpl-rmid",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
};

function streamChunk(json: Record<string, unknown>): string {
  return JSON.stringify({
    id: "chatcmpl-rmid-stream",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    ...json,
  });
}

const STREAM_EVENTS = [
  streamChunk({ choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }] }),
  streamChunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
  streamChunk({ choices: [], usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 } }),
  "[DONE]",
];

async function waitFor<T>(what: string, probe: () => T | undefined, timeoutMs = 15_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = probe();
    if (hit !== undefined) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`timed out waiting for ${what}`);
}

/** The request's attempt spans, once its SERVER span and `n` attempts arrived. */
async function attemptSpans(otlp: MockOtlp, requestId: string, n: number): Promise<CapturedSpan[]> {
  const mine = () => otlp.spans.filter((s) => s.attributes["aisix.request_id"] === requestId);
  const attempts = () => mine().filter((s) => s.attributes["gen_ai.operation.name"] !== undefined);
  await waitFor(`SERVER span and ${n} attempt span(s) of ${requestId}`, () =>
    mine().some((s) => s.kind === KIND_SERVER) && attempts().length >= n ? true : undefined,
  );
  return attempts();
}

describe("OTLP spans carry the addressed model entry's id", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let otlp: MockOtlp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  const ids: Record<string, string> = {};

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const failing = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "primary down", type: "server_error" } },
    });
    const ok = await startOpenAiUpstream({ nonStreamBody: OK_BODY });
    const streaming = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS });
    upstreams.push(failing, ok, streaming);
    otlp = await startMockOtlp();

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "rmid-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });

    const direct = async (name: string, upstream: OpenAiUpstream) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        provider: "openai",
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      const model = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
        cooldown: { enabled: false },
      });
      ids[name] = model.id;
    };
    await direct(PRIMARY, failing);
    await direct(SECONDARY, ok);
    await direct(STREAM_TARGET, streaming);
    await direct(DIRECT, ok);
    const group = await seed.createModel({
      display_name: GROUP,
      routing: {
        strategy: "failover",
        targets: [{ model: PRIMARY }, { model: SECONDARY }],
        retries: 0,
        max_fallbacks: 1,
      },
    });
    ids[GROUP] = group.id;
    const streamGroup = await seed.createModel({
      display_name: STREAM_GROUP,
      routing: { strategy: "failover", targets: [{ model: STREAM_TARGET }] },
    });
    ids[STREAM_GROUP] = streamGroup.id;

    // Seeded last: its authenticating implies every row above is loaded.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.text();
      return res.status === 200;
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await otlp?.close();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function chat(model: string, stream = false): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model, stream, messages: [{ role: "user", content: "hi" }] }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");
    return requestId;
  }

  test("a routing group's attempts carry the group's id beside each target's", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const requestId = await chat(GROUP);
    const spans = await attemptSpans(otlp!, requestId, 2);

    expect(spans.map((s) => s.attributes["aisix.model_id"]).sort()).toEqual(
      [ids[PRIMARY], ids[SECONDARY]].sort(),
    );
    for (const span of spans) {
      expect(span.attributes["aisix.requested_model"]).toBe(GROUP);
      expect(span.attributes["aisix.requested_model_id"], `${span.attributes["aisix.model_id"]}`).toBe(
        ids[GROUP],
      );
    }
  });

  test("a streamed routing-group request carries the group's id", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const requestId = await chat(STREAM_GROUP, true);
    const [span] = await attemptSpans(otlp!, requestId, 1);

    expect(span.attributes["aisix.model_id"]).toBe(ids[STREAM_TARGET]);
    expect(span.attributes["aisix.requested_model_id"]).toBe(ids[STREAM_GROUP]);
  });

  test("a direct model is both the addressed and the dispatched entry", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const requestId = await chat(DIRECT);
    const [span] = await attemptSpans(otlp!, requestId, 1);

    expect(span.attributes["aisix.model_id"]).toBe(ids[DIRECT]);
    expect(span.attributes["aisix.requested_model_id"]).toBe(ids[DIRECT]);
  });
});
