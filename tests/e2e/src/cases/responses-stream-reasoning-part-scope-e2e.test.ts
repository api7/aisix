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

// E2E: generated reasoning is out of the output-guardrail scope on a
// streamed `/v1/responses` reply, as it already is on a buffered one. An upstream that streams raw
// reasoning (gpt-oss behind a native Responses endpoint does) closes the
// reasoning item's `reasoning_text` part with `response.content_part.done`
// — the event a message's `output_text` part closes with too. Only the
// message part may be masked or judged; the reasoning part must pass
// untouched and must not trip a block rule.

const CALLER = "sk-resp-reasoning-part-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");
const REASONING_EMAIL = "thinker@example.com";
const REASONING_TERM = "ZEPHYRWORD";
const REASONING = `Consider ${REASONING_EMAIL} and ${REASONING_TERM}.`;
const ANSWER_EMAIL = "answer@example.com";
const ANSWER = `Write to ${ANSWER_EMAIL}.`;
const MASKED = "[EMAIL_REDACTED]";

const REASONING_ITEM = {
  type: "reasoning",
  id: "rs_1",
  summary: [],
  content: [{ type: "reasoning_text", text: REASONING }],
};
const MESSAGE_ITEM = {
  type: "message",
  id: "msg_1",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text: ANSWER, annotations: [] }],
};

// Event sequence as a gpt-oss Responses stream emits it.
const STREAM_EVENTS = [
  { type: "response.created", response: { id: "resp_rp", object: "response", status: "in_progress", model: "gpt-oss-120b", output: [] } },
  { type: "response.output_item.added", output_index: 0, item: { ...REASONING_ITEM, content: [] } },
  { type: "response.content_part.added", item_id: "rs_1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: "" } },
  { type: "response.reasoning_text.delta", item_id: "rs_1", output_index: 0, content_index: 0, delta: REASONING },
  { type: "response.reasoning_text.done", item_id: "rs_1", output_index: 0, content_index: 0, text: REASONING },
  { type: "response.content_part.done", item_id: "rs_1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: REASONING } },
  { type: "response.output_item.done", output_index: 0, item: REASONING_ITEM },
  { type: "response.output_item.added", output_index: 1, item: { ...MESSAGE_ITEM, content: [] } },
  { type: "response.content_part.added", item_id: "msg_1", output_index: 1, content_index: 0, part: { type: "output_text", text: "", annotations: [] } },
  { type: "response.output_text.delta", item_id: "msg_1", output_index: 1, content_index: 0, delta: ANSWER },
  { type: "response.output_text.done", item_id: "msg_1", output_index: 1, content_index: 0, text: ANSWER },
  { type: "response.content_part.done", item_id: "msg_1", output_index: 1, content_index: 0, part: { type: "output_text", text: ANSWER, annotations: [] } },
  { type: "response.output_item.done", output_index: 1, item: MESSAGE_ITEM },
  {
    type: "response.completed",
    response: {
      id: "resp_rp",
      object: "response",
      status: "completed",
      model: "gpt-oss-120b",
      output: [REASONING_ITEM, MESSAGE_ITEM],
      usage: { input_tokens: 5, output_tokens: 12, total_tokens: 17 },
    },
  },
].map((e) => JSON.stringify(e));

// Control: the same stream with the term moved into the answer, so the
// block rule is shown to be live on this route.
const CONTROL_EVENTS = STREAM_EVENTS.map((e) => e.replaceAll("Write to ", `${REASONING_TERM} `));

type Frame = { type?: string; part?: { type?: string; text?: string } };

function frames(body: string): Frame[] {
  return body
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
    .map((l) => JSON.parse(l.slice("data: ".length)) as Frame);
}

describe("native /v1/responses stream leaves a reasoning_text content part out of output scope", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let control: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS });
    control = await startOpenAiUpstream({ streamEvents: CONTROL_EVENTS });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "resp-reasoning-part-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const masked = await seed.createModel({
      display_name: "resp-reasoning-mask",
      provider: "openai",
      model_name: "gpt-oss-120b",
      provider_key_id: pk.id,
    });
    const blocked = await seed.createModel({
      display_name: "resp-reasoning-block",
      provider: "openai",
      model_name: "gpt-oss-120b",
      provider_key_id: pk.id,
    });
    const controlPk = await seed.createProviderKey({
      display_name: "resp-reasoning-control-pk",
      secret: "sk-mock",
      api_base: `${control.baseUrl}/v1`,
    });
    const blockedControl = await seed.createModel({
      display_name: "resp-reasoning-block-control",
      provider: "openai",
      model_name: "gpt-oss-120b",
      provider_key_id: controlPk.id,
    });
    const mask = await seed.createGuardrail(
      {
        name: "resp-reasoning-part-mask",
        enabled: true,
        hook_point: "output",
        kind: "pii",
        detectors: [{ type: "email", action: "mask" }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(mask.id, masked.id);
    const block = await seed.createGuardrail(
      {
        name: "resp-reasoning-part-block",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: REASONING_TERM }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(block.id, blocked.id);
    await seed.attachGuardrailToModel(block.id, blockedControl.id);
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({
      key_hash: HASH,
      allowed_models: ["resp-reasoning-mask", "resp-reasoning-block", "resp-reasoning-block-control"],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await control?.close();
  });

  async function stream(model: string): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER}` },
      body: JSON.stringify({ model, input: "hi", stream: true }),
    });
  }

  test("a mask rule rewrites the message part and leaves the reasoning part as generated", async (ctx) => {
    if (!etcdReachable || !app) ctx.skip();
    const res = await stream("resp-reasoning-mask");
    expect(res.status).toBe(200);
    const body = await res.text();
    const parts = frames(body).filter((f) => f.type === "response.content_part.done");
    expect(parts.map((f) => f.part?.type), body).toEqual(["reasoning_text", "output_text"]);
    expect(parts[0].part?.text).toBe(REASONING);
    expect(parts[1].part?.text).toBe(`Write to ${MASKED}.`);
    expect(body, "the answer's address never reaches the caller").not.toContain(ANSWER_EMAIL);
  });

  test("a block rule matching only the reasoning does not refuse the stream", async (ctx) => {
    if (!etcdReachable || !app) ctx.skip();
    const res = await stream("resp-reasoning-block");
    const body = await res.text();
    expect(res.status, body).toBe(200);
    expect(frames(body).some((f) => f.type === "response.completed"), body).toBe(true);
  });

  test("the same block rule refuses the stream when the answer carries the term", async (ctx) => {
    if (!etcdReachable || !app) ctx.skip();
    const res = await stream("resp-reasoning-block-control");
    const body = await res.text();
    expect(res.status, body).toBe(422);
    expect(body).toContain("resp-reasoning-part-block");
  });
});
