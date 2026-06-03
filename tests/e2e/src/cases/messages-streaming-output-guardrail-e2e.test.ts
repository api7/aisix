import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: STREAMING /v1/messages runs output guardrails at end-of-stream
// (#448 #22). The Anthropic passthrough forwards bytes verbatim, so a
// blocked response is signalled with a terminal `error` event (mirroring
// /v1/chat/completions and the common streaming-guardrail pattern). We stream
// Anthropic SSE whose text_delta carries a forbidden token and require
// the response to end with a content_filter error event.

const CALLER = "sk-msgstream-gr-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");
const FORBIDDEN = "forbiddenstreamtoken";
const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: { id: "msg_s", role: "assistant", content: [], model: "claude-3-5-haiku-20241022", stop_reason: null, usage: { input_tokens: 5, output_tokens: 1 } },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: `here is ${FORBIDDEN} in the stream` } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 12 } }),
  JSON.stringify({ type: "message_stop" }),
];

describe("streaming /v1/messages output guardrail (#448)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS, eventDelayMs: 2 });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    const pk = await admin.createProviderKey({
      display_name: "msgstream-gr-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "msgstream-gr",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({ key_hash: HASH, allowed_models: ["msgstream-gr"] });
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "msgstream-gr-output-keyword",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const stream = () =>
    fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": CALLER },
      body: JSON.stringify({
        model: "msgstream-gr",
        max_tokens: 64,
        stream: true,
        messages: [{ role: "user", content: "go" }],
      }),
    });

  test("a forbidden streamed response ends with a content_filter error event", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const r = await stream();
      const b = await r.text();
      return b.includes("content_filter");
    });

    const res = await stream();
    expect(res.status).toBe(200); // stream starts 200; the block is in-band
    const body = await res.text();
    expect(body, "streamed content is forwarded verbatim").toContain(FORBIDDEN);
    expect(body, "stream must end with a content_filter error event").toContain("content_filter");
  });
});
