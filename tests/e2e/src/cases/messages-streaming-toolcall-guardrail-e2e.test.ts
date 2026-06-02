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

// E2E: streaming /v1/messages output guardrails also cover tool_use
// payloads (#448 #22). Anthropic streams tool_use args as input_json_delta
// (and the name in content_block_start), with no text_delta. The
// passthrough end-of-stream guardrail must scan that too — a forbidden
// token in the tool arguments must trigger a terminal content_filter
// error, matching the non-streaming path.

const CALLER = "sk-msgtool-gr-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");
const FORBIDDEN = "forbiddentoolarg";
const STREAM_EVENTS = [
  JSON.stringify({
    type: "message_start",
    message: { id: "msg_t", role: "assistant", content: [], model: "claude-3-5-haiku-20241022", stop_reason: null, usage: { input_tokens: 5, output_tokens: 1 } },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "tu_1", name: "run_query", input: {} } }),
  JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: `{"q":"${FORBIDDEN}"}` } }),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 10 } }),
  JSON.stringify({ type: "message_stop" }),
];

describe("streaming /v1/messages tool_use output guardrail (#448)", () => {
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
      display_name: "msgtool-gr-pk",
      secret: "sk-anth-mock",
      api_base: upstream.baseUrl,
    });
    await admin.createModel({
      display_name: "msgtool-gr",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({ key_hash: HASH, allowed_models: ["msgtool-gr"] });
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "msgtool-gr-output-keyword",
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
      body: JSON.stringify({ model: "msgtool-gr", max_tokens: 64, stream: true, messages: [{ role: "user", content: "go" }] }),
    });

  test("forbidden tool_use arguments in a stream trigger a content_filter error", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => (await stream().then((r) => r.text())).includes("content_filter"));

    const body = await stream().then((r) => r.text());
    expect(body, "tool_use arguments are forwarded verbatim").toContain(FORBIDDEN);
    expect(body, "stream must end with a content_filter error event").toContain("content_filter");
  });
});
