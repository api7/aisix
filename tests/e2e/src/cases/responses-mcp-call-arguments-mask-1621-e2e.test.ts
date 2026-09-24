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

// E2E (AISIX-Cloud#1621): a streamed native `/v1/responses` reply that calls
// an MCP tool carries the call's arguments twice — on the
// `response.mcp_call_arguments.delta` / `.done` pair as they stream, and on
// the terminal `mcp_call` item. An output mask rule must rewrite both; a
// stream that masks the item and relays the argument deltas raw hands the
// caller the value the guardrail was configured to hide.

const CALLER = "sk-resp-mcp-args-1621-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");
const EMAIL = "mcp-args@example.com";
const MASKED = "[EMAIL_REDACTED]";
const ARGS = JSON.stringify({ to: EMAIL, subject: "hi" });
// Split inside the address, so no single delta carries it whole.
const SPLIT = ARGS.indexOf("@") + 3;

const MCP_ITEM = {
  type: "mcp_call",
  id: "mcp_1621",
  server_label: "mail",
  name: "send_mail",
  arguments: ARGS,
  output: "sent",
};

const STREAM_EVENTS = [
  JSON.stringify({
    type: "response.created",
    response: { id: "resp_1621", object: "response", status: "in_progress", model: "gpt-4o-mini", output: [] },
  }),
  JSON.stringify({
    type: "response.output_item.added",
    output_index: 0,
    item: { ...MCP_ITEM, arguments: "", output: null },
  }),
  JSON.stringify({
    type: "response.mcp_call_arguments.delta",
    item_id: MCP_ITEM.id,
    output_index: 0,
    delta: ARGS.slice(0, SPLIT),
  }),
  JSON.stringify({
    type: "response.mcp_call_arguments.delta",
    item_id: MCP_ITEM.id,
    output_index: 0,
    delta: ARGS.slice(SPLIT),
  }),
  JSON.stringify({
    type: "response.mcp_call_arguments.done",
    item_id: MCP_ITEM.id,
    output_index: 0,
    arguments: ARGS,
  }),
  JSON.stringify({ type: "response.output_item.done", output_index: 0, item: MCP_ITEM }),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_1621",
      object: "response",
      status: "completed",
      model: "gpt-4o-mini",
      output: [MCP_ITEM],
      usage: { input_tokens: 5, output_tokens: 12, total_tokens: 17 },
    },
  }),
];

type Frame = { type?: string; delta?: string; arguments?: string };

describe("native /v1/responses stream masks mcp_call arguments (AISIX-Cloud#1621)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream({ streamEvents: STREAM_EVENTS });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "resp-mcp-1621-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const m = await seed.createModel({
      display_name: "resp-mcp-1621",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const g = await seed.createGuardrail(
      {
        name: "resp-mcp-1621-mask",
        enabled: true,
        hook_point: "output",
        kind: "pii",
        detectors: [{ type: "email", action: "mask" }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(g.id, m.id);
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: HASH, allowed_models: ["resp-mcp-1621"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("the argument deltas and done frame are masked like the terminal item", async (ctx) => {
    if (!etcdReachable || !app) ctx.skip();
    const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER}` },
      body: JSON.stringify({ model: "resp-mcp-1621", input: "mail them", stream: true }),
    });
    expect(res.status).toBe(200);
    const body = await res.text();

    const frames: Frame[] = body
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice("data: ".length)) as Frame);
    const deltas = frames.filter((f) => f.type === "response.mcp_call_arguments.delta");
    const done = frames.find((f) => f.type === "response.mcp_call_arguments.done");
    expect(deltas.length, body).toBeGreaterThan(0);
    expect(done, body).toBeDefined();

    // What a client assembles from the deltas is what the done frame and
    // the terminal item say: the masked arguments.
    const assembled = deltas.map((d) => d.delta ?? "").join("");
    expect(JSON.parse(assembled)).toEqual({ to: MASKED, subject: "hi" });
    expect(JSON.parse(done!.arguments!)).toEqual({ to: MASKED, subject: "hi" });
    expect(body, "no frame carries the raw address or a fragment of it").not.toContain("mcp-args@");
  });
});
