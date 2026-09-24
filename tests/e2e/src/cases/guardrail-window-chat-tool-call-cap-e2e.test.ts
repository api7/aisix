import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  ProxyClient,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a streamed /v1/chat/completions response under a `window`-mode output
// guardrail, where the model streams tool-call arguments. Every window scan
// re-reads the tool-call text collected so far, so that text is held under the
// row's own `max_buffer_bytes`, and outgrowing it is a buffer trip that
// follows the row's `on_buffer_exceeded`:
//   - `fail_closed` refuses the stream and names the row
//     (`blocked_buffer_exceeded`);
//   - `fail_open` lets the unscanned remainder through and records the
//     `output_buffer_exceeded` bypass.
// It is never truncated silently.

const CALLER = "sk-window-chat-tool-cap-e2e";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "window-chat-tool-cap-events";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CLOSED_ROW = "window-chat-tool-cap-closed";
const OPEN_ROW = "window-chat-tool-cap-open";

// 30 argument pieces of 100 bytes: three times the rows' cap. No assistant
// content, so no window fills and the tool-call chunks stay held.
const CAP = 1_000;
const ARGS = Array.from({ length: 30 }, (_, i) => `${String(i).padStart(2, "0")}${"a".repeat(98)}`);

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-window-tool-cap",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
const toolCallEvents = [
  chatChunk({
    role: "assistant",
    tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "lookup", arguments: "" } }],
  }),
  ...ARGS.map((a) => chatChunk({ tool_calls: [{ index: 0, function: { arguments: a } }] })),
  chatChunk({}, "tool_calls"),
  "[DONE]",
];

// Azure Content Safety `text:analyze` that finds nothing.
async function startMockAzure(): Promise<{ url: string; close: () => Promise<void> }> {
  const server: Server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ categoriesAnalysis: [{ category: "Violence", severity: 0 }], blocklistsMatch: [] }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

interface EnforcedHit {
  guardrail_name: string;
  hook: string;
  action: string;
}

describe("streamed chat tool-call arguments are held under a window row's own buffer cap", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let azure: { url: string; close: () => Promise<void> } | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  const models = { closed: "wnd-chat-tool-closed", open: "wnd-chat-tool-open" } as const;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    azure = await startMockAzure();
    upstream = await startOpenAiUpstream({ streamEvents: toolCallEvents });
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-window-chat-tool-cap",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const pk = await seed.createProviderKey({
      display_name: "wnd-chat-tool-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    for (const [variant, rowName, policy] of [
      ["closed", CLOSED_ROW, "fail_closed"],
      ["open", OPEN_ROW, "fail_open"],
    ] as const) {
      // No `stream_processing_mode`: the kind's default, `window`.
      const row = await seed.createGuardrail(
        {
          name: rowName,
          enabled: true,
          kind: "azure_content_safety_text_moderation",
          hook_point: "output",
          endpoint: azure.url,
          api_key: "azure-mock-key",
          max_buffer_bytes: CAP,
          on_buffer_exceeded: policy,
        },
        { attach: false },
      );
      const m = await seed.createModel({
        display_name: models[variant],
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      await seed.attachGuardrailToModel(row.id, m.id);
    }

    // Caller key LAST: it authenticating implies every row above is live.
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: ["*"] });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await azure?.close();
    await sls?.close();
  });

  const send = async (model: string) => {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${CALLER}` },
      body: JSON.stringify({ model, stream: true, messages: [{ role: "user", content: "go" }] }),
    });
    return res.text();
  };
  const byModel = (model: string) => (l: Map<string, string>) => l.get("requested_model") === model;

  test("past the row's cap under fail_closed, the stream is refused and the row named", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const body = await send(models.closed);
    expect(body).toContain("output_buffer_exceeded");
    expect(body, "tool-call text past the cap reached the caller").not.toContain(ARGS[29]);
    const log = await waitForSlsLog(sls, LOGSTORE, byModel(models.closed), "chat closed");
    expect(log.get("guardrail_blocked")).toBe("true");
    const hits = JSON.parse(log.get("guardrail_enforced_hits") ?? "[]") as EnforcedHit[];
    expect(
      hits.map(({ guardrail_name, hook, action }) => ({ guardrail_name, hook, action })),
      "the refusal names no row, or the wrong one",
    ).toEqual([{ guardrail_name: CLOSED_ROW, hook: "output", action: "blocked_buffer_exceeded" }]);
  });

  test("past the row's cap under fail_open, the remainder is released and the bypass recorded", async (ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const body = await send(models.open);
    expect(body, "fail_open lets the unscanned remainder through").toContain(ARGS[29]);
    expect(body).not.toContain("output_buffer_exceeded");
    const log = await waitForSlsLog(sls, LOGSTORE, byModel(models.open), "chat open");
    expect(log.get("guardrail_blocked") ?? "false").not.toBe("true");
    expect(log.get("guardrail_bypassed_reason"), "the unscanned remainder is not recorded").toBe(
      "output_buffer_exceeded",
    );
  });
});
