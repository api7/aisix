import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { EtcdClient, ProxyClient, SeedClient, spawnApp, waitConfigPropagation, type SpawnedApp } from "../harness/index.js";

// E2E: a streamed ensemble request carrying the OpenAI SDK's
// `stream_options: {include_usage: true}`. The panel members are called
// without streaming, and an OpenAI-protocol upstream refuses
// `stream_options` on such a call, so it must not reach them; the caller
// still gets its usage chunk.

const CALLER = "sk-ensemble-stream-options";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const ENSEMBLE = "ens-stream-options";

interface Received {
  model: string;
  body: Record<string, unknown>;
}

// Answers the way OpenAI does: `stream_options` without `stream` is a 400.
function startUpstream(received: Received[]): Server {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as Record<string, unknown>;
      const model = String(body.model ?? "");
      received.push({ model, body });
      if (body.stream_options !== undefined && body.stream !== true) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            error: {
              message: "The 'stream_options' parameter is only allowed when 'stream' is enabled.",
              type: "invalid_request_error",
              param: "stream_options",
              code: null,
            },
          }),
        );
        return;
      }
      if (body.stream === true) {
        const frames = [
          { id: "chatcmpl-j", model, choices: [{ index: 0, delta: { role: "assistant", content: "Paris." } }] },
          { id: "chatcmpl-j", model, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
          { id: "chatcmpl-j", model, choices: [], usage: { prompt_tokens: 30, completion_tokens: 7, total_tokens: 37 } },
        ];
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end(frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("") + "data: [DONE]\n\n");
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          id: "chatcmpl-p",
          object: "chat.completion",
          model,
          choices: [{ index: 0, message: { role: "assistant", content: "Paris" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 11, total_tokens: 16 },
        }),
      );
    });
  });
}

describe("streamed ensemble with stream_options", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  const received: Received[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = startUpstream(received);
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "ens-stream-options-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${addr.port}/v1`,
    });
    for (const [display, upstreamModel] of [
      ["ens-so-member-a", "member-a"],
      ["ens-so-member-b", "member-b"],
      ["ens-so-judge", "judge"],
    ]) {
      await seed.createModel({ display_name: display, provider: "openai", model_name: upstreamModel, provider_key_id: pk.id });
    }
    await seed.createModel({
      display_name: ENSEMBLE,
      ensemble: {
        panel: [{ model: "ens-so-member-a" }, { model: "ens-so-member-b" }],
        judge: { model: "ens-so-judge" },
        min_responses: 2,
      },
    });
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: [ENSEMBLE] });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  test("the panel is not sent stream_options, and the caller still gets one usage chunk", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = received.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: ENSEMBLE,
        stream: true,
        stream_options: { include_usage: true },
        messages: [{ role: "user", content: "What is the capital of France?" }],
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(text).toContain("data: [DONE]");

    const sent = received.slice(before);
    const panel = sent.filter((r) => r.model.startsWith("member-"));
    expect(panel, JSON.stringify(sent)).toHaveLength(2);
    for (const call of panel) {
      expect(call.body.stream ?? false).toBe(false);
      expect(call.body).not.toHaveProperty("stream_options");
    }

    const frames = text
      .split("\n")
      .filter((l) => l.startsWith("data: ") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice(6)) as Record<string, any>);
    const content = frames.map((f) => f.choices?.[0]?.delta?.content ?? "").join("");
    expect(content).toBe("Paris.");
    const withUsage = frames.filter((f) => f.usage != null);
    expect(withUsage, text).toHaveLength(1);
    expect(frames.at(-1)).toBe(withUsage[0]);
    expect(withUsage[0]!.choices).toEqual([]);
    // Judge 30 / 7 / 37 plus the two members' 5 / 11 / 16 each.
    expect(withUsage[0]!.usage).toMatchObject({ prompt_tokens: 40, completion_tokens: 29, total_tokens: 69 });
  });
});
