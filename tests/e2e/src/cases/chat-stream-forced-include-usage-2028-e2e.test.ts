import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (AISIX-Cloud#2028): a streamed /v1/chat/completions request to an
// OpenAI-protocol upstream always asks the upstream for usage, whatever the
// caller put in `stream_options`, so the gateway accounts the upstream's own
// token counts instead of estimating them. The caller still gets the response
// it asked for: usage only when it sent `include_usage: true`.
//
// The mock upstream behaves like OpenAI: it reports usage only when the
// request carries `stream_options.include_usage: true`. One provider reports
// it on a usage-only chunk after the finish chunk (OpenAI), the other on the
// finish chunk itself (several OpenAI-compatible servers).

const CALLER = "sk-forced-include-usage-2028";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "e2e";
const LOGSTORE = "forced-include-usage-store";

const MODELS = {
  separateChunk: "forced-usage-separate",
  onFinishChunk: "forced-usage-on-finish",
} as const;

const UPSTREAM_PATHS: Record<string, string> = {
  [MODELS.separateChunk]: "/separate/chat/completions",
  [MODELS.onFinishChunk]: "/on-finish/chat/completions",
};

const USAGE = { prompt_tokens: 21, completion_tokens: 5, total_tokens: 26 };

function upstreamStream(usageOnFinish: boolean, includeUsage: boolean): string {
  const base = { id: "chatcmpl-2028", object: "chat.completion.chunk", model: "gpt-4o-mini" };
  const frames: Array<Record<string, unknown>> = [
    { ...base, choices: [{ index: 0, delta: { role: "assistant", content: "Paris." }, finish_reason: null }] },
    {
      ...base,
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      ...(includeUsage && usageOnFinish ? { usage: USAGE } : {}),
    },
  ];
  if (includeUsage && !usageOnFinish) frames.push({ ...base, choices: [], usage: USAGE });
  return frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("") + "data: [DONE]\n\n";
}

interface Received {
  path: string;
  body: Record<string, any>;
}

function startUpstream(received: Received[]): Server {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const path = req.url ?? "";
      let body: Record<string, any>;
      try {
        body = JSON.parse(raw) as Record<string, any>;
      } catch {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: `mock upstream: invalid JSON body: ${raw}` } }));
        return;
      }
      received.push({ path, body });
      const usageOnFinish = UPSTREAM_PATHS[MODELS.onFinishChunk] === path;
      if (!usageOnFinish && UPSTREAM_PATHS[MODELS.separateChunk] !== path) {
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: `mock upstream: unknown path ${path}` } }));
        return;
      }
      const includeUsage = body.stream_options?.include_usage === true;
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(upstreamStream(usageOnFinish, includeUsage));
    });
  });
}

function sseData(raw: string): string[] {
  return raw
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => l.slice(6));
}

function parseChatSse(raw: string): Array<Record<string, any>> {
  return sseData(raw)
    .filter((d) => d !== "[DONE]")
    .map((d) => JSON.parse(d));
}

describe("streamed chat to an OpenAI-protocol upstream always asks it for usage", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  let sls: MockSls | undefined;
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
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-ak-id",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-ak-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "forced-include-usage-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "forced-include-usage-proj",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });
    for (const [model, prefix] of [
      [MODELS.separateChunk, "separate"],
      [MODELS.onFinishChunk, "on-finish"],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${model}-pk`,
        secret: "sk-upstream-mock",
        api_base: `${baseUrl}/${prefix}`,
      });
      await seed.createModel({
        display_name: model,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    // Seeded last: this key authenticating implies the whole seed set landed.
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: Object.values(MODELS) });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  async function streamChat(model: string, extra: Record<string, unknown>) {
    const before = received.length;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        messages: [{ role: "user", content: "What is the capital of France?" }],
        ...extra,
      }),
    });
    const raw = await res.text();
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    const sent = received.slice(before);
    return { status: res.status, raw, requestId, sent, chunks: res.status === 200 ? parseChatSse(raw) : [] };
  }

  async function usageRow(requestId: string) {
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId,
      `usage row for ${requestId}`,
      15_000,
    );
  }

  for (const [label, model] of Object.entries(MODELS)) {
    const withoutUsage: Array<[string, Record<string, unknown>]> = [
      ["stream_options: {}", { stream_options: {} }],
      ["include_usage: false", { stream_options: { include_usage: false, continuous_usage_stats: true } }],
      ["no stream_options", {}],
    ];
    for (const [caseLabel, extra] of withoutUsage) {
      test(`${label}, ${caseLabel}: upstream is asked for usage, the caller sees none, the event has the upstream's`, async (ctx) => {
        if (!etcdReachable || !app) return ctx.skip();
        const out = await streamChat(model, extra);
        expect(out.status, out.raw).toBe(200);

        expect(out.sent).toHaveLength(1);
        expect(out.sent[0]!.path).toBe(UPSTREAM_PATHS[model]);
        const sentOptions = out.sent[0]!.body.stream_options;
        expect(sentOptions?.include_usage, JSON.stringify(out.sent[0]!.body)).toBe(true);
        // The caller's other stream_options keys reach the upstream untouched.
        const callerOptions = (extra.stream_options ?? {}) as Record<string, unknown>;
        expect(sentOptions).toEqual({ ...callerOptions, include_usage: true });

        // Neither a usage-only chunk nor a usage field on a content chunk.
        expect(out.chunks.every((c) => (c.choices ?? []).length === 1), out.raw).toBe(true);
        expect(out.raw).not.toContain('"usage"');
        const content = out.chunks.map((c) => c.choices[0].delta?.content ?? "").join("");
        expect(content).toBe("Paris.");

        const row = await usageRow(out.requestId);
        const dump = JSON.stringify([...row]);
        expect(row.get("prompt_tokens"), dump).toBe("21");
        expect(row.get("completion_tokens"), dump).toBe("5");
        expect(row.get("usage_estimated"), dump).not.toBe("true");
      });
    }

    test(`${label}, include_usage: true: the caller gets one usage-only chunk`, async (ctx) => {
      if (!etcdReachable || !app) return ctx.skip();
      const out = await streamChat(model, { stream_options: { include_usage: true } });
      expect(out.status, out.raw).toBe(200);
      expect(out.sent[0]!.path).toBe(UPSTREAM_PATHS[model]);
      expect(out.sent[0]!.body.stream_options).toEqual({ include_usage: true });

      const withUsage = out.chunks.filter((c) => c.usage != null);
      expect(withUsage, out.raw).toHaveLength(1);
      const last = out.chunks.at(-1)!;
      expect(last.choices, out.raw).toEqual([]);
      expect(last.usage).toMatchObject(USAGE);
      // The usage-only chunk is the last frame before the terminator.
      const frames = sseData(out.raw);
      expect(frames.at(-1)).toBe("[DONE]");
      expect(JSON.parse(frames.at(-2)!)).toEqual(last);

      const row = await usageRow(out.requestId);
      expect(row.get("prompt_tokens")).toBe("21");
      expect(row.get("usage_estimated")).not.toBe("true");
    });
  }
});
