import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import OpenAI, { toFile } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { WebSocketServer, type WebSocket as WsSocket } from "ws";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// The UsageEvent's `total_tokens` is the total the upstream itself reported,
// stored as received and never computed by the gateway. A row left without it
// while the upstream reported one reads as "this provider reports no total".
//
// Every upstream below reports a total that is deliberately NOT the sum of
// the counters beside it, and a different one per family: a row carrying the
// sum would be a total the gateway computed, and a row carrying another
// family's number would be crossed wiring. The embeddings upstream that
// reports no total is the other half of the contract — the gateway must not
// make one up.

const CALLER = "sk-usage-total-tokens-every-family";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");
const CREDENTIAL_REF = "totaltokens";
const LOGSTORE = "total-tokens";

const M = {
  chat: "tt-chat",
  responses: "tt-responses",
  responsesBridged: "tt-responses-bridged",
  messages: "tt-messages",
  embeddings: "tt-embeddings",
  embeddingsNoTotal: "tt-embeddings-no-total",
  rerank: "tt-rerank",
  transcription: "tt-transcription",
  transcriptionStream: "tt-transcription-stream",
  images: "tt-images",
  realtime: "tt-realtime",
  batch: "tt-batch",
} as const;

const ROUTE = {
  embeddings: { name: "tt-ptr-embeddings", prefix: "/tt-emb" },
  rerank: { name: "tt-ptr-rerank", prefix: "/tt-rerank" },
  images: { name: "tt-ptr-images", prefix: "/tt-img" },
} as const;

const chatCompletion = (prompt: number, completion: number, total: number) => ({
  id: "chatcmpl-tt",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "mock-model",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: total },
});

const embeddingList = (usage: Record<string, number>) => ({
  object: "list",
  model: "mock-embed",
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
  usage,
});

const imageResult = (input: number, output: number, total: number) => ({
  created: 1_700_000_000,
  data: [{ b64_json: "aGVsbG8=" }],
  usage: { input_tokens: input, output_tokens: output, total_tokens: total },
});

// CRLF framing, as OpenAI writes a streamed transcription.
const FRAME_END = "\r\n\r\n";
const TRANSCRIPT_FRAMES = [
  `data: ${JSON.stringify({ type: "transcript.text.delta", delta: "hello" })}${FRAME_END}`,
  `data: ${JSON.stringify({
    type: "transcript.text.done",
    text: "hello",
    usage: { type: "tokens", input_tokens: 26, output_tokens: 12, total_tokens: 18 },
  })}${FRAME_END}`,
  `data: [DONE]${FRAME_END}`,
];

/** Mock OpenAI Realtime upstream: one `response.done` per client frame. */
async function startRealtimeUpstream() {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.on("message", () => {
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: {
            usage: {
              input_tokens: 9,
              output_tokens: 4,
              total_tokens: 21,
              input_token_details: { cached_tokens: 0 },
            },
          },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    close: () =>
      new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

/** Jobs-surface mock whose completed batch output reports a total per line. */
async function startBatchUpstream() {
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      const json = (payload: unknown) => {
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(payload));
      };
      if (req.method === "POST" && path === "/v1/files") {
        return json({ id: "file-tt-in", object: "file", purpose: "batch", filename: "in.jsonl" });
      }
      if (req.method === "GET" && path === "/v1/files/file-tt-out/content") {
        res.statusCode = 200;
        res.setHeader("content-type", "application/jsonl");
        const line = (id: string, prompt: number, completion: number, total: number) =>
          JSON.stringify({
            id,
            custom_id: id,
            response: {
              status_code: 200,
              body: {
                model: "gpt-4o-2024-08-06",
                usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: total },
              },
            },
          });
        return res.end(`${line("r1", 11, 4, 20)}\n${line("r2", 2, 1, 6)}\n`);
      }
      if (req.method === "POST" && path === "/v1/batches") {
        return json({ id: "batch_tt", object: "batch", status: "validating" });
      }
      if (req.method === "GET" && path === "/v1/batches/batch_tt") {
        return json({
          id: "batch_tt",
          object: "batch",
          status: "completed",
          endpoint: "/v1/chat/completions",
          input_file_id: "file-tt-in",
          output_file_id: "file-tt-out",
        });
      }
      res.statusCode = 404;
      res.end("{}");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    close: () =>
      new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

describe("UsageEvent total_tokens: the upstream's own total on every family", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let realtime: Awaited<ReturnType<typeof startRealtimeUpstream>> | undefined;
  let batch: Awaited<ReturnType<typeof startBatchUpstream>> | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    realtime = await startRealtimeUpstream();
    batch = await startBatchUpstream();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-total-tokens",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const upstream = async (opts: Parameters<typeof startOpenAiUpstream>[0]) => {
      const u = await startOpenAiUpstream(opts);
      upstreams.push(u);
      return u;
    };
    const model = async (
      name: string,
      apiBase: string,
      modelName = "mock-model",
      provider = "openai",
    ) => {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: apiBase,
        provider,
      });
      await seed.createModel({
        display_name: name,
        provider,
        model_name: modelName,
        provider_key_id: pk.id,
      });
    };
    const v1 = (u: OpenAiUpstream) => `${u.baseUrl}/v1`;

    await model(M.chat, v1(await upstream({ nonStreamBody: chatCompletion(5, 3, 11) })));
    await model(
      M.responses,
      v1(
        await upstream({
          nonStreamBody: {
            id: "resp_tt",
            object: "response",
            status: "completed",
            model: "mock-model",
            output: [
              {
                type: "message",
                id: "msg_tt",
                role: "assistant",
                content: [{ type: "output_text", text: "ok" }],
              },
            ],
            usage: { input_tokens: 5, output_tokens: 3, total_tokens: 12 },
          },
        }),
      ),
    );
    // A provider with no Responses API: `/v1/responses` bridges to its chat
    // completions, whose usage carries the total.
    await model(
      M.responsesBridged,
      v1(await upstream({ nonStreamBody: chatCompletion(6, 2, 13) })),
      "deepseek-chat",
      "deepseek",
    );
    // `/v1/messages` addressed to an OpenAI-protocol model bridges to chat.
    await model(M.messages, v1(await upstream({ nonStreamBody: chatCompletion(7, 2, 14) })));
    await model(
      M.embeddings,
      v1(await upstream({ nonStreamBody: embeddingList({ prompt_tokens: 4, total_tokens: 15 }) })),
      "text-embedding-3-small",
    );
    await model(
      M.embeddingsNoTotal,
      v1(await upstream({ nonStreamBody: embeddingList({ prompt_tokens: 4 }) })),
      "text-embedding-3-small",
    );
    await model(
      M.rerank,
      v1(
        await upstream({
          nonStreamBody: {
            results: [{ index: 0, relevance_score: 0.9 }],
            usage: { prompt_tokens: 12, total_tokens: 16 },
          },
        }),
      ),
      "rerank-mock",
    );
    await model(
      M.transcription,
      v1(
        await upstream({
          nonStreamBody: {
            text: "hello",
            usage: { type: "tokens", input_tokens: 9, output_tokens: 2, total_tokens: 17 },
          },
        }),
      ),
      "gpt-4o-transcribe",
    );
    await model(
      M.transcriptionStream,
      v1(await upstream({ rawStreamFrames: TRANSCRIPT_FRAMES })),
      "gpt-4o-transcribe",
    );
    await model(
      M.images,
      v1(await upstream({ nonStreamBody: imageResult(50, 60, 19) })),
      "gpt-image-1",
    );
    await model(M.realtime, `http://127.0.0.1:${realtime.port}/v1`, "gpt-realtime-mock");
    await model(M.batch, `${batch.baseUrl}/v1`, "gpt-4o");

    const routePk = await seed.createProviderKey({
      display_name: "tt-ptr-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    const route = async (r: { name: string; prefix: string }, u: OpenAiUpstream) =>
      seed.createPassthroughRoute({
        name: r.name,
        path_prefix: r.prefix,
        target_url: u.baseUrl,
        provider_key_id: routePk.id,
      });
    await route(
      ROUTE.embeddings,
      await upstream({ nonStreamBody: embeddingList({ prompt_tokens: 3, total_tokens: 22 }) }),
    );
    await route(
      ROUTE.rerank,
      await upstream({
        nonStreamBody: {
          model: "jina-reranker-v2-base-multilingual",
          results: [{ index: 0, relevance_score: 0.9 }],
          usage: { total_tokens: 23 },
        },
      }),
    );
    await route(ROUTE.images, await upstream({ nonStreamBody: imageResult(1, 2, 24) }));

    // Seeded last: its authenticating implies everything above is loaded.
    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await realtime?.close();
    await batch?.close();
    await sls?.close();
  });

  const auth = { authorization: `Bearer ${CALLER}` };

  async function post(path: string, body: unknown): Promise<string> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { ...auth, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    expect(res.status, `${path}: ${text}`).toBe(200);
    return res.headers.get("x-aisix-request-id") ?? "";
  }

  async function postForm(path: string, parts: Record<string, string | Blob>): Promise<string> {
    const form = new FormData();
    for (const [name, value] of Object.entries(parts)) {
      if (value instanceof Blob) form.set(name, value, "a.bin");
      else form.set(name, value);
    }
    const res = await fetch(`${app!.proxyUrl}${path}`, { method: "POST", headers: auth, body: form });
    const text = await res.text();
    expect(res.status, `${path}: ${text}`).toBe(200);
    return res.headers.get("x-aisix-request-id") ?? "";
  }

  const rowWhere = (pred: (log: Map<string, string>) => boolean, what: string) =>
    waitForSlsLog(sls!, LOGSTORE, pred, what);
  const byModel = (model: string) =>
    rowWhere((log) => log.get("requested_model") === model, `a row for ${model}`);
  const byRequest = (id: string) => {
    expect(id).not.toBe("");
    return rowWhere((log) => log.get("request_id") === id, `a row for request ${id}`);
  };
  const byRoute = (name: string) =>
    rowWhere((log) => log.get("passthrough_route_name") === name, `a row for route ${name}`);

  async function runRealtimeSession(): Promise<void> {
    const ws = new WebSocket(
      `${app!.proxyUrl.replace("http", "ws")}/v1/realtime?model=${M.realtime}`,
      ["realtime", `openai-insecure-api-key.${CALLER}`],
    );
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("realtime session timed out")), 15_000);
      ws.addEventListener("open", () => ws.send(JSON.stringify({ type: "response.create" })));
      ws.addEventListener("message", () => {
        clearTimeout(timer);
        ws.close();
        resolve();
      });
      ws.addEventListener("error", (e) => {
        clearTimeout(timer);
        reject(new Error(`realtime session errored: ${String(e)}`));
      });
    });
  }

  async function runBatch(): Promise<void> {
    const client = new OpenAI({ apiKey: CALLER, baseURL: `${app!.proxyUrl}/v1` });
    const file = await client.files.create(
      {
        file: await toFile(Buffer.from('{"custom_id":"r1"}\n'), "in.jsonl"),
        purpose: "batch",
      },
      { headers: { "x-aisix-model": M.batch } },
    );
    const created = await client.batches.create({
      input_file_id: file.id,
      endpoint: "/v1/chat/completions",
      completion_window: "24h",
    });
    // Retrieving the completed batch triggers the usage attribution.
    expect((await client.batches.retrieve(created.id)).status).toBe("completed");
  }

  test(
    "each family records the reported total verbatim, and none where none was reported",
    async (ctx) => {
      if (!etcdReachable || !app || !sls) {
        ctx.skip();
        return;
      }
      const proxy = new ProxyClient(app.proxyUrl, CALLER);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

      const hi = [{ role: "user", content: "hi" }];
      await post("/v1/chat/completions", { model: M.chat, messages: hi });
      await post("/v1/responses", { model: M.responses, input: "hi" });
      await post("/v1/responses", { model: M.responsesBridged, input: "hi" });
      await post("/v1/messages", { model: M.messages, max_tokens: 16, messages: hi });
      await post("/v1/embeddings", { model: M.embeddings, input: "hello" });
      await post("/v1/embeddings", { model: M.embeddingsNoTotal, input: "hello" });
      await post("/v1/rerank", { model: M.rerank, query: "q", documents: ["a", "b"] });
      const audio = () => new Blob(["ID3fake-audio"], { type: "audio/mpeg" });
      await postForm("/v1/audio/transcriptions", { model: M.transcription, file: audio() });
      await postForm("/v1/audio/transcriptions", {
        model: M.transcriptionStream,
        file: audio(),
        stream: "true",
      });
      const generationId = await post("/v1/images/generations", {
        model: M.images,
        prompt: "a cat",
      });
      const editId = await postForm("/v1/images/edits", {
        model: M.images,
        prompt: "make it blue",
        image: new Blob(["fake-png-bytes"], { type: "image/png" }),
      });
      await runRealtimeSession();
      await runBatch();
      await post(`${ROUTE.embeddings.prefix}/v1/embeddings`, {
        model: "text-embedding-3-small",
        input: "hello",
      });
      await post(`${ROUTE.rerank.prefix}/v1/rerank`, {
        model: "jina-reranker-v2-base-multilingual",
        query: "q",
        documents: ["a", "b"],
      });
      await post(`${ROUTE.images.prefix}/v1/images/generations`, {
        model: "gpt-image-1",
        prompt: "a cat",
      });

      const total = (log: Map<string, string>) => log.get("total_tokens");
      const observed: Record<string, string | undefined> = {
        chat: total(await byModel(M.chat)),
        responses: total(await byModel(M.responses)),
        responsesBridged: total(await byModel(M.responsesBridged)),
        messages: total(await byModel(M.messages)),
        embeddings: total(await byModel(M.embeddings)),
        embeddingsNoTotal: total(await byModel(M.embeddingsNoTotal)),
        rerank: total(await byModel(M.rerank)),
        transcription: total(await byModel(M.transcription)),
        transcriptionStream: total(await byModel(M.transcriptionStream)),
        imageGeneration: total(await byRequest(generationId)),
        imageEdit: total(await byRequest(editId)),
        realtime: total(await byModel(M.realtime)),
        batch: total(
          await rowWhere(
            (log) => log.get("requested_model") === M.batch && log.get("inbound_protocol") === "batch",
            "the batch attribution row",
          ),
        ),
        passthroughEmbeddings: total(await byRoute(ROUTE.embeddings.name)),
        passthroughRerank: total(await byRoute(ROUTE.rerank.name)),
        passthroughImages: total(await byRoute(ROUTE.images.name)),
      };
      // Compared as one object so a run reports every family it got wrong.
      expect(observed).toEqual({
        chat: "11",
        responses: "12",
        responsesBridged: "13",
        messages: "14",
        embeddings: "15",
        embeddingsNoTotal: undefined,
        rerank: "16",
        transcription: "17",
        transcriptionStream: "18",
        imageGeneration: "19",
        imageEdit: "19",
        realtime: "21",
        // Two output lines reporting 20 and 6.
        batch: "26",
        passthroughEmbeddings: "22",
        passthroughRerank: "23",
        passthroughImages: "24",
      });
    },
    180_000,
  );
});
