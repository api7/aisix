import { createHash, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { join } from "node:path";
import { WebSocketServer, type WebSocket as WsSocket } from "ws";
import { WebSocket } from "undici";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  metricDelta,
  scrapeMetrics,
  spawnApp,
  startA2aUpstream,
  startMcpUpstream,
  waitConfigPropagation,
  type A2aUpstream,
  type McpUpstream,
  type MetricSample,
  type SpawnedApp,
} from "../harness/index.js";

// Every client-facing proxy route emits every per-request metric family,
// unless the quantity the family measures does not exist on that route
// (AISIX-Cloud#1935). The table below is that contract: one row per route
// (and per streaming variant), naming every family the route must move.
//
// A NEW ROUTE NEEDS A ROW. `every registered route has a row` reads the
// route list out of `build_router` and fails until it does, and the row's
// family list is where a family a route legitimately lacks is declared —
// by leaving it out, with the reason beside the row.

const CALLER = "sk-metric-parity-caller";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** A per-request family and how its samples are matched for one row. */
type Family =
  | "proxy_requests"
  | "llm_requests"
  | "e2e_downstream"
  | "e2e_upstream"
  | "ttft"
  | "tokens"
  | "deployment"
  | "ratelimit_remaining";

interface Row {
  /** The router's path template this row drives. */
  route: string;
  /** What the row is called in failure messages. */
  name: string;
  /** The metric `endpoint` label the route records under. */
  endpoint: string;
  /** The model the row's model-keyed families attribute to, if any. */
  model?: string;
  families: Family[];
  drive: () => Promise<void>;
}

const REQUEST = ["proxy_requests", "e2e_downstream"] as const;
const LLM = [...REQUEST, "llm_requests"] as const;
/** A model route whose terminal attempt was dispatched. */
const MODEL_ROUTE = [...LLM, "e2e_upstream", "deployment", "ratelimit_remaining"] as const;

// ── Mock upstream: one server answering every OpenAI-shaped path ─────────────

const MP4 = "MP4-metric-parity";
const SPEECH = "ID3-metric-parity-audio";
const BATCH_OUTPUT =
  [
    { custom_id: "r1", response: { status_code: 200, body: { model: "gpt-4o", usage: { prompt_tokens: 11, completion_tokens: 4 } } } },
  ]
    .map((l) => JSON.stringify(l))
    .join("\n") + "\n";

function sse(res: ServerResponse, frames: string[]) {
  res.statusCode = 200;
  res.setHeader("content-type", "text/event-stream");
  for (const f of frames) res.write(f);
  res.end();
}

const data = (v: unknown) => `data: ${typeof v === "string" ? v : JSON.stringify(v)}\n\n`;
const event = (type: string, v: unknown) => `event: ${type}\ndata: ${JSON.stringify(v)}\n\n`;

async function startUpstream(): Promise<{ baseUrl: string; close(): Promise<void> }> {
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      let body: any = {};
      try {
        body = raw ? JSON.parse(raw) : {};
      } catch {
        body = {};
      }
      const path = (req.url ?? "/").split("?")[0]!.replace(/^\/v1(?=\/)/, "");
      const json = (status: number, payload: unknown) => {
        res.statusCode = status;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(payload));
      };
      const method = req.method ?? "GET";

      if (path === "/chat/completions") {
        if (body.stream) {
          return sse(res, [
            data({ id: "c1", object: "chat.completion.chunk", model: "gpt-4o", choices: [{ index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: null }] }),
            data({ id: "c1", object: "chat.completion.chunk", model: "gpt-4o", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } }),
            data("[DONE]"),
          ]);
        }
        return json(200, {
          id: "c1",
          object: "chat.completion",
          created: 0,
          model: "gpt-4o",
          choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        });
      }
      if (path === "/completions") {
        return json(200, {
          id: "cmpl-1",
          object: "text_completion",
          created: 0,
          model: "gpt-3.5-turbo-instruct",
          choices: [{ index: 0, text: "hi", finish_reason: "stop", logprobs: null }],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        });
      }
      if (path === "/embeddings") {
        return json(200, {
          object: "list",
          model: "text-embedding-3-small",
          data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
          usage: { prompt_tokens: 5, total_tokens: 5 },
        });
      }
      if (path === "/images/generations" || path === "/images/edits") {
        return json(200, {
          created: 0,
          data: [{ b64_json: "aGk=" }],
          usage: { input_tokens: 5, output_tokens: 7, total_tokens: 12 },
        });
      }
      if (path === "/rerank") {
        return json(200, {
          results: [{ index: 0, relevance_score: 0.9 }],
          usage: { total_tokens: 9 },
        });
      }
      if (path === "/systemone") {
        return json(200, {
          model: "jev-1",
          answers: { is_bug: { type: "noul", noul: 0.9 } },
          usage: { input_tokens: 30, output_tokens: 4 },
        });
      }
      if (path === "/decisions") {
        return json(200, {
          model: "gpt-6-luna",
          answers: [{ type: "predicate", name: "is_bug", probability: 0.9 }],
          usage: { input_tokens: 30, output_tokens: 4, total_tokens: 34 },
        });
      }
      if (path === "/responses") {
        const response = {
          id: "resp_1",
          object: "response",
          created_at: 0,
          status: "completed",
          model: "gpt-4o",
          output: [{ id: "msg_1", type: "message", role: "assistant", content: [{ type: "output_text", text: "hi" }] }],
          usage: { input_tokens: 5, output_tokens: 2, total_tokens: 7 },
        };
        if (body.stream) {
          return sse(res, [
            event("response.created", { type: "response.created", response: { ...response, status: "in_progress", output: [], usage: null } }),
            event("response.output_text.delta", { type: "response.output_text.delta", item_id: "msg_1", output_index: 0, content_index: 0, delta: "hi" }),
            event("response.completed", { type: "response.completed", response }),
          ]);
        }
        return json(200, response);
      }
      if (path === "/messages/count_tokens") {
        return json(200, { input_tokens: 42 });
      }
      if (path === "/messages") {
        if (body.stream) {
          return sse(res, [
            event("message_start", { type: "message_start", message: { id: "msg_1", type: "message", role: "assistant", content: [], model: "claude-x", stop_reason: null, usage: { input_tokens: 5, output_tokens: 0 } } }),
            event("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
            event("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } }),
            event("content_block_stop", { type: "content_block_stop", index: 0 }),
            event("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } }),
            event("message_stop", { type: "message_stop" }),
          ]);
        }
        return json(200, {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-x",
          content: [{ type: "text", text: "hi" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 5, output_tokens: 2 },
        });
      }
      if (path === "/audio/transcriptions" || path === "/audio/translations") {
        if (/name="stream"\r\n\r\ntrue/.test(raw)) {
          return sse(res, [
            data({ type: "transcript.text.delta", delta: "hi" }),
            data({ type: "transcript.text.done", text: "hi", usage: { type: "tokens", input_tokens: 6, output_tokens: 2, total_tokens: 8 } }),
            data("[DONE]"),
          ]);
        }
        return json(200, { text: "hi", usage: { type: "tokens", input_tokens: 6, output_tokens: 2, total_tokens: 8 } });
      }
      if (path === "/audio/speech") {
        res.statusCode = 200;
        res.setHeader("content-type", "audio/mpeg");
        return res.end(SPEECH);
      }
      const video = (status: string) => ({
        id: "video_par_1",
        object: "video",
        model: "sora-2",
        status,
        progress: status === "completed" ? 100 : 0,
        created_at: 1770000000,
      });
      if (method === "POST" && path === "/videos") return json(200, video("queued"));
      if (method === "GET" && path === "/videos/video_par_1") return json(200, video("completed"));
      if (method === "GET" && path === "/videos/video_par_1/content") {
        res.statusCode = 200;
        res.setHeader("content-type", "video/mp4");
        return res.end(MP4);
      }
      // Files / batches / fine-tuning.
      if (method === "POST" && path === "/files") {
        return json(200, { id: "file-par-in", object: "file", purpose: "batch", filename: "in.jsonl", bytes: 1 });
      }
      if (method === "GET" && path === "/files") return json(200, { object: "list", data: [] });
      if (method === "GET" && path === "/files/file-par-out/content") {
        res.statusCode = 200;
        res.setHeader("content-type", "application/jsonl");
        return res.end(BATCH_OUTPUT);
      }
      if (method === "GET" && /^\/files\/[^/]+\/content$/.test(path)) {
        res.statusCode = 200;
        res.setHeader("content-type", "application/jsonl");
        return res.end("{}\n");
      }
      if (method === "GET" && path.startsWith("/files/")) {
        return json(200, { id: path.split("/")[2], object: "file", purpose: "batch", filename: "in.jsonl" });
      }
      if (method === "DELETE" && path.startsWith("/files/")) {
        return json(200, { id: path.split("/")[2], object: "file", deleted: true });
      }
      if (method === "POST" && path === "/batches") {
        return json(200, { id: "batch_par_1", object: "batch", status: "validating", endpoint: "/v1/chat/completions", input_file_id: body.input_file_id });
      }
      if (method === "GET" && path === "/batches") return json(200, { object: "list", data: [] });
      if (method === "GET" && path === "/batches/batch_par_1") {
        return json(200, { id: "batch_par_1", object: "batch", status: "completed", endpoint: "/v1/chat/completions", input_file_id: "file-par-in", output_file_id: "file-par-out" });
      }
      if (method === "POST" && path === "/batches/batch_par_1/cancel") {
        return json(200, { id: "batch_par_1", object: "batch", status: "cancelling" });
      }
      if (method === "POST" && path === "/fine_tuning/jobs") {
        return json(200, { id: "ftjob-par-1", object: "fine_tuning.job", model: body.model, training_file: body.training_file, status: "validating_files" });
      }
      if (method === "GET" && path === "/fine_tuning/jobs") return json(200, { object: "list", data: [] });
      if (method === "GET" && path === "/fine_tuning/jobs/ftjob-par-1") {
        return json(200, { id: "ftjob-par-1", object: "fine_tuning.job", status: "running" });
      }
      if (method === "POST" && path === "/fine_tuning/jobs/ftjob-par-1/cancel") {
        return json(200, { id: "ftjob-par-1", object: "fine_tuning.job", status: "cancelled" });
      }
      // Passthrough routes.
      if (path === "/agent/stream") {
        return sse(res, [data({ step: 1 }), data({ step: 2 })]);
      }
      return json(404, { error: { message: `no mock for ${method} ${path}` } });
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    close: () => new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve()))),
  };
}

/** Mock OpenAI Realtime upstream answering one usage-bearing frame. */
async function startRealtimeUpstream() {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.on("message", () => {
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: { usage: { input_tokens: 9, output_tokens: 4, input_token_details: { cached_tokens: 0 } } },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    close: () => new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

/** The `.route("…")` paths `build_router` registers, read from its source. */
function registeredRoutes(): string[] {
  const src = readFileSync(
    join(process.cwd(), "..", "..", "crates", "aisix-proxy", "src", "lib.rs"),
    "utf8",
  );
  const start = src.indexOf("pub fn build_router(");
  const end = src.indexOf(".fallback(", start);
  expect(start, "build_router not found").toBeGreaterThan(0);
  const body = src.slice(start, end);
  return [...body.matchAll(/\.route\(\s*"([^"]+)"/g)].map((m) => m[1]!);
}

// ── The spec ─────────────────────────────────────────────────────────────────

describe("every proxy route emits every per-request metric family", () => {
  let app: SpawnedApp | undefined;
  let upstream: Awaited<ReturnType<typeof startUpstream>> | undefined;
  let realtime: Awaited<ReturnType<typeof startRealtimeUpstream>> | undefined;
  let mcp: McpUpstream | undefined;
  let a2a: A2aUpstream | undefined;
  let etcdReachable = false;
  /** Filled by the jobs rows as they run, in table order. */
  const jobs = { fileId: "", batchId: "", ftId: "" };
  let videoId = "";

  const auth = (token = CALLER) => ({ authorization: `Bearer ${token}` });

  async function call(path: string, init: RequestInit = {}, token = CALLER): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      ...init,
      headers: { ...auth(token), ...(init.headers ?? {}) },
    });
    return res;
  }

  /** Drive one request and read its body to the end; require `status`. */
  async function expectCall(
    path: string,
    init: RequestInit = {},
    status = 200,
  ): Promise<string> {
    const res = await call(path, init);
    const text = await res.text();
    expect(res.status, `${init.method ?? "GET"} ${path}: ${text.slice(0, 300)}`).toBe(status);
    return text;
  }

  const postJson = (path: string, body: unknown, extra: Record<string, string> = {}) =>
    expectCall(path, {
      method: "POST",
      headers: { "content-type": "application/json", ...extra },
      body: JSON.stringify(body),
    });

  const form = (fields: Record<string, string>, file?: [string, string]) => {
    const f = new FormData();
    for (const [k, v] of Object.entries(fields)) f.set(k, v);
    if (file) f.set(file[0], new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), file[1]);
    return f;
  };

  const mcpRpc = (path: string, body: unknown) =>
    expectCall(path, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
      body: JSON.stringify(body),
    });
  const mcpToolCall = async (path: string) => {
    await mcpRpc(path, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2025-11-25", capabilities: {}, clientInfo: { name: "parity", version: "0.1" } },
    });
    await mcpRpc(path, { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "alpha__echo", arguments: { text: "hi" } } });
  };

  const a2aMessage = (method: string) => ({
    jsonrpc: "2.0",
    id: "1",
    method,
    params: { message: { role: "user", parts: [{ kind: "text", text: "invoice 42" }], messageId: "m-1" } },
  });

  const jobsModel = { "x-aisix-model": "par-jobs" };

  const ROWS: Row[] = [
    { route: "/v1/models", name: "GET /v1/models", endpoint: "/v1/models", families: [...REQUEST], drive: async () => void (await expectCall("/v1/models")) },
    {
      route: "/v1/chat/completions",
      name: "chat",
      endpoint: "/v1/chat/completions",
      model: "par-chat",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () => void (await postJson("/v1/chat/completions", { model: "par-chat", messages: [{ role: "user", content: "hi" }] })),
    },
    {
      route: "/v1/chat/completions",
      name: "chat stream",
      endpoint: "/v1/chat/completions",
      model: "par-chat",
      families: [...MODEL_ROUTE, "tokens", "ttft"],
      drive: async () =>
        void (await postJson("/v1/chat/completions", { model: "par-chat", stream: true, stream_options: { include_usage: true }, messages: [{ role: "user", content: "hi" }] })),
    },
    {
      route: "/v1/completions",
      name: "completions",
      endpoint: "/v1/completions",
      model: "par-completions",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () => void (await postJson("/v1/completions", { model: "par-completions", prompt: "hi" })),
    },
    {
      route: "/v1/embeddings",
      name: "embeddings",
      endpoint: "/v1/embeddings",
      model: "par-embeddings",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () => void (await postJson("/v1/embeddings", { model: "par-embeddings", input: "hi" })),
    },
    {
      route: "/v1/images/generations",
      name: "images generations",
      endpoint: "/v1/images/generations",
      model: "par-images",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () => void (await postJson("/v1/images/generations", { model: "par-images", prompt: "a cat" })),
    },
    {
      route: "/v1/images/edits",
      name: "images edits",
      endpoint: "/v1/images/edits",
      model: "par-images",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () =>
        void (await expectCall("/v1/images/edits", { method: "POST", body: form({ model: "par-images", prompt: "a hat" }, ["image", "a.png"]) })),
    },
    {
      route: "/v1/messages",
      name: "messages",
      endpoint: "/v1/messages",
      model: "par-claude",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () =>
        void (await postJson("/v1/messages", { model: "par-claude", max_tokens: 16, messages: [{ role: "user", content: "hi" }] }, { "anthropic-version": "2023-06-01" })),
    },
    {
      route: "/v1/messages",
      name: "messages stream",
      endpoint: "/v1/messages",
      model: "par-claude",
      families: [...MODEL_ROUTE, "tokens", "ttft"],
      drive: async () =>
        void (await postJson("/v1/messages", { model: "par-claude", max_tokens: 16, stream: true, messages: [{ role: "user", content: "hi" }] }, { "anthropic-version": "2023-06-01" })),
    },
    {
      // Counts a prompt; consumes no tokens.
      route: "/v1/messages/count_tokens",
      name: "count_tokens",
      endpoint: "/v1/messages/count_tokens",
      model: "par-claude",
      families: [...MODEL_ROUTE],
      drive: async () =>
        void (await postJson("/v1/messages/count_tokens", { model: "par-claude", messages: [{ role: "user", content: "hi" }] }, { "anthropic-version": "2023-06-01" })),
    },
    {
      route: "/v1/rerank",
      name: "rerank",
      endpoint: "/v1/rerank",
      model: "par-rerank",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () => void (await postJson("/v1/rerank", { model: "par-rerank", query: "q", documents: ["a"] })),
    },
    {
      route: "/v1/systemone",
      name: "systemone",
      endpoint: "/v1/systemone",
      model: "par-systemone",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () =>
        void (await postJson("/v1/systemone", {
          model: "par-systemone",
          state: "The API returns 500.",
          questions: { is_bug: { type: "noul", instructions: "Is this a bug?", criteria: { true: "misbehaves", false: null } } },
        })),
    },
    {
      route: "/v1/decisions",
      name: "decisions",
      endpoint: "/v1/decisions",
      model: "par-decisions",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () =>
        void (await postJson("/v1/decisions", {
          model: "par-decisions",
          input: "The API returns 500.",
          questions: [{ type: "predicate", name: "is_bug", instructions: "Is this a bug?" }],
        })),
    },
    {
      route: "/v1/responses",
      name: "responses",
      endpoint: "/v1/responses",
      model: "par-responses",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () => void (await postJson("/v1/responses", { model: "par-responses", input: "hi" })),
    },
    {
      route: "/v1/responses",
      name: "responses stream",
      endpoint: "/v1/responses",
      model: "par-responses",
      families: [...MODEL_ROUTE, "tokens", "ttft"],
      drive: async () => void (await postJson("/v1/responses", { model: "par-responses", input: "hi", stream: true })),
    },
    {
      route: "/v1/audio/transcriptions",
      name: "transcriptions",
      endpoint: "/v1/audio/transcriptions",
      model: "par-transcribe",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () =>
        void (await expectCall("/v1/audio/transcriptions", { method: "POST", body: form({ model: "par-transcribe" }, ["file", "a.mp3"]) })),
    },
    {
      route: "/v1/audio/transcriptions",
      name: "transcriptions stream",
      endpoint: "/v1/audio/transcriptions",
      model: "par-transcribe",
      families: [...MODEL_ROUTE, "tokens", "ttft"],
      drive: async () =>
        void (await expectCall("/v1/audio/transcriptions", { method: "POST", body: form({ model: "par-transcribe", stream: "true" }, ["file", "a.mp3"]) })),
    },
    {
      route: "/v1/audio/translations",
      name: "translations",
      endpoint: "/v1/audio/translations",
      model: "par-transcribe",
      families: [...MODEL_ROUTE, "tokens"],
      drive: async () =>
        void (await expectCall("/v1/audio/translations", { method: "POST", body: form({ model: "par-transcribe" }, ["file", "a.mp3"]) })),
    },
    {
      // Billed per character, reports no tokens; a chunked binary body is
      // not a stream of tokens, so no TTFT either.
      route: "/v1/audio/speech",
      name: "speech",
      endpoint: "/v1/audio/speech",
      model: "par-tts",
      families: [...MODEL_ROUTE],
      drive: async () => void (await postJson("/v1/audio/speech", { model: "par-tts", input: "hi", voice: "alloy" })),
    },
    {
      // Billed per video: no tokens.
      route: "/v1/videos",
      name: "video submit",
      endpoint: "/v1/videos",
      model: "par-video",
      families: [...MODEL_ROUTE],
      drive: async () => {
        const text = await postJson("/v1/videos", { model: "par-video", prompt: "a boat" });
        videoId = JSON.parse(text).id;
        expect(videoId).toBeTruthy();
      },
    },
    {
      // A poll files no usage event, so no upstream side; resolves no model
      // for the quota gate, so no remaining-quota gauge.
      route: "/v1/videos/:id",
      name: "video poll",
      endpoint: "/v1/videos/:id",
      model: "par-video",
      families: [...LLM, "deployment"],
      drive: async () => void (await expectCall(`/v1/videos/${videoId}`)),
    },
    {
      route: "/v1/videos/:id/content",
      name: "video content",
      endpoint: "/v1/videos/:id",
      model: "par-video",
      families: [...LLM, "deployment"],
      drive: async () => expect(await expectCall(`/v1/videos/${videoId}/content`)).toBe(MP4),
    },
    {
      // A WebSocket session: no request/response round trip to time and no
      // first token of a response.
      route: "/v1/realtime",
      name: "realtime",
      endpoint: "/v1/realtime",
      model: "par-realtime",
      families: ["proxy_requests", "llm_requests", "tokens", "deployment", "ratelimit_remaining"],
      drive: async () => {
        const ws = new WebSocket(`${app!.proxyUrl.replace("http", "ws")}/v1/realtime?model=par-realtime`, [
          "realtime",
          `openai-insecure-api-key.${CALLER}`,
        ]);
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
      },
    },
    // Files / batches / fine-tuning: management calls, outside the LLM tier.
    {
      route: "/v1/files",
      name: "file upload",
      endpoint: "/v1/files",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => {
        const text = await expectCall("/v1/files", {
          method: "POST",
          headers: jobsModel,
          body: (() => {
            const f = new FormData();
            f.set("purpose", "batch");
            f.set("file", new Blob(['{"custom_id":"r1"}\n']), "in.jsonl");
            return f;
          })(),
        });
        jobs.fileId = JSON.parse(text).id;
      },
    },
    {
      route: "/v1/files",
      name: "file list",
      endpoint: "/v1/files",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall("/v1/files", { headers: jobsModel })),
    },
    {
      route: "/v1/files/:id",
      name: "file get",
      endpoint: "/v1/files/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall(`/v1/files/${jobs.fileId}`)),
    },
    {
      route: "/v1/files/:id/content",
      name: "file content",
      endpoint: "/v1/files/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall(`/v1/files/${jobs.fileId}/content`)),
    },
    {
      route: "/v1/batches",
      name: "batch create",
      endpoint: "/v1/batches",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => {
        const text = await postJson("/v1/batches", { input_file_id: jobs.fileId, endpoint: "/v1/chat/completions", completion_window: "24h" });
        jobs.batchId = JSON.parse(text).id;
      },
    },
    {
      route: "/v1/batches",
      name: "batch list",
      endpoint: "/v1/batches",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall("/v1/batches", { headers: jobsModel })),
    },
    {
      // Retrieving a completed batch files its usage events, but its tokens
      // stay out of the token families: nothing makes that attribution
      // exactly-once across gateway replicas or restarts.
      route: "/v1/batches/:id",
      name: "batch get (completed)",
      endpoint: "/v1/batches/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall(`/v1/batches/${jobs.batchId}`)),
    },
    {
      route: "/v1/batches/:id/cancel",
      name: "batch cancel",
      endpoint: "/v1/batches/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await postJson(`/v1/batches/${jobs.batchId}/cancel`, {})),
    },
    {
      route: "/v1/fine_tuning/jobs",
      name: "ft create",
      endpoint: "/v1/fine_tuning/jobs",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => {
        const text = await postJson("/v1/fine_tuning/jobs", { model: "par-jobs", training_file: jobs.fileId });
        jobs.ftId = JSON.parse(text).id;
      },
    },
    {
      route: "/v1/fine_tuning/jobs",
      name: "ft list",
      endpoint: "/v1/fine_tuning/jobs",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall("/v1/fine_tuning/jobs", { headers: jobsModel })),
    },
    {
      route: "/v1/fine_tuning/jobs/:id",
      name: "ft get",
      endpoint: "/v1/fine_tuning/jobs/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall(`/v1/fine_tuning/jobs/${jobs.ftId}`)),
    },
    {
      route: "/v1/fine_tuning/jobs/:id/cancel",
      name: "ft cancel",
      endpoint: "/v1/fine_tuning/jobs/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await postJson(`/v1/fine_tuning/jobs/${jobs.ftId}/cancel`, {})),
    },
    {
      route: "/v1/files/:id",
      name: "file delete",
      endpoint: "/v1/files/:id",
      model: "par-jobs",
      families: [...REQUEST, "e2e_upstream", "deployment", "ratelimit_remaining"],
      drive: async () => void (await expectCall(`/v1/files/${jobs.fileId}`, { method: "DELETE" })),
    },
    // Discovery documents: no model call behind them.
    {
      route: "/.well-known/oauth-protected-resource",
      name: "protected-resource metadata",
      endpoint: "/.well-known/oauth-protected-resource",
      families: [...REQUEST],
      // Answered 404 while no MCP OAuth is configured — still a request.
      drive: async () => void (await expectCall("/.well-known/oauth-protected-resource", {}, 404)),
    },
    {
      route: "/.well-known/oauth-protected-resource/mcp",
      name: "protected-resource metadata (mcp)",
      endpoint: "/.well-known/oauth-protected-resource/mcp",
      families: [...REQUEST],
      drive: async () => void (await expectCall("/.well-known/oauth-protected-resource/mcp", {}, 404)),
    },
    // MCP: a tool call, no model and no token usage.
    { route: "/mcp", name: "mcp tools/call", endpoint: "/mcp", families: [...REQUEST, "e2e_upstream"], drive: () => mcpToolCall("/mcp") },
    { route: "/mcp/", name: "mcp/ tools/call", endpoint: "/mcp", families: [...REQUEST, "e2e_upstream"], drive: () => mcpToolCall("/mcp/") },
    { route: "/mcp/:server", name: "scoped mcp tools/call", endpoint: "/mcp/{server}", families: [...REQUEST, "e2e_upstream"], drive: () => mcpToolCall("/mcp/alpha") },
    // A2A: an agent call, no model; its tokens are the gateway's estimate.
    {
      route: "/a2a/:agent",
      name: "a2a message/send",
      endpoint: "/a2a",
      model: "a2a",
      families: [...REQUEST, "e2e_upstream", "tokens"],
      drive: async () => void (await postJson("/a2a/par-agent", a2aMessage("message/send"))),
    },
    {
      route: "/a2a/:agent",
      name: "a2a message/stream",
      endpoint: "/a2a",
      model: "a2a",
      families: [...REQUEST, "e2e_upstream", "tokens", "ttft"],
      drive: async () => void (await postJson("/a2a/par-agent", a2aMessage("message/stream"), { accept: "text/event-stream" })),
    },
    {
      route: "/a2a/:agent/.well-known/agent-card.json",
      name: "a2a agent card",
      endpoint: "/a2a/{agent}/.well-known/agent-card.json",
      families: [...REQUEST],
      drive: async () => void (await expectCall("/a2a/par-agent/.well-known/agent-card.json")),
    },
    // Passthrough routes (the router's fallback).
    {
      route: "<fallback>",
      name: "passthrough identified chat",
      endpoint: "/passthrough_route",
      // A route resolves no Model: its model label is the body's alias,
      // bounded to the configured names.
      model: "par-chat",
      families: [...LLM, "e2e_upstream", "tokens"],
      drive: async () =>
        void (await postJson("/passthrough/parity/chat/completions", { model: "par-chat", messages: [{ role: "user", content: "hi" }] })),
    },
    {
      // An opaque relay: no protocol to read tokens from, not model inference.
      route: "<fallback>",
      name: "passthrough opaque stream",
      endpoint: "/passthrough_route",
      families: [...REQUEST, "e2e_upstream", "ttft"],
      drive: async () => void (await postJson("/passthrough/parity/agent/stream", { payload: "opaque" })),
    },
  ];

  /** Routes that authenticate through the `AuthenticatedKey` extractor. */
  const AUTH_ROUTES: Array<[string, string, string]> = [
    ["GET", "/v1/models", "/v1/models"],
    ["POST", "/v1/chat/completions", "/v1/chat/completions"],
    ["POST", "/v1/completions", "/v1/completions"],
    ["POST", "/v1/embeddings", "/v1/embeddings"],
    ["POST", "/v1/images/generations", "/v1/images/generations"],
    ["POST", "/v1/images/edits", "/v1/images/edits"],
    ["POST", "/v1/messages", "/v1/messages"],
    ["POST", "/v1/messages/count_tokens", "/v1/messages/count_tokens"],
    ["POST", "/v1/rerank", "/v1/rerank"],
    ["POST", "/v1/systemone", "/v1/systemone"],
    ["POST", "/v1/decisions", "/v1/decisions"],
    ["POST", "/v1/responses", "/v1/responses"],
    ["POST", "/v1/audio/transcriptions", "/v1/audio/transcriptions"],
    ["POST", "/v1/audio/translations", "/v1/audio/translations"],
    ["POST", "/v1/audio/speech", "/v1/audio/speech"],
    ["POST", "/v1/videos", "/v1/videos"],
    ["GET", "/v1/videos/x", "/v1/videos/:id"],
    ["GET", "/v1/files", "/v1/files"],
    ["GET", "/v1/files/x", "/v1/files/:id"],
    ["GET", "/v1/batches", "/v1/batches"],
    ["GET", "/v1/batches/x", "/v1/batches/:id"],
    ["GET", "/v1/fine_tuning/jobs", "/v1/fine_tuning/jobs"],
    ["GET", "/v1/fine_tuning/jobs/x", "/v1/fine_tuning/jobs/:id"],
    ["POST", "/mcp", "/mcp"],
    ["POST", "/mcp/alpha", "/mcp/{server}"],
    ["POST", "/a2a/par-agent", "/a2a"],
    ["GET", "/a2a/par-agent/.well-known/agent-card.json", "/a2a/{agent}/.well-known/agent-card.json"],
  ];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startUpstream();
    realtime = await startRealtimeUpstream();
    mcp = await startMcpUpstream("alpha");
    a2a = await startA2aUpstream({ streamAnswer: true });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const openaiPk = await seed.createProviderKey({ display_name: "par-openai-pk", secret: "sk-mock", api_base: `${upstream.baseUrl}/v1`, provider: "openai" });
    const soraPk = await seed.createProviderKey({ display_name: "par-sora-pk", secret: "sk-mock", api_base: upstream.baseUrl, provider: "openai" });
    const anthropicPk = await seed.createProviderKey({ display_name: "par-anthropic-pk", secret: "sk-ant", api_base: upstream.baseUrl });
    const typesafePk = await seed.createProviderKey({ display_name: "par-typesafe-pk", provider: "typesafe", adapter: undefined, secret: "ts", api_base: upstream.baseUrl });
    const realtimePk = await seed.createProviderKey({ display_name: "par-realtime-pk", secret: "sk-mock", api_base: `http://127.0.0.1:${realtime.port}/v1` });

    const model = (display_name: string, model_name: string, provider_key_id: string, provider = "openai") =>
      seed.createModel({ display_name, provider, model_name, provider_key_id });
    await model("par-chat", "gpt-4o", openaiPk.id);
    await model("par-completions", "gpt-3.5-turbo-instruct", openaiPk.id);
    await model("par-embeddings", "text-embedding-3-small", openaiPk.id);
    await model("par-images", "gpt-image-1", openaiPk.id);
    await model("par-claude", "claude-x", anthropicPk.id, "anthropic");
    await model("par-rerank", "rerank-1", openaiPk.id);
    await model("par-systemone", "jev-latest", typesafePk.id, "typesafe");
    await model("par-decisions", "gpt-6-luna", openaiPk.id);
    await model("par-responses", "gpt-4o", openaiPk.id);
    await model("par-transcribe", "gpt-4o-transcribe", openaiPk.id);
    await model("par-tts", "tts-1", openaiPk.id);
    await model("par-video", "sora-2", soraPk.id);
    await model("par-realtime", "gpt-realtime-mock", realtimePk.id);
    await model("par-jobs", "gpt-4o", openaiPk.id);

    await seed.update("mcp_servers", randomUUID(), { display_name: "alpha", url: mcp.url, enabled: true });
    await seed.update("a2a_agents", randomUUID(), { name: "par-agent", url: a2a.url, protocol_version: "1.0", auth_type: "none", enabled: true });
    await seed.createPassthroughRoute({ name: "par-ptr", path_prefix: "/passthrough/parity", target_url: `${upstream.baseUrl}/v1`, provider_key_id: openaiPk.id });

    // Seeded last: its authenticating implies everything above landed. A
    // generous key-level limit makes the remaining-quota gauges exist.
    await seed.createApiKey({
      key_hash: sha256(CALLER),
      allowed_models: ["*"],
      allowed_agents: ["*"],
      allowed_routes: ["par-ptr"],
      mcp_access: { allow: ["*"] },
      rate_limit: { rpm: 1_000_000 },
    });
    const probe = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await realtime?.close();
    await mcp?.close();
    await a2a?.close();
  });

  const scrape = () => scrapeMetrics(app!.metricsUrl);

  /** How far `family` moved for `row` between two scrapes. */
  function moved(row: Row, family: Family, before: MetricSample[], after: MetricSample[]): number {
    const d = (name: string, want: Record<string, string>) => metricDelta(before, after, name, want);
    const ep = { endpoint: row.endpoint };
    switch (family) {
      case "proxy_requests":
        return d("aisix_proxy_requests_total", ep);
      case "llm_requests":
        return d("aisix_llm_requests_total", ep);
      case "e2e_downstream":
        return d("aisix_request_e2e_latency_seconds_count", { ...ep, side: "downstream" });
      case "e2e_upstream":
        return d("aisix_request_e2e_latency_seconds_count", { ...ep, side: "upstream" });
      case "ttft":
        return Math.min(
          d("aisix_request_ttft_seconds_count", { ...ep, side: "upstream" }),
          d("aisix_llm_time_to_first_token_seconds_count", { ...ep, side: "upstream" }),
        );
      case "tokens": {
        const model = { model: row.model ?? "" };
        return Math.min(
          d("aisix_llm_input_tokens_total", ep),
          d("aisix_tokens_consumed_total", model),
          d("aisix_llm_tokens_by_client_total", model),
        );
      }
      case "deployment":
        return d("aisix_deployment_requests_total", { model: row.model ?? "" });
      case "ratelimit_remaining":
        // A gauge: present for the caller and the row's model after the drive.
        return after.some(
          (s) => s.name === "aisix_ratelimit_remaining_requests" && s.labels.model === row.model,
        )
          ? 1
          : 0;
    }
  }

  test("every registered route has a row", () => {
    const covered = new Set(ROWS.map((r) => r.route));
    // Platform probes are not client traffic; they are pinned below instead.
    const probes = new Set(["/livez", "/readyz"]);
    const missing = registeredRoutes().filter((r) => !covered.has(r) && !probes.has(r));
    expect(missing, "routes without a row in the parity table").toEqual([]);
  });

  test("each route moves each family it must emit", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const failures: string[] = [];
    for (const row of ROWS) {
      const before = await scrape();
      await row.drive();
      // Streamed and detached emits land after the client's last read.
      let after = await scrape();
      const deadline = Date.now() + 5_000;
      while (row.families.some((f) => moved(row, f, before, after) <= 0) && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 100));
        after = await scrape();
      }
      for (const family of row.families) {
        if (moved(row, family, before, after) <= 0) failures.push(`${row.name}: ${family}`);
      }
    }
    expect(failures, "route × family cells that did not move").toEqual([]);
  }, 180_000);

  test("an authentication denial is counted on every extractor-authenticated route", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const failures: string[] = [];
    for (const [method, path, endpoint] of AUTH_ROUTES) {
      const before = await scrape();
      const res = await call(
        path,
        method === "POST" ? { method, headers: { "content-type": "application/json" }, body: "{}" } : { method },
        "sk-metric-parity-not-a-key",
      );
      await res.text();
      expect(res.status, `${method} ${path}`).toBe(401);
      const after = await scrape();
      if (metricDelta(before, after, "aisix_proxy_requests_total", { endpoint, status: "401" }) <= 0) {
        failures.push(`${method} ${path}`);
      }
    }
    expect(failures, "401s missing from aisix_proxy_requests_total").toEqual([]);
  }, 60_000);

  test("platform probes are not counted as requests", async (ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = await scrape();
    for (const probe of ["/livez", "/readyz"]) {
      const res = await fetch(`${app.proxyUrl}${probe}`);
      await res.text();
    }
    const after = await scrape();
    for (const endpoint of ["/livez", "/readyz"]) {
      expect(metricDelta(before, after, "aisix_proxy_requests_total", { endpoint }), endpoint).toBe(0);
      expect(
        metricDelta(before, after, "aisix_request_e2e_latency_seconds_count", { endpoint }),
        endpoint,
      ).toBe(0);
    }
  });
});
