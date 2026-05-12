import { createServer, type Server } from "node:http";
import { pickFreePort } from "./ports.js";

export interface OpenAiUpstreamOptions {
  /** Returned for non-streaming chat/completions. */
  nonStreamBody?: unknown;
  /** Sequence of SSE event payloads (already-stringified JSON or `[DONE]`). */
  streamEvents?: string[];
  /** Inserted before the response is written. */
  responseDelayMs?: number;
  /** Inserted between SSE events. */
  eventDelayMs?: number;
  /** Status code to return (default 200). */
  status?: number;
  /** Body to return when `status` >= 400. */
  errorBody?: unknown;
  /** Drop the connection after writing this many SSE events. */
  disconnectAfterEvents?: number;
  /** Per-request response script; used in order before static opts. */
  scriptedResponses?: OpenAiUpstreamStep[];
}

export interface OpenAiUpstreamStep {
  nonStreamBody?: unknown;
  streamEvents?: string[];
  responseDelayMs?: number;
  eventDelayMs?: number;
  status?: number;
  errorBody?: unknown;
  disconnectAfterEvents?: number;
}

export interface OpenAiUpstream {
  baseUrl: string;
  receivedRequests: ReceivedRequest[];
  close(): Promise<void>;
}

export interface ReceivedRequest {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: string;
}

/**
 * Spins a node http server that mimics the OpenAI surface tightly enough
 * for our tests: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
 * `/v1/models`, `/v1/responses`, `/v1/rerank`. All routes echo the same
 * canned response, so a single mock can serve any endpoint the test cares
 * about.
 */
export async function startOpenAiUpstream(
  opts: OpenAiUpstreamOptions = {},
): Promise<OpenAiUpstream> {
  const received: ReceivedRequest[] = [];
  let requestIndex = 0;

  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", async () => {
      const step = opts.scriptedResponses?.[requestIndex++] ?? opts;
      received.push({
        method: req.method ?? "GET",
        path: req.url ?? "/",
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(",") : (v ?? "")]),
        ),
        body: raw,
      });

      if (step.responseDelayMs) await sleep(step.responseDelayMs);

      const status = step.status ?? 200;
      if (status >= 400) {
        res.statusCode = status;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(step.errorBody ?? { error: { message: "mock error" } }));
        return;
      }

      const isStream = !!step.streamEvents;
      if (isStream) {
        res.statusCode = 200;
        res.setHeader("content-type", "text/event-stream");
        res.setHeader("cache-control", "no-cache");
        const events = step.streamEvents ?? [];
        for (let i = 0; i < events.length; i++) {
          if (
            step.disconnectAfterEvents !== undefined &&
            i >= step.disconnectAfterEvents
          ) {
            res.destroy();
            return;
          }
          res.write(`data: ${events[i]}\n\n`);
          if (step.eventDelayMs) await sleep(step.eventDelayMs);
        }
        res.end();
        return;
      }

      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify(
          step.nonStreamBody ?? {
            id: "mock-1",
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: "mock-model",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: "mock reply" },
                finish_reason: "stop",
              },
            ],
            usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
          },
        ),
      );
    });
  });

  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${port}`;

  return {
    baseUrl,
    receivedRequests: received,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
