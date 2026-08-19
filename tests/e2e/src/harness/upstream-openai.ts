import { createServer, type Server } from "node:http";
import { createServer as createTlsServer } from "node:https";
import { pickFreePort } from "./ports.js";

export interface OpenAiUpstreamOptions {
  /** Returned for non-streaming chat/completions. */
  nonStreamBody?: unknown;
  /** Sequence of SSE event payloads (already-stringified JSON or `[DONE]`). */
  streamEvents?: string[];
  /**
   * Complete SSE frames written VERBATIM, terminators included. Unlike
   * `streamEvents` (which the mock wraps in `data: …\n\n`) this lets a spec
   * reproduce a server that labels its frames — e.g. the `event: token_usage`
   * frame an IDE agent backend reports its token counts on. Takes precedence
   * over `streamEvents` when both are set.
   */
  rawStreamFrames?: string[];
  /** Inserted before the response is written (delays status + headers). */
  responseDelayMs?: number;
  /**
   * Inserted AFTER the SSE headers are flushed but BEFORE the first event.
   * Models "connection + headers fast, first token slow", i.e. the TTFT
   * timeout scenario (#554). Distinct from `responseDelayMs` (delays the
   * headers too) and `eventDelayMs` (only applies between events).
   */
  firstEventDelayMs?: number;
  /** Inserted between SSE events. */
  eventDelayMs?: number;
  /** Status code to return (default 200). */
  status?: number;
  /** Body to return when `status` >= 400. */
  errorBody?: unknown;
  /**
   * Content-Type for the error body (default `application/json`). Lets
   * tests reproduce upstreams / edge layers that return a JSON error
   * body labelled with a non-JSON Content-Type (e.g. OpenAI's 401
   * `invalid_api_key` path — see #543).
   */
  errorContentType?: string;
  /** Drop the connection after writing this many SSE events. */
  disconnectAfterEvents?: number;
  /**
   * Raw (non-JSON) 200 response body — e.g. MP4 bytes for the `/v1/videos`
   * content-proxy path. When set (and `status` < 400), the reply is these
   * bytes with `rawContentType` (default `application/octet-stream`) and an
   * auto Content-Length, instead of the JSON body. Lets a test assert the
   * gateway streams provider bytes back and injected the provider bearer on
   * the content GET.
   */
  rawBody?: string;
  /**
   * `rawBody` split into chunks written one at a time, `eventDelayMs`
   * apart, so a spec can tell a relayed body from a buffered one: an
   * upstream that trickles its bytes is only visibly trickling
   * downstream if the gateway forwards them as they arrive. Takes
   * precedence over `rawBody`; no Content-Length is sent.
   */
  rawBodyChunks?: string[];
  /** Content-Type for `rawBody` (default `application/octet-stream`). */
  rawContentType?: string;
  /** Per-request response script; used in order before static opts. */
  scriptedResponses?: OpenAiUpstreamStep[];
  /**
   * Extra response headers to set on every reply. Used by the cooldown
   * contract tests to assert that the gateway honors `Retry-After`
   * from the upstream when computing the cooldown TTL.
   */
  responseHeaders?: Record<string, string>;
  /**
   * Serve HTTPS with this key/cert pair (PEM) instead of plain HTTP, so
   * `baseUrl` is `https://…`. Used by the outbound-TLS specs to stand up
   * an upstream whose certificate is signed by a private CA the gateway
   * does not trust out of the box.
   */
  tls?: { key: string | Buffer; cert: string | Buffer };
}

export interface OpenAiUpstreamStep {
  nonStreamBody?: unknown;
  streamEvents?: string[];
  /** See `OpenAiUpstreamOptions.rawStreamFrames`. */
  rawStreamFrames?: string[];
  responseDelayMs?: number;
  /** See `OpenAiUpstreamOptions.firstEventDelayMs`. */
  firstEventDelayMs?: number;
  eventDelayMs?: number;
  status?: number;
  errorBody?: unknown;
  /** Content-Type for the error body (default `application/json`). See #543. */
  errorContentType?: string;
  disconnectAfterEvents?: number;
  /** Extra response headers, same semantics as on the top-level options. */
  responseHeaders?: Record<string, string>;
  /** Raw (non-JSON) 200 body — see `OpenAiUpstreamOptions.rawBody`. */
  rawBody?: string;
  /** See `OpenAiUpstreamOptions.rawBodyChunks`. */
  rawBodyChunks?: string[];
  /** Content-Type for `rawBody` (default `application/octet-stream`). */
  rawContentType?: string;
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

  const handler = (
    req: import("node:http").IncomingMessage,
    res: import("node:http").ServerResponse,
  ) => {
    // When the gateway abandons a slow upstream (e.g. a #554 request/stream
    // timeout fires and the client connection is dropped), a later
    // `res.write`/`res.end` here would emit an error on a closed socket.
    // Swallow it so a deliberately-slow mock can't surface as an unhandled
    // exception that fails the run.
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", async () => {
      const step = opts.scriptedResponses?.[requestIndex++] ?? opts;
      received.push({
        method: req.method ?? "GET",
        path: req.url ?? "/",
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [
            k,
            Array.isArray(v) ? v.join(",") : (v ?? ""),
          ]),
        ),
        body: raw,
      });

      if (step.responseDelayMs) await sleep(step.responseDelayMs);

      const extraHeaders = {
        ...(opts.responseHeaders ?? {}),
        ...(step.responseHeaders ?? {}),
      };
      for (const [k, v] of Object.entries(extraHeaders)) {
        res.setHeader(k, v);
      }

      const status = step.status ?? 200;
      if (status >= 400) {
        res.statusCode = status;
        res.setHeader(
          "content-type",
          step.errorContentType ?? opts.errorContentType ?? "application/json",
        );
        res.end(
          JSON.stringify(
            step.errorBody ?? { error: { message: "mock error" } },
          ),
        );
        return;
      }

      if (step.rawBodyChunks !== undefined) {
        res.statusCode = 200;
        res.setHeader(
          "content-type",
          step.rawContentType ?? "application/octet-stream",
        );
        res.flushHeaders();
        for (const chunk of step.rawBodyChunks) {
          if (res.writableEnded || res.destroyed) return;
          res.write(Buffer.from(chunk));
          if (step.eventDelayMs) await sleep(step.eventDelayMs);
        }
        if (!res.writableEnded && !res.destroyed) res.end();
        return;
      }

      if (step.rawBody !== undefined) {
        res.statusCode = 200;
        res.setHeader(
          "content-type",
          step.rawContentType ?? "application/octet-stream",
        );
        // Buffer.from sets Content-Length automatically, exercising the
        // gateway's content-length relay on the proxy path.
        res.end(Buffer.from(step.rawBody));
        return;
      }

      const isStream = !!step.streamEvents || !!step.rawStreamFrames;
      if (isStream) {
        res.statusCode = 200;
        res.setHeader("content-type", "text/event-stream");
        res.setHeader("cache-control", "no-cache");
        // Flush the 200 + headers immediately so the gateway's connect
        // phase completes fast; `firstEventDelayMs` then models a slow
        // first token (TTFT timeout) independently of the headers (#554).
        res.flushHeaders();
        if (step.firstEventDelayMs) await sleep(step.firstEventDelayMs);
        // Verbatim frames carry their own `event:`/`data:` lines and
        // terminators; `streamEvents` are payloads the mock wraps.
        const raw = step.rawStreamFrames;
        const events = raw ?? step.streamEvents ?? [];
        for (let i = 0; i < events.length; i++) {
          // The gateway may have abandoned a stalled stream (#554 read
          // timeout) and closed the connection; stop writing rather than
          // throwing on a dead socket.
          if (res.writableEnded || res.destroyed) return;
          if (
            step.disconnectAfterEvents !== undefined &&
            i >= step.disconnectAfterEvents
          ) {
            res.destroy();
            return;
          }
          res.write(raw ? events[i] : `data: ${events[i]}\n\n`);
          if (step.eventDelayMs) await sleep(step.eventDelayMs);
        }
        if (!res.writableEnded && !res.destroyed) res.end();
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
  };

  const server: Server = opts.tls
    ? createTlsServer({ key: opts.tls.key, cert: opts.tls.cert }, handler)
    : createServer(handler);

  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  const baseUrl = `${opts.tls ? "https" : "http"}://127.0.0.1:${port}`;

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
