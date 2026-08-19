import { createServer, type Server } from "node:http";

/**
 * Mock OTLP/HTTP trace receiver: accepts the `ExportTraceServiceRequest`
 * bodies the `otlp_http` observability exporter POSTs (JSON, uncompressed) and
 * keeps every span it was sent, flattened, so a test can assert what a real
 * trace backend would have seen.
 */

/** One exported span, with its attributes flattened to a plain object. */
export interface CapturedSpan {
  name: string;
  attributes: Record<string, string | number | boolean>;
  /** 32-hex W3C trace id; empty when the body omitted it. */
  traceId: string;
  /** 16-hex span id; empty when the body omitted it. */
  spanId: string;
  /** 16-hex parent span id; empty for a root span. */
  parentSpanId: string;
  /** OTLP SpanKind (2 = SERVER, 3 = CLIENT); 0 when omitted. */
  kind: number;
  /** OTLP span flags (uint32); 0 when omitted. */
  flags: number;
  /** W3C tracestate carried on the span; empty when absent. */
  traceState: string;
  /** Nanosecond boundaries, as strings (OTLP/JSON int64 encoding). */
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  /** 0-based index of the POST this span arrived in (delivery-retry tests). */
  postIndex: number;
}

export interface MockOtlpOptions {
  /**
   * Respond to the first N POSTs with 503 (their spans are still recorded),
   * then accept. Models a transiently failing receiver so a test can assert
   * the sink's delivery retry re-sends byte-identical spans.
   */
  failFirst?: number;
}

export interface MockOtlp {
  url: string;
  spans: CapturedSpan[];
  /** Number of POSTs received so far (including ones answered 503). */
  posts: number;
  /**
   * Bodies this receiver could not parse, empty on a healthy run. Without it a
   * malformed export and a never-sent one look identical: both end as "no
   * matching span arrived".
   */
  parseFailures: string[];
  close(): Promise<void>;
}

/** Read an OTLP `AnyValue` back to the scalar it wraps. */
function anyValue(value: Record<string, unknown>): string | number | boolean {
  if (typeof value.stringValue === "string") return value.stringValue;
  if (typeof value.intValue === "string") return Number(value.intValue);
  if (typeof value.intValue === "number") return value.intValue;
  if (typeof value.doubleValue === "number") return value.doubleValue;
  if (typeof value.boolValue === "boolean") return value.boolValue;
  if (value.arrayValue !== undefined) {
    const values = (value.arrayValue as { values?: Record<string, unknown>[] })
      .values;
    return (values ?? []).map((v) => String(anyValue(v))).join(",");
  }
  return "";
}

export async function startMockOtlp(
  options: MockOtlpOptions = {},
): Promise<MockOtlp> {
  const spans: CapturedSpan[] = [];
  const parseFailures: string[] = [];
  const state = { posts: 0, failuresLeft: options.failFirst ?? 0 };
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const postIndex = state.posts;
      state.posts += 1;
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
          resourceSpans?: {
            scopeSpans?: {
              spans?: {
                name: string;
                traceId?: string;
                spanId?: string;
                parentSpanId?: string;
                kind?: number;
                flags?: number;
                traceState?: string;
                startTimeUnixNano?: string;
                endTimeUnixNano?: string;
                attributes?: { key: string; value: Record<string, unknown> }[];
              }[];
            }[];
          }[];
        };
        for (const resource of body.resourceSpans ?? []) {
          for (const scope of resource.scopeSpans ?? []) {
            for (const span of scope.spans ?? []) {
              const attributes: Record<string, string | number | boolean> = {};
              for (const attr of span.attributes ?? []) {
                attributes[attr.key] = anyValue(attr.value);
              }
              spans.push({
                name: span.name,
                attributes,
                traceId: span.traceId ?? "",
                spanId: span.spanId ?? "",
                parentSpanId: span.parentSpanId ?? "",
                kind: span.kind ?? 0,
                flags: span.flags ?? 0,
                traceState: span.traceState ?? "",
                startTimeUnixNano: span.startTimeUnixNano ?? "0",
                endTimeUnixNano: span.endTimeUnixNano ?? "0",
                postIndex,
              });
            }
          }
        }
      } catch (err) {
        // A body this receiver cannot parse is a failure of the thing under
        // test, so it is kept rather than swallowed — otherwise it reaches the
        // test as an indistinguishable "no span arrived".
        parseFailures.push(String(err));
      }
      if (state.failuresLeft > 0) {
        state.failuresLeft -= 1;
        res.writeHead(503, { "content-type": "application/json" });
        res.end("{}");
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{}");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("otlp mock: no listen address");
  }
  return {
    url: `http://127.0.0.1:${address.port}/v1/traces`,
    spans,
    get posts() {
      return state.posts;
    },
    parseFailures,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}
