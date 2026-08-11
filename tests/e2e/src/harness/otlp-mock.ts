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
}

export interface MockOtlp {
  url: string;
  spans: CapturedSpan[];
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

export async function startMockOtlp(): Promise<MockOtlp> {
  const spans: CapturedSpan[] = [];
  const parseFailures: string[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
          resourceSpans?: {
            scopeSpans?: {
              spans?: {
                name: string;
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
              spans.push({ name: span.name, attributes });
            }
          }
        }
      } catch (err) {
        // A body this receiver cannot parse is a failure of the thing under
        // test, so it is kept rather than swallowed — otherwise it reaches the
        // test as an indistinguishable "no span arrived".
        parseFailures.push(String(err));
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
    parseFailures,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}
