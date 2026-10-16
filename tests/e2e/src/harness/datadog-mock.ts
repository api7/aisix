import { createServer, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { pickFreePort } from "./ports.js";

const INTAKE_PATH = "/api/v2/logs";

/** A mock Datadog Logs intake: every gunzipped log object it received. */
export interface MockDatadog {
  /** `host:port` for the exporter's `site`. */
  site: string;
  logs: Record<string, unknown>[];
  close(): Promise<void>;
}

export async function startMockDatadog(): Promise<MockDatadog> {
  const logs: Record<string, unknown>[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      if (req.method === "POST" && (req.url ?? "").split("?")[0] === INTAKE_PATH) {
        try {
          const parsed: unknown = JSON.parse(gunzipSync(Buffer.concat(chunks)).toString("utf8"));
          if (Array.isArray(parsed)) logs.push(...(parsed as Record<string, unknown>[]));
        } catch {
          // An undecodable body leaves nothing captured; the caller's wait fails.
        }
      }
      res.statusCode = 202;
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    site: `127.0.0.1:${port}`,
    logs,
    async close() {
      await new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve())));
    },
  };
}

/** Poll until a log matching `pred` arrives (or time out). */
export async function waitForDatadogLog(
  dd: MockDatadog,
  pred: (log: Record<string, unknown>) => boolean,
  what: string,
  timeoutMs = 10_000,
): Promise<Record<string, unknown>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = dd.logs.find(pred);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no Datadog log matching: ${what}`);
}
