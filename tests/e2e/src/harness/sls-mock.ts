import { createServer, type Server } from "node:http";
import { pickFreePort } from "./ports.js";

/**
 * Mock Aliyun SLS PutLogs endpoint shared by the content-capture e2e suites
 * (#687 chat/messages, AISIX-Cloud#947 responses/completions). Captures each
 * PutLogs body (lz4 block compressed) per logstore so tests can decompress
 * and search for planted tokens.
 */

export interface CapturedPutLogs {
  logstore: string;
  rawSize: number;
  body: Buffer;
}

export interface MockSls {
  url: string;
  requests: CapturedPutLogs[];
  close(): Promise<void>;
}

/** Decompress an lz4 *block* (no frame header) given the raw output size. */
export function lz4DecompressBlock(src: Buffer, rawSize: number): Buffer {
  const dest = Buffer.alloc(rawSize);
  let s = 0;
  let d = 0;
  while (s < src.length) {
    const token = src[s++];
    let litLen = token >> 4;
    if (litLen === 15) {
      let b: number;
      do {
        b = src[s++];
        litLen += b;
      } while (b === 255);
    }
    src.copy(dest, d, s, s + litLen);
    s += litLen;
    d += litLen;
    if (s >= src.length) break;
    const offset = src[s] | (src[s + 1] << 8);
    s += 2;
    let matchLen = token & 0x0f;
    if (matchLen === 15) {
      let b: number;
      do {
        b = src[s++];
        matchLen += b;
      } while (b === 255);
    }
    matchLen += 4;
    let m = d - offset;
    for (let i = 0; i < matchLen; i++) {
      dest[d++] = dest[m++];
    }
  }
  return dest;
}

function logstoreFromPath(path: string): string {
  // /logstores/<logstore>/shards/lb
  const m = path.match(/^\/logstores\/([^/]+)\/shards\/lb$/);
  return m ? m[1] : "";
}

export async function startMockSls(): Promise<MockSls> {
  const requests: CapturedPutLogs[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const logstore = logstoreFromPath((req.url ?? "").split("?")[0]);
      if (logstore) {
        const rawSizeHeader = req.headers["x-log-bodyrawsize"];
        const rawSize = Number(Array.isArray(rawSizeHeader) ? rawSizeHeader[0] : rawSizeHeader);
        requests.push({ logstore, rawSize, body: Buffer.concat(chunks) });
      }
      res.statusCode = 200;
      res.end();
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

/** Decompress every PutLogs body for a logstore and join as one searchable string. */
export function decodedTextFor(sls: MockSls, logstore: string): string {
  return sls.requests
    .filter((r) => r.logstore === logstore && r.rawSize > 0 && r.body.length > 0)
    .map((r) => lz4DecompressBlock(r.body, r.rawSize).toString("utf8"))
    .join(" ");
}

export async function waitForLogstore(
  sls: MockSls,
  logstore: string,
  timeoutMs = 10_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (sls.requests.some((r) => r.logstore === logstore)) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no PutLogs to logstore '${logstore}' within ${timeoutMs}ms`);
}

/** Poll until the decoded logstore text contains `token` (or time out). */
export async function waitForToken(
  sls: MockSls,
  logstore: string,
  token: string,
  timeoutMs = 10_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (decodedTextFor(sls, logstore).includes(token)) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`token '${token}' not seen in logstore '${logstore}' within ${timeoutMs}ms`);
}
