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

/**
 * Decompress PutLogs bodies for a logstore and join as one searchable string.
 *
 * `fromIndex` skips everything the mock had already received at that point.
 * A test that drives warm-up traffic to wait for config propagation exports
 * those warm-up requests too, and they were served by whatever config was
 * live at the time — asserting over them means asserting about a
 * deliberately-unconverged gateway. Snapshot `sls.requests.length` once the
 * config is confirmed live and pass it here.
 */
export function decodedTextFor(
  sls: MockSls,
  logstore: string,
  fromIndex = 0,
): string {
  return sls.requests
    .slice(fromIndex)
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

/**
 * Poll until the decoded logstore text contains `token` (or time out).
 * `fromIndex` has the same meaning as in [`decodedTextFor`].
 */
export async function waitForToken(
  sls: MockSls,
  logstore: string,
  token: string,
  timeoutMs = 10_000,
  fromIndex = 0,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (decodedTextFor(sls, logstore, fromIndex).includes(token)) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`token '${token}' not seen in logstore '${logstore}' within ${timeoutMs}ms`);
}

// --- LogGroup protobuf reader (see aisix-obs sink/sls.rs encoder) --------
// LogGroup { Logs = 1 (message) { Time = 1 (varint), Contents = 2 (message)
// { Key = 1 (string), Value = 2 (string) } } }; unknown fields skipped.

function readVarint(buf: Buffer, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  for (;;) {
    // Past the end `buf[pos]` is `undefined`, and `undefined & 0x80` is 0 —
    // so the loop would exit with a wrong value and an advanced position,
    // and the mis-parse would surface much later as an opaque
    // `waitForSlsLog` timeout instead of naming the truncated payload.
    if (pos >= buf.length) {
      throw new Error(`truncated varint at offset ${pos} of ${buf.length} bytes`);
    }
    const b = buf[pos]!;
    pos += 1;
    result += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) return [result, pos];
    shift += 7;
  }
}

function skipField(buf: Buffer, pos: number, wireType: number): number {
  if (wireType === 0) return readVarint(buf, pos)[1];
  if (wireType === 2) {
    const [len, p] = readVarint(buf, pos);
    return p + len;
  }
  if (wireType === 5) return pos + 4;
  if (wireType === 1) return pos + 8;
  throw new Error(`unsupported wire type ${wireType}`);
}

function parseContentPair(buf: Buffer): [string, string] {
  let pos = 0;
  let key = "";
  let value = "";
  while (pos < buf.length) {
    const [tag, p] = readVarint(buf, pos);
    pos = p;
    const field = tag >>> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const [len, q] = readVarint(buf, pos);
      const bytes = buf.subarray(q, q + len);
      pos = q + len;
      if (field === 1) key = bytes.toString("utf8");
      else if (field === 2) value = bytes.toString("utf8");
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }
  return [key, value];
}

function parseLog(buf: Buffer): Map<string, string> {
  const out = new Map<string, string>();
  let pos = 0;
  while (pos < buf.length) {
    const [tag, p] = readVarint(buf, pos);
    pos = p;
    const field = tag >>> 3;
    const wireType = tag & 7;
    if (field === 2 && wireType === 2) {
      const [len, q] = readVarint(buf, pos);
      const [k, v] = parseContentPair(buf.subarray(q, q + len));
      out.set(k, v);
      pos = q + len;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }
  return out;
}

/**
 * Every log delivered to `logstore`, decoded into flat key→value maps.
 *
 * [`decodedTextFor`] answers "did this token reach the logstore"; this
 * answers "what does THIS request's row say", which a substring search
 * cannot — a field like `guardrail_blocked` is present on every row, so
 * only a per-record read can tell one row's value from another's.
 *
 * `fromIndex` has the same meaning as in [`decodedTextFor`].
 */
export function slsLogsFor(
  sls: MockSls,
  logstore: string,
  fromIndex = 0,
): Map<string, string>[] {
  const logs: Map<string, string>[] = [];
  for (const r of sls.requests.slice(fromIndex)) {
    if (r.logstore !== logstore || r.rawSize === 0 || r.body.length === 0) continue;
    const group = lz4DecompressBlock(r.body, r.rawSize);
    let pos = 0;
    while (pos < group.length) {
      const [tag, p] = readVarint(group, pos);
      pos = p;
      const field = tag >>> 3;
      const wireType = tag & 7;
      if (field === 1 && wireType === 2) {
        const [len, q] = readVarint(group, pos);
        logs.push(parseLog(group.subarray(q, q + len)));
        pos = q + len;
      } else {
        pos = skipField(group, pos, wireType);
      }
    }
  }
  return logs;
}

/** Poll until a `logstore` record matching `pred` arrives (or time out). */
export async function waitForSlsLog(
  sls: MockSls,
  logstore: string,
  pred: (log: Map<string, string>) => boolean,
  what: string,
  timeoutMs = 10_000,
): Promise<Map<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = slsLogsFor(sls, logstore).find(pred);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`no SLS log in '${logstore}' matching: ${what}`);
}
