import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  lz4DecompressBlock,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogstore,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1014: when a captured prompt/response exceeds the exporter's
// `content_max_bytes`, the logged field must stay VALID JSON — structure
// preserved, long arrays sampled head+tail around an explicit
// `{"_aisix_truncated": true, "omitted_items": N}` placeholder — instead of
// the old blunt byte cut that left half an object behind. This drives a
// real `aisix` binary + etcd + mock upstream, captures through a real
// `aliyun_sls` exporter (content_mode=full, tiny cap) into a mock SLS
// endpoint, decodes the delivered protobuf, and JSON-parses the `prompt`
// field back.

const CALLER_PLAINTEXT = "sk-sls-trunc-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "LTAI_mock_ak";
const MOCK_AK_SECRET = "mock_ak_secret";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "structured-trunc-events";

const CONTENT_MAX_BYTES = 2048;
const HEAD_SENTINEL = "head-sentinel-1a2b3c";
const TAIL_SENTINEL = "tail-sentinel-9x8y7z";
const MIDDLE_SENTINEL = "middle-sentinel-5f6e7d";

// --- Minimal SLS LogGroup protobuf reader -------------------------------
// LogGroup { Logs = 1 (message) { Time = 1 (varint), Contents = 2 (message)
// { Key = 1 (string), Value = 2 (string) } } }; unknown fields skipped.

function readVarint(buf: Buffer, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  for (;;) {
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

/** Decode every log delivered to `logstore` into flat key→value maps. */
function logsFor(sls: MockSls, logstore: string): Map<string, string>[] {
  const logs: Map<string, string>[] = [];
  for (const r of sls.requests) {
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

// -------------------------------------------------------------------------

describe("sls e2e: oversized captured content is truncated structurally (#1014)", () => {
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-trunc",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "short reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 500, completion_tokens: 3, total_tokens: 503 },
      },
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const admin = new AdminClient(app.adminUrl, app.adminKey);

    await admin.createObservabilityExporter({
      name: "sls-structured-trunc",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
      content_max_bytes: CONTENT_MAX_BYTES,
    });

    const pk = await admin.createProviderKey({
      display_name: "sls-trunc-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "sls-trunc-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["sls-trunc-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  test("oversized JSON prompt lands in SLS as valid JSON with array sampling markers", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }

    // ~40 messages x ~130 bytes ≈ 5 KiB serialized — well over the 2 KiB cap.
    const filler = "conversational filler text to inflate the message body ";
    const messages = [
      { role: "user" as const, content: `${HEAD_SENTINEL} ${filler}` },
      ...Array.from({ length: 38 }, (_, i) => ({
        role: "user" as const,
        content: `${MIDDLE_SENTINEL} number ${i} ${filler}`,
      })),
      { role: "user" as const, content: `${TAIL_SENTINEL} ${filler}` },
    ];

    const send = async () =>
      fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "sls-trunc-model", messages }),
      });

    await waitConfigPropagation(async () => {
      const r = await send();
      await r.text();
      return r.status === 200;
    });

    const res = await send();
    expect(res.status).toBe(200);
    await res.text();

    await waitForLogstore(sls, LOGSTORE);
    // Poll until a content-bearing log (with a prompt) arrives.
    const deadline = Date.now() + 10_000;
    let captured: Map<string, string> | undefined;
    while (Date.now() < deadline && !captured) {
      captured = logsFor(sls, LOGSTORE).find((l) => l.has("prompt"));
      if (!captured) await new Promise((r) => setTimeout(r, 100));
    }
    expect(captured, "a full-content log with a prompt field").toBeDefined();

    const prompt = captured!.get("prompt")!;
    expect(Buffer.byteLength(prompt, "utf8")).toBeLessThanOrEqual(CONTENT_MAX_BYTES);
    expect(captured!.get("content_truncated")).toBe("true");

    // The core #1014 contract: the truncated prompt is still valid JSON.
    const parsed = JSON.parse(prompt) as {
      messages: Array<Record<string, unknown>>;
    };
    expect(Array.isArray(parsed.messages)).toBe(true);

    // Head and tail survive; the middle is sampled out through an explicit
    // placeholder that accounts for every omitted element.
    expect(JSON.stringify(parsed.messages[0])).toContain(HEAD_SENTINEL);
    expect(
      JSON.stringify(parsed.messages[parsed.messages.length - 1]),
    ).toContain(TAIL_SENTINEL);
    const placeholder = parsed.messages.find(
      (m) => m._aisix_truncated === true,
    );
    expect(placeholder, "array sampling placeholder").toBeDefined();
    const omitted = placeholder!.omitted_items as number;
    expect(omitted).toBeGreaterThan(0);
    expect(parsed.messages.length - 1 + omitted).toBe(messages.length);
  });
});
