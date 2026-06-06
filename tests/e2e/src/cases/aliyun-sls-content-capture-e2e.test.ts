import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Content-capture e2e (AISIX-Cloud#687): a `content_mode = full` aliyun_sls
// exporter captures the request prompt + assembled response into its logstore,
// while a `metadata_only` exporter on the same DP receives metadata only. This
// pins the per-exporter content gating end-to-end through the real binary, and
// proves the captured content actually reaches SLS (decompressed + searched).
//
// We mark the request and the mock upstream's response with unique tokens, then
// lz4-decompress each captured PutLogs body and assert which tokens appear.

const CALLER_PLAINTEXT = "sk-content-capture-caller-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-content-capture";

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const FULL_LOGSTORE = "full-events";
const META_LOGSTORE = "meta-events";

// Unique tokens planted in the request + the mock upstream response. The DP
// captures the prompt (request body, carrying PROMPT_TOKEN) and the assembled
// response (assistant content, carrying RESPONSE_TOKEN).
const PROMPT_TOKEN = "prompt-tok-9f3a2b";
const RESPONSE_TOKEN = "response-tok-7c1d8e";
// Separate tokens for the streaming path: the response is assembled across SSE
// delta chunks rather than read from a single body.
const STREAM_PROMPT_TOKEN = "stream-prompt-tok-3e8a11";
const STREAM_RESPONSE_TOKEN = "stream-response-tok-b22f70";
// Tokens for the Anthropic /v1/messages streaming path (cross-provider: an
// Anthropic-shape request streamed off an OpenAI upstream).
const ANTHROPIC_PROMPT_TOKEN = "anthropic-prompt-tok-5d1f9c";
const ANTHROPIC_RESPONSE_TOKEN = "anthropic-response-tok-a4e823";

interface CapturedPutLogs {
  logstore: string;
  rawSize: number;
  body: Buffer;
}

interface MockSls {
  url: string;
  requests: CapturedPutLogs[];
  close(): Promise<void>;
}

/** Decompress an lz4 *block* (no frame header) given the raw output size. */
function lz4DecompressBlock(src: Buffer, rawSize: number): Buffer {
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

async function startMockSls(): Promise<MockSls> {
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

async function seedRouting(
  admin: AdminClient,
  upstream: OpenAiUpstream,
  streamUpstream: OpenAiUpstream,
  messagesUpstream: OpenAiUpstream,
) {
  const pk = await admin.createProviderKey({
    display_name: "content-capture-pk",
    secret: PROVIDER_SECRET,
    api_base: `${upstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: "content-capture-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  const streamPk = await admin.createProviderKey({
    display_name: "content-capture-stream-pk",
    secret: PROVIDER_SECRET,
    api_base: `${streamUpstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: "content-capture-stream-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: streamPk.id,
  });
  const msgPk = await admin.createProviderKey({
    display_name: "content-capture-messages-pk",
    secret: PROVIDER_SECRET,
    api_base: `${messagesUpstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: "content-capture-messages-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: msgPk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: [
      "content-capture-model",
      "content-capture-stream-model",
      "content-capture-messages-model",
    ],
  });
}

/** Streaming Anthropic /v1/messages request (cross-provider off an OpenAI upstream). */
async function messagesStream(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.proxyUrl}/v1/messages`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "content-capture-messages-model",
      max_tokens: 100,
      stream: true,
      messages: [{ role: "user", content: `note the token ${ANTHROPIC_PROMPT_TOKEN}` }],
    }),
  });
  return res.text();
}

async function chat(app: SpawnedApp, model = "content-capture-model"): Promise<Response> {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: `please remember the token ${PROMPT_TOKEN}` }],
    }),
  });
}

/** Streaming chat carrying the stream-specific prompt token. */
async function chatStream(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "content-capture-stream-model",
      stream: true,
      messages: [{ role: "user", content: `recall the token ${STREAM_PROMPT_TOKEN}` }],
    }),
  });
  return res.text();
}

/** Decompress every PutLogs body for a logstore and join as one searchable string. */
function decodedTextFor(sls: MockSls, logstore: string): string {
  return sls.requests
    .filter((r) => r.logstore === logstore && r.rawSize > 0 && r.body.length > 0)
    .map((r) => lz4DecompressBlock(r.body, r.rawSize).toString("utf8"))
    .join(" ");
}

async function waitForLogstore(sls: MockSls, logstore: string, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (sls.requests.some((r) => r.logstore === logstore)) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`no PutLogs to logstore '${logstore}' within ${timeoutMs}ms`);
}

describe("aliyun_sls content capture e2e (#687): full vs metadata_only", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let streamUpstream: OpenAiUpstream | undefined;
  let messagesUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    // Plant the response token in the mock upstream's assistant content.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "mock-content-1",
        object: "chat.completion",
        created: 1_700_000_000,
        model: "mock-model",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `sure, ${RESPONSE_TOKEN} noted` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });
    // Streaming upstream: the response token arrives split across delta chunks,
    // so it's only captured if the DP assembles the stream correctly.
    streamUpstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "mock-stream-1",
          object: "chat.completion.chunk",
          model: "mock-model",
          choices: [{ index: 0, delta: { role: "assistant" } }],
        }),
        JSON.stringify({
          id: "mock-stream-1",
          object: "chat.completion.chunk",
          model: "mock-model",
          choices: [{ index: 0, delta: { content: `streamed ${STREAM_RESPONSE_TOKEN}` } }],
        }),
        JSON.stringify({
          id: "mock-stream-1",
          object: "chat.completion.chunk",
          model: "mock-model",
          choices: [{ index: 0, delta: { content: " done" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        }),
        "[DONE]",
      ],
    });
    // Cross-provider /v1/messages streams off an OpenAI upstream; the DP
    // assembles the OpenAI delta chunks and re-encodes them as Anthropic SSE.
    messagesUpstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({
          id: "mock-msg-1",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { role: "assistant" } }],
        }),
        JSON.stringify({
          id: "mock-msg-1",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content: `here is ${ANTHROPIC_RESPONSE_TOKEN}` } }],
        }),
        JSON.stringify({
          id: "mock-msg-1",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [{ index: 0, delta: { content: " ok" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
        }),
        "[DONE]",
      ],
    });
    sls = await startMockSls();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await upstream?.close();
    await streamUpstream?.close();
    await messagesUpstream?.close();
    await sls?.close();
  });

  test(
    "full-capture exporter logs prompt + response; metadata_only logs neither",
    async (ctx) => {
      if (!etcdReachable || !upstream || !streamUpstream || !messagesUpstream || !sls) {
        ctx.skip();
        return;
      }
      const app = await spawnApp({
        extraEnv: {
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
        },
      });
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "sls-full",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: FULL_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "full",
      });
      await admin.createObservabilityExporter({
        name: "sls-meta",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: SLS_PROJECT,
        logstore: META_LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: "metadata_only",
      });
      await seedRouting(admin, upstream, streamUpstream, messagesUpstream);

      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app);
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const res = await chat(app);
      expect(res.status).toBe(200);
      await res.text();

      // Both exporters deliver on every chat; wait until each logstore is seen.
      await waitForLogstore(sls, FULL_LOGSTORE);
      await waitForLogstore(sls, META_LOGSTORE);

      const fullText = decodedTextFor(sls, FULL_LOGSTORE);
      const metaText = decodedTextFor(sls, META_LOGSTORE);

      // Full-capture logstore carries both the prompt and the response content.
      expect(fullText).toContain(PROMPT_TOKEN);
      expect(fullText).toContain(RESPONSE_TOKEN);

      // Metadata-only logstore carries neither - content gating holds end-to-end.
      // (Both still log the request_id etc.; only prompt/response are withheld.)
      expect(metaText).not.toContain(PROMPT_TOKEN);
      expect(metaText).not.toContain(RESPONSE_TOKEN);

      // -- Streaming path --
      // The response token is split across SSE delta chunks; the DP must
      // assemble them into the captured response (C3b).
      const streamBody = await chatStream(app);
      expect(streamBody).toContain(STREAM_RESPONSE_TOKEN); // the client did receive it

      // The streaming fan-out runs at stream-end, so poll until it lands.
      const deadline = Date.now() + 10_000;
      while (
        Date.now() < deadline &&
        !decodedTextFor(sls, FULL_LOGSTORE).includes(STREAM_RESPONSE_TOKEN)
      ) {
        await new Promise((r) => setTimeout(r, 50));
      }

      const fullStream = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullStream).toContain(STREAM_PROMPT_TOKEN);
      expect(fullStream).toContain(STREAM_RESPONSE_TOKEN);
      expect(decodedTextFor(sls, META_LOGSTORE)).not.toContain(STREAM_PROMPT_TOKEN);
      expect(decodedTextFor(sls, META_LOGSTORE)).not.toContain(STREAM_RESPONSE_TOKEN);

      // -- Anthropic /v1/messages streaming path (cross-provider, C3b) --
      const msgBody = await messagesStream(app);
      expect(msgBody).toContain(ANTHROPIC_RESPONSE_TOKEN); // client received the assembled text

      const deadline2 = Date.now() + 10_000;
      while (
        Date.now() < deadline2 &&
        !decodedTextFor(sls, FULL_LOGSTORE).includes(ANTHROPIC_PROMPT_TOKEN)
      ) {
        await new Promise((r) => setTimeout(r, 50));
      }

      const fullMsg = decodedTextFor(sls, FULL_LOGSTORE);
      expect(fullMsg).toContain(ANTHROPIC_PROMPT_TOKEN);
      expect(fullMsg).toContain(ANTHROPIC_RESPONSE_TOKEN);
      expect(decodedTextFor(sls, META_LOGSTORE)).not.toContain(ANTHROPIC_PROMPT_TOKEN);
      expect(decodedTextFor(sls, META_LOGSTORE)).not.toContain(ANTHROPIC_RESPONSE_TOKEN);
    },
    60_000,
  );
});
