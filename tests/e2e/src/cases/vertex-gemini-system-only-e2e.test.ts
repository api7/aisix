import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a chat request holding only a system message, sent to a Gemini model
// on Vertex, is answered.
//
// The system message becomes Gemini's `systemInstruction`, which leaves no
// conversation turn in `contents` — and Gemini rejects an empty `contents`.
// The gateway sends a single blank user turn in its place, so the model
// answers the instructions alone.

const CALLER_PLAINTEXT = "sk-vertex-gemini-system-only";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const MODEL = "vertex-gemini-system-only";
const INSTRUCTIONS = "Reply with the single word: ready.";

const GEMINI_REPLY = {
  candidates: [{ content: { role: "model", parts: [{ text: "ready" }] }, finishReason: "STOP" }],
  usageMetadata: { promptTokenCount: 9, candidatesTokenCount: 1, totalTokenCount: 10 },
  modelVersion: "gemini-2.5-flash",
};

describe("Vertex Gemini: a system-only chat request", () => {
  let app: SpawnedApp | undefined;
  let upstream: Server | undefined;
  const received: Array<{ path: string; body: any }> = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = createServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => {
        received.push({ path: req.url ?? "", body: JSON.parse(raw || "{}") });
        if (req.url?.includes(":streamGenerateContent")) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          res.end(`data: ${JSON.stringify(GEMINI_REPLY)}\r\n\r\n`);
        } else {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(GEMINI_REPLY));
        }
      });
    });
    await new Promise<void>((resolve) => upstream!.listen(0, "127.0.0.1", resolve));
    const addr = upstream.address();
    if (addr === null || typeof addr === "string") throw new Error("no port");

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "vertex-gemini-system-only-pk",
      provider: "google",
      adapter: "vertex",
      secret: JSON.stringify({
        access_token: "ya29.system-only",
        project: "proj-e2e",
        region: "us-central1",
      }),
      api_base: `http://127.0.0.1:${addr.port}`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "google",
      model_name: "gemini-2.5-flash",
      provider_key_id: pk.id,
    });
    // Seeded last, so this key authenticating implies the whole seed set
    // has reached the gateway's snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await new Promise<void>((resolve) => (upstream ? upstream.close(() => resolve()) : resolve()));
  });

  test.for([false, true])("stream=%s is answered", async (stream, ctx) => {
    if (!etcdReachable || !app) return ctx.skip();
    const before = received.length;
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        stream,
        messages: [{ role: "system", content: INSTRUCTIONS }],
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(text).toContain("ready");

    const sent = received.slice(before);
    expect(sent).toHaveLength(1);
    expect(sent[0].path).toContain(stream ? ":streamGenerateContent" : ":generateContent");
    expect(sent[0].body.contents).toEqual([{ role: "user", parts: [{ text: " " }] }]);
    expect(JSON.stringify(sent[0].body.systemInstruction)).toContain(INSTRUCTIONS);
  });
});
