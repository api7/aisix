import { WebSocket } from "undici";
import { WebSocketServer, type WebSocket as WsSocket } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `api_base` is the upstream root — whatever path it carries is
// preserved, and every endpoint is appended to it (AISIX-Cloud#1244).
//
// Before the fix the gateway used two different conventions on one
// provider key: the OpenAI-family bridge appended `/chat/completions` to
// the configured base verbatim, while the proxy-side handlers injected a
// `/v1` segment unless the base already ended in `/v1`. A key pointing at
// a non-`/v1` root — Baidu Qianfan's `…/v2`, Zhipu's `…/api/paas/v4`,
// Volcengine Ark's `…/api/v3`, Gemini's `…/v1beta/openai` — therefore
// served chat correctly while responses, rerank, audio, realtime and the
// batch/files routes all dispatched to `…/v2/v1/<endpoint>` and 404'd.
//
// The mock upstream answers every path with the same canned body, so the
// assertion is purely on the path the gateway dialled.

const CALLER_PLAINTEXT = "sk-apibase-e2e-caller";
const CALLER_KEY_ENV = "APIBASE_E2E_CALLER_KEY";

/** `/v2` — the reported Qianfan shape; also stands in for every other
 *  vendor whose OpenAI-compatible root is not `/v1`. */
function resources(upstreamBase: string, realtimeBase: string): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: pk-v2-root
    provider: openai
    adapter: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v2
  - display_name: pk-bare-host
    provider: openai
    adapter: openai
    api_key: sk-mock
    api_base: ${upstreamBase}
  - display_name: pk-anthropic-mounted
    provider: anthropic
    adapter: anthropic
    api_key: sk-mock
    api_base: ${upstreamBase}/anthropic
  - display_name: pk-realtime-v2-root
    provider: openai
    adapter: openai
    api_key: sk-upstream-realtime
    api_base: ${realtimeBase}/v2
models:
  - display_name: m-v2-root
    provider: openai
    model_name: gpt-4o-mini
    provider_key: pk-v2-root
  - display_name: m-bare-host
    provider: openai
    model_name: gpt-4o-mini
    provider_key: pk-bare-host
  - display_name: m-anthropic-mounted
    provider: anthropic
    model_name: claude-3-5-sonnet
    provider_key: pk-anthropic-mounted
  - display_name: m-realtime-v2-root
    provider: openai
    model_name: gpt-realtime-mock
    provider_key: pk-realtime-v2-root
api_keys:
  - display_name: apibase-caller
    key_env: ${CALLER_KEY_ENV}
    allowed_models:
      ["m-v2-root", "m-bare-host", "m-anthropic-mounted", "m-realtime-v2-root"]
`;
}

interface RealtimeUpstream {
  baseUrl: string;
  handshakes: string[];
  close(): Promise<void>;
}

/** Records the upgrade path of every realtime handshake it receives. */
async function startRealtimeUpstream(): Promise<RealtimeUpstream> {
  const handshakes: string[] = [];
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket, req) => {
    handshakes.push(req.url ?? "");
    socket.close();
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    handshakes,
    close: () =>
      new Promise<void>((resolve, reject) =>
        wss.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

describe("api_base path prefix is preserved on every endpoint", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let realtime: RealtimeUpstream | undefined;

  const auth = { authorization: `Bearer ${CALLER_PLAINTEXT}` };

  /** Path the gateway dialled for the request this call brackets. */
  async function upstreamPathOf(run: () => Promise<unknown>): Promise<string> {
    if (!upstream) throw new Error("setup failed");
    const before = upstream.receivedRequests.length;
    await run();
    const seen = upstream.receivedRequests.slice(before);
    expect(seen.length).toBe(1);
    return seen[0].path;
  }

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    realtime = await startRealtimeUpstream();
    // File mode: resources load synchronously at boot, no etcd needed.
    app = await spawnApp({
      resourcesFile: resources(upstream.baseUrl, realtime.baseUrl),
      extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await realtime?.close();
  });

  test("chat completions keeps the /v2 root (the leg that always worked)", async () => {
    if (!app) throw new Error("setup failed");
    const path = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({
          model: "m-v2-root",
          messages: [{ role: "user", content: "hi" }],
        }),
      }),
    );
    expect(path).toBe("/v2/chat/completions");
  });

  test("responses keeps the /v2 root instead of building /v2/v1/responses", async () => {
    if (!app) throw new Error("setup failed");
    const path = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/responses`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({ model: "m-v2-root", input: "hello" }),
      }),
    );
    expect(path).toBe("/v2/responses");
  });

  test("rerank keeps the /v2 root", async () => {
    if (!app) throw new Error("setup failed");
    const path = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/rerank`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({
          model: "m-v2-root",
          query: "q",
          documents: ["a", "b"],
        }),
      }),
    );
    expect(path).toBe("/v2/rerank");
  });

  test("audio speech keeps the /v2 root", async () => {
    if (!app) throw new Error("setup failed");
    const path = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/audio/speech`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({
          model: "m-v2-root",
          input: "hi",
          voice: "alloy",
        }),
      }),
    );
    expect(path).toBe("/v2/audio/speech");
  });

  test("audio transcriptions keeps the /v2 root", async () => {
    if (!app) throw new Error("setup failed");
    const form = new FormData();
    form.set("model", "m-v2-root");
    form.set("file", new Blob(["fake audio"]), "clip.mp3");
    const path = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/audio/transcriptions`, {
        method: "POST",
        headers: auth,
        body: form,
      }),
    );
    expect(path).toBe("/v2/audio/transcriptions");
  });

  test("the batch/files family keeps the /v2 root", async () => {
    if (!app) throw new Error("setup failed");

    const listPath = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/files?model=m-v2-root`, { headers: auth }),
    );
    expect(listPath).toBe("/v2/files");

    const batchPath = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/batches`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({
          model: "m-v2-root",
          input_file_id: "file-1",
          endpoint: "/v1/chat/completions",
          completion_window: "24h",
        }),
      }),
    );
    expect(batchPath).toBe("/v2/batches");
  });

  test("realtime keeps the /v2 root on the upstream upgrade", async () => {
    if (!app || !realtime) throw new Error("setup failed");
    const ws = new WebSocket(
      `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=m-realtime-v2-root`,
      ["realtime", `openai-insecure-api-key.${CALLER_PLAINTEXT}`],
    );
    await new Promise<void>((resolve) => {
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener("error", () => resolve(), { once: true });
    });
    // The relay dials upstream during the client handshake; give the
    // recorded upgrade a moment to land before asserting on it.
    for (let i = 0; i < 50 && realtime.handshakes.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 20));
    }
    ws.close();
    expect(realtime.handshakes.length).toBe(1);
    expect(realtime.handshakes[0]).toBe("/v2/realtime?model=gpt-realtime-mock");
  });

  test("a bare host still gets the canonical /v1 synthesized", async () => {
    if (!app) throw new Error("setup failed");
    // Several vendor defaults ship the bare host (deepseek, cohere,
    // runwayml) — dropping the synthesis would break all of them.
    const path = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/responses`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({ model: "m-bare-host", input: "hello" }),
      }),
    );
    expect(path).toBe("/v1/responses");
  });

  test("anthropic keeps /v1 in the endpoint even under a mounted path", async () => {
    if (!app) throw new Error("setup failed");
    // Mirror image of the OpenAI family: Anthropic documents the bare
    // host and owns `/v1` in the endpoint, so a gateway mounted at
    // `/anthropic` serves `/anthropic/v1/messages` — the path must NOT
    // suppress the version segment here.
    const messagesPath = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({
          model: "m-anthropic-mounted",
          max_tokens: 16,
          messages: [{ role: "user", content: "hi" }],
        }),
      }),
    );
    expect(messagesPath).toBe("/anthropic/v1/messages");

    const countPath = await upstreamPathOf(() =>
      fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { ...auth, "content-type": "application/json" },
        body: JSON.stringify({
          model: "m-anthropic-mounted",
          messages: [{ role: "user", content: "hi" }],
        }),
      }),
    );
    expect(countPath).toBe("/anthropic/v1/messages/count_tokens");
  });
});
