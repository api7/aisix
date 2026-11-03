import { createHash } from "node:crypto";
import { createServer as createTcpServer, type Server as TcpServer } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1921 and #1948. Under a full-content exporter every failure row
// carries the request body and the attempt's own response body; the wording of
// `error_message` and the access log names why an upstream gave no detail, and
// what the caller receives does not change.

const CALLER = "sk-failure-row-caller";
const LIMITED_CALLER = "sk-failure-row-limited";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const FULL = "failure-row-full";
const META = "failure-row-meta";

const NO_DETAIL = (status: number) =>
  `upstream returned HTTP ${status} with no error detail in the response`;

type Row = Map<string, string>;

describe("failure rows carry request and response bodies", () => {
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let truncatingUpstream: TcpServer | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;
  let seed: SeedClient;

  const headers = {
    authorization: `Bearer ${CALLER}`,
    "content-type": "application/json",
  };

  async function failingModel(
    name: string,
    opts: Parameters<typeof startOpenAiUpstream>[0],
  ): Promise<void> {
    const upstream = await startOpenAiUpstream(opts);
    upstreams.push(upstream);
    const pk = await seed.createProviderKey({
      display_name: `${name}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: name,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      cooldown: { enabled: false },
    });
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        SLS_CRED_MOCK_AK_ID: "LTAI_mock_ak",
        SLS_CRED_MOCK_AK_SECRET: "mock_ak_secret",
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    for (const [name, logstore, content_mode] of [
      ["failure-row-full", FULL, "full"],
      ["failure-row-meta", META, "metadata_only"],
    ] as const) {
      await seed.createObservabilityExporter({
        name,
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: "aisix-e2e-obs",
        logstore,
        credential_ref: CREDENTIAL_REF,
        content_mode,
      });
    }

    for (const status of [400, 401, 403]) {
      await failingModel(`row-${status}`, {
        status,
        errorBody: { error: { message: `sentinel-body-${status}`, type: "upstream_type" } },
      });
    }
    await failingModel("row-first-fails", {
      status: 500,
      errorBody: { error: { message: "sentinel-first-target" } },
    });
    await failingModel("row-second-fails", {
      status: 503,
      errorBody: { error: { message: "sentinel-second-target" } },
    });
    await failingModel("row-ok", {
      nonStreamBody: {
        id: "cmpl-ok",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "fine" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });
    await seed.createModel({
      display_name: "row-all-fail",
      routing: {
        strategy: "failover",
        targets: [{ model: "row-first-fails" }, { model: "row-second-fails" }],
      },
    });
    await seed.createModel({
      display_name: "row-recovers",
      routing: {
        strategy: "failover",
        targets: [{ model: "row-first-fails" }, { model: "row-ok" }],
      },
    });
    const delta = (content: string) =>
      JSON.stringify({
        id: "c",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-mini",
        choices: [{ index: 0, delta: { content }, finish_reason: null }],
      });
    await failingModel("row-stream-cut", {
      streamEvents: [delta("partial-answer"), delta(" more"), delta(" more")],
      disconnectAfterEvents: 1,
      eventDelayMs: 100,
    });
    await failingModel("row-in-band", {
      streamEvents: [JSON.stringify({ error: { message: "sentinel-in-band", code: 500 } })],
    });
    // Configured, and not on the caller's list.
    await failingModel("row-unlisted", { status: 400 });
    await failingModel("row-empty-404", { status: 404, rawErrorBody: "" });
    await failingModel("row-blank-message", {
      status: 404,
      errorBody: { error: { message: "" } },
    });
    await failingModel("row-control-only", {
      status: 500,
      rawErrorBody: "\n\t\r\n",
      errorContentType: "text/plain",
    });
    await failingModel("row-html", {
      status: 502,
      rawErrorBody: "<html>sentinel fronting proxy page</html>",
      errorContentType: "text/html",
    });

    // Declares more bytes than it sends, then drops the connection.
    truncatingUpstream = createTcpServer((socket) => {
      socket.once("data", () => {
        socket.write(
          "HTTP/1.1 502 Bad Gateway\r\ncontent-type: application/json\r\n" +
            'content-length: 4096\r\n\r\n{"error":{"message":"sentinel-cut',
          () => socket.destroy(),
        );
      });
      socket.on("error", () => {});
    });
    await new Promise<void>((r) => truncatingUpstream!.listen(0, "127.0.0.1", r));
    const port = (truncatingUpstream.address() as { port: number }).port;
    const truncatedPk = await seed.createProviderKey({
      display_name: "row-truncated-pk",
      secret: "sk-mock",
      api_base: `http://127.0.0.1:${port}/v1`,
    });
    await seed.createModel({
      display_name: "row-truncated",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: truncatedPk.id,
      cooldown: { enabled: false },
    });

    const passthroughUpstream = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "sentinel-passthrough-answer" } },
    });
    upstreams.push(passthroughUpstream);
    await seed.createPassthroughRoute({
      name: "row-passthrough",
      path_prefix: "/row-passthrough",
      target_url: passthroughUpstream.baseUrl,
      provider_key_id: truncatedPk.id,
    });

    await seed.createApiKey({
      key_hash: hash(LIMITED_CALLER),
      allowed_models: ["row-ok"],
      rate_limit: { rpm: 1 },
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_routes: ["*"],
      allowed_models: [
        "row-400",
        "row-401",
        "row-403",
        "row-first-fails",
        "row-second-fails",
        "row-ok",
        "row-all-fail",
        "row-recovers",
        "row-empty-404",
        "row-blank-message",
        "row-control-only",
        "row-html",
        "row-truncated",
        "row-stream-cut",
        "row-in-band",
      ],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, { headers });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    for (const upstream of upstreams) await upstream.close();
    truncatingUpstream?.close();
    await sls?.close();
  });

  async function chat(model: string, content: string) {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
    });
    return { res, requestId: res.headers.get("x-aisix-request-id")!, body: await res.text() };
  }

  function fullRows(requestId: string): Row[] {
    return slsLogsFor(sls!, FULL).filter((l) => l.get("request_id") === requestId);
  }

  async function rowOf(requestId: string, what: string): Promise<Row> {
    return waitForSlsLog(sls!, FULL, (l) => l.get("request_id") === requestId, what);
  }

  async function rowsOf(requestId: string, n: number): Promise<Row[]> {
    const deadline = Date.now() + 10_000;
    while (fullRows(requestId).length < n && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
    }
    const rows = fullRows(requestId);
    expect(rows.length, `rows of ${requestId}`).toBeGreaterThanOrEqual(n);
    return rows.sort((a, b) => Number(a.get("attempt_index")) - Number(b.get("attempt_index")));
  }

  const requestText = (row: Row) => row.get("input_messages") ?? row.get("prompt") ?? "";

  test.each([400, 401, 403])(
    "chat upstream %i carries the request and the upstream's own body",
    async (status) => {
      if (!etcdReachable) return;
      const prompt = `chat-prompt-${status}`;
      const { res, requestId } = await chat(`row-${status}`, prompt);
      expect(res.status).toBe(status);
      const row = await rowOf(requestId, `the ${status} row`);
      expect(row.get("status_code")).toBe(String(status));
      expect(requestText(row)).toContain(prompt);
      expect(JSON.parse(row.get("response")!)).toEqual({
        error: { message: `sentinel-body-${status}`, type: "upstream_type" },
      });
    },
  );

  test("a model the key may not use still records what was sent and answered", async () => {
    if (!etcdReachable) return;
    const { res, requestId, body } = await chat("row-unlisted", "forbidden-prompt");
    expect(res.status).toBe(403);
    const row = await rowOf(requestId, "the refusal row");
    expect(requestText(row)).toContain("forbidden-prompt");
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  test("every attempt of an all-failed group carries its own body", async () => {
    if (!etcdReachable) return;
    const { requestId } = await chat("row-all-fail", "group-prompt");
    const [first, second] = await rowsOf(requestId, 2);
    for (const row of [first!, second!]) expect(requestText(row)).toContain("group-prompt");
    expect(first!.get("response")).toContain("sentinel-first-target");
    expect(second!.get("response")).toContain("sentinel-second-target");
  });

  test("a failed attempt before a winner carries its own body; the winner keeps its answer", async () => {
    if (!etcdReachable) return;
    const { res, requestId } = await chat("row-recovers", "recover-prompt");
    expect(res.status).toBe(200);
    const [failed, winner] = await rowsOf(requestId, 2);
    expect(requestText(failed!)).toContain("recover-prompt");
    expect(failed!.get("response")).toContain("sentinel-first-target");
    expect(winner!.get("status_code")).toBe("200");
    expect(winner!.get("output_messages") ?? winner!.get("response")).toContain("fine");
    expect(winner!.get("response") ?? "").not.toContain("sentinel-first-target");
  });

  test("a local rate-limit refusal carries the request and the error the caller got", async () => {
    if (!etcdReachable) return;
    const send = () =>
      fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { ...headers, authorization: `Bearer ${LIMITED_CALLER}` },
        body: JSON.stringify({
          model: "row-ok",
          messages: [{ role: "user", content: "limited-prompt" }],
        }),
      });
    let refused: Response | undefined;
    for (let i = 0; i < 5 && !refused; i++) {
      const res = await send();
      if (res.status === 429) refused = res;
      else await res.text();
    }
    expect(refused, "the key's rpm refused a request").toBeDefined();
    const body = await refused!.text();
    const row = await rowOf(refused!.headers.get("x-aisix-request-id")!, "the 429 row");
    expect(row.get("status_code")).toBe("429");
    expect(requestText(row)).toContain("limited-prompt");
    expect(JSON.parse(row.get("response")!)).toEqual(JSON.parse(body));
  });

  test("embeddings upstream 400 carries both bodies", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: "row-400", input: "embed-prompt" }),
    });
    await res.text();
    expect(res.status).toBe(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the embeddings row");
    expect(row.get("prompt")).toContain("embed-prompt");
    expect(row.get("response")).toContain("sentinel-body-400");
  });

  test("a multipart failure reports text fields and a file checksum, never the file", async () => {
    if (!etcdReachable) return;
    const form = new FormData();
    form.set("model", "row-400");
    form.set("prompt", "transcribe-prompt");
    form.set("file", new Blob(["sentinel-audio-bytes"]), "clip.mp3");
    const res = await fetch(`${app!.proxyUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}` },
      body: form,
    });
    await res.text();
    expect(res.status).toBe(400);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the transcription row");
    const prompt = JSON.parse(row.get("prompt")!) as Record<string, string>;
    expect(prompt.prompt).toBe("transcribe-prompt");
    expect(prompt.file_sha256).toBe(hash("sentinel-audio-bytes"));
    expect(row.get("prompt")).not.toContain("sentinel-audio-bytes");
    expect(row.get("response")).toContain("sentinel-body-400");
  });

  test("a passthrough route's upstream failure carries both bodies", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/row-passthrough/anything`, {
      method: "POST",
      headers,
      body: JSON.stringify({ note: "passthrough-prompt" }),
    });
    await res.text();
    expect(res.status).toBe(500);
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the passthrough row");
    expect(row.get("prompt")).toContain("passthrough-prompt");
    expect(row.get("response")).toContain("sentinel-passthrough-answer");
  });

  test("count_tokens and videos, which capture nothing on success, report a failure", async () => {
    if (!etcdReachable) return;
    const counted = await fetch(`${app!.proxyUrl}/v1/messages/count_tokens`, {
      method: "POST",
      headers: { ...headers, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({
        model: "row-unlisted",
        messages: [{ role: "user", content: "count-prompt" }],
      }),
    });
    const countedBody = await counted.text();
    expect(counted.status).toBe(403);
    const countedRow = await rowOf(counted.headers.get("x-aisix-request-id")!, "the count row");
    expect(requestText(countedRow)).toContain("count-prompt");
    // The Anthropic shape this route answers with, not the OpenAI one.
    expect(JSON.parse(countedRow.get("response")!)).toEqual(JSON.parse(countedBody));
    expect(JSON.parse(countedBody).type).toBe("error");

    const video = await fetch(`${app!.proxyUrl}/v1/videos`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: "row-unlisted", prompt: "video-prompt" }),
    });
    await video.text();
    expect(video.status).toBe(403);
    const videoRow = await rowOf(video.headers.get("x-aisix-request-id")!, "the video row");
    expect(requestText(videoRow)).toContain("video-prompt");
  });

  test.each([
    ["an empty body", "row-empty-404", 404, 404, "upstream returned 404 Not Found"],
    ["a blank message", "row-blank-message", 404, 404, "upstream returned 404 Not Found"],
    ["a body of control characters", "row-control-only", 500, 502, "upstream returned 500 Internal Server Error"],
  ] as const)(
    "%s names the missing detail; the client's answer does not",
    async (_what, model, status, clientStatus, clientMessage) => {
      if (!etcdReachable) return;
      const { res, requestId, body } = await chat(model, "no-detail-prompt");
      expect(res.status).toBe(clientStatus);
      expect((JSON.parse(body) as { error: { message: string } }).error.message).toBe(
        clientMessage,
      );
      expect(body).not.toContain("no error detail");
      const row = await rowOf(requestId, `the ${model} row`);
      expect(row.get("error_message")).toBe(NO_DETAIL(status));
      await waitForLogLine(
        app!,
        (l) => l.includes(requestId) && l.includes(NO_DETAIL(status)),
        `the ${model} access-log line`,
      );
    },
  );

  test("a non-JSON body stays the detail, for the row and the caller", async () => {
    if (!etcdReachable) return;
    const page = "<html>sentinel fronting proxy page</html>";
    const { requestId, body } = await chat("row-html", "html-prompt");
    expect((JSON.parse(body) as { error: { message: string } }).error.message).toBe(page);
    const row = await rowOf(requestId, "the html row");
    expect(row.get("error_message")).toBe(`upstream returned HTTP 502: ${page}`);
    expect(row.get("response")).toBe(page);
  });

  test.each([
    ["chat", "/v1/chat/completions", { messages: [{ role: "user", content: "cut-prompt" }] }],
    ["responses", "/v1/responses", { input: "cut-prompt" }],
  ] as const)(
    "%s: a body the upstream cut off names the read failure; the caller gets the status",
    async (_what, path, extra) => {
      if (!etcdReachable) return;
      const res = await fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify({ model: "row-truncated", ...extra }),
      });
      const body = await res.text();
      const requestId = res.headers.get("x-aisix-request-id")!;
      expect(res.status).toBe(502);
      expect((JSON.parse(body) as { error: { message: string } }).error.message).toBe(
        "upstream returned 502 Bad Gateway",
      );
      const row = await rowOf(requestId, "the truncated row");
      expect(row.get("error_message")).toMatch(
        /^upstream returned HTTP 502: failed to read response body: .+/,
      );
      expect(requestText(row)).toContain("cut-prompt");
      // What did arrive is still reported.
      expect(row.get("response")).toContain("sentinel-cut");
      const warn = await waitForLogLine(
        app!,
        (l) =>
          l.includes("failed to read upstream error response body") && l.includes(requestId),
        "the read-failure warning",
      );
      expect(warn).toContain("502");
    },
  );

  test("a stream cut off mid-way keeps the request and what had arrived", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "row-stream-cut",
        stream: true,
        messages: [{ role: "user", content: "stream-cut-prompt" }],
      }),
    });
    await res.text().catch(() => "");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the cut stream row");
    expect(Number(row.get("status_code"))).toBeGreaterThanOrEqual(400);
    expect(requestText(row)).toContain("stream-cut-prompt");
    expect(row.get("output_messages") ?? row.get("response")).toContain("partial-answer");
  });

  test("an in-band stream error before any content reports the request and the error", async () => {
    if (!etcdReachable) return;
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: "row-in-band",
        stream: true,
        messages: [{ role: "user", content: "in-band-prompt" }],
      }),
    });
    await res.text().catch(() => "");
    const row = await rowOf(res.headers.get("x-aisix-request-id")!, "the in-band row");
    expect(Number(row.get("status_code"))).toBeGreaterThanOrEqual(400);
    expect(requestText(row)).toContain("in-band-prompt");
    expect(row.get("response")).toContain("sentinel-in-band");
  });

  // Last: every sentinel above has been sent by now.
  test("a metadata_only exporter receives none of it", async () => {
    if (!etcdReachable) return;
    const failed = () =>
      slsLogsFor(sls!, META).filter((l) => Number(l.get("status_code") ?? "0") >= 400);
    const deadline = Date.now() + 10_000;
    while (failed().length < 5 && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
    }
    expect(failed().length).toBeGreaterThanOrEqual(5);
    for (const row of slsLogsFor(sls!, META)) {
      for (const field of ["prompt", "response", "input_messages", "output_messages"]) {
        expect(row.get(field), field).toBeUndefined();
      }
      for (const value of row.values()) expect(value).not.toContain("-prompt");
    }
  });
});
