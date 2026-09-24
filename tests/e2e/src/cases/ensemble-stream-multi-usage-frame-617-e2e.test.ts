import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { slsLogsFor, startMockSls, waitForSlsLog, type MockSls } from "../harness/sls-mock.js";

// E2E for api7/aisix#617: a streaming ensemble whose judge stamps `usage` on
// more than one chunk (the Gemini/Vertex shape — cumulative counts on every
// chunk) must add the panel's usage to the client stream exactly once, on the
// terminal usage frame. Every earlier frame carries the judge's own usage
// unchanged, and the billed UsageEvents stay one per sub-call at the judge's
// final (not summed) counts.

const CALLER_PLAINTEXT = "sk-ensemble-multi-usage-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "ens-multi-usage-events";

const ENSEMBLE = "ens617-council";
const MEMBER_A = "ens617-member-a";
const MEMBER_B = "ens617-member-b";
const JUDGE = "ens617-judge";

// Each panel member: prompt 5 / completion 11 / total 16 ⇒ panel sum 10/22/32.
const PANEL_BODY = {
  id: "chatcmpl-panel",
  object: "chat.completion",
  created: 0,
  model: "panel-upstream",
  choices: [
    { index: 0, message: { role: "assistant", content: "a panel answer" }, finish_reason: "stop" },
  ],
  usage: { prompt_tokens: 5, completion_tokens: 11, total_tokens: 16 },
};

// Judge: cumulative usage on every chunk, final 30 / 7 / 37.
const judgeChunk = (delta: object, finish: string | null, completion: number) =>
  JSON.stringify({
    id: "chatcmpl-judge",
    object: "chat.completion.chunk",
    model: "judge-upstream",
    choices: [{ index: 0, delta, finish_reason: finish }],
    usage: { prompt_tokens: 30, completion_tokens: completion, total_tokens: 30 + completion },
  });
const JUDGE_STREAM = [
  judgeChunk({ role: "assistant", content: "Hello" }, null, 1),
  judgeChunk({ content: " world" }, null, 3),
  judgeChunk({}, "stop", 7),
  "[DONE]",
];

type Usage = { prompt_tokens: number; completion_tokens: number; total_tokens: number };

function dataFrames(text: string): Array<Record<string, unknown>> {
  const frames: Array<Record<string, unknown>> = [];
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(6).trim();
    if (data === "[DONE]") continue;
    frames.push(JSON.parse(data) as Record<string, unknown>);
  }
  return frames;
}

describe("streaming ensemble folds the panel usage once with a multi-usage-frame judge (api7/aisix#617)", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let memberAUp: OpenAiUpstream | undefined;
  let memberBUp: OpenAiUpstream | undefined;
  let judgeUp: OpenAiUpstream | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    memberAUp = await startOpenAiUpstream({ nonStreamBody: PANEL_BODY });
    memberBUp = await startOpenAiUpstream({ nonStreamBody: PANEL_BODY });
    judgeUp = await startOpenAiUpstream({ streamEvents: JUDGE_STREAM });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "ens617-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
    });

    for (const [display, upstream] of [
      [MEMBER_A, memberAUp],
      [MEMBER_B, memberBUp],
      [JUDGE, judgeUp],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: display,
        provider: "openai",
        model_name: "gpt-4o",
        provider_key_id: pk.id,
      });
    }
    await seed.createModel({
      display_name: ENSEMBLE,
      ensemble: {
        panel: [{ model: MEMBER_A }, { model: MEMBER_B }],
        judge: { model: JUDGE },
        min_responses: 2,
      },
    });

    // Caller key last: the moment it authenticates, the whole seed set is in
    // the snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: [ENSEMBLE] });
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((row) => row.id === ENSEMBLE);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await memberAUp?.close();
    await memberBUp?.close();
    await judgeUp?.close();
  });

  test("panel sum lands once, on the terminal usage frame; UsageEvents are not over-counted", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: ENSEMBLE,
        messages: [{ role: "user", content: "hi" }],
        stream: true,
        stream_options: { include_usage: true },
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    const text = await res.text();
    expect(text).toContain("data: [DONE]");

    const frames = dataFrames(text);
    const content = frames
      .map((f) => (f.choices as Array<{ delta?: { content?: string } }> | undefined)?.[0]?.delta?.content ?? "")
      .join("");
    expect(content).toBe("Hello world");

    const usages = frames.filter((f) => f.usage != null).map((f) => f.usage as Usage);
    // The judge stamped usage on three chunks; each still reaches the client.
    expect(usages).toHaveLength(3);
    // Intermediate frames: the judge's own cumulative usage, no panel folded in.
    expect(usages[0]).toMatchObject({ prompt_tokens: 30, completion_tokens: 1, total_tokens: 31 });
    expect(usages[1]).toMatchObject({ prompt_tokens: 30, completion_tokens: 3, total_tokens: 33 });
    // Terminal frame: judge final + panel sum (10 / 22 / 32), exactly once.
    expect(usages[2]).toMatchObject({ prompt_tokens: 40, completion_tokens: 29, total_tokens: 69 });
    // …and it is the last data frame the client sees.
    expect(frames[frames.length - 1]?.usage).toMatchObject({ total_tokens: 69 });
    const panelFolds = usages.filter((u) => u.prompt_tokens >= 40).length;
    expect(panelFolds).toBe(1);

    // Billing: one row per sub-call, the judge at its final counts (not summed
    // across its three usage frames, and never carrying the panel's).
    await waitForSlsLog(
      sls!,
      LOGSTORE,
      (l) => l.get("request_id") === requestId && l.get("attempt_kind") === "judge",
      `judge usage row for ${requestId}`,
      15_000,
    );
    const rows = slsLogsFor(sls!, LOGSTORE).filter((l) => l.get("request_id") === requestId);
    const judgeRows = rows.filter((l) => l.get("attempt_kind") === "judge");
    const panelRows = rows.filter((l) => l.get("attempt_kind") === "panel");
    expect(judgeRows).toHaveLength(1);
    expect(judgeRows[0]!.get("prompt_tokens")).toBe("30");
    expect(judgeRows[0]!.get("completion_tokens")).toBe("7");
    expect(panelRows).toHaveLength(2);
    for (const row of panelRows) {
      expect(row.get("prompt_tokens")).toBe("5");
      expect(row.get("completion_tokens")).toBe("11");
    }
  }, 60_000);
});
