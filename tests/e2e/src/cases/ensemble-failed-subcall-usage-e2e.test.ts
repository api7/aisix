import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Every ensemble sub-call is a usage row of its own, including the ones that
// fail: a panel member or a judge whose upstream answers 500 leaves a
// zero-token row naming that model and its error, at its own attempt index
// (a member's panel slot; the panel size for the judge). A request that
// fails then ends with one terminal row at the next index. Observed through
// the SLS exporter, which receives the rows as the control plane does.

const CALLER = "sk-ensemble-failed-subcall";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "ensemble-failed-subcall-events";
const PANEL_USAGE = { prompt_tokens: 5, completion_tokens: 11, total_tokens: 16 };
const PANEL_ERROR = "panel upstream exploded";
const JUDGE_ERROR = "judge upstream exploded";

const PANEL_BODY = {
  id: "chatcmpl-efs-panel",
  object: "chat.completion",
  created: 1_700_000_000,
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "a candidate" }, finish_reason: "stop" }],
  usage: PANEL_USAGE,
};
const JUDGE_BODY = {
  ...PANEL_BODY,
  id: "chatcmpl-efs-judge",
  choices: [{ index: 0, message: { role: "assistant", content: "synthesized" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 40, completion_tokens: 6, total_tokens: 46 },
};
const JUDGE_STREAM = [
  JSON.stringify({
    id: "chatcmpl-efs-judge",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { role: "assistant", content: "synthesized" }, finish_reason: null }],
  }),
  JSON.stringify({
    id: "chatcmpl-efs-judge",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 40, completion_tokens: 6, total_tokens: 46 },
  }),
  "[DONE]",
];

// Stands for the judge that answers: the mock serves either a stream or a
// buffered body, so each mode has its own.
const JUDGE_OK = "<judge-ok>";
const judgeOk = (stream: boolean) => (stream ? "efs-judge-ok-stream" : "efs-judge-ok");

// [attempt_index, attempt_kind, status_code, attempt_model]; the terminal
// row of a failed request names no kind and no model.
type Row = [string, string, string, string];

interface Scenario {
  name: string;
  panel: string[];
  judge: string;
  status: number;
  rows: Row[];
}

const SCENARIOS: Scenario[] = [
  {
    name: "one of three members fails, the other two are enough",
    panel: ["efs-ok-a", "efs-fail-a", "efs-ok-b"],
    judge: JUDGE_OK,
    status: 200,
    rows: [
      ["0", "panel", "200", "efs-ok-a"],
      ["1", "panel", "502", "efs-fail-a"],
      ["2", "panel", "200", "efs-ok-b"],
      ["3", "judge", "200", JUDGE_OK],
    ],
  },
  {
    name: "every member fails, so the panel falls short",
    panel: ["efs-fail-a", "efs-fail-b"],
    judge: JUDGE_OK,
    status: 502,
    rows: [
      ["0", "panel", "502", "efs-fail-a"],
      ["1", "panel", "502", "efs-fail-b"],
      ["2", "", "502", ""],
    ],
  },
  {
    name: "the panel answers and the judge fails",
    panel: ["efs-ok-a", "efs-ok-b"],
    judge: "efs-judge-fail",
    status: 502,
    rows: [
      ["0", "panel", "200", "efs-ok-a"],
      ["1", "panel", "200", "efs-ok-b"],
      ["2", "judge", "502", "efs-judge-fail"],
      ["3", "", "502", ""],
    ],
  },
];

const cases = SCENARIOS.flatMap((s, i) =>
  [false, true].map((stream) => ({
    ...s,
    stream,
    judge: s.judge === JUDGE_OK ? judgeOk(stream) : s.judge,
    rows: s.rows.map(([index, kind, status, model]): Row => [
      index,
      kind,
      status,
      model === JUDGE_OK ? judgeOk(stream) : model,
    ]),
    council: `efs-council-${i}-${stream ? "stream" : "buffered"}`,
    label: `${stream ? "streaming" : "non-streaming"}: ${s.name}`,
  })),
);

describe("an ensemble's failed sub-calls each leave a usage row", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  const upstreams: OpenAiUpstream[] = [];
  const modelIds = new Map<string, string>();

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-ensemble-failed-subcall",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e-obs",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const failing = (message: string) => ({
      status: 500,
      errorBody: { error: { message, type: "server_error" } },
    });
    const members: Array<[string, Parameters<typeof startOpenAiUpstream>[0]]> = [
      ["efs-ok-a", { nonStreamBody: PANEL_BODY }],
      ["efs-ok-b", { nonStreamBody: PANEL_BODY }],
      ["efs-fail-a", failing(PANEL_ERROR)],
      ["efs-fail-b", failing(PANEL_ERROR)],
      ["efs-judge-ok", { nonStreamBody: JUDGE_BODY }],
      ["efs-judge-ok-stream", { streamEvents: JUDGE_STREAM }],
      ["efs-judge-fail", failing(JUDGE_ERROR)],
    ];
    for (const [name, opts] of members) {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      const model = await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
        // One upstream call per sub-call, so a failure is final at once.
        retries: 0,
      });
      modelIds.set(name, model.id);
    }
    for (const c of cases) {
      const model = await seed.createModel({
        display_name: c.council,
        ensemble: {
          panel: c.panel.map((model) => ({ model })),
          judge: { model: c.judge },
          min_responses: 2,
        },
      });
      modelIds.set(c.council, model.id);
    }
    // Seeded last: the key authenticating implies the whole set is live.
    await seed.createApiKey({
      key_hash: createHash("sha256").update(CALLER).digest("hex"),
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      await res.arrayBuffer();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of upstreams) await u.close();
    await sls?.close();
  });

  async function rowsFor(requestId: string, count: number): Promise<Map<string, string>[]> {
    const matching = () =>
      slsLogsFor(sls!, LOGSTORE)
        .filter((l) => l.get("request_id") === requestId)
        .sort((a, b) => Number(a.get("attempt_index") ?? 0) - Number(b.get("attempt_index") ?? 0));
    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline && matching().length < count) {
      await new Promise((r) => setTimeout(r, 100));
    }
    // Settle past the count: a surplus row would ride the same export.
    await new Promise((r) => setTimeout(r, 500));
    return matching();
  }

  test.for(cases)("$label", { timeout: 60_000 }, async (c, ctx) => {
    if (!etcdReachable || !app || !sls) return ctx.skip();
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: c.council,
        messages: [{ role: "user", content: "ask the council" }],
        stream: c.stream,
      }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(c.status);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");

    const rows = await rowsFor(requestId, c.rows.length);
    expect(
      rows.map((r) => [
        r.get("attempt_index") ?? "0",
        r.get("attempt_kind") ?? "",
        r.get("status_code"),
        r.get("attempt_model") ?? "",
      ]),
    ).toEqual(c.rows);

    for (const [row, [, kind, status, model]] of rows.map((r, i) => [r, c.rows[i]] as const)) {
      expect(row.get("requested_model")).toBe(c.council);
      if (kind === "") {
        // The terminal row of the failed request names the ensemble.
        expect(row.get("model_id")).toBe(modelIds.get(c.council));
        continue;
      }
      // Every sub-call row names the sub-call's own model.
      expect(row.get("model_id"), `${kind} ${model}`).toBe(modelIds.get(model));
      if (status === "200") {
        expect(row.get("error_class") ?? "", `${kind} ${model}`).toBe("");
        continue;
      }
      expect(row.get("error_class"), `${kind} ${model}`).toBe("upstream_status");
      expect(row.get("error_message") ?? "", `${kind} ${model}`).toContain(
        kind === "judge" ? JUDGE_ERROR : PANEL_ERROR,
      );
      expect(row.get("prompt_tokens") ?? "0", `${kind} ${model}`).toBe("0");
      expect(row.get("completion_tokens") ?? "0", `${kind} ${model}`).toBe("0");
    }
  });
});
