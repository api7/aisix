import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  awaitWindowHeadroom,
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  slsLogsFor,
  waitConfigPropagation,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// An ensemble whose caller hangs up after the panel answered but before the
// judge did: both panel members already round-tripped their upstreams, so the
// request owes their usage rows next to its 499 row, and their tokens count
// against the caller's rate limit like a completed request's would.
const PROVIDER_SECRET = "sk-mock-ensemble-cancel";
const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "ensemble-cancel-events";

const NONSTREAM_CALLER = "sk-ensemble-cancel-nonstream";
const STREAM_CALLER = "sk-ensemble-cancel-stream";
const COUNCIL = "ec-council";
const PANEL_USAGE = { prompt_tokens: 5, completion_tokens: 11, total_tokens: 16 };

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

function panelBody(id: string) {
  return {
    id,
    object: "chat.completion",
    created: 1_700_000_000,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content: `answer ${id}` }, finish_reason: "stop" },
    ],
    usage: PANEL_USAGE,
  };
}

describe("an ensemble abandoned after its panel answered", () => {
  let etcdReachable = false;
  let panelA: OpenAiUpstream | undefined;
  let panelB: OpenAiUpstream | undefined;
  let judge: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    panelA = await startOpenAiUpstream({ nonStreamBody: panelBody("chatcmpl-ec-a") });
    panelB = await startOpenAiUpstream({ nonStreamBody: panelBody("chatcmpl-ec-b") });
    // Never answers within the test: the caller is gone long before.
    judge = await startOpenAiUpstream({ responseDelayMs: 30_000, streamEvents: ["[DONE]"] });
    sls = await startMockSls();

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-ensemble-cancel",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    for (const [name, upstream] of [
      ["ec-panel-a", panelA],
      ["ec-panel-b", panelB],
      ["ec-judge", judge],
    ] as const) {
      const pk = await seed.createProviderKey({
        display_name: `${name}-pk`,
        secret: PROVIDER_SECRET,
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    }
    await seed.createModel({
      display_name: COUNCIL,
      ensemble: {
        panel: [{ model: "ec-panel-a" }, { model: "ec-panel-b" }],
        judge: { model: "ec-judge" },
        min_responses: 2,
      },
    });
    // Capped below what one panel member costs, so whether the abandoned
    // request was charged shows in the next request's admission. Seeded
    // last, so authenticating implies everything above is in the snapshot.
    for (const caller of [NONSTREAM_CALLER, STREAM_CALLER]) {
      await seed.createApiKey({
        key_hash: sha256(caller),
        allowed_models: ["*"],
        rate_limit: { tpm: 10 },
      });
    }
    for (const caller of [NONSTREAM_CALLER, STREAM_CALLER]) {
      const proxy = new ProxyClient(app.proxyUrl, caller);
      await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
    }
  });

  afterAll(async () => {
    await app?.exit();
    await panelA?.close();
    await panelB?.close();
    await judge?.close();
    await sls?.close();
  });

  const chat = (caller: string, stream: boolean, signal?: AbortSignal) =>
    fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: COUNCIL,
        messages: [{ role: "user", content: "hang up on me" }],
        stream,
      }),
      signal,
    });

  for (const stream of [false, true]) {
    test(
      `${stream ? "streaming" : "non-streaming"}: the panel's rows precede the 499 and its tokens are charged`,
      async (ctx) => {
        if (!etcdReachable || !app || !judge || !sls) {
          ctx.skip();
          return;
        }
        const caller = stream ? STREAM_CALLER : NONSTREAM_CALLER;
        // The abandoned request's charge and the admission that must refuse
        // the next one have to land in the same per-minute TPM window.
        await awaitWindowHeadroom();
        const judgeCallsBefore = judge.receivedRequests.length;
        const slsFrom = sls.requests.length;
        const controller = new AbortController();
        const inflight = chat(caller, stream, controller.signal);
        // Abort once the judge has been called: the panel has answered, and
        // the judge is 30s from answering, so no response head exists yet.
        for (let i = 0; i < 400 && judge.receivedRequests.length === judgeCallsBefore; i++) {
          await new Promise((r) => setTimeout(r, 25));
        }
        expect(judge.receivedRequests.length, "the judge was never called").toBeGreaterThan(
          judgeCallsBefore,
        );
        controller.abort();
        await expect(inflight).rejects.toThrow();

        // This request's rows, joined on the 499 row's request id (the
        // aborted fetch never read the response's request-id header).
        let rows: Map<string, string>[] = [];
        const deadline = Date.now() + 20_000;
        while (Date.now() < deadline) {
          const logs = slsLogsFor(sls, LOGSTORE, slsFrom).filter(
            (l) => l.get("requested_model") === COUNCIL,
          );
          const cancelled = logs.find((l) => l.get("status_code") === "499");
          rows = cancelled ? logs.filter((l) => l.get("request_id") === cancelled.get("request_id")) : [];
          if (rows.length >= 3) break;
          await new Promise((r) => setTimeout(r, 100));
        }
        rows.sort((a, b) => Number(a.get("attempt_index") ?? "0") - Number(b.get("attempt_index") ?? "0"));
        expect(
          rows.map((r) => [r.get("attempt_index") ?? "0", r.get("attempt_kind") ?? "", r.get("status_code")]),
        ).toEqual([
          ["0", "panel", "200"],
          ["1", "panel", "200"],
          ["2", "", "499"],
        ]);
        for (const panel of rows.slice(0, 2)) {
          expect(panel.get("prompt_tokens")).toBe("5");
          expect(panel.get("completion_tokens")).toBe("11");
        }

        // 32 panel tokens against a 10-token budget: refused at admission.
        const next = await chat(caller, false, AbortSignal.timeout(5_000));
        expect(next.status).toBe(429);
      },
      60_000,
    );
  }
});
