import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { lz4DecompressBlock, startMockSls, type MockSls } from "../harness/sls-mock.js";

// E2E for api7/aisix#796 (follow-up to AISIX-Cloud#1074): the non-streaming
// ensemble sub-call events (every panel member + the judge) must run through
// the token-estimation fallback, and so must the panel members on the
// streaming path (the judge there was already wired by #794). With a
// usage-less backend, each sub-call event previously recorded silent zeros;
// now each is locally counted (prompt from the sub-call's own request,
// completion from its answer text) and flagged `usage_estimated`.
//
// Observable contract: an ensemble fans out to one usage event per sub-call
// (attempt_kind "panel" x N, then "judge"), all sharing the request_id and
// carrying the ensemble model as requested_model. They are telemetry-only
// (usage sink -> SLS), so the metadata_only SLS exporter is the wire we
// assert on: a `usage_estimated` marker appears once per flagged record
// (serialized only when true) and each sub-call's `attempt_model` is present.

const CALLER_PLAINTEXT = "sk-ensemble-est-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "mock-akid";
const MOCK_AK_SECRET = "mock-secret";
const SLS_PROJECT = "aisix-e2e-obs";
const META_LOGSTORE = "ens-est-meta-events";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Non-streaming 200 with NO usage block — the relay/transit-backend shape
// that #1074 targets. Panel members are always dispatched non-streaming
// (the executor buffers them), and the non-streaming judge answers this way
// too.
const nonStreamNoUsage = (content: string) => ({
  id: "chatcmpl-ens",
  object: "chat.completion",
  created: 0,
  model: "relay-compat-x",
  choices: [
    { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
  ],
  // deliberately no `usage`
});

// Streaming judge answer with no terminal usage chunk (relay ignores the
// gateway-injected stream_options.include_usage). Deltas are the only
// completion signal → estimated from "Hello world".
const JUDGE_STREAM_NO_USAGE = [
  '{"id":"j","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
  '{"id":"j","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
  '{"id":"j","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
  '{"id":"j","object":"chat.completion.chunk","model":"relay-compat-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
  "[DONE]",
];

describe("ensemble usage estimation e2e (api7/aisix#796): every panel + judge sub-call is estimated and flagged when the backend omits usage", () => {
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let memberAUp: OpenAiUpstream | undefined;
  let memberBUp: OpenAiUpstream | undefined;
  let judgeNsUp: OpenAiUpstream | undefined;
  let judgeStUp: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    memberAUp = await startOpenAiUpstream({
      nonStreamBody: nonStreamNoUsage("Panel answer A"),
    });
    memberBUp = await startOpenAiUpstream({
      nonStreamBody: nonStreamNoUsage("Panel answer B"),
    });
    judgeNsUp = await startOpenAiUpstream({
      nonStreamBody: nonStreamNoUsage("Final synthesized answer"),
    });
    judgeStUp = await startOpenAiUpstream({
      streamEvents: JUDGE_STREAM_NO_USAGE,
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "ens-est-sls-meta",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: META_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    // Direct sub-call models. All upstream model names are non-OpenAI on
    // purpose so the estimator falls back to cl100k_base.
    const seedModel = async (display: string, upstream: OpenAiUpstream) => {
      const pk = await seed!.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed!.createModel({
        display_name: display,
        provider: "openai",
        model_name: "relay-compat-x",
        provider_key_id: pk.id,
      });
    };
    await seedModel("est-ens-member-1", memberAUp);
    await seedModel("est-ens-member-2", memberBUp);
    await seedModel("est-ens-judge-ns", judgeNsUp);
    await seedModel("est-ens-judge-st", judgeStUp);

    // Two ensembles sharing the same panel; one judged non-streaming, one
    // judged streaming — the two sub-call emit families #796 wires.
    await seed.createModel({
      display_name: "est-ens-nonstream",
      ensemble: {
        panel: [{ model: "est-ens-member-1" }, { model: "est-ens-member-2" }],
        judge: { model: "est-ens-judge-ns" },
        min_responses: 2,
      },
    });
    await seed.createModel({
      display_name: "est-ens-stream",
      ensemble: {
        panel: [{ model: "est-ens-member-1" }, { model: "est-ens-member-2" }],
        judge: { model: "est-ens-judge-st" },
        min_responses: 2,
      },
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "est-ens-nonstream",
        "est-ens-stream",
        "est-ens-member-1",
        "est-ens-member-2",
        "est-ens-judge-ns",
        "est-ens-judge-st",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await sls?.close();
    await memberAUp?.close();
    await memberBUp?.close();
    await judgeNsUp?.close();
    await judgeStUp?.close();
  });

  function openai(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  /** Decode only the SLS PutLogs bodies appended at/after `startIndex`, so a
   * per-call assertion sees that call's sub-call records rather than the
   * shared logstore's whole history. */
  function decodedSince(startIndex: number): string {
    return sls!.requests
      .slice(startIndex)
      .filter((r) => r.logstore === META_LOGSTORE && r.rawSize > 0 && r.body.length > 0)
      .map((r) => lz4DecompressBlock(r.body, r.rawSize).toString("utf8"))
      .join(" ");
  }

  /** Split the decoded metadata into per-record chunks on the `schema_version`
   * anchor (SinkRecord serializes it once per record, ahead of the flattened
   * usage fields), so each chunk holds exactly one sub-call event. */
  function records(text: string): string[] {
    return text.split("schema_version").slice(1);
  }

  /** True when some record for `attemptModel` ALSO carries `usage_estimated`
   * (serialized only when the record was flagged). Checked per-record so an
   * already-flagged judge (via #794) cannot stand in for an unflagged panel
   * member — the exact gap #796 closes. */
  function subcallFlagged(text: string, attemptModel: string): boolean {
    return records(text).some(
      (r) => r.includes(attemptModel) && r.includes("usage_estimated"),
    );
  }

  /** Poll until every named sub-call has its own flagged record among the
   * events appended after `mark`. Pre-fix the non-streaming sub-calls carry
   * no flag at all, and the streaming panel members carry none (only the
   * #794 judge is flagged), so this fails until #796 wires them. */
  async function waitEnsembleFlagged(mark: number, attemptModels: string[]): Promise<void> {
    const deadline = Date.now() + 15_000;
    let text = "";
    while (Date.now() < deadline) {
      text = decodedSince(mark);
      if (attemptModels.every((m) => subcallFlagged(text, m))) return;
      await sleep(100);
    }
    const flagged = attemptModels.filter((m) => subcallFlagged(text, m));
    throw new Error(
      `ensemble sub-calls not all flagged within 15s: flagged [${flagged.join(",")}] ` +
        `of [${attemptModels.join(",")}]`,
    );
  }

  test("non-streaming ensemble: panel members + judge all estimated and flagged", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const client = openai();
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "est-ens-nonstream",
          messages: [{ role: "user", content: "hi" }],
        });
        return true;
      } catch {
        return false;
      }
    });

    const mark = sls!.requests.length;
    const resp = await client.chat.completions.create({
      model: "est-ens-nonstream",
      messages: [{ role: "user", content: "hi" }],
    });
    // The client-facing answer is the judge's synthesis; the aggregate usage
    // stays zero (no upstream reported any, and the gateway never fabricates
    // client-visible usage — estimation feeds telemetry only).
    expect(resp.choices[0]?.message?.content).toBe("Final synthesized answer");
    expect(resp.usage?.prompt_tokens ?? 0).toBe(0);
    expect(resp.usage?.completion_tokens ?? 0).toBe(0);

    await waitEnsembleFlagged(mark, [
      "est-ens-member-1",
      "est-ens-member-2",
      "est-ens-judge-ns",
    ]);
  }, 60_000);

  test("streaming ensemble: buffered panel members estimated alongside the streamed judge", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const client = openai();
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "est-ens-stream",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        });
        for await (const _ of probe) {
          /* drain */
        }
        return true;
      } catch {
        return false;
      }
    });

    const mark = sls!.requests.length;
    const stream = await client.chat.completions.create({
      model: "est-ens-stream",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    let content = "";
    for await (const chunk of stream) {
      content += chunk.choices[0]?.delta?.content ?? "";
      // Estimation must never fabricate a usage chunk on the client stream.
      expect(chunk.usage ?? null).toBeNull();
    }
    // The streamed answer is the judge's synthesis, restamped with the
    // ensemble model name.
    expect(content).toBe("Hello world");

    // Panel members (buffered, emitted on stream drop) are the #796 lockstep
    // extension; the judge was already flagged by #794. All three flagged.
    await waitEnsembleFlagged(mark, [
      "est-ens-member-1",
      "est-ens-member-2",
      "est-ens-judge-st",
    ]);
  }, 60_000);
});
