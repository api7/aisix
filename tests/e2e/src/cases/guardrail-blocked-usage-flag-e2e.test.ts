import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  slsLogsFor,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1428: a guardrail refusal on /v1/responses must be
// findable under `guardrail_blocked = true`.
//
// The report: an input guardrail attached to the model a caller addresses
// refuses the request with 422 `content_filter`, and the usage row it
// produces carries `guardrail_blocked` at its `false` default. The row IS
// in the unfiltered feed, so the dashboard's Logs "Guardrail blocks" view
// — whose whole predicate is `guardrail_blocked = true` — comes back empty
// while the caller is being refused. To an operator that reads as "the
// gateway records no guardrail activity at all", which is a worse answer
// than a missing row.
//
// Driven over the four combinations the report names — direct model or
// routing-group parent, streaming or not — because they take different
// code inside the handler even though the input hook runs before target
// selection on all four. Read back off a real Aliyun-SLS export from a
// real `aisix` binary, so what is asserted is the row a consumer actually
// receives, not an in-process struct.
//
// The clean control at the end is what gives the four assertions meaning:
// the same models and the same guardrail, text it does not match, must
// produce a row with the flag OFF. A flag that merely tracked "a guardrail
// was configured" — or "the request failed" — would pass the first four
// and fail that one.

const CALLER_PLAINTEXT = "sk-guardrail-blocked-flag-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "LTAI_mock_ak";
const MOCK_AK_SECRET = "mock_ak_secret";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "guardrail-blocked-flag";

const FORBIDDEN_WORD = "blockedflagsentinel";
const DIRECT_MODEL = "gbf-direct";
const GROUP_MODEL = "gbf-group";

/** Each blocked probe plants a unique marker so its row can be found. */
interface Probe {
  model: string;
  stream: boolean;
  marker: string;
}

const PROBES: Probe[] = [
  { model: DIRECT_MODEL, stream: false, marker: "gbf-direct-nonstream-4a91" },
  { model: DIRECT_MODEL, stream: true, marker: "gbf-direct-stream-7c02" },
  { model: GROUP_MODEL, stream: false, marker: "gbf-group-nonstream-2e58" },
  { model: GROUP_MODEL, stream: true, marker: "gbf-group-stream-9b13" },
];

describe("guardrail_blocked e2e: a /v1/responses refusal reaches the Blocked view (#1428)", () => {
  let upstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  async function responses(
    model: string,
    input: string,
    stream: boolean,
  ): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, input, stream }),
    });
    // Drain: a streaming response only completes its telemetry once the
    // body is consumed.
    await res.text();
    return res;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_gbf",
        object: "response",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "fine" }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 1 },
      },
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "gbf-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
    });

    const pk = await seed.createProviderKey({
      display_name: "gbf-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: DIRECT_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // A group over the one direct member. The report's own workaround for
    // the separate member-guardrail gap (#1090) is to attach the rule to
    // the group instead, so the group parent is the shape an affected
    // operator is most likely to be running.
    await seed.createModel({
      display_name: GROUP_MODEL,
      routing: {
        strategy: "failover",
        targets: [{ model: DIRECT_MODEL }],
      },
    });

    // Env-scoped input guardrail: it governs the entry the caller
    // addresses, group parent included.
    await seed.createGuardrail({
      name: "gbf-guard",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });

    // The caller key is written LAST, after every other resource above.
    // The gateway runs one etcd watch over one prefix and applies its
    // events in revision order (`aisix-etcd`), so the moment this key
    // authenticates, everything written ahead of it — models, the routing
    // group, the guardrail, the exporter — is already in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [DIRECT_MODEL, GROUP_MODEL],
    });

    // ...which is why the readiness gate can stay independent of what the
    // tests assert. Gating on the guardrail's own 422 would make a
    // guardrail regression surface as a propagation timeout in `beforeAll`
    // instead of a failed assertion pointing at the cause.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  for (const { model, stream, marker } of PROBES) {
    test(`input block on ${model} (stream=${stream}) is recorded as a guardrail block`, async (ctx) => {
      if (!etcdReachable || !app || !sls) {
        ctx.skip();
        return;
      }
      const res = await responses(model, `${marker} ${FORBIDDEN_WORD}`, stream);
      expect(res.status).toBe(422);

      // The row exists in the unfiltered feed…
      const log = await waitForSlsLog(
        sls,
        LOGSTORE,
        (l) => (l.get("prompt") ?? "").includes(marker),
        `${model} stream=${stream} usage row`,
      );
      // …and the Blocked view's predicate finds it.
      expect(log.get("guardrail_blocked")).toBe("true");
      expect(log.get("status_code")).toBe("422");
      // Nothing was sent upstream, so nothing is billed.
      expect(log.get("prompt_tokens") ?? "0").toBe("0");
      expect(log.get("completion_tokens") ?? "0").toBe("0");
      // The caller-addressed entry, group parent included.
      expect(log.get("requested_model")).toBe(model);

      // Exactly one row per refusal: an input block refuses before any
      // target is contacted, so a group parent must not also emit a
      // per-attempt row.
      const rows = slsLogsFor(sls, LOGSTORE).filter((l) =>
        (l.get("prompt") ?? "").includes(marker),
      );
      expect(rows).toHaveLength(1);
    });
  }

  test("a request the same guardrail allows is not marked blocked", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const marker = "gbf-clean-6d47";
    const res = await responses(DIRECT_MODEL, `${marker} an ordinary question`, false);
    expect(res.status).toBe(200);

    const log = await waitForSlsLog(
      sls,
      LOGSTORE,
      (l) => (l.get("prompt") ?? "").includes(marker),
      "allowed request usage row",
    );
    expect(log.get("guardrail_blocked")).not.toBe("true");
    expect(log.get("status_code")).toBe("200");
  });
});
