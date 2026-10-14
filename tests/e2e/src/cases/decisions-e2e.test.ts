import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type ReceivedRequest,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the two decision endpoints.
//
//   POST /v1/systemone — System One format: `state` + NAMED questions
//     (noul / choice / score), answered by an object keyed by name.
//   POST /v1/decisions — OpenAI Decisions format: `input` + an ORDERED
//     array of questions (predicate / choice / score), answered in order.
//
// The model's provider key decides the upstream: typesafe, azure (Microsoft
// Foundry) and openrouter speak System One; openai speaks Decisions. A
// request in the upstream's own format must reach it untouched but for
// `model`, and come back untouched but for `model` — so fields the gateway
// does not know pass in both directions. A request in the other format is
// converted both ways, and whatever cannot be represented is refused up
// front rather than silently changing the question being asked.
//
// Upstream wire shapes are the documented ones:
//   https://developers.openai.com/api/reference/resources/decisions
//   TypeSafe System One `POST /v1/systemone`.

const CALLER = "sk-decisions-e2e-caller";
const TPM_CALLER = "sk-decisions-e2e-tpm";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "decision-events";

/** A System One answer document, as TypeSafe returns it. */
const SYSTEMONE_ANSWER = {
  model: "jev-1.13.0",
  answers: {
    is_bug: { type: "noul", noul: 0.92 },
    team: {
      type: "choice",
      choice: "engineering",
      confidence: 1.0,
      probabilities: { billing: 0.0, engineering: 1.0 },
    },
    severity: {
      type: "score",
      score: 2.0,
      confidence: 1.0,
      legend: { "0": "low", "1": "medium: degraded service", "2": "high" },
      probabilities: { "1": 0.25, "2": 0.75 },
    },
    // Positional keys, for questions that carry no usable names.
    "0": { type: "noul", noul: 0.4 },
    "1": {
      type: "choice",
      choice: "true",
      confidence: 0.7,
      probabilities: { true: 0.7, false: 0.3 },
    },
  },
  usage: { input_tokens: 381, output_tokens: 62 },
  vendor_extension: { kept: true },
};

/** An OpenAI Decisions answer document. */
const DECISIONS_ANSWER = {
  model: "gpt-6-luna",
  answers: [
    { type: "predicate", name: "is_bug", probability: 0.99 },
    { type: "refusal", name: "team" },
    {
      type: "score",
      name: "severity",
      score: 1.98,
      probabilities: [
        { value: 0, label: "low", probability: 0.0 },
        { value: 1, label: "medium", probability: 0.02 },
        { value: 2, label: "high", probability: 0.98 },
      ],
      confidence: 0.97,
    },
  ],
  usage: {
    input_tokens: 388,
    input_tokens_details: { cached_tokens: 100, cache_write_tokens: 7 },
    output_tokens: 20,
    output_tokens_details: { reasoning_tokens: 5 },
    total_tokens: 408,
  },
  vendor_extension: "kept",
};

/** OpenRouter's System One answer adds an id, a provider and a cost. */
const OPENROUTER_ANSWER = {
  model: "typesafe/jev-1.13-20260917",
  answers: { is_bug: { type: "noul", noul: 0.93 } },
  usage: { input_tokens: 381, output_tokens: 62, cost: 0.000016002 },
  id: "gen-dec-1791966046",
  provider: "TypeSafe",
};

/** The canonical System One request used across the suite. */
function systemOneBody(model: string, extra: Record<string, unknown> = {}) {
  return {
    model,
    state: "The API returns 500 on every call since this morning.",
    questions: {
      is_bug: {
        type: "noul",
        instructions: "Is this a software bug report?",
        criteria: { true: "the software misbehaves", false: null },
      },
      team: {
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: { billing: "Charges, invoices, refunds", engineering: "Bugs, errors, outages" },
      },
      severity: {
        type: "score",
        instructions: "How severe is this?",
        criteria: ["low", { level: "medium" }, "high"],
      },
    },
    ...extra,
  };
}

/** The canonical Decisions request used across the suite. */
function decisionsBody(model: string, extra: Record<string, unknown> = {}) {
  return {
    model,
    input: "The API returns 500 on every call since this morning.",
    questions: [
      { type: "predicate", name: "is_bug", instructions: "Is this a software bug report?" },
      {
        type: "choice",
        name: "team",
        instructions: "Which team should handle this?",
        choices: [
          { value: "billing", description: "Charges, invoices, refunds" },
          { value: "engineering", description: "Bugs, errors, outages" },
        ],
      },
      {
        type: "score",
        name: "severity",
        instructions: "How severe is this?",
        levels: [
          { label: "low" },
          { label: "medium", description: "degraded service" },
          { label: "high" },
        ],
      },
    ],
    ...extra,
  };
}

interface Posted {
  status: number;
  body: any;
  requestId: string;
}

describe("decision endpoints e2e: /v1/systemone + /v1/decisions", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let typesafe: OpenAiUpstream | undefined;
  let openai: OpenAiUpstream | undefined;
  let foundry: OpenAiUpstream | undefined;
  let openrouter: OpenAiUpstream | undefined;
  let failing: OpenAiUpstream | undefined;
  let rejecting: OpenAiUpstream | undefined;
  let anthropic: OpenAiUpstream | undefined;

  async function post(path: string, body: unknown, caller = CALLER): Promise<Posted> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = JSON.parse(text);
    } catch {
      // Left as text: the assertion that reads it will say what came back.
    }
    return {
      status: res.status,
      body: parsed,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
    };
  }

  /** The single upstream call `upstream` received since `from`. */
  function onlyCallSince(upstream: OpenAiUpstream, from: number): ReceivedRequest {
    const calls = upstream.receivedRequests.slice(from);
    expect(calls).toHaveLength(1);
    return calls[0]!;
  }

  async function rowFor(requestId: string): Promise<Map<string, string>> {
    expect(requestId).not.toBe("");
    return waitForSlsLog(
      sls!,
      LOGSTORE,
      (log) => log.get("request_id") === requestId && log.get("status_code") === "200",
      `the usage row of ${requestId}`,
    );
  }

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    typesafe = await startOpenAiUpstream({ nonStreamBody: SYSTEMONE_ANSWER });
    openai = await startOpenAiUpstream({ nonStreamBody: DECISIONS_ANSWER });
    foundry = await startOpenAiUpstream({ nonStreamBody: SYSTEMONE_ANSWER });
    openrouter = await startOpenAiUpstream({ nonStreamBody: OPENROUTER_ANSWER });
    failing = await startOpenAiUpstream({
      status: 503,
      errorBody: { detail: "overloaded" },
    });
    rejecting = await startOpenAiUpstream({
      status: 400,
      errorBody: { detail: "Noul question must have criteria or instructions: is_bug" },
    });
    anthropic = await startOpenAiUpstream({ nonStreamBody: SYSTEMONE_ANSWER });
    sls = await startMockSls();

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-decisions",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const direct = async (
      name: string,
      modelName: string,
      pk: Record<string, unknown>,
      extra: Record<string, unknown> = {},
    ) => {
      const key = await seed.createProviderKey({ display_name: `${name}-pk`, ...pk });
      await seed.createModel({
        display_name: name,
        provider: pk.provider,
        model_name: modelName,
        provider_key_id: key.id,
        ...extra,
      });
    };
    await direct("dec-typesafe", "jev-latest", {
      provider: "typesafe",
      adapter: undefined,
      secret: "ts-secret",
      api_base: typesafe.baseUrl,
    });
    await direct("dec-openai", "gpt-6-luna", {
      provider: "openai",
      secret: "sk-openai",
      api_base: `${openai.baseUrl}/v1`,
    });
    await direct("dec-foundry", "jev-foundry-deployment", {
      provider: "azure",
      adapter: "azure-openai",
      secret: "foundry-resource-key",
      api_base: `${foundry.baseUrl}/api/projects/proj-1`,
    });
    await direct("dec-openrouter", "typesafe/jev-1.13", {
      provider: "openrouter",
      secret: "sk-or",
      api_base: `${openrouter.baseUrl}/api/v1`,
    });
    await direct("dec-failing", "jev-latest", {
      provider: "typesafe",
      adapter: undefined,
      secret: "ts-secret",
      api_base: failing.baseUrl,
    });
    await direct("dec-rejecting", "jev-latest", {
      provider: "typesafe",
      adapter: undefined,
      secret: "ts-secret",
      api_base: rejecting.baseUrl,
    });
    await direct("dec-anthropic", "claude-haiku-4-5", {
      provider: "anthropic",
      adapter: "anthropic",
      secret: "sk-ant",
      api_base: anthropic.baseUrl,
    });
    await direct(
      "dec-embedding",
      "text-embedding-3-small",
      { provider: "openai", secret: "sk-openai", api_base: `${openai.baseUrl}/v1` },
      { embedding: { dimensions: 4, normalize: true } },
    );
    await seed.createModel({
      display_name: "dec-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "dec-failing" }, { model: "dec-typesafe" }],
        max_fallbacks: 1,
      },
    });
    await seed.createModel({
      display_name: "dec-semantic",
      semantic: {
        embedding_model: "dec-embedding",
        routes: [
          { name: "bugs", target: "dec-typesafe", examples: ["the api is down"], threshold: 0.5 },
        ],
        default: "dec-typesafe",
        match: { threshold: 0.5 },
      },
    });
    await seed.createModel({
      display_name: "dec-ensemble",
      ensemble: {
        panel: [{ model: "dec-typesafe" }, { model: "dec-openai" }],
        judge: { model: "dec-openai" },
        min_responses: 2,
      },
    });

    // Caller keys last: the gate on the last one implies everything above.
    await seed.createApiKey({ key_hash: hash(CALLER), allowed_models: ["*"] });
    await seed.createApiKey({
      key_hash: hash(TPM_CALLER),
      allowed_models: ["dec-typesafe"],
      rate_limit: { tpm: 400 },
    });
    const probe = new ProxyClient(app.proxyUrl, TPM_CALLER);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    for (const u of [typesafe, openai, foundry, openrouter, failing, rejecting, anthropic]) {
      await u?.close();
    }
    await sls?.close();
  });

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return false;
    }
    return true;
  };

  test("System One to a System One upstream: verbatim both ways but for model", async (ctx) => {
    if (!ready(ctx)) return;
    const from = typesafe!.receivedRequests.length;
    const sent = systemOneBody("dec-typesafe", { x_trace: "abc" });
    const res = await post("/v1/systemone", sent);

    expect(res.status).toBe(200);
    const call = onlyCallSince(typesafe!, from);
    expect(call.path).toBe("/v1/systemone");
    expect(call.headers["authorization"]).toBe("Bearer ts-secret");
    expect(JSON.parse(call.body)).toEqual({ ...sent, model: "jev-latest" });

    expect(res.body).toEqual({ ...SYSTEMONE_ANSWER, model: "dec-typesafe" });

    const row = await rowFor(res.requestId);
    expect(row.get("operation")).toBe("systemone");
    expect(row.get("prompt_tokens")).toBe("381");
    expect(row.get("completion_tokens")).toBe("62");
    expect(row.get("provider_model_version")).toBe("jev-1.13.0");
    expect(row.get("requested_model")).toBe("dec-typesafe");
  });

  test("Decisions to an OpenAI upstream: verbatim both ways but for model", async (ctx) => {
    if (!ready(ctx)) return;
    const from = openai!.receivedRequests.length;
    const sent = decisionsBody("dec-openai", { safety_identifier: "user-7", x_trace: "abc" });
    const res = await post("/v1/decisions", sent);

    expect(res.status).toBe(200);
    const call = onlyCallSince(openai!, from);
    expect(call.path).toBe("/v1/decisions");
    expect(call.headers["authorization"]).toBe("Bearer sk-openai");
    expect(JSON.parse(call.body)).toEqual({ ...sent, model: "gpt-6-luna" });

    expect(res.body).toEqual({ ...DECISIONS_ANSWER, model: "dec-openai" });

    const row = await rowFor(res.requestId);
    expect(row.get("operation")).toBe("decisions");
    expect(row.get("prompt_tokens")).toBe("388");
    expect(row.get("completion_tokens")).toBe("20");
    expect(row.get("cached_prompt_tokens")).toBe("100");
    expect(row.get("cache_write_tokens")).toBe("7");
    expect(row.get("reasoning_tokens")).toBe("5");
    expect(row.get("total_tokens")).toBe("408");
    expect(row.get("provider_model_version")).toBe("gpt-6-luna");
  });

  test("Decisions to a System One upstream: converted request and answers", async (ctx) => {
    if (!ready(ctx)) return;
    const from = typesafe!.receivedRequests.length;
    const res = await post("/v1/decisions", decisionsBody("dec-typesafe"));

    expect(res.status).toBe(200);
    const call = onlyCallSince(typesafe!, from);
    expect(call.path).toBe("/v1/systemone");
    // Names are distinct, so they become the keys; score levels travel as
    // `label: description`; nothing the System One format lacks is sent.
    expect(JSON.parse(call.body)).toEqual({
      model: "jev-latest",
      state: "The API returns 500 on every call since this morning.",
      questions: {
        is_bug: { type: "noul", instructions: "Is this a software bug report?" },
        team: {
          type: "choice",
          instructions: "Which team should handle this?",
          criteria: { billing: "Charges, invoices, refunds", engineering: "Bugs, errors, outages" },
        },
        severity: {
          type: "score",
          instructions: "How severe is this?",
          criteria: ["low", "medium: degraded service", "high"],
        },
      },
    });

    // Answers come back in question order, labelled from the request, with a
    // level the upstream gave no probability reported as 0.
    expect(res.body).toEqual({
      model: "dec-typesafe",
      answers: [
        { type: "predicate", name: "is_bug", probability: 0.92 },
        {
          type: "choice",
          name: "team",
          choice: "engineering",
          probabilities: [
            { value: "billing", probability: 0 },
            { value: "engineering", probability: 1 },
          ],
          confidence: 1,
        },
        {
          type: "score",
          name: "severity",
          score: 2,
          probabilities: [
            { value: 0, label: "low", probability: 0 },
            { value: 1, label: "medium", probability: 0.25 },
            { value: 2, label: "high", probability: 0.75 },
          ],
          confidence: 1,
        },
      ],
      usage: {
        input_tokens: 381,
        input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
        output_tokens: 62,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 443,
      },
    });

    const row = await rowFor(res.requestId);
    expect(row.get("operation")).toBe("decisions");
    expect(row.get("prompt_tokens")).toBe("381");
    expect(row.get("completion_tokens")).toBe("62");
  });

  test("Decisions without distinct names: positional keys, typed booleans, missing answer refused", async (ctx) => {
    if (!ready(ctx)) return;
    const from = typesafe!.receivedRequests.length;
    const res = await post("/v1/decisions", {
      model: "dec-typesafe",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "first part" },
            { type: "input_text", text: "second part" },
          ],
        },
        { role: "user", content: "third part" },
      ],
      questions: [
        { type: "predicate", name: "dup", instructions: "Is it urgent?" },
        {
          type: "choice",
          name: "dup",
          instructions: "Should we page someone?",
          choices: [{ value: true, description: "yes, now" }, { value: false }],
        },
        {
          type: "score",
          instructions: "How confident is the reporter?",
          levels: [{ label: "unsure" }, { label: "certain" }],
        },
      ],
    });

    expect(res.status).toBe(200);
    const sent = JSON.parse(onlyCallSince(typesafe!, from).body);
    expect(sent.state).toBe("first part\n\nsecond part\n\nthird part");
    expect(Object.keys(sent.questions)).toEqual(["0", "1", "2"]);
    expect(sent.questions["1"].criteria).toEqual({ true: "yes, now", false: null });

    expect(res.body.answers).toEqual([
      { type: "predicate", name: "dup", probability: 0.4 },
      {
        type: "choice",
        name: "dup",
        choice: true,
        probabilities: [
          { value: true, probability: 0.7 },
          { value: false, probability: 0.3 },
        ],
        confidence: 0.7,
      },
      // The upstream answered no "2".
      { type: "refusal", name: null },
    ]);
  });

  test("System One to an OpenAI upstream: converted request, refusal omitted", async (ctx) => {
    if (!ready(ctx)) return;
    const from = openai!.receivedRequests.length;
    const res = await post("/v1/systemone", systemOneBody("dec-openai"));

    expect(res.status).toBe(200);
    const call = onlyCallSince(openai!, from);
    expect(call.path).toBe("/v1/decisions");
    expect(JSON.parse(call.body)).toEqual({
      model: "gpt-6-luna",
      input: "The API returns 500 on every call since this morning.",
      questions: [
        {
          type: "predicate",
          name: "is_bug",
          instructions: "Is this a software bug report?\n\nAnswer true when: the software misbehaves",
        },
        {
          type: "choice",
          name: "team",
          instructions: "Which team should handle this?",
          choices: [
            { value: "billing", description: "Charges, invoices, refunds" },
            { value: "engineering", description: "Bugs, errors, outages" },
          ],
        },
        {
          type: "score",
          name: "severity",
          instructions: "How severe is this?",
          levels: [{ label: "low" }, { label: '{"level":"medium"}' }, { label: "high" }],
        },
      ],
    });

    // `team` was refused upstream, so it is absent; the score legend carries
    // the request's own criteria.
    expect(res.body).toEqual({
      model: "dec-openai",
      answers: {
        is_bug: { type: "noul", noul: 0.99 },
        severity: {
          type: "score",
          score: 1.98,
          confidence: 0.97,
          legend: { "0": "low", "1": { level: "medium" }, "2": "high" },
          probabilities: { "0": 0, "1": 0.02, "2": 0.98 },
        },
      },
      usage: { input_tokens: 388, output_tokens: 20 },
    });

    const row = await rowFor(res.requestId);
    expect(row.get("operation")).toBe("systemone");
    expect(row.get("cached_prompt_tokens")).toBe("100");
  });

  test("conversions that would change the question are refused before the upstream", async (ctx) => {
    if (!ready(ctx)) return;
    const tsFrom = typesafe!.receivedRequests.length;
    const oaFrom = openai!.receivedRequests.length;

    const image = await post("/v1/decisions", {
      model: "dec-typesafe",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "is this a cat?" },
            { type: "input_image", image_url: "data:image/png;base64,AAAA" },
          ],
        },
      ],
      questions: [{ type: "predicate", instructions: "Is it a cat?" }],
    });
    expect(image.status).toBe(400);
    expect(image.body.error.message).toContain("input_image");

    const sid = await post("/v1/decisions", decisionsBody("dec-typesafe", { safety_identifier: "u1" }));
    expect(sid.status).toBe(400);
    expect(sid.body.error.message).toContain("safety_identifier");

    const single = await post("/v1/systemone", {
      model: "dec-openai",
      state: "x",
      questions: { pick: { type: "choice", criteria: { only: null } } },
    });
    expect(single.status).toBe(400);
    expect(single.body.error.message).toContain("at least 2");

    expect(typesafe!.receivedRequests.length).toBe(tsFrom);
    expect(openai!.receivedRequests.length).toBe(oaFrom);
  });

  test("Microsoft Foundry: derived endpoint, api-key auth, deployment as model", async (ctx) => {
    if (!ready(ctx)) return;
    const from = foundry!.receivedRequests.length;
    const res = await post("/v1/systemone", systemOneBody("dec-foundry"));

    expect(res.status).toBe(200);
    const call = onlyCallSince(foundry!, from);
    expect(call.path).toBe("/providers/microsoft/v1/systemone");
    expect(call.headers["api-key"]).toBe("foundry-resource-key");
    expect(call.headers["authorization"]).toBeUndefined();
    expect(JSON.parse(call.body).model).toBe("jev-foundry-deployment");
    expect(res.body.model).toBe("dec-foundry");
  });

  test("OpenRouter: its extra response fields are tolerated on both entries", async (ctx) => {
    if (!ready(ctx)) return;
    const from = openrouter!.receivedRequests.length;
    const s1 = await post("/v1/systemone", {
      model: "dec-openrouter",
      state: "x",
      questions: { is_bug: { type: "noul", instructions: "Bug?" } },
    });
    expect(s1.status).toBe(200);
    expect(s1.body).toEqual({ ...OPENROUTER_ANSWER, model: "dec-openrouter" });
    const call = onlyCallSince(openrouter!, from);
    expect(call.path).toBe("/api/v1/systemone");
    expect(call.headers["authorization"]).toBe("Bearer sk-or");

    const dec = await post("/v1/decisions", {
      model: "dec-openrouter",
      input: "x",
      questions: [{ type: "predicate", name: "is_bug", instructions: "Bug?" }],
    });
    expect(dec.status).toBe(200);
    expect(dec.body.answers).toEqual([{ type: "predicate", name: "is_bug", probability: 0.93 }]);
    expect(dec.body.usage.total_tokens).toBe(443);
  });

  test("a model group fails over a 5xx target to the next one", async (ctx) => {
    if (!ready(ctx)) return;
    const failFrom = failing!.receivedRequests.length;
    const okFrom = typesafe!.receivedRequests.length;
    const res = await post("/v1/systemone", systemOneBody("dec-group"));

    expect(res.status).toBe(200);
    expect(failing!.receivedRequests.length).toBeGreaterThan(failFrom);
    expect(typesafe!.receivedRequests.length).toBe(okFrom + 1);
    expect(res.body.model).toBe("dec-group");
  });

  test("an upstream 4xx reaches the caller with the upstream's message", async (ctx) => {
    if (!ready(ctx)) return;
    const res = await post("/v1/systemone", systemOneBody("dec-rejecting"));
    expect(res.status).toBe(400);
    expect(res.body.error.message).toBe(
      "Noul question must have criteria or instructions: is_bug",
    );
  });

  test("unsupported kinds and providers are refused with a 400 that names them", async (ctx) => {
    if (!ready(ctx)) return;
    for (const [model, kind] of [
      ["dec-semantic", "semantic"],
      ["dec-ensemble", "ensemble"],
      ["dec-embedding", "embedding"],
    ]) {
      for (const path of ["/v1/systemone", "/v1/decisions"]) {
        const res = await post(path, { model, state: "x", input: "x", questions: {} });
        expect(res.status, `${path} ${model}`).toBe(400);
        expect(res.body.error.message).toContain(`\`${kind}\``);
        expect(res.body.error.message).toContain(path);
      }
    }

    const from = anthropic!.receivedRequests.length;
    const res = await post("/v1/decisions", decisionsBody("dec-anthropic"));
    expect(res.status).toBe(400);
    expect(res.body.error.type).toBe("invalid_request_error");
    expect(res.body.error.message).toContain("anthropic");
    expect(res.body.error.message).toContain("/v1/decisions");
    expect(anthropic!.receivedRequests.length).toBe(from);
  });

  test("the reported tokens are committed against the caller's TPM", async (ctx) => {
    if (!ready(ctx)) return;
    // 381 + 62 = 443 tokens against a 400-token minute: the first call is
    // admitted, and only a committed count can refuse the second.
    const first = await post("/v1/systemone", systemOneBody("dec-typesafe"), TPM_CALLER);
    expect(first.status).toBe(200);
    const second = await post("/v1/systemone", systemOneBody("dec-typesafe"), TPM_CALLER);
    expect(second.status).toBe(429);
  });
});
