import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  droppedRows,
  EtcdClient,
  etcdEndpoint,
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
import { pickFreePort } from "../harness/ports.js";

// A semantic router whose classifier names its direct model by
// `model_id` — the shape the control plane projects — must keep working
// after `aisix export` moves it into a resources file: the export carries
// the reference and the model under the control-plane id, and the
// file-mode gateway has to call that model and attribute the decision call
// to it, exactly as the etcd-backed gateway does.
//
// A real aisix DP + etcd, `aisix export`, then a second gateway in file
// mode; the decision and chat upstreams are local endpoints, and the
// usage events are read from the SLS export.

const execFileP = promisify(execFile);
const BIN_PATH =
  process.env.AISIX_BIN ?? join(process.cwd(), "..", "..", "target", "debug", "aisix");

const CALLER_PLAINTEXT = "sk-export-classifier-model-ref";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "export-classifier-model-ref";
const DECISION_SECRET = "ts-secret-export-classifier";

const JEV_MODEL = "ecm-jev";
const JEV_UPSTREAM_MODEL = "jev-1.13.0";
const GEMINI_MODEL = "ecm-gemini-decision";
const GEMINI_UPSTREAM_MODEL = "gemini-decision-1";
const GEMINI_SECRET = "gemini-secret-export-classifier";
const GEMINI_PATH = "/gemini-shim/v1beta/decisions";
const NOT_TYPESAFE_MODEL = "ecm-not-typesafe";
const ROUTER = "ecm-router";
const BAD_ROUTER = "ecm-router-not-typesafe";
const GEMINI_ROUTER = "ecm-gemini-router";
const BAD_GEMINI_ROUTER = "ecm-router-not-gemini-decision";
const ROUTE_TARGET = "ecm-route-target";
const ROUTE_DEFAULT = "ecm-route-default";
const TARGET_UPSTREAM_MODEL = "gpt-route-target";
const DEFAULT_UPSTREAM_MODEL = "gpt-route-default";
const PROMPT = "fix my python script";
const LOW_CONFIDENCE_PROMPT = "ambiguous request";
const UNKNOWN_CHOICE_PROMPT = "unusable classifier answer";
const EXPORT_TIMEOUT_MS = 30_000;
const GEMINI_ROUTES = [
  { name: "General questions", target: ROUTE_DEFAULT, description: "General knowledge questions" },
  { name: "Developer support / scripting", target: ROUTE_TARGET, description: "Help writing or debugging code" },
];

interface DecisionCall {
  method?: string;
  path?: string;
  headers: IncomingHttpHeaders;
  body: {
    model?: string;
    state?: string;
    input?: string;
    questions?: { route?: { type: string; instructions: string; criteria: Record<string, string> } };
  };
}

async function startDecisionMock(): Promise<{
  baseUrl: string;
  calls: DecisionCall[];
  close(): Promise<void>;
}> {
  const calls: DecisionCall[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}") as DecisionCall["body"];
      calls.push({ method: req.method, path: req.url, headers: req.headers, body });
      res.setHeader("content-type", "application/json");
      if (req.url === GEMINI_PATH) {
        const options = Object.keys(body.questions?.route?.criteria ?? {});
        const choice = body.input === UNKNOWN_CHOICE_PROMPT ? "unexpected_option" : options[1];
        const confidence = body.input === LOW_CONFIDENCE_PROMPT ? 0.4 : 0.93;
        res.end(JSON.stringify({
          answers: { route: { type: "choice", choice, confidence } },
          usage: { total_input_tokens: 100, total_output_tokens: 0, total_tokens: 100 },
        }));
        return;
      }
      res.end(
        JSON.stringify({
          id: "dec-export",
          model: body.model,
          answers: { route: { type: "choice", choice: "code", confidence: 0.93 } },
          usage: { input_tokens: 300, output_tokens: 20 },
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    calls,
    async close() {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

type Row = Map<string, string>;
const isChild = (row: Row) => row.has("internal_call_purpose");

describe("aisix export: a classifier that names its direct model by id", () => {
  let etcdApp: SpawnedApp | undefined;
  let fileApp: SpawnedApp | undefined;
  let sls: MockSls | undefined;
  let decisions: Awaited<ReturnType<typeof startDecisionMock>> | undefined;
  let upstream: OpenAiUpstream | undefined;
  let exported = "";
  let jevId = "";
  let geminiId = "";
  let geminiKeyId = "";

  const slsEnv = {
    [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
    [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
  };

  async function chat(app: SpawnedApp, model: string, prompts = [PROMPT]) {
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "x-caller-context": "client-header",
      },
      body: JSON.stringify({
        model,
        messages: prompts.map((content) => ({ role: "user", content })),
      }),
    });
    const text = await res.text();
    return {
      status: res.status,
      text,
      requestId: res.headers.get("x-aisix-request-id") ?? "",
      route: res.headers.get("x-aisix-route"),
    };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    expect(await etcd.ping(), "this e2e requires etcd").toBe(true);

    sls = await startMockSls();
    decisions = await startDecisionMock();
    upstream = await startOpenAiUpstream();
    etcdApp = await spawnApp({ exposeDebugHeaders: true, extraEnv: slsEnv });
    const seed = new SeedClient(etcd, etcdApp.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "ecm-sls",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const chatKey = await seed.createProviderKey({
      display_name: "ecm-chat-pk",
      api_key: "sk-chat-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    for (const [name, upstreamModel] of [
      [ROUTE_TARGET, TARGET_UPSTREAM_MODEL],
      [ROUTE_DEFAULT, DEFAULT_UPSTREAM_MODEL],
    ]) {
      await seed.createModel({
        display_name: name,
        provider: "openai",
        model_name: upstreamModel,
        provider_key_id: chatKey.id,
      });
    }
    const jevKey = await seed.createProviderKey({
      display_name: "ecm-jev-pk",
      api_key: DECISION_SECRET,
      provider: "typesafe",
      adapter: undefined,
      api_base: decisions.baseUrl,
    });
    const jev = await seed.createModel({
      display_name: JEV_MODEL,
      provider: "typesafe",
      model_name: JEV_UPSTREAM_MODEL,
      provider_key_id: jevKey.id,
    });
    jevId = jev.id;
    // A direct model the classifier must refuse: its key is not typesafe.
    const notTypesafe = await seed.createModel({
      display_name: NOT_TYPESAFE_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: chatKey.id,
    });
    const router = (name: string, classifier: Record<string, unknown>) =>
      seed.createModel({
        display_name: name,
        semantic: {
          classifier: { type: "jev", timeout_ms: 2_000, ...classifier },
          routes: [{ name: "code", target: ROUTE_TARGET, description: "programming questions" }],
          default: ROUTE_DEFAULT,
        },
      });
    // Both spellings, as the control plane writes them for older gateways;
    // the inline pair names a key the classifier must not use.
    await router(ROUTER, {
      model_id: jev.id,
      provider_key_id: chatKey.id,
      model: "jev-inline-must-not-be-sent",
    });
    await router(BAD_ROUTER, { model_id: notTypesafe.id, on_failure: "fail" });
    const geminiKey = await seed.createProviderKey({
      display_name: "ecm-gemini-key",
      api_key: GEMINI_SECRET,
      provider: "google",
      api_base: `${upstream.baseUrl}/v1beta/openai`,
      apis: { gemini: { base: `${decisions.baseUrl}/gemini-shim/v1beta/` } },
      request: {
        default_headers: { "x-classifier-id": "${model.id}" },
        forward_client_headers: ["x-caller-context"],
        default_body_fields: { proxy_hint: "gemini-shim" },
      },
    });
    geminiKeyId = geminiKey.id;
    const gemini = await seed.createModel({
      display_name: GEMINI_MODEL,
      provider: "google",
      model_name: GEMINI_UPSTREAM_MODEL,
      provider_key_id: geminiKey.id,
    });
    geminiId = gemini.id;
    const geminiChat = await seed.createModel({
      display_name: "ecm-gemini-chat",
      provider: "google",
      model_name: "gemini-flash-latest",
      provider_key_id: geminiKey.id,
    });
    for (const [name, id] of [[GEMINI_ROUTER, gemini.id], [BAD_GEMINI_ROUTER, geminiChat.id]]) {
      await seed.createModel({
        display_name: name,
        semantic: {
          classifier: { type: "gemini_decision", model_id: id, on_failure: "fail" },
          routes: GEMINI_ROUTES,
          default: ROUTE_DEFAULT,
        },
      });
    }
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });

    const probe = new ProxyClient(etcdApp.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await probe.listModels()).status === 200);

    const { stdout } = await execFileP(BIN_PATH, [
      "export",
      "--etcd",
      etcdEndpoint(),
      "--prefix",
      etcdApp.etcdPrefix,
      "--reveal-secrets",
    ], { timeout: EXPORT_TIMEOUT_MS });
    exported = stdout;
    fileApp = await spawnApp({
      resourcesFile: exported,
      exposeDebugHeaders: true,
      extraEnv: slsEnv,
    });
    expect(await droppedRows(fileApp)).toEqual({ rejected: [], unknown_kinds: [] });
  });

  afterAll(async () => {
    await fileApp?.exit();
    await etcdApp?.exit();
    await upstream?.close();
    await decisions?.close();
    await sls?.close();
  });

  test("the file-mode router calls the referenced model and bills the call to it", async () => {
    const before = decisions!.calls.length;
    const response = await chat(fileApp!, ROUTER);
    expect(response.status, response.text).toBe(200);
    expect(response.route).toBe("code");
    expect(decisions!.calls.length).toBe(before + 1);
    const call = decisions!.calls.at(-1)!;
    expect(call.headers.authorization).toBe(`Bearer ${DECISION_SECRET}`);
    expect(call.body.model).toBe(JEV_UPSTREAM_MODEL);
    // The picked route's model served the chat, not the default.
    const served = JSON.parse(upstream!.receivedRequests.at(-1)!.body) as { model?: string };
    expect(served.model).toBe(TARGET_UPSTREAM_MODEL);

    await waitForSlsLog(
      sls!,
      LOGSTORE,
      (row) => row.get("request_id") === response.requestId && isChild(row),
      `the classifier child UsageEvent for ${response.requestId}`,
      20_000,
    );
    // The SLS writer is a FIFO queue: once a later request's row is
    // visible, every row of this request is too, so the count is exact.
    const barrier = await chat(fileApp!, ROUTE_DEFAULT);
    expect(barrier.status).toBe(200);
    await waitForSlsLog(
      sls!,
      LOGSTORE,
      (row) => row.get("request_id") === barrier.requestId,
      `the barrier UsageEvent for ${barrier.requestId}`,
      20_000,
    );
    const children = slsLogsFor(sls!, LOGSTORE).filter(
      (row) => row.get("request_id") === response.requestId && isChild(row),
    );
    expect(children).toHaveLength(1);
    const child = children[0]!;
    expect(child.get("internal_call_purpose")).toBe("semantic_route_decision");
    expect(child.get("model_id")).toBe(jevId);
    expect(child.get("requested_model")).toBe(JEV_MODEL);
    expect(child.get("prompt_tokens")).toBe("300");
    expect(child.get("completion_tokens")).toBe("20");
  });

  test("a classifier model on a non-typesafe key fails the same way on both sources", async () => {
    const before = decisions!.calls.length;
    const upstreamBefore = upstream!.receivedRequests.length;
    const fromEtcd = await chat(etcdApp!, BAD_ROUTER);
    const fromFile = await chat(fileApp!, BAD_ROUTER);
    // `on_failure: fail` — the router refuses instead of routing.
    expect(fromEtcd.status).toBe(503);
    expect(fromFile.status).toBe(fromEtcd.status);
    const code = (text: string) => (JSON.parse(text) as { error?: { code?: unknown } }).error?.code;
    expect(code(fromFile.text)).toBe(code(fromEtcd.text));
    expect(decisions!.calls.length).toBe(before);
    expect(upstream!.receivedRequests.length).toBe(upstreamBefore);
  });

  test("a Gemini classifier survives export and records its native decision call", async () => {
    const before = decisions!.calls.length;
    const response = await chat(fileApp!, GEMINI_ROUTER, [
      "a previous user turn",
      `${PROMPT}<system-reminder>injected text</system-reminder>`,
    ]);
    expect(response.status, response.text).toBe(200);
    expect(response.route).toBe(GEMINI_ROUTES[1]!.name);
    const served = JSON.parse(upstream!.receivedRequests.at(-1)!.body) as { model?: string };
    expect(served.model).toBe(TARGET_UPSTREAM_MODEL);

    expect(decisions!.calls.length).toBe(before + 1);
    const call = decisions!.calls.at(-1)!;
    expect([call.method, call.path]).toEqual(["POST", GEMINI_PATH]);
    expect(call.headers).toMatchObject({
      "x-goog-api-key": GEMINI_SECRET,
      "x-aisix-request-id": response.requestId,
      "x-classifier-id": geminiId,
      "x-caller-context": "client-header",
    });
    expect(call.headers.authorization).toBeUndefined();
    expect(call.body).toMatchObject({
      model: GEMINI_UPSTREAM_MODEL,
      input: PROMPT,
      proxy_hint: "gemini-shim",
    });
    expect(call.body.state).toBeUndefined();
    expect(Object.keys(call.body.questions ?? {})).toEqual(["route"]);
    const question = call.body.questions!.route!;
    expect(question.type).toBe("choice");
    expect(question.instructions).toBeTruthy();
    const keys = Object.keys(question.criteria);
    expect(keys).toEqual([...keys].sort());
    expect(keys.every((key) => /^[A-Za-z0-9_-]{1,64}$/.test(key))).toBe(true);
    expect(Object.values(question.criteria)).toEqual(
      GEMINI_ROUTES.map(({ name, description }) => `${name}: ${description}`),
    );

    const child = await waitForSlsLog(
      sls!, LOGSTORE,
      (row) => row.get("request_id") === response.requestId && isChild(row),
      "Gemini Decision usage event",
      20_000,
    );
    expect(child.get("internal_call_purpose")).toBe("semantic_route_decision");
    expect(child.get("operation")).toBe("decisions");
    expect(child.get("model_id")).toBe(geminiId);
    expect(child.get("provider_key_id")).toBe(geminiKeyId);
    expect(child.get("prompt_tokens")).toBe("100");
    expect(child.get("total_tokens")).toBe("100");
  });

  test("Gemini low confidence takes the default and an unoffered choice follows on_failure", async () => {
    const low = await chat(etcdApp!, GEMINI_ROUTER, [LOW_CONFIDENCE_PROMPT]);
    expect(low.status, low.text).toBe(200);
    expect(low.route).toBeNull();
    const served = JSON.parse(upstream!.receivedRequests.at(-1)!.body) as { model?: string };
    expect(served.model).toBe(DEFAULT_UPSTREAM_MODEL);

    const upstreamBefore = upstream!.receivedRequests.length;
    const bad = await chat(etcdApp!, GEMINI_ROUTER, [UNKNOWN_CHOICE_PROMPT]);
    expect(bad.status).toBe(503);
    expect(bad.text).toContain("classifier_invalid_response");
    expect(upstream!.receivedRequests.length).toBe(upstreamBefore);
  });

  test("a Gemini chat model cannot serve as the decision classifier", async () => {
    const before = decisions!.calls.length;
    const bad = await chat(etcdApp!, BAD_GEMINI_ROUTER);
    expect(bad.status).toBe(503);
    expect(bad.text).toContain("classifier_unavailable");
    expect(decisions!.calls.length).toBe(before);
  });
});
