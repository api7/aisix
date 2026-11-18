import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
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

// A semantic router whose JEV classifier names its direct model by
// `model_id` — the shape the control plane projects — must keep working
// after `aisix export` moves it into a resources file. A file derives its
// ids from entry names, so the control-plane id cannot resolve there; the
// export has to carry the reference by name, and the file-mode gateway has
// to call that model and attribute the decision call to it, exactly as the
// etcd-backed gateway does.
//
// A real aisix DP + etcd, `aisix export`, then a second gateway in file
// mode; System One and chat upstreams are local endpoints, and the usage
// events are read from the SLS export.

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
const NOT_TYPESAFE_MODEL = "ecm-not-typesafe";
const ROUTER = "ecm-router";
const BAD_ROUTER = "ecm-router-not-typesafe";
const ROUTE_TARGET = "ecm-route-target";
const ROUTE_DEFAULT = "ecm-route-default";
const TARGET_UPSTREAM_MODEL = "gpt-route-target";
const DEFAULT_UPSTREAM_MODEL = "gpt-route-default";
const PROMPT = "fix my python script";
const EXPORT_TIMEOUT_MS = 30_000;

// The resources file's id derivation: UUIDv5 of "<kind>/<identity>" in the
// file-resource namespace.
const FILE_RESOURCE_NAMESPACE = "63e50ab2-677a-54d3-8d1e-c0cb29ceae94";
function fileResourceId(kind: string, identity: string): string {
  const ns = Buffer.from(FILE_RESOURCE_NAMESPACE.replaceAll("-", ""), "hex");
  const hash = createHash("sha1").update(ns).update(`${kind}/${identity}`).digest();
  hash[6] = (hash[6]! & 0x0f) | 0x50;
  hash[8] = (hash[8]! & 0x3f) | 0x80;
  const hex = hash.subarray(0, 16).toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

interface DecisionCall {
  headers: IncomingHttpHeaders;
  body: { model?: string; state?: string };
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
      calls.push({ headers: req.headers, body });
      res.setHeader("content-type", "application/json");
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

  const slsEnv = {
    [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
    [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
  };

  async function chat(app: SpawnedApp, model: string) {
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: PROMPT }] }),
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
      secret: "sk-chat-mock",
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
      secret: DECISION_SECRET,
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
    expect(child.get("model_id")).toBe(fileResourceId("models", JEV_MODEL));
    expect(child.get("requested_model")).toBe(JEV_MODEL);
    expect(child.get("prompt_tokens")).toBe("300");
    expect(child.get("completion_tokens")).toBe("20");

    // The inline pair the control plane writes for older gateways is not
    // read beside `model_id`, so it is not exported either.
    expect(exported).not.toContain("jev-inline-must-not-be-sent");
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
});
