import { createHash, randomUUID } from "node:crypto";
import { rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  eventStream,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: how a Bedrock provider key and a Bedrock guardrail authenticate.
//
// - `auth_type: "api_key"` sends the Bedrock API key as a bearer token, with
//   no SigV4 signature, on every Bedrock operation the gateway calls.
// - `auth_type: "default_chain"` signs with the gateway process's own AWS
//   credentials (here: AWS_* environment variables).
// - `role_arn` exchanges the base credentials through sts:AssumeRole (a mock
//   STS reached through AWS_ENDPOINT_URL_STS), skips the exchange when the
//   base credentials already are that role, and fetches new credentials
//   before the old ones expire.
// - A secret whose fields do not match its auth_type is refused before any
//   upstream is called, without echoing the secret.
// - The Bedrock guardrail's `aws_credentials` offers the same modes.

const CALLER = "sk-bedrock-auth-modes-e2e";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const ACCOUNT = "111122223333";
const AMBIENT_KEY = "AKIDAMBIENTCHAIN0001";
const STATIC_BASE_KEY = "AKIDSTATICBASE000001";
const PLAIN_STATIC_KEY = "AKIDPLAINSTATIC00001";
const API_KEY = "ABSK-e2e-bedrock-api-key";
const LEAK_MARKER = "LEAKMARKER-must-not-echo";

const CREDENTIAL_RE =
  /^AWS4-HMAC-SHA256 Credential=([^/]+)\/\d{8}\/([^/]+)\/([^/]+)\/aws4_request/;

type Received = { path: string; headers: Record<string, string>; raw: string; body: any };

function signer(r: { headers: Record<string, string> }): string | undefined {
  return CREDENTIAL_RE.exec(r.headers.authorization ?? "")?.[1];
}

async function listen(
  handler: (r: Received) => { status?: number; headers?: Record<string, string>; body: unknown },
): Promise<{ server: Server; url: string; received: Received[] }> {
  const received: Received[] = [];
  const server = createServer((req: IncomingMessage, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let body: any = {};
      try {
        body = JSON.parse(raw || "{}");
      } catch {
        body = {};
      }
      const r: Received = {
        path: decodeURIComponent(req.url ?? ""),
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(",") : (v ?? "")]),
        ),
        raw,
        body,
      };
      received.push(r);
      const reply = handler(r);
      res.writeHead(reply.status ?? 200, { "content-type": "application/json", ...(reply.headers ?? {}) });
      res.end(
        typeof reply.body === "string" || Buffer.isBuffer(reply.body) ? reply.body : JSON.stringify(reply.body),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return { server, url: `http://127.0.0.1:${addr.port}`, received };
}

async function close(server: Server | undefined) {
  if (!server) return;
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

/** A minimal STS: AssumeRole and GetCallerIdentity over the query protocol. */
interface StsCall {
  action: string;
  signer?: string;
  roleArn?: string;
  externalId?: string;
  sessionName?: string;
}

async function startSts() {
  const calls: StsCall[] = [];
  const issued = new Map<string, number>();
  const xml = (body: string) => ({ headers: { "content-type": "text/xml" }, body });
  const meta = "<ResponseMetadata><RequestId>req-sts</RequestId></ResponseMetadata>";
  const ns = 'xmlns="https://sts.amazonaws.com/doc/2011-06-15/"';
  const mock = await listen((r) => {
    const form = new URLSearchParams(r.raw);
    const call: StsCall = {
      action: form.get("Action") ?? "",
      signer: signer(r),
      roleArn: form.get("RoleArn") ?? undefined,
      externalId: form.get("ExternalId") ?? undefined,
      sessionName: form.get("RoleSessionName") ?? undefined,
    };
    calls.push(call);
    if (call.action === "GetCallerIdentity") {
      // The ambient credentials belong to the role `AmbientRole`; every
      // other key is a plain IAM user.
      const arn =
        call.signer === AMBIENT_KEY
          ? `arn:aws:sts::${ACCOUNT}:assumed-role/AmbientRole/i-0123`
          : `arn:aws:iam::${ACCOUNT}:user/base-user`;
      return xml(
        `<GetCallerIdentityResponse ${ns}><GetCallerIdentityResult><Arn>${arn}</Arn>` +
          `<UserId>AIDA0000</UserId><Account>${ACCOUNT}</Account></GetCallerIdentityResult>${meta}</GetCallerIdentityResponse>`,
      );
    }
    if (call.action === "AssumeRoleWithWebIdentity") {
      const n = (issued.get("web") ?? 0) + 1;
      issued.set("web", n);
      return xml(
        `<AssumeRoleWithWebIdentityResponse ${ns}><AssumeRoleWithWebIdentityResult><Credentials>` +
          `<AccessKeyId>ASIA-web-${n}</AccessKeyId><SecretAccessKey>secret-web-${n}</SecretAccessKey>` +
          `<SessionToken>token-web-${n}</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration></Credentials>` +
          `<AssumedRoleUser><Arn>arn:aws:sts::${ACCOUNT}:assumed-role/WebRole/s</Arn><AssumedRoleId>AROA:s</AssumedRoleId></AssumedRoleUser>` +
          `<SubjectFromWebIdentityToken>system:serviceaccount:aisix:gw</SubjectFromWebIdentityToken>` +
          `</AssumeRoleWithWebIdentityResult>${meta}</AssumeRoleWithWebIdentityResponse>`,
      );
    }
    if (call.action === "AssumeRole") {
      const role = (call.roleArn ?? "").split("/").pop() ?? "";
      // `Flaky` issues one set inside the refresh margin, then fails.
      const failing = role === "Flaky" && (issued.get(role) ?? 0) >= 1;
      if (role === "Denied" || role === "Broken" || failing) {
        const [status, code] = role === "Denied" ? [403, "AccessDenied"] : [500, "InternalFailure"];
        return {
          status,
          headers: { "content-type": "text/xml" },
          body:
            `<ErrorResponse ${ns}><Error><Type>Sender</Type><Code>${code}</Code>` +
            `<Message>refused by the mock</Message></Error><RequestId>req-sts</RequestId></ErrorResponse>`,
        };
      }
      const n = (issued.get(role) ?? 0) + 1;
      issued.set(role, n);
      // `ShortLived` credentials are already inside the refresh margin.
      const expiration =
        role === "ShortLived"
          ? new Date(Date.now() + 30_000).toISOString()
          : role === "Flaky"
            ? new Date(Date.now() + 45_000).toISOString()
            : "2099-01-01T00:00:00Z";
      return xml(
        `<AssumeRoleResponse ${ns}><AssumeRoleResult><Credentials>` +
          `<AccessKeyId>ASIA-${role}-${n}</AccessKeyId><SecretAccessKey>secret-${role}-${n}</SecretAccessKey>` +
          `<SessionToken>token-${role}-${n}</SessionToken><Expiration>${expiration}</Expiration></Credentials>` +
          `<AssumedRoleUser><Arn>arn:aws:sts::${ACCOUNT}:assumed-role/${role}/s</Arn><AssumedRoleId>AROA:s</AssumedRoleId></AssumedRoleUser>` +
          `</AssumeRoleResult>${meta}</AssumeRoleResponse>`,
      );
    }
    return { status: 400, body: "unknown action" };
  });
  return { ...mock, calls };
}

/** Answers every Bedrock operation the provider bridge calls. */
function bedrockReply(r: Received): { status?: number; headers?: Record<string, string>; body: unknown } {
  if (r.path.endsWith("/openai/v1/chat/completions")) {
    return {
      body: {
        id: "chatcmpl-auth",
        object: "chat.completion",
        model: "global.openai.gpt-6.1-sol",
        choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    };
  }
  if (r.path.endsWith("/converse-stream")) {
    return {
      headers: { "content-type": "application/vnd.amazon.eventstream" },
      body: eventStream([
        ["messageStart", { role: "assistant" }],
        ["contentBlockDelta", { contentBlockIndex: 0, delta: { text: "streamed" } }],
        ["contentBlockStop", { contentBlockIndex: 0 }],
        ["messageStop", { stopReason: "end_turn" }],
        ["metadata", { usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 }, metrics: { latencyMs: 1 } }],
      ]),
    };
  }
  if (r.path.endsWith("/converse")) {
    return {
      body: {
        output: { message: { role: "assistant", content: [{ text: "converse answer" }] } },
        stopReason: "end_turn",
        usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
        metrics: { latencyMs: 1 },
      },
    };
  }
  if (r.path.endsWith("/count-tokens")) {
    // The model only Bedrock Mantle can count.
    if (r.path.includes("opus-5-5")) return { status: 400, body: { message: "model not supported" } };
    return { body: { inputTokens: 11 } };
  }
  if (r.path.endsWith("/anthropic/v1/messages/count_tokens")) return { body: { input_tokens: 77 } };
  if (r.path.endsWith("/invoke")) {
    if (r.body.texts) {
      return {
        body: {
          id: "e1",
          response_type: "embeddings_by_type",
          embeddings: { float: [[0.1, 0.2]] },
          meta: { billed_units: { input_tokens: 2 } },
        },
      };
    }
    return {
      body: {
        id: "msg_auth",
        type: "message",
        role: "assistant",
        model: "claude",
        content: [{ type: "text", text: "invoke answer" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 3, output_tokens: 1 },
      },
    };
  }
  return { status: 404, body: { message: `no route ${r.path}` } };
}

const MODELS: Record<string, string> = {
  llama: "meta.llama3-3-70b-instruct-v1:0",
  claude: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  gpt: "global.openai.gpt-6.1-sol",
  mantle: "anthropic.claude-opus-5-5",
  cohere: "cohere.embed-v4:0",
};

const PROVIDER_KEYS: Record<string, Record<string, unknown>> = {
  apikey: { auth_type: "api_key", api_key: API_KEY, region: "us-east-1" },
  chain: { auth_type: "default_chain", region: "us-east-1" },
  chainrole: {
    auth_type: "default_chain",
    role_arn: `arn:aws:iam::${ACCOUNT}:role/Target`,
    external_id: "ext-e2e",
    role_session_name: "gw-session",
    region: "us-east-1",
  },
  chainself: {
    auth_type: "default_chain",
    role_arn: `arn:aws:iam::${ACCOUNT}:role/AmbientRole`,
    region: "us-east-1",
  },
  staticrole: {
    access_key_id: STATIC_BASE_KEY,
    secret_access_key: "static-base-secret",
    role_arn: `arn:aws:iam::${ACCOUNT}:role/ShortLived`,
    region: "us-east-1",
  },
  denied: { auth_type: "default_chain", role_arn: `arn:aws:iam::${ACCOUNT}:role/Denied`, region: "us-east-1" },
  flaky: { auth_type: "default_chain", role_arn: `arn:aws:iam::${ACCOUNT}:role/Flaky`, region: "us-east-1" },
  broken: { auth_type: "default_chain", role_arn: `arn:aws:iam::${ACCOUNT}:role/Broken`, region: "us-east-1" },
  plain: { access_key_id: PLAIN_STATIC_KEY, secret_access_key: "plain-secret", region: "us-east-1" },
  badapikey: { auth_type: "api_key", region: "us-east-1" },
  badchain: {
    auth_type: "default_chain",
    access_key_id: "AKIDBADCHAIN",
    secret_access_key: LEAK_MARKER,
    region: "us-east-1",
  },
};

/** Environment for a gateway whose default chain is env credentials and
 * whose STS is the mock. Profile files and IMDS are kept out of the chain. */
function awsEnv(stsUrl: string): Record<string, string> {
  return {
    AWS_ACCESS_KEY_ID: AMBIENT_KEY,
    AWS_SECRET_ACCESS_KEY: "ambient-secret",
    AWS_ENDPOINT_URL_STS: stsUrl,
    AWS_EC2_METADATA_DISABLED: "true",
    AWS_CONFIG_FILE: "/nonexistent/aisix-e2e-aws-config",
    AWS_SHARED_CREDENTIALS_FILE: "/nonexistent/aisix-e2e-aws-credentials",
  };
}

describe("Bedrock provider key authentication modes", () => {
  let app: SpawnedApp | undefined;
  let bedrock: Awaited<ReturnType<typeof listen>> | undefined;
  let sts: Awaited<ReturnType<typeof startSts>> | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    bedrock = await listen(bedrockReply);
    sts = await startSts();
    app = await spawnApp({
      extraEnv: awsEnv(sts.url),
      extra: { upstream: { bedrock: { mantle_endpoint: bedrock.url } } },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const allowed: string[] = [];
    for (const [pkName, secret] of Object.entries(PROVIDER_KEYS)) {
      const pk = await seed.createProviderKey({
        display_name: `auth-${pkName}`,
        provider: "bedrock",
        adapter: "bedrock",
        secret: JSON.stringify(secret),
        api_base: bedrock.url,
      });
      for (const [alias, name] of Object.entries(MODELS)) {
        await seed.createModel({
          display_name: `${pkName}-${alias}`,
          provider: "bedrock",
          model_name: name,
          provider_key_id: pk.id,
        });
        allowed.push(`${pkName}-${alias}`);
      }
    }
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: allowed });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await r.text();
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await close(bedrock?.server);
    await close(sts?.server);
  });

  function skip(ctx: { skip: () => void }) {
    if (!etcdReachable || !app) {
      ctx.skip();
      return true;
    }
    return false;
  }

  async function post(path: string, body: Record<string, unknown>) {
    const before = bedrock!.received.length;
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    return { status: res.status, text, sent: bedrock!.received.slice(before) };
  }

  const chat = (model: string, stream = false) =>
    post("/v1/chat/completions", { model, stream, messages: [{ role: "user", content: "hello" }] });

  test("an API key is sent as a bearer token, never SigV4, on every Bedrock operation", async (ctx) => {
    if (skip(ctx)) return;
    const run = async (label: string, p: Promise<{ status: number; text: string; sent: Received[] }>, paths: string[]) => {
      const r = await p;
      expect(r.status, `${label}: ${r.text}`).toBe(200);
      expect(r.sent.map((s) => s.path), label).toEqual(paths);
      for (const s of r.sent) {
        expect(s.headers.authorization, `${label} ${s.path}`).toBe(`Bearer ${API_KEY}`);
        expect(s.headers["x-amz-date"], `${label} ${s.path}`).toBeUndefined();
        expect(s.headers["x-amz-security-token"], `${label} ${s.path}`).toBeUndefined();
      }
    };
    await run("converse", chat("apikey-llama"), [`/model/${MODELS.llama}/converse`]);
    await run("converse-stream", chat("apikey-llama", true), [`/model/${MODELS.llama}/converse-stream`]);
    await run("invoke", chat("apikey-claude"), [`/model/${MODELS.claude}/invoke`]);
    await run("chat completions", chat("apikey-gpt"), ["/openai/v1/chat/completions"]);
    await run(
      "messages",
      post("/v1/messages", { model: "apikey-claude", max_tokens: 16, messages: [{ role: "user", content: "hi" }] }),
      [`/model/${MODELS.claude}/invoke`],
    );
    await run(
      "count tokens",
      post("/v1/messages/count_tokens", { model: "apikey-claude", messages: [{ role: "user", content: "hi" }] }),
      ["/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens"],
    );
    await run(
      "mantle count tokens",
      post("/v1/messages/count_tokens", { model: "apikey-mantle", messages: [{ role: "user", content: "hi" }] }),
      [`/model/${MODELS.mantle}/count-tokens`, "/anthropic/v1/messages/count_tokens"],
    );
    await run("embeddings", post("/v1/embeddings", { model: "apikey-cohere", input: ["a"] }), [
      `/model/${MODELS.cohere}/invoke`,
    ]);
  });

  test("the default chain signs with the gateway's own AWS credentials", async (ctx) => {
    if (skip(ctx)) return;
    for (const [label, req] of [
      ["converse", () => chat("chain-llama")],
      ["chat completions", () => chat("chain-gpt")],
      [
        "mantle",
        () => post("/v1/messages/count_tokens", { model: "chain-mantle", messages: [{ role: "user", content: "hi" }] }),
      ],
    ] as const) {
      const r = await req();
      expect(r.status, `${label}: ${r.text}`).toBe(200);
      expect(r.sent.length, label).toBeGreaterThan(0);
      for (const s of r.sent) expect(signer(s), `${label} ${s.path}`).toBe(AMBIENT_KEY);
    }
  });

  test("explicit access keys still sign as themselves next to ambient credentials", async (ctx) => {
    if (skip(ctx)) return;
    const r = await chat("plain-llama");
    expect(r.status, r.text).toBe(200);
    expect(signer(r.sent[0])).toBe(PLAIN_STATIC_KEY);
  });

  test("role_arn on the default chain assumes the role once and signs with its credentials", async (ctx) => {
    if (skip(ctx)) return;
    for (const model of ["chainrole-llama", "chainrole-gpt", "chainrole-mantle", "chainrole-llama"]) {
      const r =
        model === "chainrole-mantle"
          ? await post("/v1/messages/count_tokens", { model, messages: [{ role: "user", content: "hi" }] })
          : await chat(model);
      expect(r.status, `${model}: ${r.text}`).toBe(200);
      for (const s of r.sent) {
        expect(signer(s), `${model} ${s.path}`).toBe("ASIA-Target-1");
        expect(s.headers["x-amz-security-token"], `${model} ${s.path}`).toBe("token-Target-1");
      }
    }
    const assumes = sts!.calls.filter((c) => c.action === "AssumeRole" && c.roleArn?.endsWith("/Target"));
    expect(assumes).toHaveLength(1);
    expect(assumes[0]).toMatchObject({ signer: AMBIENT_KEY, externalId: "ext-e2e", sessionName: "gw-session" });
  });

  test("a role the ambient identity already is is used without AssumeRole", async (ctx) => {
    if (skip(ctx)) return;
    const stsBefore = sts!.calls.length;
    const r = await chat("chainself-llama");
    expect(r.status, r.text).toBe(200);
    expect(signer(r.sent[0])).toBe(AMBIENT_KEY);
    expect(sts!.calls.slice(stsBefore)).toEqual([
      expect.objectContaining({ action: "GetCallerIdentity", signer: AMBIENT_KEY }),
    ]);
  });

  test("a failed refresh keeps serving credentials that have not expired yet", async (ctx) => {
    if (skip(ctx)) return;
    // The first set is already inside the refresh margin, so the second
    // request refreshes; STS fails that refresh.
    for (let i = 0; i < 2; i++) {
      const r = await chat("flaky-llama");
      expect(r.status, r.text).toBe(200);
      expect(signer(r.sent[0])).toBe("ASIA-Flaky-1");
    }
    const assumes = sts!.calls.filter((c) => c.action === "AssumeRole" && c.roleArn?.endsWith("/Flaky"));
    expect(assumes.length).toBeGreaterThanOrEqual(2);
  });

  test("assumed credentials close to expiry are refreshed before the next request signs", async (ctx) => {
    if (skip(ctx)) return;
    const signers: string[] = [];
    for (let i = 0; i < 3; i++) {
      const r = await chat("staticrole-llama");
      expect(r.status, r.text).toBe(200);
      signers.push(signer(r.sent[0]) ?? "");
    }
    // Every set STS issued expires inside the refresh margin, so each
    // request is signed with a newly assumed set.
    expect(new Set(signers).size).toBe(3);
    for (const s of signers) expect(s).toMatch(/^ASIA-ShortLived-\d+$/);
    const assumes = sts!.calls.filter((c) => c.action === "AssumeRole" && c.roleArn?.endsWith("/ShortLived"));
    expect(assumes.length).toBeGreaterThanOrEqual(3);
    for (const a of assumes) {
      expect(a.signer).toBe(STATIC_BASE_KEY);
      expect(a.sessionName).toMatch(/^aisix-session-\d+$/);
    }
  });

  test("a secret that does not match its auth_type is refused before any upstream call", async (ctx) => {
    if (skip(ctx)) return;
    const stsBefore = sts!.calls.length;
    for (const [model, field] of [
      ["badapikey-llama", "`api_key`"],
      ["badchain-llama", "`access_key_id`"],
      ["badapikey-gpt", "`api_key`"],
    ]) {
      const r = await chat(model);
      expect(r.status, `${model}: ${r.text}`).toBe(401);
      expect(r.sent, model).toEqual([]);
      expect(JSON.parse(r.text).error.message, model).toContain(field);
      expect(r.text).not.toContain(LEAK_MARKER);
    }
    expect(sts!.calls.length).toBe(stsBefore);
  });

  test("a refused AssumeRole is a credential error; a failing STS is not", async (ctx) => {
    if (skip(ctx)) return;
    const denied = await chat("denied-llama");
    expect(denied.status, denied.text).toBe(401);
    expect(denied.sent).toEqual([]);
    // The caller learns which setting failed, not what STS said: a real
    // refusal names the gateway's own account and role.
    const message = JSON.parse(denied.text).error.message;
    expect(message).toContain("role_arn");
    expect(message).not.toContain("refused by the mock");
    expect(denied.text).not.toContain(ACCOUNT);

    const broken = await chat("broken-llama");
    expect(broken.sent).toEqual([]);
    expect(broken.status, broken.text).toBeGreaterThanOrEqual(500);
  });
});

describe("Bedrock guardrail authentication modes", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let guardrails: Awaited<ReturnType<typeof listen>> | undefined;
  let sts: Awaited<ReturnType<typeof startSts>> | undefined;
  let etcdReachable = false;

  const GUARDS = {
    session: "gsession0001",
    apikey: "gapikey00001",
    chain: "gchain000001",
    role: "grole0000001",
  };
  const guardOf = (r: Received) => /^\/guardrail\/([^/]+)\//.exec(r.path)?.[1];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    guardrails = await listen(() => ({ body: { action: "NONE", outputs: [] } }));
    sts = await startSts();
    app = await spawnApp({ extraEnv: awsEnv(sts.url), extra: { bedrock_endpoint_url: guardrails.url } });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const credentials: Record<keyof typeof GUARDS, Record<string, unknown>> = {
      session: {
        kind: "static",
        access_key_id: "ASIAGUARDSESSION001",
        secret_access_key: "guard-session-secret",
        session_token: "guard-session-token",
      },
      apikey: { kind: "api_key", api_key: API_KEY },
      chain: { kind: "default_chain" },
      role: {
        kind: "static",
        access_key_id: STATIC_BASE_KEY,
        secret_access_key: "static-base-secret",
        role_arn: `arn:aws:iam::${ACCOUNT}:role/GuardRole`,
        external_id: "guard-ext",
      },
    };
    for (const [mode, id] of Object.entries(GUARDS) as Array<[keyof typeof GUARDS, string]>) {
      await seed.createGuardrail({
        name: `gr-auth-${mode}`,
        enabled: true,
        hook_point: "input",
        fail_open: false,
        kind: "bedrock",
        guardrail_id: id,
        guardrail_version: "DRAFT",
        region: "us-east-1",
        aws_credentials: credentials[mode],
        latency_mode: { kind: "serial" },
      });
    }
    const pk = await seed.createProviderKey({
      display_name: "gr-auth-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gr-auth-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["gr-auth-model"] });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await r.text();
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await close(guardrails?.server);
    await close(sts?.server);
  });

  test("each credential mode authenticates its ApplyGuardrail call", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Every guardrail is attached to the environment, so one request runs
    // all four; wait until each one has been called.
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
        body: JSON.stringify({ model: "gr-auth-model", messages: [{ role: "user", content: "hello" }] }),
      });
      await r.text();
      const seen = new Set(guardrails!.received.map(guardOf));
      return r.status === 200 && Object.values(GUARDS).every((id) => seen.has(id));
    });
    const calls = (id: string) => guardrails!.received.filter((r) => guardOf(r) === id);

    for (const c of calls(GUARDS.session)) {
      expect(signer(c)).toBe("ASIAGUARDSESSION001");
      expect(c.headers["x-amz-security-token"]).toBe("guard-session-token");
    }
    for (const c of calls(GUARDS.apikey)) {
      expect(c.headers.authorization).toBe(`Bearer ${API_KEY}`);
      expect(c.headers["x-amz-date"]).toBeUndefined();
    }
    for (const c of calls(GUARDS.chain)) expect(signer(c)).toBe(AMBIENT_KEY);
    for (const c of calls(GUARDS.role)) {
      expect(signer(c)).toBe("ASIA-GuardRole-1");
      expect(c.headers["x-amz-security-token"]).toBe("token-GuardRole-1");
    }
    const assumes = sts!.calls.filter((c) => c.action === "AssumeRole");
    expect(assumes).toHaveLength(1);
    expect(assumes[0]).toMatchObject({
      signer: STATIC_BASE_KEY,
      roleArn: `arn:aws:iam::${ACCOUNT}:role/GuardRole`,
      externalId: "guard-ext",
    });
  });
});

describe("Bedrock default chain on web identity (IRSA)", () => {
  let app: SpawnedApp | undefined;
  let bedrock: Awaited<ReturnType<typeof listen>> | undefined;
  let sts: Awaited<ReturnType<typeof startSts>> | undefined;
  let etcdReachable = false;
  const tokenFile = join(tmpdir(), `aisix-e2e-web-identity-${randomUUID()}`);

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    writeFileSync(tokenFile, "header.payload.signature");
    bedrock = await listen(bedrockReply);
    sts = await startSts();
    const env = awsEnv(sts.url);
    delete env.AWS_ACCESS_KEY_ID;
    delete env.AWS_SECRET_ACCESS_KEY;
    app = await spawnApp({
      extraEnv: {
        ...env,
        AWS_ROLE_ARN: `arn:aws:iam::${ACCOUNT}:role/WebRole`,
        AWS_WEB_IDENTITY_TOKEN_FILE: tokenFile,
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "auth-web-pk",
      provider: "bedrock",
      adapter: "bedrock",
      secret: JSON.stringify({
        auth_type: "default_chain",
        role_arn: `arn:aws:iam::${ACCOUNT}:role/WebRole`,
        region: "us-east-1",
      }),
      api_base: bedrock.url,
    });
    await seed.createModel({
      display_name: "web-llama",
      provider: "bedrock",
      model_name: MODELS.llama,
      provider_key_id: pk.id,
    });
    await seed.createApiKey({ key_hash: CALLER_HASH, allowed_models: ["web-llama"] });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, { headers: { authorization: `Bearer ${CALLER}` } });
      await r.text();
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await close(bedrock?.server);
    await close(sts?.server);
    rmSync(tokenFile, { force: true });
  });

  test("the web-identity role is used as it is, without asking STS who it is", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "web-llama", messages: [{ role: "user", content: "hello" }] }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    expect(signer(bedrock!.received[bedrock!.received.length - 1])).toBe("ASIA-web-1");
    expect(sts!.calls.map((c) => c.action)).toEqual(["AssumeRoleWithWebIdentity"]);
  });
});
