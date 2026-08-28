import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startOpenAiUpstream,
  waitConfigPropagation,
  type McpUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a request that carries no scannable text still gets a guardrail
// VERDICT. Found by release QA on v0.11.0-rc.5 and reproducing back to
// rc.4 — pre-existing, not a regression.
//
// The gateway used to treat "nothing to scan" as "nothing to decide", in
// four separate places. An operator who attached an unconditional `block`
// policy watched these succeed:
//
//   - MCP `tools/call` with `"arguments": {}` — the tool EXECUTED (the
//     reported bug). The segment pass collected zero string leaves under
//     `params.arguments` and returned early without consulting the chain.
//   - `/v1/messages/count_tokens` — ran no chain at all, while shipping
//     the caller's whole `system` + `messages` + `tools` payload to the
//     provider.
//   - `/v1/audio/transcriptions` with no `prompt` form field — i.e. the
//     ordinary shape of that endpoint.
//   - `/v1/images/edits` with no `prompt` part.
//
// A guardrail whose verdict depends on the text (keyword, pii) legitimately
// allows an empty request — it matched nothing. A guardrail that decides
// about the CALL does not, and that difference belongs to the guardrail,
// never to the call site. Both halves are pinned here: the same textless
// requests are driven against an unconditional `kind: custom` policy (must
// refuse) and against a `kind: keyword` rule (must pass), so a fix that
// simply started blocking every empty request would fail this file too.
//
// The upstream request recorder is the load-bearing assertion: a refusal
// that still contacted the provider would have leaked exactly the payload
// the guardrail exists to keep in.

const BLOCKING_KEY = "sk-textless-blocking";
const KEYWORD_KEY = "sk-textless-keyword";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** No `ctx` read at all: the verdict cannot come from the text. */
const BLOCK_EVERYTHING = `
export function checkInput() {
  return { action: "block", reason_code: "textless-e2e" };
}
`;

const KEYWORD_PATTERN = "textless-e2e-forbidden";

interface RpcReply {
  status: number;
  json?: {
    result?: {
      content?: Array<{ type: string; text?: string }>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

/** One gateway plus its upstreams, wired with a single guardrail row. */
interface Env {
  app: SpawnedApp;
  llm: OpenAiUpstream;
  mcp: McpUpstream;
  key: string;
}

const startEnv = async (
  etcd: EtcdClient,
  key: string,
  guardrail: Record<string, unknown>,
): Promise<Env> => {
  const llm = await startOpenAiUpstream({ nonStreamBody: { input_tokens: 7 } });
  const mcp = await startMcpUpstream("textless");
  const app = await spawnApp();
  const seed = new SeedClient(etcd, app.etcdPrefix);

  const anthropicPk = await seed.createProviderKey({
    display_name: "textless-anthropic",
    secret: "sk-ant-mock",
    // The Anthropic bridge appends to the BARE host — no /v1 suffix here.
    api_base: llm.baseUrl,
    provider: "anthropic",
    adapter: "anthropic",
  });
  const openaiPk = await seed.createProviderKey({
    display_name: "textless-openai",
    secret: "sk-mock",
    api_base: `${llm.baseUrl}/v1`,
  });
  await seed.createModel({
    display_name: "textless-anthropic",
    provider: "anthropic",
    model_name: "claude-haiku-4-5-20251001",
    provider_key_id: anthropicPk.id,
  });
  await seed.createModel({
    display_name: "textless-audio",
    provider: "openai",
    model_name: "gpt-4o-transcribe",
    provider_key_id: openaiPk.id,
  });
  await seed.update("mcp_servers", randomUUID(), {
    display_name: "textless",
    url: mcp.url,
    enabled: true,
  });
  // No attachment row: a guardrail with none falls back to implicit
  // env scope, which is what an operator gets by default.
  await seed.createGuardrail(guardrail);

  // Written LAST, so the key authenticating implies every row above it
  // landed (etcd applies in revision order). The gate touches neither a
  // guarded endpoint nor a guardrail, so a broken assertion below fails as
  // an assertion rather than as a gate timeout.
  await seed.createApiKey({
    key_hash: sha256(key),
    allowed_models: ["*"],
    mcp_access: { allow: ["*"] },
  });
  const proxy = new ProxyClient(app.proxyUrl, key);
  await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);

  return { app, llm, mcp, key };
};

/** `tools/call` with NO arguments at all — the reported shape. */
const callToolWithoutArguments = async (env: Env): Promise<RpcReply> => {
  const post = async (body: unknown): Promise<RpcReply> => {
    const res = await fetch(`${env.app.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${env.key}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: RpcReply["json"];
    try {
      json = text ? (JSON.parse(text) as RpcReply["json"]) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, json };
  };
  // The endpoint is stateless: every operation re-handshakes.
  await post({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "textless-e2e", version: "0.1" },
    },
  });
  return post({
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: { name: "textless__echo", arguments: {} },
  });
};

const countTokens = (env: Env) =>
  fetch(`${env.app.proxyUrl}/v1/messages/count_tokens`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": env.key,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "textless-anthropic",
      messages: [{ role: "user", content: "my SSN is 123-45-6789" }],
    }),
  });

/** A transcription upload with no `prompt` part — the ordinary shape. */
const transcribe = (env: Env) => {
  const form = new FormData();
  form.set("model", "textless-audio");
  form.set(
    "file",
    new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }),
    "a.mp3",
  );
  return fetch(`${env.app.proxyUrl}/v1/audio/transcriptions`, {
    method: "POST",
    headers: { authorization: `Bearer ${env.key}` },
    body: form,
  });
};

describe("textless requests still get a guardrail verdict", () => {
  let etcdReachable = false;
  let blocking: Env | undefined;
  let keyword: Env | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    blocking = await startEnv(etcd, BLOCKING_KEY, {
      name: "textless-block-all",
      enabled: true,
      kind: "custom",
      hook_point: "input",
      fail_open: false,
      script: BLOCK_EVERYTHING,
      timeout_ms: 5000,
    });
    keyword = await startEnv(etcd, KEYWORD_KEY, {
      name: "textless-keyword",
      enabled: true,
      kind: "keyword",
      hook_point: "input",
      patterns: [{ kind: "literal", value: KEYWORD_PATTERN }],
    });
  }, 90_000);

  afterAll(async () => {
    await blocking?.app.exit();
    await blocking?.llm.close();
    await blocking?.mcp.close();
    await keyword?.app.exit();
    await keyword?.llm.close();
    await keyword?.mcp.close();
  });

  test("MCP tools/call with empty arguments is refused, and the tool never runs", async (ctx) => {
    if (!etcdReachable || !blocking) return ctx.skip();

    const before = blocking.mcp.received.length;
    const reply = await callToolWithoutArguments(blocking);

    // The MCP block contract: HTTP 200 + a TOOL-execution error, so the
    // calling agent reads it as tool output rather than a protocol fault.
    expect(reply.status).toBe(200);
    expect(reply.json?.error).toBeUndefined();
    expect(reply.json?.result?.isError).toBe(true);
    expect(reply.json?.result?.content?.[0]?.text).toContain("content policy");
    expect(reply.json?.result?.content?.[0]?.text).toContain("textless-block-all");

    // The whole point: the tool did not execute. Pre-fix it did.
    const relayed = blocking.mcp.received
      .slice(before)
      .filter((body) => body.includes("tools/call"));
    expect(relayed).toEqual([]);
  }, 30_000);

  test("count_tokens is refused, and the payload never reaches the provider", async (ctx) => {
    if (!etcdReachable || !blocking) return ctx.skip();

    const before = blocking.llm.receivedRequests.length;
    const res = await countTokens(blocking);

    // Anthropic-shaped envelope, like every other error on this route.
    expect(res.status).toBe(422);
    const body = (await res.json()) as {
      type?: string;
      error?: { type?: string; message?: string };
    };
    expect(body.type).toBe("error");
    expect(body.error?.message ?? "").toContain("content policy");
    expect(body.error?.message ?? "").toContain("textless-block-all");

    const forwarded = blocking.llm.receivedRequests
      .slice(before)
      .filter((r) => r.path.includes("count_tokens"));
    expect(forwarded).toEqual([]);
  }, 30_000);

  test("a transcription with no prompt field is refused", async (ctx) => {
    if (!etcdReachable || !blocking) return ctx.skip();

    const before = blocking.llm.receivedRequests.length;
    const res = await transcribe(blocking);

    expect(res.status).toBe(422);
    const body = (await res.json()) as { error?: { type?: string; message?: string } };
    expect(body.error?.type).toBe("content_filter");
    expect(body.error?.message ?? "").toContain("textless-block-all");

    const forwarded = blocking.llm.receivedRequests
      .slice(before)
      .filter((r) => r.path.includes("transcriptions"));
    expect(forwarded).toEqual([]);
  }, 30_000);

  // The negative control. Removing the short-circuits must not turn "no
  // text" into a blanket refusal: a rule that decides by matching text has
  // matched nothing, and an empty request is clean to it.
  test("a text-matching guardrail lets the same textless requests through", async (ctx) => {
    if (!etcdReachable || !keyword) return ctx.skip();

    const reply = await callToolWithoutArguments(keyword);
    expect(reply.status).toBe(200);
    expect(reply.json?.result?.isError).toBeFalsy();

    const counted = await countTokens(keyword);
    expect(counted.status).toBe(200);

    const transcribed = await transcribe(keyword);
    expect(transcribed.status).toBe(200);
  }, 30_000);
});
