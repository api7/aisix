import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  lz4DecompressBlock,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1013: non-200 requests must also record the (post-mask)
// request body in full-content SLS logs — previously content was attached
// only on the 200 success path, so a 4xx/5xx row showed status + error
// class but never WHAT was sent, making triage guesswork. This drives a
// real `aisix` binary + etcd against a hard-failing mock upstream and a
// keyword guardrail, and reads the delivered SLS protobuf back:
//
//   1. upstream failure (chat)        → record carries `prompt`
//   2. guardrail input block 422      → record carries `prompt`
//   3. 403 model-forbidden            → record exists WITHOUT `prompt`
//      (auth-class failures stay body-less by design)
//   4. /v1/messages upstream failure  → record carries `prompt`
//   5. /v1/responses upstream failure → record carries `prompt`
//
// A metadata_only exporter runs alongside and must never see any prompt.

const CALLER_PLAINTEXT = "sk-failure-content-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const CREDENTIAL_REF = "mock";
const MOCK_AK_ID = "LTAI_mock_ak";
const MOCK_AK_SECRET = "mock_ak_secret";
const SLS_PROJECT = "aisix-e2e-obs";
const FULL_LOGSTORE = "failure-content-full";
const META_LOGSTORE = "failure-content-meta";

const FORBIDDEN_WORD = "failurecontentforbidden";
const UPSTREAM_FAIL_SENTINEL = "upstream-fail-prompt-7c1d2e";
const GUARDRAIL_SENTINEL = `${FORBIDDEN_WORD} plus context 4b9f0a`;
const FORBIDDEN_MODEL_SENTINEL = "forbidden-model-prompt-9e3a1b";
const MESSAGES_SENTINEL = "messages-fail-prompt-5d8c4f";
const RESPONSES_SENTINEL = "responses-fail-prompt-2a6e9d";

// --- Minimal SLS LogGroup protobuf reader (see sink/sls.rs encoder) -----
// LogGroup { Logs = 1 (message) { Time = 1 (varint), Contents = 2 (message)
// { Key = 1 (string), Value = 2 (string) } } }; unknown fields skipped.

function readVarint(buf: Buffer, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  for (;;) {
    const b = buf[pos]!;
    pos += 1;
    result += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) return [result, pos];
    shift += 7;
  }
}

function skipField(buf: Buffer, pos: number, wireType: number): number {
  if (wireType === 0) return readVarint(buf, pos)[1];
  if (wireType === 2) {
    const [len, p] = readVarint(buf, pos);
    return p + len;
  }
  if (wireType === 5) return pos + 4;
  if (wireType === 1) return pos + 8;
  throw new Error(`unsupported wire type ${wireType}`);
}

function parseContentPair(buf: Buffer): [string, string] {
  let pos = 0;
  let key = "";
  let value = "";
  while (pos < buf.length) {
    const [tag, p] = readVarint(buf, pos);
    pos = p;
    const field = tag >>> 3;
    const wireType = tag & 7;
    if (wireType === 2) {
      const [len, q] = readVarint(buf, pos);
      const bytes = buf.subarray(q, q + len);
      pos = q + len;
      if (field === 1) key = bytes.toString("utf8");
      else if (field === 2) value = bytes.toString("utf8");
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }
  return [key, value];
}

function parseLog(buf: Buffer): Map<string, string> {
  const out = new Map<string, string>();
  let pos = 0;
  while (pos < buf.length) {
    const [tag, p] = readVarint(buf, pos);
    pos = p;
    const field = tag >>> 3;
    const wireType = tag & 7;
    if (field === 2 && wireType === 2) {
      const [len, q] = readVarint(buf, pos);
      const [k, v] = parseContentPair(buf.subarray(q, q + len));
      out.set(k, v);
      pos = q + len;
    } else {
      pos = skipField(buf, pos, wireType);
    }
  }
  return out;
}

/** Decode every log delivered to `logstore` into flat key→value maps. */
function logsFor(sls: MockSls, logstore: string): Map<string, string>[] {
  const logs: Map<string, string>[] = [];
  for (const r of sls.requests) {
    if (r.logstore !== logstore || r.rawSize === 0 || r.body.length === 0) continue;
    const group = lz4DecompressBlock(r.body, r.rawSize);
    let pos = 0;
    while (pos < group.length) {
      const [tag, p] = readVarint(group, pos);
      pos = p;
      const field = tag >>> 3;
      const wireType = tag & 7;
      if (field === 1 && wireType === 2) {
        const [len, q] = readVarint(group, pos);
        logs.push(parseLog(group.subarray(q, q + len)));
        pos = q + len;
      } else {
        pos = skipField(group, pos, wireType);
      }
    }
  }
  return logs;
}

/** Poll until a FULL_LOGSTORE log matching `pred` arrives (or time out). */
async function waitForLog(
  sls: MockSls,
  pred: (l: Map<string, string>) => boolean,
  what: string,
  timeoutMs = 10_000,
): Promise<Map<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = logsFor(sls, FULL_LOGSTORE).find(pred);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`no SLS log matching: ${what}`);
}

// -------------------------------------------------------------------------

describe("sls e2e: failed requests record the request body (#1013)", () => {
  let okUpstream: OpenAiUpstream | undefined;
  let failUpstream: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    sls = await startMockSls();
    // Healthy upstream — used only to gate config propagation.
    okUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-ok",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "fine" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });
    // Hard-failing upstream: every call returns 500.
    failUpstream = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "mock upstream exploded", type: "server_error" } },
    });

    app = await spawnApp({
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: MOCK_AK_ID,
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: MOCK_AK_SECRET,
      },
    });
    const admin = new AdminClient(app.adminUrl, app.adminKey);

    await admin.createObservabilityExporter({
      name: "sls-failure-full",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: FULL_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "full",
    });
    await admin.createObservabilityExporter({
      name: "sls-failure-meta",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: META_LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const okPk = await admin.createProviderKey({
      display_name: "failure-content-ok-pk",
      secret: "sk-mock",
      api_base: `${okUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "failure-content-ok",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: okPk.id,
    });
    const failPk = await admin.createProviderKey({
      display_name: "failure-content-fail-pk",
      secret: "sk-mock",
      api_base: `${failUpstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "failure-content-fail",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: failPk.id,
    });
    // A model the caller is NOT allowed to use (403 case).
    await admin.createModel({
      display_name: "failure-content-offlimits",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: okPk.id,
    });

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["failure-content-ok", "failure-content-fail"],
    });

    // Input-side keyword guardrail (block mode) for the 422 case.
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "failure-content-guard",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });

    // Gate: benign chat succeeds AND the guardrail is live (risky 422).
    await waitConfigPropagation(async () => {
      const ok = await chat("failure-content-ok", "a plain benign question");
      if (ok.status !== 200) return false;
      const blocked = await chat("failure-content-ok", `probe ${FORBIDDEN_WORD}`);
      return blocked.status === 422;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await okUpstream?.close();
    await failUpstream?.close();
    await sls?.close();
  });

  async function chat(model: string, content: string): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
    });
    await res.text();
    return res;
  }

  test("upstream failure: the failed chat request's record carries the prompt", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await chat("failure-content-fail", UPSTREAM_FAIL_SENTINEL);
    expect(res.status).toBeGreaterThanOrEqual(500);

    const log = await waitForLog(
      sls,
      (l) => (l.get("prompt") ?? "").includes(UPSTREAM_FAIL_SENTINEL),
      "failed-upstream chat record with prompt",
    );
    // It is the FAILED request's record: non-2xx status, no response text.
    expect(Number(log.get("status_code"))).toBeGreaterThanOrEqual(400);
    expect(log.get("response") ?? "").toBe("");
    // The prompt is the request body — valid JSON with the messages array.
    const prompt = JSON.parse(log.get("prompt")!) as {
      messages: Array<{ content: string }>;
    };
    expect(prompt.messages[0]!.content).toContain(UPSTREAM_FAIL_SENTINEL);
  });

  test("guardrail input block (422): the record carries the prompt", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await chat("failure-content-ok", GUARDRAIL_SENTINEL);
    expect(res.status).toBe(422);

    const log = await waitForLog(
      sls,
      (l) => (l.get("prompt") ?? "").includes(GUARDRAIL_SENTINEL),
      "guardrail-blocked record with prompt",
    );
    expect(log.get("status_code")).toBe("422");
    expect(log.get("guardrail_blocked")).toBe("true");
  });

  test("403 model-forbidden: the record exists but stays body-less", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await chat("failure-content-offlimits", FORBIDDEN_MODEL_SENTINEL);
    expect(res.status).toBe(403);

    // The 403 event lands in SLS…
    const log = await waitForLog(
      sls,
      (l) =>
        l.get("status_code") === "403" &&
        (l.get("requested_model") ?? "") === "failure-content-offlimits",
      "403 record",
    );
    // …but carries no prompt, and the sentinel never reaches the logstore.
    expect(log.get("prompt")).toBeUndefined();
    for (const l of logsFor(sls, FULL_LOGSTORE)) {
      expect(l.get("prompt") ?? "").not.toContain(FORBIDDEN_MODEL_SENTINEL);
    }
  });

  test("/v1/messages upstream failure: the record carries the prompt", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "failure-content-fail",
        max_tokens: 32,
        messages: [{ role: "user", content: MESSAGES_SENTINEL }],
      }),
    });
    await res.text();
    expect(res.status).toBeGreaterThanOrEqual(400);

    const log = await waitForLog(
      sls,
      (l) => (l.get("prompt") ?? "").includes(MESSAGES_SENTINEL),
      "failed /v1/messages record with prompt",
    );
    expect(Number(log.get("status_code"))).toBeGreaterThanOrEqual(400);
  });

  test("/v1/responses upstream failure: the record carries the prompt", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "failure-content-fail",
        input: RESPONSES_SENTINEL,
      }),
    });
    await res.text();
    expect(res.status).toBeGreaterThanOrEqual(400);

    const log = await waitForLog(
      sls,
      (l) => (l.get("prompt") ?? "").includes(RESPONSES_SENTINEL),
      "failed /v1/responses record with prompt",
    );
    expect(Number(log.get("status_code"))).toBeGreaterThanOrEqual(400);
  });

  test("metadata_only exporter never receives any failed-request prompt", async (ctx) => {
    if (!etcdReachable || !app || !sls) {
      ctx.skip();
      return;
    }
    // Runs last: every sentinel above has already been sent and captured
    // into the FULL logstore. None may appear in the metadata logstore.
    const metaText = logsFor(sls, META_LOGSTORE)
      .flatMap((l) => [...l.values()])
      .join(" ");
    for (const sentinel of [
      UPSTREAM_FAIL_SENTINEL,
      GUARDRAIL_SENTINEL,
      MESSAGES_SENTINEL,
      RESPONSES_SENTINEL,
    ]) {
      expect(metaText).not.toContain(sentinel);
    }
    for (const l of logsFor(sls, META_LOGSTORE)) {
      expect(l.get("prompt")).toBeUndefined();
    }
  });
});
