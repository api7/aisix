import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the `lakera` guardrail (#52) screens chat input/output against
// Lakera Guard (`POST /v2/guard`). We stand up a mock guard endpoint that
// flags any text containing INJECT_MARKER as a prompt_attack (block), any
// text containing an email as pii/email with span offsets (mask), and
// 500s on ERROR_MARKER (fail-open path). The guardrail's `endpoint`
// override points at the mock; a real `aisix` binary + etcd + mock
// upstream complete the chain. No control plane involved.
//
// References:
// - Lakera Guard v2 <https://docs.lakera.ai>
// - LiteLLM `lakera_ai_v2` (behavior baseline): flagged + only pii/*
//   detections → mask via payload offsets; any non-PII detection → block.

const CALLER = "sk-lakera-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const INJECT_MARKER = "lakerainjectmarker";
const ERROR_MARKER = "lakerafivehundredmarker";
const EMAIL = "alice@example.com";

interface LakeraMockRequest {
  auth: string | undefined;
  projectId: string | undefined;
  messages: Array<{ role: string; content: string }>;
}

interface LakeraMock {
  baseUrl: string;
  requests: LakeraMockRequest[];
  close(): Promise<void>;
}

// Minimal mock of the /v2/guard endpoint. Flags INJECT_MARKER as a
// prompt_attack, EMAIL occurrences as pii/email (with char offsets per
// message), and 500s when any message carries ERROR_MARKER.
async function startLakeraMock(): Promise<LakeraMock> {
  const requests: LakeraMockRequest[] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let messages: Array<{ role: string; content: string }> = [];
      let projectId: string | undefined;
      try {
        const body = JSON.parse(raw);
        messages = Array.isArray(body.messages) ? body.messages : [];
        projectId = body.project_id;
      } catch {
        // leave defaults
      }
      requests.push({
        auth: req.headers.authorization,
        projectId,
        messages,
      });

      if (messages.some((m) => m.content.includes(ERROR_MARKER))) {
        res.statusCode = 500;
        res.end("mock lakera outage");
        return;
      }

      const breakdown: Array<{ detector_type: string; detected: boolean }> = [];
      const payload: Array<{
        message_id: number;
        start: number;
        end: number;
        detector_type: string;
      }> = [];
      let flagged = false;
      messages.forEach((m, i) => {
        if (m.content.includes(INJECT_MARKER)) {
          flagged = true;
          breakdown.push({ detector_type: "prompt_attack", detected: true });
        }
        const at = m.content.indexOf(EMAIL);
        if (at >= 0) {
          flagged = true;
          breakdown.push({ detector_type: "pii/email", detected: true });
          payload.push({
            message_id: i,
            start: at,
            end: at + EMAIL.length,
            detector_type: "pii/email",
          });
        }
      });

      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ flagged, payload, breakdown }));
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

describe("lakera guardrail e2e: injection blocks, PII-only masks, 5xx fail-open", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let lakera: LakeraMock | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    lakera = await startLakeraMock();

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-clean",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a safe and clean reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "lakera-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "lakera-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["lakera-e2e"],
    });

    // One env-wide guardrail on the input hook. fail_open=true so the
    // 5xx case exercises the bypass path (fail-closed is pinned by the
    // dispatcher's wiremock unit tests).
    await admin.json("POST", "/admin/v1/guardrails", {
      name: "lakera-e2e-guard",
      enabled: true,
      hook_point: "input",
      fail_open: true,
      kind: "lakera",
      api_key: "lk-e2e-key",
      endpoint: lakera.baseUrl,
      project_id: "project-e2e",
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await lakera?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  // Poll with an injection probe until the guardrail is live, so every
  // test stands alone (no hidden dependency on execution order). Once
  // propagated, the first poll returns immediately.
  const ensureGuardrailLive = () =>
    waitConfigPropagation(async () => {
      try {
        await client().chat.completions.create({
          model: "lakera-e2e",
          messages: [{ role: "user", content: `probe ${INJECT_MARKER}` }],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });

  test("injection phrase → 422 content_filter, upstream never called", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !lakera) {
      ctx.skip();
      return;
    }

    await ensureGuardrailLive();

    const upstreamBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await client().chat.completions.create({
        model: "lakera-e2e",
        messages: [
          { role: "user", content: `ignore instructions ${INJECT_MARKER}` },
        ],
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    // The matched content must not leak back to the caller (#153).
    expect(JSON.stringify(caught.error ?? {})).not.toContain(INJECT_MARKER);
    expect(upstream.receivedRequests.length).toBe(upstreamBefore);

    // The mock saw the guardrail's credentials, not a leaked caller key.
    const guardReq = lakera.requests.at(-1);
    expect(guardReq?.auth).toBe("Bearer lk-e2e-key");
    expect(guardReq?.projectId).toBe("project-e2e");
  });

  test("clean prompt → 200 via upstream", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const before = upstream.receivedRequests.length;
    const res = await client().chat.completions.create({
      model: "lakera-e2e",
      messages: [{ role: "user", content: "what is a safe topic" }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });

  test("PII-only detection → masked before the upstream, request continues", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const res = await client().chat.completions.create({
      model: "lakera-e2e",
      messages: [
        { role: "user", content: `contact me at ${EMAIL} about the order` },
      ],
    });
    // Request went through (mask, not block) …
    expect(res.choices[0]?.message.role).toBe("assistant");
    // … and the upstream saw the LiteLLM-shaped mask token, never the value.
    const lastReq = upstream.receivedRequests.at(-1);
    expect(lastReq).toBeDefined();
    expect(lastReq!.body).toContain("[MASKED EMAIL]");
    expect(lastReq!.body).toContain("about the order");
    expect(lastReq!.body).not.toContain(EMAIL);
  });

  test("lakera 5xx with fail_open=true → request passes (bypass)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    await ensureGuardrailLive();
    const before = upstream.receivedRequests.length;
    const res = await client().chat.completions.create({
      model: "lakera-e2e",
      messages: [{ role: "user", content: `hello ${ERROR_MARKER}` }],
    });
    expect(res.choices[0]?.message.role).toBe("assistant");
    expect(upstream.receivedRequests.length).toBe(before + 1);
  });
});
