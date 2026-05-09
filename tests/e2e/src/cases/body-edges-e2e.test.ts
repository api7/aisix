import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: request body edge cases. Two user journeys that prior
// coverage skipped — every existing chat-completions test sends a
// single one-message array, so the gateway's behavior on real-world
// shapes was unverified:
//
//   1. Multi-turn 10+ messages — long conversation history with
//      system/user/assistant interleave must reach the upstream
//      byte-for-byte. A regression that truncated, dropped, or
//      reordered messages would silently lose context for every
//      conversational caller.
//
//   2. Empty `messages: []` — OpenAI Chat Completions spec requires
//      a non-empty messages array. Gateway must reject with a
//      4xx error envelope, NOT 500 / panic / hang.
//
// (The "body exceeds size limit → 413" case is tracked as a
// separate test pending a product fix; the gateway currently
// resets the connection rather than emitting 413. See follow-up
// issue.)
//
// References:
// - OpenAI Chat Completions API spec
//   <https://platform.openai.com/docs/api-reference/chat/create>
// - OpenAI error envelope spec
//   <https://platform.openai.com/docs/guides/error-codes/api-errors>

const CALLER_PLAINTEXT = "sk-body-edges-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("body edges e2e: multi-turn, empty messages", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "body-edges-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "body-edges",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["body-edges"],
    });

    // Confirm propagation with a one-message happy-path call so
    // the downstream tests run against a fully-loaded snapshot.
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "body-edges",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("multi-turn: 12-message history (system + 5×user/assistant + final user) reaches upstream byte-for-byte", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Realistic conversation shape: system primer + 5 turns of
    // back-and-forth + final user query awaiting response. Caller
    // SDKs build histories like this all the time.
    const history = [
      { role: "system" as const, content: "You are a helpful assistant." },
      { role: "user" as const, content: "Hi, what's 2+2?" },
      { role: "assistant" as const, content: "It's 4." },
      { role: "user" as const, content: "What about 3+3?" },
      { role: "assistant" as const, content: "That's 6." },
      { role: "user" as const, content: "Now 4+4?" },
      { role: "assistant" as const, content: "Eight." },
      { role: "user" as const, content: "And 5+5?" },
      { role: "assistant" as const, content: "Ten." },
      { role: "user" as const, content: "Last one: 6+6?" },
      { role: "assistant" as const, content: "Twelve." },
      { role: "user" as const, content: "Thanks. Now summarise." },
    ];

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    const baseline = upstream.receivedRequests.length;
    const completion = await client.chat.completions.create({
      model: "body-edges",
      messages: history,
    });

    // Caller-side: 200 success with assistant role.
    expect(completion.choices[0]?.message.role).toBe("assistant");

    // Upstream-side: every message reached the upstream with role
    // and content intact. A regression that truncated to the last
    // message (or dropped the system primer) would fail here.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      model?: string;
      messages?: Array<{ role?: string; content?: string }>;
    };
    // Gateway must translate the caller's display name into the
    // upstream-supplied model_name. A regression that forwarded the
    // caller's name to the upstream would 4xx in production
    // (upstream doesn't recognise "body-edges") but pass against a
    // permissive mock — pinning this catches that wire-shape gap.
    expect(sentBody.model).toBe("gpt-4o-mini");
    expect(sentBody.messages).toHaveLength(history.length);
    for (let i = 0; i < history.length; i++) {
      expect(sentBody.messages?.[i]?.role).toBe(history[i]!.role);
      expect(sentBody.messages?.[i]?.content).toBe(history[i]!.content);
    }
  });

  test("empty messages array: 4xx with OpenAI-shape error envelope, upstream untouched", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Path-filtered baseline (matches the convention used across
    // the suite): a regression that triggered an unrelated upstream
    // call would not falsely inflate this counter.
    const upstreamChatHitsBefore = upstream.receivedRequests.filter(
      (r) => r.path === "/v1/chat/completions",
    ).length;

    let caught: unknown;
    try {
      await client.chat.completions.create({
        model: "body-edges",
        // OpenAI Chat Completions spec requires a non-empty
        // messages array. Empty must be rejected at the validation
        // boundary, not bubbled up as a 500 / panic.
        messages: [],
      });
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) {
      throw new Error("unreachable: caught is not APIError");
    }
    // OpenAI Chat Completions request schema declares
    // `messages: minItems: 1` — a schema-violation 400 is the only
    // spec-conformant outcome. 401/403/404/422 here would all
    // signal a different bug (auth ordering, model resolution,
    // schema choice).
    expect(caught.status).toBe(400);
    // Pin the OpenAI error vocabulary: the gateway is rejecting
    // on OpenAI's behalf, so it must use OpenAI's published value
    // for schema violations rather than a gateway-internal string
    // <https://platform.openai.com/docs/guides/error-codes/api-errors>.
    const err = caught.error as { type?: unknown; message?: unknown };
    expect(err.type).toBe("invalid_request_error");
    expect(typeof err.message).toBe("string");
    expect((err.message as string).length).toBeGreaterThan(0);

    // Validation must short-circuit before dispatch.
    const upstreamChatHitsAfter = upstream.receivedRequests.filter(
      (r) => r.path === "/v1/chat/completions",
    ).length;
    expect(upstreamChatHitsAfter).toBe(upstreamChatHitsBefore);
  });
});
