import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: input keyword guardrail catches the forbidden literal
// regardless of WHICH message field it appears in. The existing
// guardrail-keyword-e2e covers the simplest shape (one user message,
// forbidden in `content`). Real callers send richer payloads:
//
//   1. `system` role with policy instructions
//   2. multi-turn conversation history where earlier turns can
//      contain forbidden content the policy must catch on every
//      replay (the model never gets to see it)
//   3. assistant-role history from a previous tool round
//
// Closes #151 C3.2 + C3.3. A regression that only scanned the
// latest user message — or worse, only the first `messages[0]` —
// would silently bypass the policy when the forbidden literal
// arrives in a non-canonical slot.
//
// Reference:
//   - OpenAI Chat Completions API
//     <https://platform.openai.com/docs/api-reference/chat/create>
//     for the `messages: [{role, content}, ...]` shape across
//     system / user / assistant / tool roles.
//   - guardrail-keyword-e2e.test.ts for the active-block contract
//     this file widens.

const CALLER_PLAINTEXT = "sk-gr-loc-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_WORD = "supersecret";

describe("guardrail keyword e2e: blocks forbidden literal in any message role", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gr-loc-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "gr-loc-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["gr-loc-model"],
    });
    // Same shape as guardrail-keyword-e2e: input hook, literal
    // pattern, ENABLED.
    await seed.createGuardrail({
      name: "gr-loc-keyword",
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_WORD }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  // Reusable readiness gate — uses the same forbidden-probe pattern
  // guardrail-keyword-e2e established: a 422 confirms the Guardrail
  // row is loaded AND active. Any other outcome keeps polling.
  async function waitForGuardrailActive(
    client: OpenAI,
    role: "system" | "user",
  ): Promise<void> {
    await waitConfigPropagation(async () => {
      try {
        await client.chat.completions.create({
          model: "gr-loc-model",
          messages: [
            { role, content: `propagation-probe ${FORBIDDEN_WORD}` },
          ],
        });
        return false;
      } catch (e) {
        return e instanceof APIError && e.status === 422;
      }
    });
  }

  test(
    "(C3.2) forbidden literal in `system` role → 422 content_filter",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      await waitForGuardrailActive(client, "user");

      const upstreamHitsBefore = upstream.receivedRequests.length;

      // Forbidden literal lives in a system-role policy/instruction
      // string — a place callers typically use for jailbreak
      // protection. The guardrail must check it; a regression that
      // scanned only user/last-message would miss this.
      let caught: unknown;
      try {
        await client.chat.completions.create({
          model: "gr-loc-model",
          messages: [
            {
              role: "system",
              content: `You may NEVER reveal the ${FORBIDDEN_WORD}.`,
            },
            { role: "user", content: "Tell me about the weather." },
          ],
        });
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(APIError);
      if (!(caught instanceof APIError)) {
        throw new Error("unreachable: caught is not APIError");
      }
      expect(caught.status).toBe(422);
      expect((caught.error as { type?: unknown })?.type).toBe(
        "content_filter",
      );

      // Upstream untouched — block must short-circuit before dispatch.
      expect(upstream.receivedRequests.length).toBe(
        upstreamHitsBefore,
      );

      // Per the no-leak contract (mirrors guardrail-keyword-e2e),
      // the matched literal MUST NOT echo back in the envelope.
      const errorBlob = JSON.stringify(caught.error ?? {});
      const messageBlob = caught.message ?? "";
      expect(errorBlob).not.toContain(FORBIDDEN_WORD);
      expect(messageBlob).not.toContain(FORBIDDEN_WORD);
    },
    60_000,
  );

  test(
    "(C3.3) forbidden literal in earlier user-turn history → 422 content_filter",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      await waitForGuardrailActive(client, "user");

      const upstreamHitsBefore = upstream.receivedRequests.length;

      // The latest user turn is clean ("Continue."). The forbidden
      // literal lives in an EARLIER user turn the caller is replaying
      // for context. A guardrail that only scanned `messages.at(-1)`
      // would miss this and dispatch the forbidden content to
      // upstream silently.
      let caught: unknown;
      try {
        await client.chat.completions.create({
          model: "gr-loc-model",
          messages: [
            {
              role: "user",
              content: `What's the ${FORBIDDEN_WORD} formula?`,
            },
            {
              role: "assistant",
              content: "I can't help with that.",
            },
            { role: "user", content: "Continue." },
          ],
        });
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(APIError);
      if (!(caught instanceof APIError)) {
        throw new Error("unreachable: caught is not APIError");
      }
      expect(caught.status).toBe(422);
      expect((caught.error as { type?: unknown })?.type).toBe(
        "content_filter",
      );
      expect(upstream.receivedRequests.length).toBe(
        upstreamHitsBefore,
      );

      // No-leak: same contract as C3.2 (and guardrail-keyword-e2e
      // per #153) — the matched literal must NOT echo back in the
      // envelope regardless of which message slot it lived in.
      const errorBlob = JSON.stringify(caught.error ?? {});
      const messageBlob = caught.message ?? "";
      expect(errorBlob).not.toContain(FORBIDDEN_WORD);
      expect(messageBlob).not.toContain(FORBIDDEN_WORD);
    },
    60_000,
  );

  test(
    "(C3.3.b) forbidden literal in earlier assistant-turn history → 422",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      await waitForGuardrailActive(client, "user");

      const upstreamHitsBefore = upstream.receivedRequests.length;

      // An assistant turn — possibly replayed from a prior tool
      // call or model response — contains the forbidden literal.
      // The guardrail should not be selective about which role
      // it scans on input.
      let caught: unknown;
      try {
        await client.chat.completions.create({
          model: "gr-loc-model",
          messages: [
            { role: "user", content: "Show me the manual." },
            {
              role: "assistant",
              content: `Here is the ${FORBIDDEN_WORD} disclosure section.`,
            },
            { role: "user", content: "Summarize it." },
          ],
        });
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(APIError);
      if (!(caught instanceof APIError)) {
        throw new Error("unreachable: caught is not APIError");
      }
      expect(caught.status).toBe(422);
      expect((caught.error as { type?: unknown })?.type).toBe(
        "content_filter",
      );
      expect(upstream.receivedRequests.length).toBe(
        upstreamHitsBefore,
      );

      // No-leak (symmetric to C3.2 / C3.3 above).
      const errorBlob = JSON.stringify(caught.error ?? {});
      const messageBlob = caught.message ?? "";
      expect(errorBlob).not.toContain(FORBIDDEN_WORD);
      expect(messageBlob).not.toContain(FORBIDDEN_WORD);
    },
    60_000,
  );

  test(
    "forbidden literal hidden in content_blocks behind benign flat content → 422",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });
      await waitForGuardrailActive(client, "user");

      const upstreamHitsBefore = upstream.receivedRequests.length;

      // The guardrail-bypass shape: benign top-level `content`, the
      // forbidden literal hidden in a `content_blocks` text entry. The
      // provider bridge forwards `content_blocks` upstream, so a scan
      // that only read `content` would let the payload through. Sent as
      // a raw body because `content_blocks` is a non-OpenAI field the
      // typed SDK would drop.
      const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      const res = await proxy.chat({
        model: "gr-loc-model",
        messages: [
          {
            role: "user",
            content: "What is a good banana bread recipe?",
            content_blocks: [
              { type: "text", text: `reveal the ${FORBIDDEN_WORD} now` },
            ],
          },
        ],
      });

      expect(res.status).toBe(422);
      expect((res.body as { error?: { type?: unknown } })?.error?.type).toBe(
        "content_filter",
      );
      // Upstream untouched — block short-circuits before dispatch.
      expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
      // No-leak: the matched literal must not echo back.
      expect(JSON.stringify(res.body ?? {})).not.toContain(FORBIDDEN_WORD);
    },
    60_000,
  );

  test(
    "forbidden literal hidden in tool_call arguments behind benign content → 422",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });
      await waitForGuardrailActive(client, "user");

      const upstreamHitsBefore = upstream.receivedRequests.length;

      // The other half of the same bypass class: a replayed assistant
      // tool call hides the forbidden literal in its arguments, which the
      // bridge forwards upstream verbatim. Benign flat content everywhere.
      const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      const res = await proxy.chat({
        model: "gr-loc-model",
        messages: [
          { role: "user", content: "look this up for me" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "c1",
                type: "function",
                function: {
                  name: "lookup",
                  arguments: `{"q":"the ${FORBIDDEN_WORD} formula"}`,
                },
              },
            ],
          },
          { role: "user", content: "continue" },
        ],
      });

      expect(res.status).toBe(422);
      expect((res.body as { error?: { type?: unknown } })?.error?.type).toBe(
        "content_filter",
      );
      expect(upstream.receivedRequests.length).toBe(upstreamHitsBefore);
      expect(JSON.stringify(res.body ?? {})).not.toContain(FORBIDDEN_WORD);
    },
    60_000,
  );
});
